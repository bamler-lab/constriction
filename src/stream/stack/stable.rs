use alloc::vec::Vec;

use core::{
    borrow::Borrow,
    convert::{Infallible, TryFrom},
    ops::{Deref, DerefMut},
};

use num::cast::AsPrimitive;

use super::{
    super::models::{DecoderModel, EncoderModel},
    AnsCoder, Code, Decode, Encode, EncoderError, TryCodingError,
};
use crate::{BitArray, CoderError, EncoderFrontendError, NonZeroBitArray, UnwrapInfallible};

#[derive(Debug, Clone)]
struct Coder<Word, State, const PRECISION: usize>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    /// The supply of bits.
    ///
    /// Satisfies the normal invariant of a `AnsCoder`.
    supply: AnsCoder<Word, State>,

    /// Remaining information not used up by decoded symbols.
    ///
    /// Satisfies different invariants than a usual `AnsCoder`:
    /// - `waste.state() >= State::one() << (State::BITS - PRECISION - Word::BITS)`
    ///   unless `waste.buf().is_empty()`; and
    /// - `waste.state() < State::one() << (State::BITS - PRECISION)`
    waste: AnsCoder<Word, State>,
}

#[derive(Debug, Clone)]
pub struct Encoder<Word, State, const PRECISION: usize>(
    Coder<Word, State, PRECISION>,
)
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>;

#[derive(Debug, Clone)]
pub struct Decoder<Word, State, const PRECISION: usize>(
    Coder<Word, State, PRECISION>,
)
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>;

impl<Word, State, const PRECISION: usize> Decoder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn new(
        ans: AnsCoder<Word, State>,
    ) -> Result<Self, Option<AnsCoder<Word, State>>> {
        assert!(Word::BITS > 0);
        assert!(State::BITS >= 2 * Word::BITS);
        assert!(State::BITS % Word::BITS == 0);
        assert!(PRECISION <= Word::BITS);
        assert!(PRECISION > 0);

        // The following also assures that `ans.buf()` is no-empty since
        // `State::BITS / Word::BITS - 1 >= 1`.
        if ans.bulk().len() < State::BITS / Word::BITS - 1 {
            // Not enough data to initialize `supply` and `waste`.
            return Err(Some(ans));
        }

        let (buf, supply_state) = ans.into_raw_parts();
        let prefix = AnsCoder::from_binary(buf).map_err(|_| None)?;
        let (supply_buf, waste_state) = prefix.into_raw_parts();

        let supply = unsafe {
            // SAFETY: `ans` had non-empty `buf`, so its state satisfies invariant.
            AnsCoder::from_raw_parts(supply_buf, supply_state)
        };

        let mut waste = AnsCoder::with_state_and_empty_bulk(waste_state);
        if PRECISION == Word::BITS {
            waste.flush_state();
        }

        Ok(Self(Coder { supply, waste }))
    }

    /// Converts the `stable::Decoder` into a new `stable::Decoder` that accepts entropy
    /// models with a different fixed-point precision.
    ///
    /// Here, "precision" refers to the number of bits with which probabilities are
    /// represented in entropy models passed to the `decode_XXX` methods.
    ///
    /// The generic argument `NEW_PRECISION` can usually be omitted because the compiler
    /// can infer its value from the first time the new `stable::Decoder` is used for
    /// decoding. The recommended usage pattern is to store the returned
    /// `stable::Decoder` in a variable that shadows the old `stable::Decoder` (since
    /// the old one gets consumed anyway), i.e.,
    /// `let mut stable_decoder = stable_decoder.change_precision()`. See example below.
    ///
    /// # Failure Case
    ///
    /// The conversion can only fail if *all* of the following conditions are true
    ///
    /// - `NEW_PRECISION < PRECISION`; and
    /// - the `stable::Decoder` originates from a [`stable::Encoder`] that was converted
    ///   with [`into_decoder`]; and
    /// - before calling `into_decoder`, the `stable::Encoder` was used incorrectly: it
    ///   must have encoded too many symbols or used the wrong sequence of entropy
    ///   models, causing it to use up just a few more bits of `waste` than available
    ///   (but also not exceeding the capacity enough for this to be detected during
    ///   encoding).
    ///
    /// In the event of this failure, `change_precision` returns `Err(self)`.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{models::LeakyQuantizer, Decode, stack::DefaultAnsCoder};
    ///
    /// // Construct two entropy models with 24 bits and 20 bits of precision, respectively.
    /// let continuous_distribution = statrs::distribution::Normal::new(0.0, 10.0).unwrap();
    /// let quantizer24 = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let quantizer20 = LeakyQuantizer::<_, _, u32, 20>::new(-100..=100);
    /// let distribution24 = quantizer24.quantize(continuous_distribution);
    /// let distribution20 = quantizer20.quantize(continuous_distribution);
    ///
    /// // Construct a `stable::Decoder` and decode some data with the 24 bit precision entropy model.
    /// let data = vec![0x0123_4567u32, 0x89ab_cdef];
    /// let mut stable_decoder = DefaultAnsCoder::from_binary(data).unwrap().into_stable_decoder().unwrap();
    /// let _symbol_a = stable_decoder.decode_symbol(distribution24);
    ///
    /// // Change the `Decoder`'s precision and decode data with the 20 bit precision entropy model.
    /// // The compiler can infer the new precision based on how the `stable_decoder` will be used.
    /// let mut stable_decoder = stable_decoder.change_precision().unwrap();
    /// let _symbol_b = stable_decoder.decode_symbol(distribution20);
    /// ```
    ///
    /// [`stable::Encoder`]: Encoder
    /// [`into_decoder`]: Encoder::into_decoder
    pub fn change_precision<const NEW_PRECISION: usize>(
        self,
    ) -> Result<
        Decoder<Word, State, NEW_PRECISION>,
        Decoder<Word, State, PRECISION>,
    > {
        match self.0.change_precision() {
            Ok(coder) => Ok(Decoder(coder)),
            Err(coder) => Err(Decoder(coder)),
        }
    }

    /// Converts the `stable::Decoder` into a [`stable::Encoder`].
    ///
    /// This is a no-op since `stable::Decoder` and [`stable::Encoder`] use the same
    /// internal representation with the same invariants. Therefore, we could in
    /// principle have merged the two into a single type. However, keeping them as
    /// separate types makes the API more clear and prevents misuse since the conversion
    /// to and from a [`AnsCoder`] is different for `stable::Decoder` and
    /// `stable::Encoder`.
    ///
    /// [`stable::Encoder`]: Encoder
    pub fn into_encoder(self) -> Encoder<Word, State, PRECISION> {
        Encoder(self.0)
    }

    pub fn finish(self) -> (Vec<Word>, AnsCoder<Word, State>) {
        let (prefix, supply_state) = self.0.supply.into_raw_parts();
        let suffix = unsafe {
            // SAFETY: `stable::Decoder` always reserves enough `supply.state`.
            AnsCoder::from_raw_parts(
                self.0.waste.into_compressed().unwrap_infallible(),
                supply_state,
            )
        };

        (prefix, suffix)
    }

    pub fn finish_and_concatenate(self) -> AnsCoder<Word, State> {
        unsafe {
            // UNSAFE: `stable::Decoder` always reserves enough `supply.state`.
            concatenate(self.finish())
        }
    }

    pub fn supply(&self) -> &AnsCoder<Word, State> {
        self.0.supply()
    }

    pub fn supply_mut(&mut self) -> &mut AnsCoder<Word, State> {
        self.0.supply_mut()
    }

    pub fn waste_mut<'a>(
        &'a mut self,
    ) -> impl DerefMut<Target = AnsCoder<Word, State>> + Drop + 'a {
        self.0.waste_mut()
    }

    pub fn into_supply_and_waste(
        self,
    ) -> (
        AnsCoder<Word, State>,
        AnsCoder<Word, State>,
    ) {
        // `self.waste` satisfies slightly different invariants than a usual `AnsCoder`.
        // We therefore first restore the usual `AnsCoder` invariant.
        self.0.into_supply_and_waste()
    }
}

impl<Word, State, const PRECISION: usize> TryFrom<AnsCoder<Word, State>>
    for Decoder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Error = Option<AnsCoder<Word, State>>;

    fn try_from(
        ans: AnsCoder<Word, State>,
    ) -> Result<Self, Option<AnsCoder<Word, State>>> {
        Self::new(ans)
    }
}

/// TODO: check if this can be made generic over the backend
impl<Word, State, const PRECISION: usize> From<Decoder<Word, State, PRECISION>>
    for AnsCoder<Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn from(decoder: Decoder<Word, State, PRECISION>) -> Self {
        decoder.finish_and_concatenate()
    }
}

impl<Word, State, const PRECISION: usize> Encoder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn new(ans: AnsCoder<Word, State>) -> Result<Self, AnsCoder<Word, State>> {
        assert!(Word::BITS > 0);
        assert!(State::BITS >= 2 * Word::BITS);
        assert!(State::BITS % Word::BITS == 0);
        assert!(PRECISION <= Word::BITS);
        assert!(PRECISION > 0);

        // The following also assures that `ans.buf()` is no-empty since
        // `State::BITS / Word::BITS - 1 >= 1`.
        if ans.bulk().len() < State::BITS / Word::BITS - 1 {
            // Not enough data to initialize `supply` and `waste`.
            return Err(ans);
        }

        if ans.bulk.last() == Some(&Word::zero()) {
            // Invalid data that couldn't have been produced by `stable::Decoder::finish()`.
            return Err(ans);
        }

        let (buf, supply_state) = ans.into_raw_parts();
        let supply = AnsCoder::with_state_and_empty_bulk(supply_state);
        let mut waste = AnsCoder::from_compressed(buf)
            .expect("We verified above that `buf` doesn't end in zero word");
        if waste.state() >= State::one() << (State::BITS - PRECISION) {
            waste.flush_state();
        }

        Ok(Self(Coder { supply, waste }))
    }

    /// Stable variant of [`AnsCoder::encode_symbols_reverse`].
    pub fn encode_symbols_reverse<S, D, I>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<(), EncoderError<WriteError>>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Word>,
        Word: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = (S, D)>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_symbols(symbols_and_models.into_iter().rev())
    }

    /// Stable variant of [`AnsCoder::try_encode_symbols_reverse`].
    pub fn try_encode_symbols_reverse<S, D, E, I>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<(), TryCodingError<EncoderError<WriteError>, E>>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Word>,
        Word: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = core::result::Result<(S, D), E>>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.try_encode_symbols(symbols_and_models.into_iter().rev())
    }

    /// Stable variant of [`AnsCoder::encode_iid_symbols_reverse`].
    pub fn encode_iid_symbols_reverse<S, D, I>(
        &mut self,
        symbols: I,
        model: &D,
    ) -> Result<(), EncoderError<WriteError>>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Word>,
        Word: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = S>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_iid_symbols(symbols.into_iter().rev(), model)
    }

    /// Converts the `stable::Encoder` into a new `stable::Encoder` that accepts entropy
    /// models with a different fixed-point precision.
    ///
    /// # Failure Case
    ///
    /// The conversion can only fail if *both* of the following two conditions are true
    ///
    /// - `NEW_PRECISION < PRECISION`; and
    /// - the `stable::Encoder` has been used incorrectly: it must have encoded too many
    ///   symbols or used the wrong sequence of entropy models, causing it to use up
    ///   just a few more bits of `waste` than available (but also not exceeding the
    ///   capacity enough for this to be detected during encoding).
    ///
    /// # See Also
    ///
    /// This method is analogous to [`stable::Decoder::change_precision`]. See its
    /// documentation for details and an example.
    ///
    /// [`stable::Decoder::change_precision`]: Decoder::change_precision
    pub fn change_precision<const NEW_PRECISION: usize>(
        self,
    ) -> Result<
        Encoder<Word, State, NEW_PRECISION>,
        Encoder<Word, State, PRECISION>,
    > {
        match self.0.change_precision() {
            Ok(coder) => Ok(Encoder(coder)),
            Err(coder) => Err(Encoder(coder)),
        }
    }

    /// Converts the `stable::Encoder` into a [`stable::Decoder`].
    ///
    /// This is a no-op since `stable::Encoder` and [`stable::Decoder`] use the same
    /// internal representation with the same invariants. Therefore, we could in
    /// principle have merged the two into a single type. However, keeping them as
    /// separate types makes the API more clear and prevents misuse since the conversion
    /// to and from a [`AnsCoder`] is different for `stable::Encoder` and
    /// `stable::Decoder`.
    ///
    /// [`stable::Decoder`]: Decoder
    pub fn into_decoder(self) -> Decoder<Word, State, PRECISION> {
        Decoder(self.0)
    }

    pub fn finish(
        mut self,
    ) -> Result<(Vec<Word>, AnsCoder<Word, State>), Self> {
        if PRECISION == Word::BITS {
            if self.0.waste.state() >> (State::BITS - 2 * Word::BITS) != State::one()
                || self.0.waste.refill_state_maybe_truncating().is_err()
            {
                // Waste's state (without leading 1 bit) doesn't fit into integer number of
                // `Word`s, or there is not enough data on `waste`.
                return Err(self);
            }
        } else {
            if self.0.waste.state() >> (State::BITS - Word::BITS) != State::one() {
                // Waste's state (without leading 1 bit) doesn't fit into integer number of
                // `Word`s, or there is not enough data on `waste`.
                return Err(self);
            }
        }

        let (mut buf, state) = self.0.supply.into_raw_parts();
        let (prefix, mut waste_state) = self.0.waste.into_raw_parts();

        while waste_state != State::one() {
            buf.push(waste_state.as_());
            waste_state = waste_state >> Word::BITS;
        }

        let suffix = unsafe {
            // SAFETY: `stable::Encoder` always reserves enough `supply.state`.
            AnsCoder::from_raw_parts(buf, state)
        };

        Ok((prefix, suffix))
    }

    pub fn finish_and_concatenate(self) -> Result<AnsCoder<Word, State>, Self> {
        unsafe {
            // UNSAFE: `stable::Encoder` always reserves enough `supply.state`.
            Ok(concatenate(self.finish()?))
        }
    }

    pub fn supply(&self) -> &AnsCoder<Word, State> {
        self.0.supply()
    }

    pub fn supply_mut(&mut self) -> &mut AnsCoder<Word, State> {
        self.0.supply_mut()
    }

    pub fn waste_mut<'a>(
        &'a mut self,
    ) -> impl DerefMut<Target = AnsCoder<Word, State>> + Drop + 'a {
        self.0.waste_mut()
    }

    pub fn into_supply_and_waste(
        self,
    ) -> (
        AnsCoder<Word, State>,
        AnsCoder<Word, State>,
    ) {
        // `self.waste` satisfies slightly different invariants than a usual `AnsCoder`.
        // We therefore first restore the usual `AnsCoder` invariant.
        self.0.into_supply_and_waste()
    }
}

impl<Word, State, const PRECISION: usize> TryFrom<AnsCoder<Word, State>>
    for Encoder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Error = AnsCoder<Word, State>;

    fn try_from(
        ans: AnsCoder<Word, State>,
    ) -> Result<Self, AnsCoder<Word, State>> {
        Self::new(ans)
    }
}

/// TODO: check if this can be made generic over the backend
impl<Word, State, const PRECISION: usize>
    TryFrom<Encoder<Word, State, PRECISION>> for AnsCoder<Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Error = Encoder<Word, State, PRECISION>;

    fn try_from(
        decoder: Encoder<Word, State, PRECISION>,
    ) -> Result<Self, Encoder<Word, State, PRECISION>> {
        decoder.finish_and_concatenate()
    }
}

impl<Word, State, const PRECISION: usize> Coder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    pub fn change_precision<const NEW_PRECISION: usize>(
        mut self,
    ) -> Result<Coder<Word, State, NEW_PRECISION>, Coder<Word, State, PRECISION>>
    {
        assert!(NEW_PRECISION <= Word::BITS);

        if NEW_PRECISION > PRECISION {
            if self.waste.state() >= State::one() << (State::BITS - NEW_PRECISION) {
                if self.waste.flush_state().is_err() {
                    return Err(self);
                }
            }
        } else if NEW_PRECISION < PRECISION {
            if self.waste.state()
                < State::one() << (State::BITS - NEW_PRECISION - Word::BITS)
            {
                if self.waste.refill_state_maybe_truncating().is_err() {
                    return Err(self);
                }
            }
        }

        Ok(Coder {
            supply: self.supply,
            waste: self.waste,
        })
    }

    pub fn supply(&self) -> &AnsCoder<Word, State> {
        &self.supply
    }

    pub fn supply_mut(&mut self) -> &mut AnsCoder<Word, State> {
        &mut self.supply
    }

    pub fn waste_mut<'a>(
        &'a mut self,
    ) -> impl DerefMut<Target = AnsCoder<Word, State>> + Drop + 'a {
        WasteGuard::<'a, _, _, PRECISION>::new(&mut self.waste)
    }

    pub fn into_supply_and_waste(
        mut self,
    ) -> (
        AnsCoder<Word, State>,
        AnsCoder<Word, State>,
    ) {
        // `self.waste` satisfies slightly different invariants than a usual `AnsCoder`.
        // We therefore first restore the usual `AnsCoder` invariant.
        let _ = self.waste.try_refill_state_if_necessary();

        (self.supply, self.waste)
    }
}

#[derive(Debug, Clone)]
pub struct CoderState<State> {
    /// Invariant: `supply >= State::one() << (State::BITS - Word::BITS)`
    pub supply: State,

    /// Invariants:
    /// - `waste >= State::one() << (State::BITS - PRECISION - Word::BITS)`
    /// - `waste < State::one() << (State::BITS - PRECISION)`
    pub waste: State,
}

impl<Word, State, const PRECISION: usize> Code for Coder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Word = Word;
    type State = CoderState<State>;

    fn state(&self) -> Self::State {
        CoderState {
            supply: self.supply.state(),
            waste: self.waste.state(),
        }
    }

    fn maybe_empty(&self) -> bool {
        // `self.supply.state()` must always be above threshold if we still want to call
        // `finish_decoding`, so we only check if `supply.buf` is empty here.
        self.supply.bulk().is_empty()
    }
}

impl<Word, State, const PRECISION: usize> Code
    for Encoder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Word = Word;
    type State = CoderState<State>;

    fn state(&self) -> Self::State {
        self.0.state()
    }

    fn maybe_empty(&self) -> bool {
        self.0.maybe_empty()
    }
}

impl<Word, State, const PRECISION: usize> Code
    for Decoder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Word = Word;
    type State = CoderState<State>;

    fn state(&self) -> Self::State {
        self.0.state()
    }

    fn maybe_empty(&self) -> bool {
        self.0.maybe_empty()
    }
}

/// Error type for a [`stable::Decoder`].
///
/// [`stable::Decoder`]: Decoder
#[derive(Debug)]
pub enum FrontendError {
    /// Not enough binary data available to decode the symbol.
    ///
    /// Note that a [`stable::Decoder<State, Word, PRECISION>`]
    /// - consumes `PRECISION` bits of compressed data for each decoded symbol, even for
    ///   symbols with low information content under the used entropy model (the
    ///   superfluous information content will be appended to the `stable::Decoder`'s
    ///   [`waste`]); and it
    /// - retains up to `2 * State::BITS - Word::BITS` bits of compressed data
    ///   that it won't decode into any symbols (these bits are needed to initialize the
    ///   `stable::Decoder`'s `waste`, and for proper "sealing" of the binary data in
    ///   [`stable::Decoder::finish`]).
    ///
    /// Thus, when you want to decode `n` symbols with a [`stable::Decoder`], you should
    /// construct it from a [`AnsCoder`] where [`num_valid_bits`] reports at least
    /// `n * PRECISION + 2 * State::BITS - Word::BITS`.
    ///
    /// [`stable::Decoder<State, Word, PRECISION>`]: Decoder
    /// [`stable::Decoder`]: Decoder
    /// [`waste`]: Decoder::waste
    /// [`stable::Decoder::finish`]: Decoder::finish
    /// [`num_valid_bits`]: AnsCoder::num_valid_bits
    OutOfData,
}

impl core::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfData => {
                write!(f, "Out of binary data.")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FrontendError {}

#[derive(Debug)]
pub enum WriteError {
    CapacityExceeded,
}

impl core::fmt::Display for WriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CapacityExceeded => {
                write!(f, "Capacity exceeded.")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WriteError {}

impl<Word, State, const PRECISION: usize> Decode<PRECISION>
    for Decoder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type FrontendError = FrontendError;

    /// TODO: this should be an enum that reports errors in either of the backends.
    type BackendError = Infallible;

    fn decode_symbol<D>(
        &mut self,
        model: D,
    ) -> Result<D::Symbol, CoderError<Self::FrontendError, Self::BackendError>>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<D::Probability>,
    {
        let quantile = self.0.supply.chop_quantile_off_state::<D, PRECISION>();
        if self.0.supply.try_refill_state_if_necessary().is_err() {
            // Restore original state and return an error.
            self.0
                .supply
                .append_quantile_to_state::<D, PRECISION>(quantile);
            return Err(CoderError::FrontendError(FrontendError::OutOfData));
        }

        let (symbol, left_sided_cumulative, probability) = model.quantile_function(quantile);
        let remainder = quantile - left_sided_cumulative;

        self.0
            .waste
            .encode_remainder_onto_state::<D, PRECISION>(remainder, probability);

        if self.0.waste.state() >= State::one() << (State::BITS - PRECISION) {
            // The invariant on `self.0.waste.state` (see its doc comment) is violated and must
            // be restored:
            self.0.waste.flush_state();
        }

        Ok(symbol)
    }
}

impl<Word, State, const PRECISION: usize> Encode<PRECISION>
    for Encoder<Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type BackendError = WriteError;

    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        model: D,
    ) -> Result<(), EncoderError<WriteError>>
    where
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::Word>,
        Word: AsPrimitive<D::Probability>,
    {
        let (left_sided_cumulative, probability) = model
            .left_cumulative_and_probability(symbol)
            .map_err(|()| EncoderFrontendError::ImpossibleSymbol.into_encoder_error())?;

        if self.0.waste.state()
            < probability.get().into().into() << (State::BITS - Word::BITS - PRECISION)
        {
            self.0
                .waste
                .refill_state_maybe_truncating()
                .map_err(|_| EncoderError::BackendError(WriteError::CapacityExceeded))?;
            // At this point, the invariant on `self.0.waste` (see its doc comment) is
            // temporarily violated (but it will be restored below). This is how
            // `decode_symbol` can detect that it has to flush `waste.state`.
        }

        // TODO: not sure if we're returning the right error here. Why are there two places
        // in this function that can return a `CapacityExceeded` error?
        let remainder = self
            .0
            .waste
            .decode_remainder_off_state::<D, PRECISION>(probability);

        if (self.0.supply.state() >> (State::BITS - PRECISION)) != State::zero() {
            self.0.supply.flush_state();
        }
        self.0
            .supply
            .append_quantile_to_state::<D, PRECISION>(left_sided_cumulative + remainder);

        Ok(())
    }
}

struct WasteGuard<'a, Word, State, const PRECISION: usize>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    waste: &'a mut AnsCoder<Word, State>,
}

impl<'a, Word, State, const PRECISION: usize>
    WasteGuard<'a, Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn new(waste: &'a mut AnsCoder<Word, State>) -> Self {
        // `stable::Coder::waste` satisfies slightly different invariants than a usual
        // `AnsCoder`. We therefore restore the usual `AnsCoder` invariant here. This is reversed
        // when the `WasteGuard` gets dropped.
        let _ = waste.try_refill_state_if_necessary();

        Self { waste }
    }
}

impl<'a, Word, State, const PRECISION: usize> Deref
    for WasteGuard<'a, Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Target = AnsCoder<Word, State>;

    fn deref(&self) -> &Self::Target {
        self.waste
    }
}

impl<'a, Word, State, const PRECISION: usize> DerefMut
    for WasteGuard<'a, Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.waste
    }
}

impl<'a, Word, State, const PRECISION: usize> Drop
    for WasteGuard<'a, Word, State, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn drop(&mut self) {
        // Reverse the mutation done in `CoderGuard::new` to restore `stable::Coder`'s
        // special invariants for `waste`.
        if self.waste.state() >= State::one() << (State::BITS - PRECISION) {
            self.waste.flush_state();
        }
    }
}

unsafe fn concatenate<Word, State>(
    (mut prefix, suffix): (Vec<Word>, AnsCoder<Word, State>),
) -> AnsCoder<Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    let (suffix_buf, state) = suffix.into_raw_parts();

    let buf = if prefix.is_empty() {
        // Avoid copying in this not-so-unlikely special case.
        suffix_buf
    } else {
        prefix.extend_from_slice(&suffix_buf);
        prefix
    };

    AnsCoder::from_raw_parts(buf, state)
}

#[cfg(test)]
mod test {
    use super::super::super::models::LeakyQuantizer;
    use super::*;

    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };
    use statrs::distribution::Normal;

    #[test]
    fn restore_none() {
        generic_restore_many::<u32, u64, u32, 24>(4, 0);
    }

    #[test]
    fn restore_one() {
        generic_restore_many::<u32, u64, u32, 24>(5, 1);
    }

    #[test]
    fn restore_two() {
        generic_restore_many::<u32, u64, u32, 24>(5, 2);
    }

    #[test]
    fn restore_ten() {
        generic_restore_many::<u32, u64, u32, 24>(20, 10);
    }

    #[test]
    fn restore_twenty() {
        generic_restore_many::<u32, u64, u32, 24>(19, 20);
    }

    #[test]
    fn restore_many_u32_u64_32() {
        generic_restore_many::<u32, u64, u32, 32>(1024, 1000);
    }

    #[test]
    fn restore_many_u32_u64_24() {
        generic_restore_many::<u32, u64, u32, 24>(1024, 1000);
    }

    #[test]
    fn restore_many_u32_u64_16() {
        generic_restore_many::<u32, u64, u16, 16>(1024, 1000);
    }

    #[test]
    fn restore_many_u16_u64_16() {
        generic_restore_many::<u16, u64, u16, 16>(1024, 1000);
    }

    #[test]
    fn restore_many_u32_u64_8() {
        generic_restore_many::<u32, u64, u8, 8>(1024, 1000);
    }

    #[test]
    fn restore_many_u16_u64_8() {
        generic_restore_many::<u16, u64, u8, 8>(1024, 1000);
    }

    #[test]
    fn restore_many_u8_u64_8() {
        generic_restore_many::<u8, u64, u8, 8>(1024, 1000);
    }

    #[test]
    fn restore_many_u16_u32_16() {
        generic_restore_many::<u16, u32, u16, 16>(1024, 1000);
    }

    #[test]
    fn restore_many_u16_u32_8() {
        generic_restore_many::<u16, u32, u8, 8>(1024, 1000);
    }

    #[test]
    fn restore_many_u8_u32_8() {
        generic_restore_many::<u8, u32, u8, 8>(1024, 1000);
    }

    fn generic_restore_many<Word, State, Probability, const PRECISION: usize>(
        amt_compressed_words: usize,
        amt_symbols: usize,
    ) where
        State: BitArray + AsPrimitive<Word>,
        Word: BitArray + Into<State> + AsPrimitive<Probability>,
        Probability: BitArray + Into<Word> + AsPrimitive<usize> + Into<f64>,
        u64: AsPrimitive<Word>,
        u32: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
        f64: AsPrimitive<Probability>,
        i32: AsPrimitive<Probability>,
    {
        let mut rng = Xoshiro256StarStar::seed_from_u64(
            (amt_compressed_words as u64).rotate_left(32) ^ amt_symbols as u64,
        );
        let mut compressed = (0..amt_compressed_words)
            .map(|_| rng.next_u64().as_())
            .collect::<Vec<_>>();

        // Make the last compressed word have a random number of leading zero bits so that
        // we test various filling levels.
        let leading_zeros = (rng.next_u32() % (Word::BITS as u32 - 1)) as usize;
        let last_word = compressed.last_mut().unwrap();
        *last_word =
            *last_word | Word::one() << (Word::BITS - leading_zeros - 1);
        *last_word = *last_word & Word::max_value() >> leading_zeros;

        let distributions = (0..amt_symbols)
            .map(|_| {
                let mean = (100.0 / u32::MAX as f64) * rng.next_u32() as f64 - 100.0;
                let std_dev = (10.0 / u32::MAX as f64) * rng.next_u32() as f64 + 0.001;
                Normal::new(mean, std_dev)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-100..=100);

        let ans = AnsCoder::from_compressed(compressed.clone()).unwrap();
        let mut stable_decoder = ans.into_stable_decoder().unwrap();

        let symbols = stable_decoder
            .decode_symbols(
                distributions
                    .iter()
                    .map(|&distribution| quantizer.quantize(distribution)),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(!stable_decoder.maybe_empty());

        // Test two ways to construct a `stable::Encoder`: direct conversion from a
        // `stable::Decoder`, and converting to a `AnsCoder` and then to a `stable::Encoder`.
        let stable_encoder1 = stable_decoder.clone().into_encoder();
        let ans = stable_decoder.finish_and_concatenate();
        let stable_encoder2 = ans.into_stable_encoder().unwrap();

        for mut stable_encoder in alloc::vec![stable_encoder1, stable_encoder2] {
            stable_encoder
                .encode_symbols_reverse(
                    symbols
                        .iter()
                        .zip(&distributions)
                        .map(|(&symbol, &distribution)| (symbol, quantizer.quantize(distribution))),
                )
                .unwrap();

            let ans = stable_encoder.finish_and_concatenate().unwrap();
            assert_eq!(ans.into_compressed().unwrap_infallible(), compressed);
        }
    }
}
