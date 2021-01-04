use std::{
    borrow::Borrow,
    convert::TryFrom,
    error::Error,
    ops::{Deref, DerefMut},
};

use num::cast::AsPrimitive;

use super::Stack;

use crate::{
    distributions::DiscreteDistribution, BitArray, Code, Decode, Encode, EncodingError,
    TryCodingError,
};

#[derive(Debug, Clone)]
struct Coder<CompressedWord, State, const PRECISION: usize>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    /// The supply of bits.
    ///
    /// Satisfies the normal invariant of a `Stack`.
    supply: Stack<CompressedWord, State>,

    /// Remaining information not used up by decoded symbols.
    ///
    /// Satisfies different invariants than a usual `Stack`:
    /// - `waste.state() >= State::one() << (State::BITS - PRECISION - CompressedWord::BITS)`
    ///   unless `waste.buf().is_empty()`; and
    /// - `waste.state() < State::one() << (State::BITS - PRECISION)`
    waste: Stack<CompressedWord, State>,
}

#[derive(Debug, Clone)]
pub struct Encoder<CompressedWord, State, const PRECISION: usize>(
    Coder<CompressedWord, State, PRECISION>,
)
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>;

#[derive(Debug, Clone)]
pub struct Decoder<CompressedWord, State, const PRECISION: usize>(
    Coder<CompressedWord, State, PRECISION>,
)
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>;

impl<CompressedWord, State, const PRECISION: usize> Decoder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    pub fn new(stack: Stack<CompressedWord, State>) -> Result<Self, Stack<CompressedWord, State>> {
        assert!(CompressedWord::BITS > 0);
        assert!(State::BITS >= 2 * CompressedWord::BITS);
        assert!(State::BITS % CompressedWord::BITS == 0);
        assert!(PRECISION <= CompressedWord::BITS);
        assert!(PRECISION > 0);

        if stack.state() < State::one() << (State::BITS - CompressedWord::BITS) {
            // Not enough data to initialize `supply`.
            return Err(stack);
        }

        let mut waste_state = State::one();
        let mut word_iter = stack.buf().iter().rev();
        while waste_state < State::one() << (State::BITS - CompressedWord::BITS - PRECISION) {
            if let Some(&word) = word_iter.next() {
                waste_state = (waste_state << CompressedWord::BITS) | word.into();
            } else {
                // Not enough data to initialize `waste`.
                return Err(stack);
            }
        }

        let remaining_words = word_iter.len();
        let (mut buf, state) = stack.into_buf_and_state();
        buf.resize(remaining_words, CompressedWord::zero());

        let supply = Stack::with_buf_and_state(buf, state)
            .expect("Original `Stack` was valid and we only shrank its `buf`.");
        let waste = Stack::with_state_and_empty_buf(waste_state);

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
    /// # Example
    ///
    /// ```
    /// use constriction::{distributions::LeakyQuantizer, Decode, stack::DefaultStack};
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
    /// let mut stable_decoder = DefaultStack::from_binary(data).into_stable_decoder().unwrap();
    /// let _symbol_a = stable_decoder.decode_symbol(distribution24);
    ///
    /// // Change the `Decoder`'s precision and decode data with the 20 bit precision entropy model.
    /// // The compiler can infer the new precision based on how the `stable_decoder` will be used.
    /// let mut stable_decoder = stable_decoder.change_precision().unwrap();
    /// let _symbol_b = stable_decoder.decode_symbol(distribution20);
    /// ```
    pub fn change_precision<const NEW_PRECISION: usize>(
        self,
    ) -> Result<
        Decoder<CompressedWord, State, NEW_PRECISION>,
        Encoder<CompressedWord, State, PRECISION>,
    > {
        match self.0.change_precision() {
            Ok(coder) => Ok(Decoder(coder)),
            Err(coder) => Err(Encoder(coder)),
        }
    }

    /// Converts the `stable::Decoder` into a [`stable::Encoder`].
    ///
    /// This is a no-op since `stable::Decoder` and [`stable::Encoder`] use the same
    /// internal representation with the same invariants. Therefore, we could in
    /// principle have merged the two into a single type. However, keeping them as
    /// separate types makes the API more clear and prevents misuse since the conversion
    /// to and from a [`Stack`] is different for `stable::Decoder` and
    /// [`stable::Encoder`].
    ///
    /// [`stable::Encoder`]: Encoder
    pub fn into_encoder(self) -> Encoder<CompressedWord, State, PRECISION> {
        Encoder(self.0)
    }

    pub fn finish(self) -> (Vec<CompressedWord>, Stack<CompressedWord, State>) {
        let (prefix, supply_state) = self.0.supply.into_buf_and_state();
        let suffix = Stack::with_buf_and_state(self.0.waste.into_compressed(), supply_state)
            .expect("`stable::Decoder` always reserves enough `supply.state`.");

        (prefix, suffix)
    }

    pub fn finish_and_concatenate(self) -> Stack<CompressedWord, State> {
        let stack = concatenate(self.finish())
            .expect("`stable::Decoder` always reserves enough `supply.state`.");
        stack
    }

    pub fn supply(&self) -> &Stack<CompressedWord, State> {
        self.0.supply()
    }

    pub fn supply_mut(&mut self) -> &mut Stack<CompressedWord, State> {
        self.0.supply_mut()
    }

    /// Get shared access to the leftover information from decoding symbols.
    ///
    /// The returned `Stack` may be in an invalid state for entropy coding since the
    /// internal representation of a `stable::Decoder` enforces slightly different
    /// invariants. This is not an issue, however, since the returned shared reference
    /// doesn't allow mutation, and all non-mutating operations are guaranteed to work
    /// as expected (including `clone`, which ensures that the cloned `Stack` is in a
    /// valid state).
    ///
    /// See [`Encoder::waste`] for details and alternatives.
    pub fn waste(&self) -> &Stack<CompressedWord, State> {
        self.0.waste()
    }

    pub fn waste_mut<'a>(
        &'a mut self,
    ) -> impl DerefMut<Target = Stack<CompressedWord, State>> + Drop + 'a {
        self.0.waste_mut()
    }

    pub fn into_supply_and_waste(
        self,
    ) -> (Stack<CompressedWord, State>, Stack<CompressedWord, State>) {
        // `self.waste` satisfies slightly different invariants than a usual `Stack`.
        // We therefore first restore the usual `Stack` invariant.
        self.0.into_supply_and_waste()
    }
}

impl<CompressedWord, State, const PRECISION: usize> TryFrom<Stack<CompressedWord, State>>
    for Decoder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type Error = Stack<CompressedWord, State>;

    fn try_from(stack: Stack<CompressedWord, State>) -> Result<Self, Stack<CompressedWord, State>> {
        Self::new(stack)
    }
}

impl<CompressedWord, State, const PRECISION: usize> From<Decoder<CompressedWord, State, PRECISION>>
    for Stack<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn from(decoder: Decoder<CompressedWord, State, PRECISION>) -> Self {
        decoder.finish_and_concatenate()
    }
}

impl<CompressedWord, State, const PRECISION: usize> Encoder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    pub fn new(stack: Stack<CompressedWord, State>) -> Result<Self, Stack<CompressedWord, State>> {
        assert!(CompressedWord::BITS > 0);
        assert!(State::BITS >= 2 * CompressedWord::BITS);
        assert!(State::BITS % CompressedWord::BITS == 0);
        assert!(PRECISION <= CompressedWord::BITS);
        assert!(PRECISION > 0);

        let (buf, state) = stack.into_buf_and_state();

        let mut waste = Stack::from_compressed(buf)
            .and_then(|waste| {
                if waste.state() < State::one() << (State::BITS - PRECISION - CompressedWord::BITS)
                {
                    // Not enough data available to initialize `waste`.
                    Err(waste.into_compressed())
                } else {
                    Ok(waste)
                }
            })
            .map_err(|buf| {
                Stack::with_buf_and_state(buf, state)
                    .expect("We're reconstructing the original stack, which was valid.")
            })?;

        // `waste` has to satisfy slightly different invariants than a usual `Stack`.
        // If they're violated then flushing one word is guaranteed to restore them.
        if waste.state() >= State::one() << (State::BITS - PRECISION) {
            waste.flush_state();
            // Now, `waste` satisfies both required invariants:
            // - waste.state() >= State::one() << (State::BITS - PRECISION - CompressedWord::BITS)
            // - waste.state() < State::one() << (State::BITS - CompressedWord::BITS)
            //                 <= State::one() << (State::BITS - PRECISION)
        }

        let supply = Stack::with_state_and_empty_buf(state);

        Ok(Self(Coder { supply, waste }))
    }

    pub fn encode_symbols_reverse<S, D, I>(
        &mut self,
        symbols_and_distributions: I,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = (S, D)>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_symbols(symbols_and_distributions.into_iter().rev())
    }

    pub fn try_encode_symbols_reverse<S, D, E, I>(
        &mut self,
        symbols_and_distributions: I,
    ) -> Result<(), TryCodingError<EncodingError, E>>
    where
        S: Borrow<D::Symbol>,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
        E: Error + 'static,
        I: IntoIterator<Item = std::result::Result<(S, D), E>>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.try_encode_symbols(symbols_and_distributions.into_iter().rev())
    }

    pub fn encode_iid_symbols_reverse<S, D, I>(
        &mut self,
        symbols: I,
        distribution: &D,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = S>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_iid_symbols(symbols.into_iter().rev(), distribution)
    }

    /// Converts the `stable::Encoder` into a new `stable::Encoder` that accepts entropy
    /// models with a different fixed-point precision.
    ///
    /// This method is analogous to [`stable::Decoder::change_precision`]. See its
    /// documentation for details and an example.
    ///
    /// [`stable::Decoder::change_precision`]: Decoder::change_precision
    pub fn change_precision<const NEW_PRECISION: usize>(
        self,
    ) -> Result<
        Encoder<CompressedWord, State, NEW_PRECISION>,
        Decoder<CompressedWord, State, PRECISION>,
    > {
        match self.0.change_precision() {
            Ok(coder) => Ok(Encoder(coder)),
            Err(coder) => Err(Decoder(coder)),
        }
    }

    /// Converts the `stable::Encoder` into a [`stable::Decoder`].
    ///
    /// This is a no-op since `stable::Encoder` and [`stable::Decoder`] use the same
    /// internal representation with the same invariants. Therefore, we could in
    /// principle have merged the two into a single type. However, keeping them as
    /// separate types makes the API more clear and prevents misuse since the conversion
    /// to and from a [`Stack`] is different for `stable::Encoder` and
    /// [`stable::Decoder`].
    ///
    /// [`stable::Decoder`]: Decoder
    pub fn into_decoder(self) -> Decoder<CompressedWord, State, PRECISION> {
        Decoder(self.0)
    }

    pub fn finish(self) -> Result<(Vec<CompressedWord>, Stack<CompressedWord, State>), Self> {
        let waste_state_bits =
            (State::BITS - 1).wrapping_sub(self.0.waste.state().leading_zeros() as usize);
        if waste_state_bits % CompressedWord::BITS != 0 || waste_state_bits == usize::max_value() {
            // Waste's state (without leading 1 bit) must fit into an integer number of words.
            return Err(self);
        }

        let (mut buf, state) = self.0.supply.into_buf_and_state();
        let (prefix, mut waste_state) = self.0.waste.into_buf_and_state();

        while waste_state != State::one() {
            buf.push(waste_state.as_());
            waste_state = waste_state >> CompressedWord::BITS;
        }

        let suffix = Stack::with_buf_and_state(buf, state)
            .expect("`stable::Encoder` always reserves enough `supply.state`.");

        Ok((prefix, suffix))
    }

    pub fn finish_and_concatenate(self) -> Result<Stack<CompressedWord, State>, Self> {
        let stack = concatenate(self.finish()?)
            .expect("`stable::Encoder` always reserves enough `supply.state`.");
        Ok(stack)
    }

    pub fn supply(&self) -> &Stack<CompressedWord, State> {
        self.0.supply()
    }

    pub fn supply_mut(&mut self) -> &mut Stack<CompressedWord, State> {
        self.0.supply_mut()
    }

    /// Get shared access to the leftover leftover information from decoding, which
    /// encoding will "recycle" into chunks of binary data.
    ///
    /// The returned `Stack` may be in an invalid state for entropy coding since the
    /// internal representation of a `stable::Encoder` enforces slightly different
    /// invariants. This is not an issue, however, since the returned shared reference
    /// doesn't allow mutation, and all non-mutating operations are guaranteed to work
    /// as expected (including `clone`, which ensures that the cloned `Stack` is in a
    /// valid state).
    ///
    /// # Example
    ///
    /// This method is mainly useful to read out `waste`s state, e.g., by calling
    /// `stable_encoder.waste().iter_compressed()`. However, if you have mutable access
    /// to or even ownership of the `stable::Encoder`, then it may be better to call
    /// [`waste_mut`] or [`into_supply_and_waste`], respectively, followed by
    /// [`Stack::get_compressed`] as in the example below:
    ///
    /// ```
    /// use constriction::{distributions::LeakyQuantizer, stack::DefaultStack, Decode};
    ///
    /// let data = vec![0x0123_4567u32, 0x89ab_cdef];
    /// let mut stable_encoder = DefaultStack::from_binary(data).into_stable_encoder::<24>().unwrap();
    ///
    /// // Calling `stable_encoder.waste()` only needs shared access to `stable_encoder`.
    /// dbg!(stable_encoder.waste()); // `Debug` implementation calls `.iter_compressed()`.
    ///
    /// // Since we have mutable access to `stable_encoder`, the following is also possible and
    /// // might be slightly more efficient in expectation:
    /// dbg!(stable_encoder.waste_mut().get_compressed()); // Prints the same compressed words as above.
    ///
    /// // If we no longer want to use `stable_encoder` then we can also deconstruct it
    /// // and call `get_compressed` on its constituents:
    /// let (_supply, mut waste) = stable_encoder.into_supply_and_waste();
    /// dbg!(waste.get_compressed()); // Prints the same compressed words as above.
    /// ```
    ///
    /// [`waste_mut`]: #method.waste_mut
    /// [`into_supply_and_waste`]: #method.into_supply_and_waste
    /// [`get_compressed`]: Stack::get_compressed
    pub fn waste(&self) -> &Stack<CompressedWord, State> {
        self.0.waste()
    }

    pub fn waste_mut<'a>(
        &'a mut self,
    ) -> impl DerefMut<Target = Stack<CompressedWord, State>> + Drop + 'a {
        self.0.waste_mut()
    }

    pub fn into_supply_and_waste(
        self,
    ) -> (Stack<CompressedWord, State>, Stack<CompressedWord, State>) {
        // `self.waste` satisfies slightly different invariants than a usual `Stack`.
        // We therefore first restore the usual `Stack` invariant.
        self.0.into_supply_and_waste()
    }
}

impl<CompressedWord, State, const PRECISION: usize> TryFrom<Stack<CompressedWord, State>>
    for Encoder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type Error = Stack<CompressedWord, State>;

    fn try_from(stack: Stack<CompressedWord, State>) -> Result<Self, Stack<CompressedWord, State>> {
        Self::new(stack)
    }
}

impl<CompressedWord, State, const PRECISION: usize>
    TryFrom<Encoder<CompressedWord, State, PRECISION>> for Stack<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type Error = Encoder<CompressedWord, State, PRECISION>;

    fn try_from(
        decoder: Encoder<CompressedWord, State, PRECISION>,
    ) -> Result<Self, Encoder<CompressedWord, State, PRECISION>> {
        decoder.finish_and_concatenate()
    }
}

impl<CompressedWord, State, const PRECISION: usize> Coder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    pub fn change_precision<const NEW_PRECISION: usize>(
        mut self,
    ) -> Result<Coder<CompressedWord, State, NEW_PRECISION>, Coder<CompressedWord, State, PRECISION>>
    {
        assert!(NEW_PRECISION <= CompressedWord::BITS);

        if NEW_PRECISION > PRECISION {
            if self.waste.state() >= State::one() << (State::BITS - NEW_PRECISION) {
                self.waste.flush_state()
            }
        } else if NEW_PRECISION < PRECISION {
            if self.waste.state()
                < State::one() << (State::BITS - NEW_PRECISION - CompressedWord::BITS)
            {
                if self.waste.try_refill_state().is_err() {
                    return Err(self);
                }
            }
        }

        Ok(Coder {
            supply: self.supply,
            waste: self.waste,
        })
    }

    pub fn supply(&self) -> &Stack<CompressedWord, State> {
        &self.supply
    }

    pub fn supply_mut(&mut self) -> &mut Stack<CompressedWord, State> {
        &mut self.supply
    }

    pub fn waste(&self) -> &Stack<CompressedWord, State> {
        &self.waste
    }

    pub fn waste_mut<'a>(
        &'a mut self,
    ) -> impl DerefMut<Target = Stack<CompressedWord, State>> + Drop + 'a {
        WasteGuard::<'a, _, _, PRECISION>::new(&mut self.waste)
    }

    pub fn into_supply_and_waste(
        mut self,
    ) -> (Stack<CompressedWord, State>, Stack<CompressedWord, State>) {
        // `self.waste` satisfies slightly different invariants than a usual `Stack`.
        // We therefore first restore the usual `Stack` invariant.
        let _ = self.waste.try_refill_state_if_necessary();

        (self.supply, self.waste)
    }
}

#[derive(Debug, Clone)]
pub struct CoderState<State> {
    /// Invariant: `supply >= State::one() << (State::BITS - CompressedWord::BITS)`
    pub supply: State,

    /// Invariants:
    /// - `waste >= State::one() << (State::BITS - PRECISION - CompressedWord::BITS)`
    /// - `waste < State::one() << (State::BITS - PRECISION)`
    pub waste: State,
}

impl<CompressedWord, State, const PRECISION: usize> Code for Coder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type CompressedWord = CompressedWord;
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
        self.supply.buf().is_empty()
    }
}

impl<CompressedWord, State, const PRECISION: usize> Code
    for Encoder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type CompressedWord = CompressedWord;
    type State = CoderState<State>;

    fn state(&self) -> Self::State {
        self.0.state()
    }

    fn maybe_empty(&self) -> bool {
        self.0.maybe_empty()
    }
}

impl<CompressedWord, State, const PRECISION: usize> Code
    for Decoder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type CompressedWord = CompressedWord;
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
pub enum DecodingError {
    /// Not enough binary data available to decode the symbol.
    ///
    /// Note that a [`stable::Decoder<State, CompressedWord, PRECISION>`]
    /// - consumes `PRECISION` bits of compressed data for each decoded symbol, even for
    ///   symbols with low information content under the used entropy model (the
    ///   superfluous information content will be appended to the `stable::Decoder`'s
    ///   [`waste`]); and it
    /// - retains up to `2 * State::BITS - CompressedWord::BITS` bits of compressed data
    ///   that it won't decode into any symbols (these bits are needed to initialize the
    ///   `stable::Decoder`'s `waste`, and for proper "sealing" of the binary data in
    ///   [`stable::Decoder::finish`]).
    ///
    /// Thus, when you want to decode `n` symbols with a [`stable::Decoder`], you should
    /// construct it from a [`Stack`] where [`num_valid_bits`] reports at least
    /// `n * PRECISION + 2 * State::BITS - CompressedWord::BITS`.
    ///
    /// [`stable::Decoder<State, CompressedWord, PRECISION>`]: Decoder
    /// [`stable::Decoder`]: Decoder
    /// [`waste`]: Decoder::waste
    /// [`stable::Decoder::finish`]: Decoder::finish
    /// [`num_valid_bits`]: Stack::num_valid_bits
    OutOfData,
}

impl std::fmt::Display for DecodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodingError::OutOfData => {
                write!(f, "Out of binary data.")
            }
        }
    }
}

impl Error for DecodingError {}

impl<CompressedWord, State, const PRECISION: usize> Decode<PRECISION>
    for Decoder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type DecodingError = DecodingError;

    fn decode_symbol<D>(&mut self, distribution: D) -> Result<D::Symbol, Self::DecodingError>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        let quantile = self.0.supply.chop_quantile_off_state::<D, PRECISION>();
        if self.0.supply.try_refill_state_if_necessary().is_err() {
            // Restore original state and return an error.
            self.0
                .supply
                .append_quantile_to_state::<D, PRECISION>(quantile);
            return Err(DecodingError::OutOfData);
        }

        let (symbol, left_sided_cumulative, probability) = distribution.quantile_function(quantile);
        let remainder = quantile - left_sided_cumulative;
        dbg!(probability, remainder);

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

impl<CompressedWord, State, const PRECISION: usize> Encode<PRECISION>
    for Encoder<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        distribution: D,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
    {
        let (left_sided_cumulative, probability) = distribution
            .left_cumulative_and_probability(symbol)
            .map_err(|()| EncodingError::ImpossibleSymbol)?;

        if self.0.waste.state()
            < probability.into().into() << (State::BITS - CompressedWord::BITS - PRECISION)
        {
            if self.0.waste.try_refill_state().is_err() {
                return Err(EncodingError::CapacityExceeded);
            }
            // At this point, the invariant on `self.0.waste` (see its doc comment) is
            // temporarily violated (but it will be restored below). This is how
            // `decode_symbol` can detect that it has to flush `waste.state`.
        }

        let remainder = self
            .0
            .waste
            .decode_remainder_off_state::<D, PRECISION>(probability)?;

        if (self.0.supply.state() >> (State::BITS - PRECISION)) != State::zero() {
            self.0.supply.flush_state();
        }
        self.0
            .supply
            .append_quantile_to_state::<D, PRECISION>(left_sided_cumulative + remainder);

        Ok(())
    }
}

struct WasteGuard<'a, CompressedWord, State, const PRECISION: usize>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    waste: &'a mut Stack<CompressedWord, State>,
}

impl<'a, CompressedWord, State, const PRECISION: usize>
    WasteGuard<'a, CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn new(waste: &'a mut Stack<CompressedWord, State>) -> Self {
        // `stable::Coder::waste` satisfies slightly different invariants than a usual
        // `Stack`. We therefore restore the usual `Stack` invariant here. This is reversed
        // when the `WasteGuard` gets dropped.
        let _ = waste.try_refill_state_if_necessary();

        Self { waste }
    }
}

impl<'a, CompressedWord, State, const PRECISION: usize> Deref
    for WasteGuard<'a, CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type Target = Stack<CompressedWord, State>;

    fn deref(&self) -> &Self::Target {
        self.waste
    }
}

impl<'a, CompressedWord, State, const PRECISION: usize> DerefMut
    for WasteGuard<'a, CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.waste
    }
}

impl<'a, CompressedWord, State, const PRECISION: usize> Drop
    for WasteGuard<'a, CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn drop(&mut self) {
        // Reverse the mutation done in `CoderGuard::new` to restore `stable::Coder`'s
        // special invariants for `waste`.
        if self.waste.state() >= State::one() << (State::BITS - PRECISION) {
            self.waste.flush_state();
        }
    }
}

fn concatenate<CompressedWord, State>(
    (mut prefix, suffix): (Vec<CompressedWord>, Stack<CompressedWord, State>),
) -> Result<Stack<CompressedWord, State>, ()>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    let (suffix_buf, state) = suffix.into_buf_and_state();
    prefix.extend_from_slice(&suffix_buf);
    Stack::with_buf_and_state(prefix, state)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::distributions::LeakyQuantizer;

    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };
    use statrs::distribution::Normal;

    #[test]
    fn restore_none() {
        generic_restore_many::<u32, u64, u32, 24>(3, 0);
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
        generic_restore_many::<u32, u64, u32, 24>(18, 20);
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

    fn generic_restore_many<CompressedWord, State, Probability, const PRECISION: usize>(
        amt_compressed_words: usize,
        amt_symbols: usize,
    ) where
        State: BitArray + AsPrimitive<CompressedWord>,
        CompressedWord: BitArray + Into<State> + AsPrimitive<Probability>,
        Probability: BitArray + Into<CompressedWord> + AsPrimitive<usize> + Into<f64>,
        u64: AsPrimitive<CompressedWord>,
        u32: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
        f64: AsPrimitive<Probability>,
        i32: AsPrimitive<Probability>,
    {
        let mut rng = Xoshiro256StarStar::seed_from_u64(
            (amt_compressed_words as u64).rotate_left(32) ^ amt_symbols as u64,
        );
        let compressed = (0..amt_compressed_words)
            .map(|_| rng.next_u64().as_())
            .collect::<Vec<_>>();

        let distributions = (0..amt_symbols)
            .map(|_| {
                let mean = (100.0 / u32::MAX as f64) * rng.next_u32() as f64 - 100.0;
                let std_dev = (10.0 / u32::MAX as f64) * rng.next_u32() as f64 + 0.001;
                Normal::new(mean, std_dev)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-100..=100);

        let mut stable_decoder = Stack::from_binary(compressed.clone())
            .into_stable_decoder()
            .unwrap();

        let symbols = stable_decoder
            .decode_symbols(
                distributions
                    .iter()
                    .map(|&distribution| quantizer.quantize(distribution)),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(!stable_decoder.maybe_empty());

        // Test exporting to stack and re-importing into a `stable::Encoder`:
        let stack = stable_decoder.finish_and_concatenate();
        let mut stable_encoder = stack.into_stable_encoder().unwrap();

        stable_encoder
            .encode_symbols_reverse(
                symbols
                    .iter()
                    .zip(distributions)
                    .map(|(&symbol, distribution)| (symbol, quantizer.quantize(distribution))),
            )
            .unwrap();

        let stack = stable_encoder.finish_and_concatenate().unwrap();
        assert_eq!(stack.into_binary().unwrap(), compressed);
    }
}
