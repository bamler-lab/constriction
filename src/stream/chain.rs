use alloc::vec::Vec;

use core::{borrow::Borrow, fmt::Display};

use num::cast::AsPrimitive;

use super::{
    models::{DecoderModel, EncoderModel},
    Code, Decode, Encode, EncoderError, TryCodingError,
};
use crate::{
    backends::{ReadBackend, Stack, WriteBackend},
    BitArray, CoderError, EncoderFrontendError, NonZeroBitArray,
};

#[derive(Debug, Clone)]
pub struct ChainCoder<Word, State, QuantilesBackend, RemaindersBackend, const PRECISION: usize>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    /// The compressed bit string. Read from by encoder, written to by decoder.
    quantiles: QuantilesBackend,

    /// Left-over information from decoding. Written to by encoder, read from by decoder.
    remainders: RemaindersBackend,

    heads: ChainCoderHeads<Word, State, PRECISION>,
}

/// Type of the internal state used by [`ChainCoder<Word, State>`]. Relevant for
/// [`Seek`]ing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChainCoderHeads<Word: BitArray, State: BitArray, const PRECISION: usize> {
    /// All bits following the highest order bit (which is a given in a `NonZero`) are
    /// leftover bits from previous reads from `quantiles` that still need to be consumed.
    /// Thus, there are at most `Word::BITS - 1` leftover bits at any time.
    quantiles: Word::NonZero,

    /// Satisfies invariants:
    /// - `heads.remainders >= 1 << (State::BITS - Word::BITS - PRECISION)`; and
    /// - `heads.remainders < 1 << (State::BITS - PRECISION)`
    remainders: State,
}

impl<Word: BitArray, State: BitArray, const PRECISION: usize>
    ChainCoderHeads<Word, State, PRECISION>
{
    /// Returns `true` iff there's currently an integer amount of `Words` on `quantiles`
    #[inline(always)]
    pub fn is_whole(self) -> bool {
        self.quantiles.get() == Word::one()
    }

    /// Private on purpose.
    fn new<B: ReadBackend<Word, Stack>>(
        source: &mut B,
    ) -> Result<ChainCoderHeads<Word, State, PRECISION>, CoderError<(), B::ReadError>>
    where
        Word: Into<State>,
    {
        assert!(State::BITS >= Word::BITS + PRECISION);
        assert!(PRECISION > 0);
        assert!(PRECISION <= Word::BITS);

        let threshold = State::one() << (State::BITS - Word::BITS - PRECISION);
        let mut remainders_head = match source.read()? {
            Some(word) if word != Word::zero() => word.into(),
            _ => return Err(CoderError::FrontendError(())),
        };
        while remainders_head < threshold {
            remainders_head = remainders_head << Word::BITS
                | source.read()?.ok_or(CoderError::FrontendError(()))?.into();
        }

        Ok(ChainCoderHeads {
            quantiles: Word::one().into_nonzero().expect("1 != 0"),
            remainders: remainders_head,
        })
    }
}

pub type DefaultChainCoder = ChainCoder<u32, u64, Vec<u32>, Vec<u32>, 24>;
pub type SmallChainCoder = ChainCoder<u16, u32, Vec<u16>, Vec<u16>, 12>;

impl<Word, State, QuantilesBackend, RemaindersBackend, const PRECISION: usize>
    ChainCoder<Word, State, QuantilesBackend, RemaindersBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    /// Creates a new `ChainCoder` ready for decoding from the provided `quantiles`, which
    /// must have enough words to initialize the chain heads and must not have a zero word
    /// at the current read position.
    ///
    /// Retuns an error if `quantiles` does not have enough words, if reading from
    /// `quantiles` lead to an error, or if the first word read from `quantiles` is zero.
    pub fn from_quantiles(
        mut quantiles: QuantilesBackend,
    ) -> Result<Self, CoderError<QuantilesBackend, QuantilesBackend::ReadError>>
    where
        QuantilesBackend: ReadBackend<Word, Stack>,
        RemaindersBackend: Default,
    {
        let heads = match ChainCoderHeads::new(&mut quantiles) {
            Ok(heads) => heads,
            Err(CoderError::FrontendError(())) => return Err(CoderError::FrontendError(quantiles)),
            Err(CoderError::BackendError(err)) => return Err(CoderError::BackendError(err)),
        };
        let remainders = RemaindersBackend::default();

        Ok(Self {
            quantiles,
            remainders,
            heads,
        })
    }

    /// Creates a new `ChainCoder` ready for encoding some symbols together with the
    /// provided `remainders`, which must have enough words to initialize the chain heads
    /// and must start with at least two nonzero words.
    ///
    /// Retuns an error if `quantiles` does not have enough words, if reading from
    /// `quantiles` lead to an error, or if the first word read from `quantiles` is zero.
    pub fn from_remainders(
        mut remainders: RemaindersBackend,
    ) -> Result<Self, CoderError<RemaindersBackend, RemaindersBackend::ReadError>>
    where
        RemaindersBackend: ReadBackend<Word, Stack>,
        QuantilesBackend: Default,
    {
        let quantiles_head = match remainders.read()?.and_then(Word::into_nonzero) {
            Some(word) => word,
            _ => return Err(CoderError::FrontendError(remainders)),
        };
        let mut heads = match ChainCoderHeads::new(&mut remainders) {
            Ok(heads) => heads,
            Err(CoderError::FrontendError(())) => {
                return Err(CoderError::FrontendError(remainders))
            }
            Err(CoderError::BackendError(err)) => return Err(CoderError::BackendError(err)),
        };
        heads.quantiles = quantiles_head;

        let quantiles = QuantilesBackend::default();

        Ok(Self {
            quantiles,
            remainders,
            heads,
        })
    }

    /// Transfers any fractional `Word` left on `quantiles` to `remainders` and returns
    /// (quantiles, remainders). You could contatenate these two and call
    /// [`from_remainders`] on them, or you could call [`from_remainders`] just on the
    /// second item of the returned tuple.
    pub fn into_remainders(
        mut self,
    ) -> Result<(QuantilesBackend, RemaindersBackend), RemaindersBackend::WriteError>
    where
        RemaindersBackend: WriteBackend<Word>,
    {
        // Flush remainders head.
        while self.heads.remainders != State::zero() {
            self.remainders.write(self.heads.remainders.as_())?;
            self.heads.remainders = self.heads.remainders >> Word::BITS;
        }

        // Transfer quantiles head onto `remainders`.
        self.remainders.write(self.heads.quantiles.get())?;

        Ok((self.quantiles, self.remainders))
    }

    /// Checks that there's currently an integer amount of `Words` in `quantiles` (see
    /// [`is_whole`]) and returns `(remainders, quantiles)`. You could concatenate these two
    /// and call [`from_quantiles`] on the result, or you could call [`from_quantiles`] just
    /// on the second item of the returned tuple.
    pub fn into_quantiles(
        mut self,
    ) -> Result<
        (RemaindersBackend, QuantilesBackend),
        CoderError<Self, RemaindersBackend::WriteError>,
    >
    where
        RemaindersBackend: WriteBackend<Word>,
    {
        if !self.is_whole() {
            return Err(CoderError::FrontendError(self));
        }

        // Flush remainders head.
        while self.heads.remainders != State::zero() {
            self.remainders.write(self.heads.remainders.as_())?;
            self.heads.remainders = self.heads.remainders >> Word::BITS;
        }

        Ok((self.remainders, self.quantiles))
    }

    /// Returns `true` iff there's currently an integer amount of `Words` on `quantiles`
    #[inline(always)]
    pub fn is_whole(&self) -> bool {
        self.heads.quantiles.get() == Word::one()
    }

    pub fn encode_symbols_reverse<S, D, I>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<
        (),
        EncoderError<
            BackendError<QuantilesBackend::WriteError, Option<RemaindersBackend::ReadError>>,
        >,
    >
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Word>,
        Word: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = (S, D)>,
        I::IntoIter: DoubleEndedIterator,
        QuantilesBackend: WriteBackend<Word>,
        RemaindersBackend: ReadBackend<Word, Stack>,
    {
        self.encode_symbols(symbols_and_models.into_iter().rev())
    }

    /// TODO: type aliases for these ridiculous error types.
    pub fn try_encode_symbols_reverse<S, D, E, I>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<
        (),
        TryCodingError<
            EncoderError<
                BackendError<QuantilesBackend::WriteError, Option<RemaindersBackend::ReadError>>,
            >,
            E,
        >,
    >
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Word>,
        Word: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = core::result::Result<(S, D), E>>,
        I::IntoIter: DoubleEndedIterator,
        QuantilesBackend: WriteBackend<Word>,
        RemaindersBackend: ReadBackend<Word, Stack>,
    {
        self.try_encode_symbols(symbols_and_models.into_iter().rev())
    }

    pub fn encode_iid_symbols_reverse<S, D, I>(
        &mut self,
        symbols: I,
        model: &D,
    ) -> Result<
        (),
        EncoderError<
            BackendError<QuantilesBackend::WriteError, Option<RemaindersBackend::ReadError>>,
        >,
    >
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Word>,
        Word: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = S>,
        I::IntoIter: DoubleEndedIterator,
        QuantilesBackend: WriteBackend<Word>,
        RemaindersBackend: ReadBackend<Word, Stack>,
    {
        self.encode_iid_symbols(symbols.into_iter().rev(), model)
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
    pub fn increase_precision<const NEW_PRECISION: usize>(
        mut self,
    ) -> Result<
        ChainCoder<Word, State, QuantilesBackend, RemaindersBackend, NEW_PRECISION>,
        RemaindersBackend::WriteError,
    >
    where
        RemaindersBackend: WriteBackend<Word>,
    {
        assert!(NEW_PRECISION >= PRECISION);
        assert!(NEW_PRECISION <= Word::BITS);
        assert!(State::BITS >= Word::BITS + NEW_PRECISION);

        if self.heads.remainders >= State::one() << (State::BITS - NEW_PRECISION) {
            self.flush_remainders_head()?;
        }

        Ok(ChainCoder {
            quantiles: self.quantiles,
            remainders: self.remainders,
            heads: ChainCoderHeads {
                quantiles: self.heads.quantiles,
                remainders: self.heads.remainders,
            },
        })
    }

    pub fn decrease_precision<const NEW_PRECISION: usize>(
        mut self,
    ) -> Result<
        ChainCoder<Word, State, QuantilesBackend, RemaindersBackend, NEW_PRECISION>,
        Option<RemaindersBackend::ReadError>,
    >
    where
        RemaindersBackend: ReadBackend<Word, Stack>,
    {
        assert!(NEW_PRECISION <= PRECISION);
        assert!(NEW_PRECISION > 0);

        if self.heads.remainders < State::one() << (State::BITS - NEW_PRECISION - Word::BITS) {
            // Won't truncate since, from the above check it follows that we satisfy the contract
            // `self.heads.remainders < 1 << (State::BITS - Word::BITS)`.
            self.refill_remainders_head()?
        }

        Ok(ChainCoder {
            quantiles: self.quantiles,
            remainders: self.remainders,
            heads: ChainCoderHeads {
                quantiles: self.heads.quantiles,
                remainders: self.heads.remainders,
            },
        })
    }

    #[inline(always)]
    pub fn change_precision<const NEW_PRECISION: usize>(
        self,
    ) -> Result<
        ChainCoder<Word, State, QuantilesBackend, RemaindersBackend, NEW_PRECISION>,
        ChangePrecisionError<RemaindersBackend, Word>,
    >
    where
        RemaindersBackend: WriteBackend<Word> + ReadBackend<Word, Stack>,
    {
        // TODO: encapsulate error type
        if NEW_PRECISION > PRECISION {
            self.increase_precision()
                .map_err(ChangePrecisionError::Write)
        } else {
            self.decrease_precision()
                .map_err(ChangePrecisionError::Read)
        }
    }

    #[inline(always)]
    /// This would flush meaningless zero bits if `self.heads.remainders < 1 << Word::BITS`.
    fn flush_remainders_head(&mut self) -> Result<(), RemaindersBackend::WriteError>
    where
        RemaindersBackend: WriteBackend<Word>,
    {
        self.remainders.write(self.heads.remainders.as_())?;
        self.heads.remainders = self.heads.remainders >> Word::BITS;
        Ok(())
    }

    /// This truncates if `self.heads.remainders >= 1 << (State::BITS - Word::BITS)`.
    #[inline(always)]
    fn refill_remainders_head(&mut self) -> Result<(), Option<RemaindersBackend::ReadError>>
    where
        RemaindersBackend: ReadBackend<Word, Stack>,
    {
        let word = self.remainders.read().map_err(Some)?.ok_or(None)?;
        self.heads.remainders = (self.heads.remainders << Word::BITS) | word.into();
        Ok(())
    }
}

impl<Word, State, QuantilesBackend, RemaindersBackend, const PRECISION: usize> Code
    for ChainCoder<Word, State, QuantilesBackend, RemaindersBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Word = Word;
    type State = ChainCoderHeads<Word, State, PRECISION>;

    fn state(&self) -> Self::State {
        self.heads
    }
}

/// Error type for misuse of a [`stable::Decoder`].
///
/// [`stable::Decoder`]: Decoder
#[derive(Debug, PartialEq, Eq)]
pub enum FrontendError {
    OutOfData,
}

impl core::fmt::Display for FrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfData => {
                write!(f, "Out of data.")
            }
        }
    }
}

/// Error type for backend errors in a [`stable::Decoder`].
///
/// [`stable::Decoder`]: Decoder
#[derive(Debug, PartialEq, Eq)]
pub enum BackendError<QuantilesBackendError, RemaindersBackendError> {
    Quantiles(QuantilesBackendError),
    Remainders(RemaindersBackendError),
}

impl<QuantilesBackendError: Display, RemaindersBackendError: Display> core::fmt::Display
    for BackendError<QuantilesBackendError, RemaindersBackendError>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Quantiles(err) => {
                write!(f, "Read/write error when accessing quantiles: {}", err)
            }
            Self::Remainders(err) => {
                write!(f, "Read/write error when accessing remainders: {}", err)
            }
        }
    }
}

#[cfg(feature = "std")]
impl<
        QuantilesBackendError: std::error::Error + 'static,
        RemaindersBackendError: std::error::Error + 'static,
    > std::error::Error for BackendError<QuantilesBackendError, RemaindersBackendError>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Quantiles(err) => Some(err),
            Self::Remainders(err) => Some(err),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ChangePrecisionError<RemaindersBackend, Word>
where
    RemaindersBackend: WriteBackend<Word> + ReadBackend<Word, Stack>,
{
    Write(RemaindersBackend::WriteError),

    /// `None` if out of data, `Some(err)` if reading lead to error.
    Read(Option<RemaindersBackend::ReadError>),
}

impl<Word, State, QuantilesBackend, RemaindersBackend, const PRECISION: usize> Decode<PRECISION>
    for ChainCoder<Word, State, QuantilesBackend, RemaindersBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    QuantilesBackend: ReadBackend<Word, Stack>,
    RemaindersBackend: WriteBackend<Word>,
{
    type FrontendError = FrontendError;

    type BackendError = BackendError<QuantilesBackend::ReadError, RemaindersBackend::WriteError>;

    fn decode_symbol<D>(
        &mut self,
        model: D,
    ) -> Result<D::Symbol, CoderError<Self::FrontendError, Self::BackendError>>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<D::Probability>,
    {
        assert!(PRECISION <= Word::BITS);
        assert!(PRECISION != 0);
        assert!(State::BITS >= Word::BITS + PRECISION);

        let word = if PRECISION == Word::BITS
            || self.heads.quantiles.get() < Word::one() << PRECISION
        {
            let word = self
                .quantiles
                .read()
                .map_err(BackendError::Quantiles)?
                .ok_or(CoderError::FrontendError(FrontendError::OutOfData))?;
            if PRECISION != Word::BITS {
                self.heads.quantiles = unsafe {
                    // SAFETY:
                    // - `0 < PRECISION < Word::BITS` as per our assertion and the above check,
                    //   therefore `Word::BITS - PRECISION > 0` and both the left-shift and
                    //   the right-shift are valid;
                    // - `heads.quantiles.get() != 0` sinze `heads.quantiles` is a `NonZero`.
                    // - `heads.quantiles.get() < 1 << PRECISION`, so all its "one" bits are
                    //   in the `PRECISION` lowest significant bits; since it we have
                    //   `Word::BITS` bits available, shifting left by `Word::BITS - PRECISION`
                    //   doesn't truncate, and thus the result is also nonzero.
                    Word::NonZero::new_unchecked(
                        self.heads.quantiles.get() << (Word::BITS - PRECISION) | word >> PRECISION,
                    )
                };
            }
            word
        } else {
            let quantile = self.heads.quantiles.get();
            self.heads.quantiles = unsafe {
                // SAFETY: `heads.quantiles.get() >= 1 << PRECISION`, so shifting right by
                // `PRECISION` doesn't result in zero.
                Word::NonZero::new_unchecked(self.heads.quantiles.get() >> PRECISION)
            };
            quantile
        };

        let quantile = if PRECISION == Word::BITS {
            word
        } else {
            word % (Word::one() << PRECISION)
        };
        let quantile = quantile.as_();

        let (symbol, left_sided_cumulative, probability) = model.quantile_function(quantile);
        let remainder = quantile - left_sided_cumulative;

        // This can't truncate because
        // - we maintain the invariant `heads.remainders < 1 << (State::BITS - PRECISION)`; and
        // - `probability <= 1 << PRECISION` and `remainder < probability`.
        // Thus, `remainders * proability + remainder < (remainders + 1) * probability`
        // which is `<= (1 << (State::BITS - PRECISION)) << PRECISION = 1 << State::BITS`.
        self.heads.remainders =
            self.heads.remainders * probability.get().into().into() + remainder.into().into();

        if self.heads.remainders >= State::one() << (State::BITS - PRECISION) {
            // The invariant on `self.heads.remainders` (see its doc comment) is violated and must
            // be restored.
            self.flush_remainders_head()
                .map_err(BackendError::Remainders)?
        }

        Ok(symbol)
    }

    fn maybe_exhausted(&self) -> bool {
        self.quantiles.maybe_exhausted() || self.remainders.maybe_full()
    }
}

impl<Word, State, QuantilesBackend, RemaindersBackend, const PRECISION: usize> Encode<PRECISION>
    for ChainCoder<Word, State, QuantilesBackend, RemaindersBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    QuantilesBackend: WriteBackend<Word>,
    RemaindersBackend: ReadBackend<Word, Stack>,
{
    type BackendError =
        BackendError<QuantilesBackend::WriteError, Option<RemaindersBackend::ReadError>>;

    // TODO: we should be allowed to return our own FrontendError here if we run out of remainders.
    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        model: D,
    ) -> Result<(), EncoderError<Self::BackendError>>
    where
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<D::Probability>,
    {
        // assert!(State::BITS >= Word::BITS + PRECISION);
        assert!(PRECISION <= Word::BITS);
        assert!(PRECISION > 0);

        let (left_sided_cumulative, probability) = model
            .left_cumulative_and_probability(symbol)
            .map_err(|()| EncoderFrontendError::ImpossibleSymbol.into_coder_error())?;

        if self.heads.remainders
            < probability.get().into().into() << (State::BITS - Word::BITS - PRECISION)
        {
            self.refill_remainders_head()
                .map_err(BackendError::Remainders)?;
            // At this point, the invariant on `self.heads.remainders` (see its doc comment) is
            // temporarily violated (but it will be restored below). This is how
            // `decode_symbol` can detect that it has to flush `remainders.state`.
        }

        let remainder = (self.heads.remainders % probability.get().into().into())
            .as_()
            .as_();
        let quantile = (left_sided_cumulative + remainder).into();
        self.heads.remainders = self.heads.remainders / probability.get().into().into();

        if PRECISION != Word::BITS
            && self.heads.quantiles.get() < Word::one() << (Word::BITS - PRECISION)
        {
            unsafe {
                // SAFETY:
                // - `heads.quantiles` is nonzero because it is a `NonZero`
                // - `heads.quantiles`, has `Word::BITS` bits and we checked above that all its one
                //   bits are within theleast significant `Word::BITS - PRECISION` bits. Thus, the
                //   most significant `PRECISION` bits are 0 and the left-shift doesn't truncate.
                // Thus, the result of the left-shift is also noznero.
                self.heads.quantiles =
                    (self.heads.quantiles.get() << PRECISION | quantile).into_nonzero_unchecked();
            }
        } else {
            let word = if PRECISION == Word::BITS {
                quantile
            } else {
                let word = self.heads.quantiles.get() << PRECISION | quantile;
                unsafe {
                    // SAFETY: if we're here then `heads.quantiles >= 1 << (Word::BITS - PRECISION).
                    // Thus, shifting right by this amount of bits leaves at least one 1 bit.
                    self.heads.quantiles = (self.heads.quantiles.get() >> (Word::BITS - PRECISION))
                        .into_nonzero_unchecked();
                }
                word
            };
            self.quantiles
                .write(word)
                .map_err(BackendError::Quantiles)?;
        }

        Ok(())
    }

    fn maybe_full(&self) -> bool {
        self.remainders.maybe_exhausted() || self.quantiles.maybe_full()
    }
}

#[cfg(testTODO)]
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
        *last_word = *last_word | Word::one() << (Word::BITS - leading_zeros - 1);
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

        assert!(!stable_decoder.maybe_exhausted());

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
