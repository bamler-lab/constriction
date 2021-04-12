//! # Motivation
//!
//! TODO: explain
//!
//! ```
//! use constriction::stream::{
//!     models::DefaultCategorical, stack::DefaultAnsCoder, chain::DefaultChainCoder, Decode
//! };
//!
//! /// Shorthand for decoding a sequence of symbols with categorical entropy models.
//! fn decode_categoricals<Decoder: Decode<24, Word = u32>>(
//!     decoder: &mut Decoder,
//!     probabilities: &[[f64; 4]],
//! ) -> Vec<usize> {
//!     let entropy_models = probabilities
//!         .iter()
//!         .map(|probs| DefaultCategorical::from_floating_point_probabilities(probs).unwrap());
//!     decoder.decode_symbols(entropy_models).collect::<Result<Vec<_>, _>>().unwrap()
//! }
//!
//! // Let's define some sample binary data and some probabilities for our entropy models
//! let data = vec![0x80d1_4131, 0xdda9_7c6c, 0x5017_a640, 0x01170a3d];
//! let mut probabilities = [
//!     [0.1, 0.7, 0.1, 0.1], // Probabilities for the entropy model of the first decoded symbol.
//!     [0.2, 0.2, 0.1, 0.5], // Probabilities for the entropy model of the second decoded symbol.
//!     [0.2, 0.1, 0.4, 0.3], // Probabilities for the entropy model of the third decoded symbol.
//! ];
//!
//! // Decoding the binary data with an `AnsCoder` results in the symbols `[0, 0, 1]`.
//! let mut ans_coder = DefaultAnsCoder::from_binary(data.clone()).unwrap();
//! let symbols = decode_categoricals(&mut ans_coder, &probabilities);
//! assert_eq!(symbols, [0, 0, 1]);
//!
//! // Even if we change only the first entropy model (slightly), *all* decoded symbols can change:
//! probabilities[0] = [0.09, 0.71, 0.1, 0.1]; // was: `[0.1, 0.7, 0.1, 0.1]`
//! let mut ans_coder = DefaultAnsCoder::from_binary(data.clone()).unwrap();
//! let symbols = decode_categoricals(&mut ans_coder, &probabilities);
//! assert_eq!(symbols, [1, 0, 3]); // (instead of `[0, 0, 1]` from above)
//! // It's no surprise that the first symbol changed since we changed its entropy model. But
//! // note that the third symbol changed too even though we hadn't changed its entropy model.
//! // --> Changes to entropy models (and also to compressed bits) have a *global* effect.
//!
//! // Let's try the same with a `ChainCoder`:
//! probabilities[0] = [0.1, 0.7, 0.1, 0.1]; // Restore original entropy model for first symbol.
//! let mut chain_coder = DefaultChainCoder::from_binary(data.clone()).unwrap();
//! let symbols = decode_categoricals(&mut chain_coder, &probabilities);
//! assert_eq!(symbols, [0, 3, 3]);
//! // We get different symbols than for the `AnsCoder`, of course, but that's not the point here.
//!
//! probabilities[0] = [0.09, 0.71, 0.1, 0.1]; // Change the first entropy model again slightly.
//! let mut chain_coder = DefaultChainCoder::from_binary(data).unwrap();
//! let symbols = decode_categoricals(&mut chain_coder, &probabilities);
//! assert_eq!(symbols, [1, 3, 3]); // (instead of `[0, 3, 3]` from above)
//! // The only symbol that changed was the one whose entropy model we had changed.
//! // --> In a `ChainCoder`, changes to entropy models (and also to compressed bits)
//! //     only have a *local* effect on the decompressed symbols.
//! ```

use alloc::vec::Vec;

use core::{borrow::Borrow, convert::Infallible, fmt::Display};

use num::cast::AsPrimitive;

use super::{
    models::{DecoderModel, EncoderModel},
    Code, Decode, Encode, TryCodingError,
};
use crate::{
    backends::{ReadWords, WriteWords},
    BitArray, CoderError, DefaultEncoderFrontendError, NonZeroBitArray, Pos, PosSeek, Seek, Stack,
};

/// # Intended Usage
///
/// A typical usage cycle goes along the following steps:
///
/// ## When compressing data using the bits-back trick
///
/// 0. Start with some stack of (typically already compressed) binary data, which you want
///    to piggy-back into the choice of certain latent variables.
/// 1. Create a `ChainCoder` by calling [`ChainCoder::from_binary`] or
///    [`ChainCoder::from_compressed`] (depending on whether you can guarantee that the
///    stack of binary data has a nonzero word on top).
/// 2. Use the `ChainCoder` and a sequence of entropy models to decode some symbols.
/// 3. Export the remaining data on the `ChainCoder` by calling [`.into_remaining()`].
///
/// ## When decompressing the data
///
/// 1. Create a `ChainCoder` by calling [`ChainCoder::from_remaining`].
/// 2. Encode the symbols you obtained in Step 2 above back onto the new chain coder (in
///    reverse order) using the same entropy models.
/// 3. Recover the original binary data from Step 0 above by calling [`.into_binary()`] or
///    [`.into_compressed()`] (using the analogous choice as in Step 1 above).
///
/// # Examples
///
/// The following two examples show two variants of the above typical usage cycle.
///
/// ```
/// use constriction::stream::{models::DefaultLeakyQuantizer, Decode, chain::DefaultChainCoder};
/// use probability::distribution::Gaussian;
///
/// // Step 0 of the compressor: Generate some sample binary data for demonstration purpose.
/// let original_data = (0..100u32).map(
///     |i| i.wrapping_mul(0xad5f_b2ed).wrapping_add(0xed55_4892)
/// ).collect::<Vec<_>>();
///
/// // Step 1 of the compressor: obtain a `ChainCoder` from the original binary data.
/// let mut coder = DefaultChainCoder::from_binary(original_data.clone()).unwrap();
///
/// // Step 2 of the compressor: decode data into symbols using some entropy models.
/// let quantizer = DefaultLeakyQuantizer::new(-100..=100);
/// let models = (0..50u32).map(|i| quantizer.quantize(Gaussian::new(i as f64, 10.0)));
/// let symbols = coder.decode_symbols(models.clone()).collect::<Result<Vec<_>, _>>().unwrap();
///
/// // Step 3 of the compressor: export the remaining data.
/// let (remaining_prefix, remaining_suffix) = coder.into_remaining().unwrap();
/// // (verify that we've indeed reduced the amount of data:)
/// assert!(remaining_prefix.len() + remaining_suffix.len() < original_data.len());
///
/// // ... do something with the `symbols`, then recover them later ...
///
/// // Step 1 of the decompressor: create a `ChainCoder` from the remaining data. We only really
/// // need the `remaining_suffix` here, but it would also be legal to use the concatenation of
/// // `remaining_prefix` with `remaining_suffix` (see other example below).
/// let mut coder = DefaultChainCoder::from_remaining(remaining_suffix).unwrap();
///
/// // Step 2 of the decompressor: re-encode the symbols in reverse order.
/// coder.encode_symbols_reverse(symbols.into_iter().zip(models));
///
/// // Step 3 of the decompressor: recover the original data.
/// let (recovered_prefix, recovered_suffix) = coder.into_binary().unwrap();
/// assert!(recovered_prefix.is_empty());  // Empty because we discarded `remaining_prefix` above.
/// let mut recovered = remaining_prefix;  // But we have to prepend it to the recovered data now.
/// recovered.extend_from_slice(&recovered_suffix);
///
/// assert_eq!(recovered, original_data);
/// ```
///
/// In Step 3 of the compressor in the example above, calling `.into_remaining()` on a
/// `ChainCoder` returns a tuple of a `remaining_prefix` and a `remaining_suffix`. The
/// `remaining_prefix` contains superflous data that we didn't need when decoding the
/// `symbols` (`remaining_prefix` is an unaltered prefix of the original `data`). We
/// therefore don't need `remaining_prefix` for re-encoding the symbols, so we didn't pass
/// it to `ChainCoder::from_remaining` in Step 1 of the decompressor above.
///
/// If we were to write out `remaining_prefix` and `remaining_suffix` to a file then it
/// would be tedious to keep track of where the prefix ends and where the suffix begins.
/// Luckly, we don't have to do this. We can just as well concatenate `remaining_prefix`
/// and `remaining_suffix` right away. The only additional change this will cause is that
/// the call to `.into_binary()` in Step 3 of the decompressor will then return a non-empty
/// `recovered_prefix` because the second `ChainCoder` will then also have some superflous
/// data. So we'll have to again concatenate the two returned buffers. The following example
/// shows how this works:
///
/// ```
/// # use constriction::stream::{models::DefaultLeakyQuantizer, Decode, chain::DefaultChainCoder};
/// # use probability::distribution::Gaussian;
/// # let original_data = (0..100u32).map(
/// #     |i| i.wrapping_mul(0xad5f_b2ed).wrapping_add(0xed55_4892)
/// # ).collect::<Vec<_>>();
/// # let mut coder = DefaultChainCoder::from_binary(original_data.clone()).unwrap();
/// # let quantizer = DefaultLeakyQuantizer::new(-100..=100);
/// # let models = (0..50u32).map(|i| quantizer.quantize(Gaussian::new(i as f64, 10.0)));
/// # let symbols = coder.decode_symbols(models.clone()).collect::<Result<Vec<_>, _>>().unwrap();
/// # let (remaining_prefix, remaining_suffix) = coder.into_remaining().unwrap();
/// // ... compressor same as in the previous example above ...
///
/// // Alternative Step 1 of the decompressor: concatenate `remaining_prefix` with
/// // `remaining_suffix` before creating a `ChainCoder` from them.
/// let mut remaining = remaining_prefix;
/// remaining.extend_from_slice(&remaining_suffix);
/// let mut coder = DefaultChainCoder::from_remaining(remaining).unwrap();
///
/// // Step 2 of the decompressor: re-encode symbols in reverse order (same as in previous example).
/// coder.encode_symbols_reverse(symbols.into_iter().zip(models));
///
/// // Alternative Step 3 of the decompressor: recover the original data by another concatenation.
/// let (recovered_prefix, recovered_suffix) = coder.into_binary().unwrap();
/// assert!(!recovered_prefix.is_empty());  // No longer empty because there was superflous data.
/// let mut recovered = recovered_prefix;   // So we have to concatenate `recovered_{pre,suf}fix`.
/// recovered.extend_from_slice(&recovered_suffix);
///
/// assert_eq!(recovered, original_data);
/// ```
///
/// [`.into_remaining()`]: Self::into_remaining
/// [`.into_binary()`]: Self::into_binary
/// [`.into_compressed()`]: Self::into_compressed
#[derive(Debug, Clone)]
pub struct ChainCoder<Word, State, CompressedBackend, RemainingBackend, const PRECISION: usize>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    /// The compressed bit string. Written to by encoder, read from by decoder.
    compressed: CompressedBackend,

    /// Leftover information from decoding. Read from by encoder, written to by decoder.
    remaining: RemainingBackend,

    heads: ChainCoderHeads<Word, State, PRECISION>,
}

/// Type of the internal state used by [`ChainCoder<Word, State>`]. Relevant for
/// [`Seek`]ing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChainCoderHeads<Word: BitArray, State: BitArray, const PRECISION: usize> {
    /// All bits following the highest order bit (which is a given in a `NonZero`) are
    /// leftover bits from previous reads from `compressed` that still need to be consumed.
    /// Thus, there are at most `Word::BITS - 1` leftover bits at any time.
    compressed: Word::NonZero,

    /// Satisfies invariants:
    /// - `heads.remaining >= 1 << (State::BITS - Word::BITS - PRECISION)`; and
    /// - `heads.remaining < 1 << (State::BITS - PRECISION)`
    remaining: State,
}

impl<Word: BitArray, State: BitArray, const PRECISION: usize>
    ChainCoderHeads<Word, State, PRECISION>
{
    /// Returns `true` iff there's currently an integer amount of `Words` on `compressed`
    #[inline(always)]
    pub fn is_whole(self) -> bool {
        self.compressed.get() == Word::one()
    }

    /// Private on purpose.
    fn new<B: ReadWords<Word, Stack>>(
        source: &mut B,
        push_one: bool,
    ) -> Result<ChainCoderHeads<Word, State, PRECISION>, CoderError<(), B::ReadError>>
    where
        Word: Into<State>,
    {
        assert!(State::BITS >= Word::BITS + PRECISION);
        assert!(PRECISION > 0);
        assert!(PRECISION <= Word::BITS);

        let threshold = State::one() << (State::BITS - Word::BITS - PRECISION);
        let mut remaining_head = if push_one {
            State::one()
        } else {
            match source.read()? {
                Some(word) if word != Word::zero() => word.into(),
                _ => return Err(CoderError::Frontend(())),
            }
        };
        while remaining_head < threshold {
            remaining_head = remaining_head << Word::BITS
                | source.read()?.ok_or(CoderError::Frontend(()))?.into();
        }

        Ok(ChainCoderHeads {
            compressed: Word::one().into_nonzero().expect("1 != 0"),
            remaining: remaining_head,
        })
    }
}

pub type DefaultChainCoder = ChainCoder<u32, u64, Vec<u32>, Vec<u32>, 24>;
pub type SmallChainCoder = ChainCoder<u16, u32, Vec<u16>, Vec<u16>, 12>;

impl<Word, State, CompressedBackend, RemainingBackend, const PRECISION: usize>
    ChainCoder<Word, State, CompressedBackend, RemainingBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    /// Creates a new `ChainCoder` for decoding from the provided `data`.
    ///
    /// The reader `data` must have enough words to initialize the chain heads but can
    /// otherwise be arbitrary. In particualar, `data` doesn't necessary have to come from
    /// an [`AnsCoder`]. If you know that `data` comes from an `AnsCoder` then it's slightly
    /// better to call [`from_compressed`] instead.
    ///
    /// Retuns an error if `data` does not have enough words to initialize the chain heads
    /// or if reading from `data` lead to an error.
    ///
    /// [`AnsCoder`]: super::stack::AnsCoder
    /// [`from_compressed`]: Self::from_compressed
    pub fn from_binary(
        mut data: CompressedBackend,
    ) -> Result<Self, CoderError<CompressedBackend, CompressedBackend::ReadError>>
    where
        CompressedBackend: ReadWords<Word, Stack>,
        RemainingBackend: Default,
    {
        let heads = match ChainCoderHeads::new(&mut data, true) {
            Ok(heads) => heads,
            Err(CoderError::Frontend(())) => return Err(CoderError::Frontend(data)),
            Err(CoderError::Backend(err)) => return Err(CoderError::Backend(err)),
        };
        let remaining = RemainingBackend::default();

        Ok(Self {
            compressed: data,
            remaining,
            heads,
        })
    }

    /// Creates a new `ChainCoder` for decoding from the compressed data of an [`AnsCoder`]
    ///
    /// The provided read backend `compressed`, must have enough words to initialize the
    /// chain heads and must not have a zero word at the current read position. The latter
    /// is always satisfied for (nonempty) data returned from [`AnsCoder::into_compressed`].
    ///
    /// Retuns an error if `compressed` does not have enough words, if reading from
    /// `compressed` lead to an error, or if the first word read from `compressed` is zero.
    ///
    /// [`AnsCoder`]: super::stack::AnsCoder
    /// [`AnsCoder::into_compressed`]: super::stack::AnsCoder::into_compressed
    pub fn from_compressed(
        mut compressed: CompressedBackend,
    ) -> Result<Self, CoderError<CompressedBackend, CompressedBackend::ReadError>>
    where
        CompressedBackend: ReadWords<Word, Stack>,
        RemainingBackend: Default,
    {
        let heads = match ChainCoderHeads::new(&mut compressed, false) {
            Ok(heads) => heads,
            Err(CoderError::Frontend(())) => return Err(CoderError::Frontend(compressed)),
            Err(CoderError::Backend(err)) => return Err(CoderError::Backend(err)),
        };
        let remaining = RemainingBackend::default();

        Ok(Self {
            compressed: compressed,
            remaining,
            heads,
        })
    }

    /// Terminates decoding and returns the remaining bit string as a tuple `(prefix,
    /// suffix)`.
    ///
    /// - The `prefix` is a shortened but otherwise unaltered variant of the data from which
    ///   you created this `ChainCoder` when you called [`ChainCoder::from_binary`] or
    ///   [`ChainCoder::from_compressed`].
    /// - The `suffix` is a stack with at least two nonzero words on top.
    ///
    /// You can use the returned tuple `(prefix, suffix)` in either of the following two
    /// ways (see examples in the [struct level documentation](ChainCoder)):
    /// - Either put `prefix` away and continue only with `suffix` as follows:
    ///   1. obtain a new `ChainCoder` by calling [`ChainCoder::from_remaining(suffix)`];
    ///   2. encode the same symbols that you decoded from the original `ChainCoder` back
    ///      onto the new `ChainCoder` (in reverse order);
    ///   3. call [`.into_binary()`] or [`.into_compressed()`] on the new `ChainCoder` to
    ///      obatain another tuple `(prefix2, suffix2)`.
    ///   4. concatenate `prefix`, `prefix2`, and `suffix2` to recover the data from which
    ///      you created the original `ChainCoder` when you constructed it with
    ///      [`ChainCoder::from_binary`] or [`ChainCoder::from_compressed`], respectively.
    /// - Or you can concatenate `prefix` with `suffix`, create a new `ChainCoder` from the
    ///   concatenation by calling `ChainCoder::from_remaining(concatenation)`, continue
    ///   with steps 2 and 3 above, and then just concatenate `prefix2` with `suffix2` to
    ///   recover the original data.
    ///
    /// [`ChainCoder::from_remaining(suffix)`]: Self::from_remaining
    /// [`.into_binary()`]: Self::into_binary
    /// [`.into_compressed()`]: Self::into_compressed
    pub fn into_remaining(
        mut self,
    ) -> Result<(CompressedBackend, RemainingBackend), RemainingBackend::WriteError>
    where
        RemainingBackend: WriteWords<Word>,
    {
        // Flush remaining head.
        while self.heads.remaining != State::zero() {
            self.remaining.write(self.heads.remaining.as_())?;
            self.heads.remaining = self.heads.remaining >> Word::BITS;
        }

        // Transfer compressed head onto `remaining`.
        self.remaining.write(self.heads.compressed.get())?;

        Ok((self.compressed, self.remaining))
    }

    /// Creates a new `ChainCoder` for encoding some symbols together with the data
    /// previously obtained from [`into_remaining`].
    ///
    /// See [`into_remaining`] for detailed explanation.
    ///
    /// [`into_remaining`]: Self::into_remaining
    pub fn from_remaining(
        mut remaining: RemainingBackend,
    ) -> Result<Self, CoderError<RemainingBackend, RemainingBackend::ReadError>>
    where
        RemainingBackend: ReadWords<Word, Stack>,
        CompressedBackend: Default,
    {
        let compressed_head = match remaining.read()?.and_then(Word::into_nonzero) {
            Some(word) => word,
            _ => return Err(CoderError::Frontend(remaining)),
        };
        let mut heads = match ChainCoderHeads::new(&mut remaining, false) {
            Ok(heads) => heads,
            Err(CoderError::Frontend(())) => return Err(CoderError::Frontend(remaining)),
            Err(CoderError::Backend(err)) => return Err(CoderError::Backend(err)),
        };
        heads.compressed = compressed_head;

        let compressed = CompressedBackend::default();

        Ok(Self {
            compressed,
            remaining: remaining,
            heads,
        })
    }

    /// Terminates encoding if possible and returns the compressed data as a tuple `(prefix,
    /// suffix)`
    ///
    /// Call this method only if the original `ChainCoder` used for decoding was constructed
    /// with [`ChainCoder::from_compressed`] (typically if the original data came from an
    /// [`AnsCoder`]). If the original `ChainCoder` was instead constructed with
    /// [`ChainCoder::from_binary`] then call [`.into_binary()`] instead.
    ///
    /// Returns an error unless there's currently an integer amount of `Words` in the
    /// compressed data (which will be the case if you've used the `ChainCoder` correctly,
    /// see also [`is_whole`]).
    ///
    /// See [`into_remaining`] for usage instructions.
    ///
    /// [`is_whole`]: Self::is_whole
    /// [`AnsCoder`]: super::stack::AnsCoder
    /// [`.into_binary()`]: Self::into_binary
    /// [`into_remaining`]: Self::into_remaining
    pub fn into_compressed(
        mut self,
    ) -> Result<
        (RemainingBackend, CompressedBackend),
        CoderError<Self, CompressedBackend::WriteError>,
    >
    where
        CompressedBackend: WriteWords<Word>,
    {
        if !self.is_whole() {
            return Err(CoderError::Frontend(self));
        }

        // Transfer remaining head onto `compressed`.
        while self.heads.remaining != State::zero() {
            self.compressed.write(self.heads.remaining.as_())?;
            self.heads.remaining = self.heads.remaining >> Word::BITS;
        }

        Ok((self.remaining, self.compressed))
    }

    /// Terminates encoding if possible and returns the compressed data as a tuple `(prefix,
    /// suffix)`
    ///
    /// Call this method only if the original `ChainCoder` used for decoding was constructed
    /// with [`ChainCoder::from_binary`]. If the original `ChainCoder` was instead
    /// constructed with [`ChainCoder::from_compressed`] then call [`.into_compressed()`]
    /// instead.
    ///
    /// Returns an error unless there's currently an integer amount of `Words` in the both
    /// the compressed data and the remaining data (which will be the case if you've used
    /// the `ChainCoder` correctly and if the original chain coder was constructed with
    /// `from_binary` rather than `from_compressed`).
    ///
    /// See [`into_remaining`] for usage instructions.
    ///
    /// [`is_whole`]: Self::is_whole
    /// [`AnsCoder`]: super::stack::AnsCoder
    /// [`.into_compressed()`]: Self::into_compressed
    /// [`into_remaining`]: Self::into_remaining
    pub fn into_binary(
        mut self,
    ) -> Result<
        (RemainingBackend, CompressedBackend),
        CoderError<Self, CompressedBackend::WriteError>,
    >
    where
        CompressedBackend: WriteWords<Word>,
    {
        if !self.is_whole()
            || (State::BITS - self.heads.remaining.leading_zeros() as usize - 1) % Word::BITS != 0
        {
            return Err(CoderError::Frontend(self));
        }

        // Transfer remaining head onto `compressed`.
        while self.heads.remaining > State::one() {
            self.compressed.write(self.heads.remaining.as_())?;
            self.heads.remaining = self.heads.remaining >> Word::BITS;
        }

        debug_assert!(self.heads.remaining == State::one());

        Ok((self.remaining, self.compressed))
    }

    /// Returns `true` iff there's currently an integer amount of `Words` in the compressed data
    #[inline(always)]
    pub fn is_whole(&self) -> bool {
        self.heads.compressed.get() == Word::one()
    }

    pub fn encode_symbols_reverse<S, D, I>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<(), EncoderError<Word, CompressedBackend, RemainingBackend>>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Word>,
        Word: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = (S, D)>,
        I::IntoIter: DoubleEndedIterator,
        CompressedBackend: WriteWords<Word>,
        RemainingBackend: ReadWords<Word, Stack>,
    {
        self.encode_symbols(symbols_and_models.into_iter().rev())
    }

    pub fn try_encode_symbols_reverse<S, D, E, I>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<(), TryCodingError<EncoderError<Word, CompressedBackend, RemainingBackend>, E>>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Word>,
        Word: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = core::result::Result<(S, D), E>>,
        I::IntoIter: DoubleEndedIterator,
        CompressedBackend: WriteWords<Word>,
        RemainingBackend: ReadWords<Word, Stack>,
    {
        self.try_encode_symbols(symbols_and_models.into_iter().rev())
    }

    pub fn encode_iid_symbols_reverse<S, D, I>(
        &mut self,
        symbols: I,
        model: &D,
    ) -> Result<(), EncoderError<Word, CompressedBackend, RemainingBackend>>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Word>,
        Word: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = S>,
        I::IntoIter: DoubleEndedIterator,
        CompressedBackend: WriteWords<Word>,
        RemainingBackend: ReadWords<Word, Stack>,
    {
        self.encode_iid_symbols(symbols.into_iter().rev(), model)
    }

    pub fn increase_precision<const NEW_PRECISION: usize>(
        mut self,
    ) -> Result<
        ChainCoder<Word, State, CompressedBackend, RemainingBackend, NEW_PRECISION>,
        CoderError<Infallible, BackendError<Infallible, RemainingBackend::WriteError>>,
    >
    where
        RemainingBackend: WriteWords<Word>,
    {
        assert!(NEW_PRECISION >= PRECISION);
        assert!(NEW_PRECISION <= Word::BITS);
        assert!(State::BITS >= Word::BITS + NEW_PRECISION);

        if self.heads.remaining >= State::one() << (State::BITS - NEW_PRECISION) {
            self.flush_remaining_head()?;
        }

        Ok(ChainCoder {
            compressed: self.compressed,
            remaining: self.remaining,
            heads: ChainCoderHeads {
                compressed: self.heads.compressed,
                remaining: self.heads.remaining,
            },
        })
    }

    pub fn decrease_precision<const NEW_PRECISION: usize>(
        mut self,
    ) -> Result<
        ChainCoder<Word, State, CompressedBackend, RemainingBackend, NEW_PRECISION>,
        CoderError<EncoderFrontendError, BackendError<Infallible, RemainingBackend::ReadError>>,
    >
    where
        RemainingBackend: ReadWords<Word, Stack>,
    {
        assert!(NEW_PRECISION <= PRECISION);
        assert!(NEW_PRECISION > 0);

        if self.heads.remaining < State::one() << (State::BITS - NEW_PRECISION - Word::BITS) {
            // Won't truncate since, from the above check it follows that we satisfy the contract
            // `self.heads.remaining < 1 << (State::BITS - Word::BITS)`.
            self.refill_remaining_head()?
        }

        Ok(ChainCoder {
            compressed: self.compressed,
            remaining: self.remaining,
            heads: ChainCoderHeads {
                compressed: self.heads.compressed,
                remaining: self.heads.remaining,
            },
        })
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
    /// use constriction::stream::{models::LeakyQuantizer, Decode, chain::DefaultChainCoder};
    ///
    /// // Construct two entropy models with 24 bits and 20 bits of precision, respectively.
    /// let continuous_distribution = probability::distribution::Gaussian::new(0.0, 10.0);
    /// let quantizer24 = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let quantizer20 = LeakyQuantizer::<_, _, u32, 20>::new(-100..=100);
    /// let distribution24 = quantizer24.quantize(continuous_distribution);
    /// let distribution20 = quantizer20.quantize(continuous_distribution);
    ///
    /// // Construct a `ChainCoder` and decode some data with the 24 bit precision entropy model.
    /// let data = vec![0x0123_4567u32, 0x89ab_cdef];
    /// let mut coder = DefaultChainCoder::from_binary(data).unwrap();
    /// let _symbol_a = coder.decode_symbol(distribution24);
    ///
    /// // Change `coder`'s precision and decode data with the 20 bit precision entropy model.
    /// // The compiler can infer the new precision based on how `coder` will be used.
    /// let mut coder = coder.change_precision().unwrap();
    /// let _symbol_b = coder.decode_symbol(distribution20);
    /// ```
    ///
    /// [`stable::Encoder`]: Encoder
    /// [`into_decoder`]: Encoder::into_decoder
    #[inline(always)]
    pub fn change_precision<const NEW_PRECISION: usize>(
        self,
    ) -> Result<
        ChainCoder<Word, State, CompressedBackend, RemainingBackend, NEW_PRECISION>,
        ChangePrecisionError<Word, RemainingBackend>,
    >
    where
        RemainingBackend: WriteWords<Word> + ReadWords<Word, Stack>,
    {
        if NEW_PRECISION > PRECISION {
            self.increase_precision()
                .map_err(ChangePrecisionError::Increase)
        } else {
            self.decrease_precision()
                .map_err(ChangePrecisionError::Decrease)
        }
    }

    #[inline(always)]
    /// This would flush meaningless zero bits if `self.heads.remaining < 1 << Word::BITS`.
    fn flush_remaining_head<FrontendError, ReadError>(
        &mut self,
    ) -> Result<(), CoderError<FrontendError, BackendError<ReadError, RemainingBackend::WriteError>>>
    where
        RemainingBackend: WriteWords<Word>,
    {
        self.remaining
            .write(self.heads.remaining.as_())
            .map_err(|err| CoderError::Backend(BackendError::Remaining(err)))?;
        self.heads.remaining = self.heads.remaining >> Word::BITS;
        Ok(())
    }

    /// This truncates if `self.heads.remaining >= 1 << (State::BITS - Word::BITS)`.
    #[inline(always)]
    fn refill_remaining_head<WriteError>(
        &mut self,
    ) -> Result<
        (),
        CoderError<EncoderFrontendError, BackendError<WriteError, RemainingBackend::ReadError>>,
    >
    where
        RemainingBackend: ReadWords<Word, Stack>,
    {
        let word = self
            .remaining
            .read()
            .map_err(|err| CoderError::Backend(BackendError::Remaining(err)))?
            .ok_or(CoderError::Frontend(EncoderFrontendError::OutOfRemaining))?;
        self.heads.remaining = (self.heads.remaining << Word::BITS) | word.into();
        Ok(())
    }
}

impl<Word, State, CompressedBackend, RemainingBackend, const PRECISION: usize> Code
    for ChainCoder<Word, State, CompressedBackend, RemainingBackend, PRECISION>
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

#[allow(type_alias_bounds)]
pub type DecoderError<
    Word,
    CompressedBackend: ReadWords<Word, Stack>,
    RemainingBackend: WriteWords<Word>,
> = CoderError<
    DecoderFrontendError,
    BackendError<CompressedBackend::ReadError, RemainingBackend::WriteError>,
>;

#[allow(type_alias_bounds)]
pub type EncoderError<
    Word,
    CompressedBackend: WriteWords<Word>,
    RemainingBackend: ReadWords<Word, Stack>,
> = CoderError<
    EncoderFrontendError,
    BackendError<CompressedBackend::WriteError, RemainingBackend::ReadError>,
>;

/// Frontend error type for misuse of a [`ChainCoder`] for decoding.
#[derive(Debug, PartialEq, Eq)]
pub enum DecoderFrontendError {
    OutOfCompressedData,
}

impl core::fmt::Display for DecoderFrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfCompressedData => {
                write!(f, "Out of compressed data.")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecoderFrontendError {}

/// Frontend error type for misuse of a [`ChainCoder`] for encoding.
#[derive(Debug, PartialEq, Eq)]
pub enum EncoderFrontendError {
    OutOfRemaining,
    ImpossibleSymbol,
}

impl core::fmt::Display for EncoderFrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfRemaining => {
                write!(f, "Out of remaining information from previous decoding.")
            }
            Self::ImpossibleSymbol => DefaultEncoderFrontendError::ImpossibleSymbol.fmt(f),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EncoderFrontendError {}

/// Error type for backend errors in a [`stable::Decoder`].
///
/// [`stable::Decoder`]: Decoder
#[derive(Debug, PartialEq, Eq)]
pub enum BackendError<CompressedBackendError, RemainingBackendError> {
    Compressed(CompressedBackendError),
    Remaining(RemainingBackendError),
}

impl<CompressedBackendError: Display, RemainingBackendError: Display> core::fmt::Display
    for BackendError<CompressedBackendError, RemainingBackendError>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Compressed(err) => {
                write!(f, "Read/write error when accessing compressed: {}", err)
            }
            Self::Remaining(err) => {
                write!(f, "Read/write error when accessing remaining: {}", err)
            }
        }
    }
}

#[cfg(feature = "std")]
impl<
        CompressedBackendError: std::error::Error + 'static,
        RemainingBackendError: std::error::Error + 'static,
    > std::error::Error for BackendError<CompressedBackendError, RemainingBackendError>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compressed(err) => Some(err),
            Self::Remaining(err) => Some(err),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum ChangePrecisionError<Word, RemainingBackend>
where
    RemainingBackend: WriteWords<Word> + ReadWords<Word, Stack>,
{
    Increase(CoderError<Infallible, BackendError<Infallible, RemainingBackend::WriteError>>),
    Decrease(
        CoderError<EncoderFrontendError, BackendError<Infallible, RemainingBackend::ReadError>>,
    ),
}

impl<Word, RemainingBackend> Display for ChangePrecisionError<Word, RemainingBackend>
where
    RemainingBackend: WriteWords<Word> + ReadWords<Word, Stack>,
    RemainingBackend::WriteError: Display,
    RemainingBackend::ReadError: Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ChangePrecisionError::Increase(err) => {
                write!(
                    f,
                    "Error while increasing precision of chain coder: {}",
                    err
                )
            }
            ChangePrecisionError::Decrease(err) => {
                write!(
                    f,
                    "Error while decreasing precision of chain coder: {}",
                    err
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl<Word, RemainingBackend> std::error::Error for ChangePrecisionError<Word, RemainingBackend>
where
    Self: core::fmt::Debug,
    RemainingBackend: WriteWords<Word> + ReadWords<Word, Stack>,
    RemainingBackend::WriteError: std::error::Error + 'static,
    RemainingBackend::ReadError: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Increase(err) => Some(err),
            Self::Decrease(err) => Some(err),
        }
    }
}

impl<Word, State, CompressedBackend, RemainingBackend, const PRECISION: usize> PosSeek
    for ChainCoder<Word, State, CompressedBackend, RemainingBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    CompressedBackend: PosSeek,
    RemainingBackend: PosSeek,
{
    type Position = (
        BackendPosition<CompressedBackend::Position, RemainingBackend::Position>,
        ChainCoderHeads<Word, State, PRECISION>,
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BackendPosition<CompressedPosition, RemainingPosition> {
    pub compressed: CompressedPosition,
    pub remaining: RemainingPosition,
}

impl<Word, State, CompressedBackend, RemainingBackend, const PRECISION: usize> Pos
    for ChainCoder<Word, State, CompressedBackend, RemainingBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    CompressedBackend: Pos,
    RemainingBackend: Pos,
{
    fn pos(&self) -> Self::Position {
        (
            BackendPosition {
                compressed: self.compressed.pos(),
                remaining: self.remaining.pos(),
            },
            self.state(),
        )
    }
}

impl<Word, State, CompressedBackend, RemainingBackend, const PRECISION: usize> Seek
    for ChainCoder<Word, State, CompressedBackend, RemainingBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    CompressedBackend: Seek,
    RemainingBackend: Seek,
{
    fn seek(&mut self, (pos, state): Self::Position) -> Result<(), ()> {
        self.compressed.seek(pos.compressed)?;
        self.remaining.seek(pos.remaining)?;

        // `state` is valid since we don't provide a public API to modify fields of
        // `ChainCoderHeads` individually.
        self.heads = state;

        Ok(())
    }
}

impl<Word, State, CompressedBackend, RemainingBackend, const PRECISION: usize> Decode<PRECISION>
    for ChainCoder<Word, State, CompressedBackend, RemainingBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    CompressedBackend: ReadWords<Word, Stack>,
    RemainingBackend: WriteWords<Word>,
{
    type FrontendError = DecoderFrontendError;

    type BackendError = BackendError<CompressedBackend::ReadError, RemainingBackend::WriteError>;

    fn decode_symbol<D>(
        &mut self,
        model: D,
    ) -> Result<D::Symbol, DecoderError<Word, CompressedBackend, RemainingBackend>>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<D::Probability>,
    {
        assert!(PRECISION <= Word::BITS);
        assert!(PRECISION != 0);
        assert!(State::BITS >= Word::BITS + PRECISION);

        let word = if PRECISION == Word::BITS
            || self.heads.compressed.get() < Word::one() << PRECISION
        {
            let word = self
                .compressed
                .read()
                .map_err(BackendError::Compressed)?
                .ok_or(CoderError::Frontend(
                    DecoderFrontendError::OutOfCompressedData,
                ))?;
            if PRECISION != Word::BITS {
                self.heads.compressed = unsafe {
                    // SAFETY:
                    // - `0 < PRECISION < Word::BITS` as per our assertion and the above check,
                    //   therefore `Word::BITS - PRECISION > 0` and both the left-shift and
                    //   the right-shift are valid;
                    // - `heads.compressed.get() != 0` sinze `heads.compressed` is a `NonZero`.
                    // - `heads.compressed.get() < 1 << PRECISION`, so all its "one" bits are
                    //   in the `PRECISION` lowest significant bits; since it we have
                    //   `Word::BITS` bits available, shifting left by `Word::BITS - PRECISION`
                    //   doesn't truncate, and thus the result is also nonzero.
                    Word::NonZero::new_unchecked(
                        self.heads.compressed.get() << (Word::BITS - PRECISION) | word >> PRECISION,
                    )
                };
            }
            word
        } else {
            let word = self.heads.compressed.get();
            self.heads.compressed = unsafe {
                // SAFETY: `heads.compressed.get() >= 1 << PRECISION`, so shifting right by
                // `PRECISION` doesn't result in zero.
                Word::NonZero::new_unchecked(self.heads.compressed.get() >> PRECISION)
            };
            word
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
        // - we maintain the invariant `heads.remaining < 1 << (State::BITS - PRECISION)`; and
        // - `probability <= 1 << PRECISION` and `remainder < probability`.
        // Thus, `remaining * proability + remainder < (remaining + 1) * probability`
        // which is `<= (1 << (State::BITS - PRECISION)) << PRECISION = 1 << State::BITS`.
        self.heads.remaining =
            self.heads.remaining * probability.get().into().into() + remainder.into().into();

        if self.heads.remaining >= State::one() << (State::BITS - PRECISION) {
            // The invariant on `self.heads.remaining` (see its doc comment) is violated and must
            // be restored.
            self.flush_remaining_head()?;
        }

        Ok(symbol)
    }

    fn maybe_exhausted(&self) -> bool {
        self.compressed.maybe_exhausted() || self.remaining.maybe_full()
    }
}

impl<Word, State, CompressedBackend, RemainingBackend, const PRECISION: usize> Encode<PRECISION>
    for ChainCoder<Word, State, CompressedBackend, RemainingBackend, PRECISION>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    CompressedBackend: WriteWords<Word>,
    RemainingBackend: ReadWords<Word, Stack>,
{
    type FrontendError = EncoderFrontendError;
    type BackendError = BackendError<CompressedBackend::WriteError, RemainingBackend::ReadError>;

    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        model: D,
    ) -> Result<(), EncoderError<Word, CompressedBackend, RemainingBackend>>
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
            .ok_or(CoderError::Frontend(EncoderFrontendError::ImpossibleSymbol))?;

        if self.heads.remaining
            < probability.get().into().into() << (State::BITS - Word::BITS - PRECISION)
        {
            self.refill_remaining_head()?;
            // At this point, the invariant on `self.heads.remaining` (see its doc comment) is
            // temporarily violated (but it will be restored below). This is how
            // `decode_symbol` can detect that it has to flush `remaining.state`.
        }

        let remainder = (self.heads.remaining % probability.get().into().into())
            .as_()
            .as_();
        let quantile = (left_sided_cumulative + remainder).into();
        self.heads.remaining = self.heads.remaining / probability.get().into().into();

        if PRECISION != Word::BITS
            && self.heads.compressed.get() < Word::one() << (Word::BITS - PRECISION)
        {
            unsafe {
                // SAFETY:
                // - `heads.compressed` is nonzero because it is a `NonZero`
                // - `heads.compressed`, has `Word::BITS` bits and we checked above that all its one
                //   bits are within theleast significant `Word::BITS - PRECISION` bits. Thus, the
                //   most significant `PRECISION` bits are 0 and the left-shift doesn't truncate.
                // Thus, the result of the left-shift is also noznero.
                self.heads.compressed =
                    (self.heads.compressed.get() << PRECISION | quantile).into_nonzero_unchecked();
            }
        } else {
            let word = if PRECISION == Word::BITS {
                quantile
            } else {
                let word = self.heads.compressed.get() << PRECISION | quantile;
                unsafe {
                    // SAFETY: if we're here then `heads.compressed >= 1 << (Word::BITS - PRECISION).
                    // Thus, shifting right by this amount of bits leaves at least one 1 bit.
                    self.heads.compressed = (self.heads.compressed.get()
                        >> (Word::BITS - PRECISION))
                        .into_nonzero_unchecked();
                }
                word
            };
            self.compressed
                .write(word)
                .map_err(BackendError::Compressed)?;
        }

        Ok(())
    }

    fn maybe_full(&self) -> bool {
        self.remaining.maybe_exhausted() || self.compressed.maybe_full()
    }
}

#[cfg(test)]
mod tests {
    use super::super::models::LeakyQuantizer;
    use super::*;

    use probability::distribution::Gaussian;
    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };

    use alloc::vec;

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
                let mean = (200.0 / u32::MAX as f64) * rng.next_u32() as f64 - 100.0;
                let std_dev = (10.0 / u32::MAX as f64) * rng.next_u32() as f64 + 0.001;
                Gaussian::new(mean, std_dev)
            })
            .collect::<Vec<_>>();
        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-100..=100);

        let mut coder =
            ChainCoder::<Word, State, Vec<Word>, Vec<Word>, PRECISION>::from_compressed(
                compressed.clone(),
            )
            .unwrap();

        let symbols = coder
            .decode_symbols(
                distributions
                    .iter()
                    .map(|&distribution| quantizer.quantize(distribution)),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(!coder.maybe_exhausted());

        let (remaining_prefix, remaining_suffix) = coder.clone().into_remaining().unwrap();
        let mut remaining = remaining_prefix.clone();
        remaining.extend_from_slice(&remaining_suffix);
        let coder2 = ChainCoder::from_remaining(remaining).unwrap();
        let coder3 = ChainCoder::from_remaining(remaining_suffix).unwrap();

        for (mut coder, prefix) in vec![
            (coder, vec![]),
            (coder2, vec![]),
            (coder3, remaining_prefix),
        ] {
            coder
                .encode_symbols_reverse(
                    symbols
                        .iter()
                        .zip(&distributions)
                        .map(|(&symbol, &distribution)| (symbol, quantizer.quantize(distribution))),
                )
                .unwrap();

            let (compressed_prefix, compressed_suffix) = coder.into_compressed().unwrap();

            let mut reconstructed = prefix;
            reconstructed.extend(compressed_prefix);
            reconstructed.extend(compressed_suffix);

            assert_eq!(reconstructed, compressed);
        }
    }
}