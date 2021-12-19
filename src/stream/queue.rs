//! Near-optimal compression on a queue ("first in first out")
//!
//! This module provides an implementation of the Range Coding algorithm [1], an entropy
//! coder with near-optimal compression effectiveness that operates as a *queue* data
//! structure. Range Coding is a more computationally efficient variant of Arithmetic
//! Coding.
//!
//! # Comparison to sister module `stack`
//!
//! Range Coding operates as a *queue*: decoding a sequence of symbols yields the symbols in
//! the same order in which they were encoded. This is unlike the case with the [`AnsCoder`]
//! in the sister module [`stack`], which decodes in reverse order. Therefore, Range Coding
//! is typically the preferred method for autoregressive models. On the other hand, the
//! provided implementation of Range Coding uses two distinct data structures,
//! [`RangeEncoder`] and [`RangeDecoder`], for encoding and decoding, respectively. This
//! means that, unlike the case with the `AnsCoder`, encoding and decoding operations on a
//! Range Coder cannot be interleaved: once you've *sealed* a `RangeEncoder` (e.g., by
//! calling [`.into_compressed()`] on it) you cannot add any more compressed data onto it.
//! This makes Range Coding difficult to use for advanced compression techniques such as
//! bits-back coding with hierarchical models.
//!
//! The parent module contains a more detailed discussion of the [differences between ANS
//! Coding and Range Coding](super#which-stream-code-should-i-use) .
//!
//! # References
//!
//! [1] Pasco, Richard Clark. Source coding algorithms for fast data compression. Diss.
//! Stanford University, 1976.
//!
//! [`AnsCoder`]: super::stack::AnsCoder
//! [`stack`]: super::stack
//! [`.into_compressed()`]: RangeEncoder::into_compressed

use alloc::vec::Vec;
use core::{
    borrow::Borrow,
    fmt::{Debug, Display},
    marker::PhantomData,
    num::NonZeroUsize,
    ops::Deref,
};

use num::cast::AsPrimitive;

use super::{
    model::{DecoderModel, EncoderModel},
    Code, Decode, Encode, IntoDecoder,
};
use crate::{
    backends::{AsReadWords, BoundedReadWords, Cursor, IntoReadWords, ReadWords, WriteWords},
    BitArray, CoderError, DefaultEncoderError, DefaultEncoderFrontendError, NonZeroBitArray, Pos,
    PosSeek, Queue, Seek, UnwrapInfallible,
};

/// Type of the internal state used by [`RangeEncoder<Word, State>`] and
/// [`RangeDecoder<Word, State>`]. Relevant for [`Seek`]ing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RangeCoderState<Word, State: BitArray> {
    lower: State,

    /// Invariant: `range >= State::one() << (State::BITS - Word::BITS)`
    /// Therefore, the highest order `Word` of `lower` is always sufficient to
    /// identify the current interval, so only it has to be flushed at the end.
    range: State::NonZero,

    /// We keep track of the `Word` type so that we can statically enforce
    /// the invariants for `lower` and `range`.
    phantom: PhantomData<Word>,
}

impl<Word, State: BitArray> RangeCoderState<Word, State> {
    /// Get the lower bound of the current range (inclusive)
    pub fn lower(&self) -> State {
        self.lower
    }

    /// Get the size of the current range
    pub fn range(&self) -> State::NonZero {
        self.range
    }
}

impl<Word: BitArray, State: BitArray> Default for RangeCoderState<Word, State> {
    fn default() -> Self {
        Self {
            lower: State::zero(),
            range: State::max_value().into_nonzero().expect("max_value() != 0"),
            phantom: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RangeEncoder<Word, State, Backend = Vec<Word>>
where
    Word: BitArray,
    State: BitArray,
    Backend: WriteWords<Word>,
{
    bulk: Backend,
    state: RangeCoderState<Word, State>,
    situation: EncoderSituation<Word>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderSituation<Word> {
    Normal,

    /// Wraps `num_inverted` and `first_inverted_lower_word`
    Inverted(NonZeroUsize, Word),
}

impl<Word> Default for EncoderSituation<Word> {
    fn default() -> Self {
        Self::Normal
    }
}

/// Type alias for an [`RangeEncoder`] with sane parameters for typical use cases.
pub type DefaultRangeEncoder<Backend = Vec<u32>> = RangeEncoder<u32, u64, Backend>;

/// Type alias for a [`RangeEncoder`] for use with [lookup models]
///
/// This encoder has a smaller word size and internal state than [`DefaultRangeEncoder`]. It
/// is optimized for use with lookup entropy models, in particular with a
/// [`DefaultEncoderArrayLookupTable`] or a [`DefaultEncoderHashLookupTable`].
///
/// # Examples
///
/// See [`DefaultEncoderArrayLookupTable`] and [`DefaultEncoderHashLookupTable`].
///
/// # See also
///
/// - [`SmallRangeDecoder`]
///
/// [lookup models]: super::model::lookup
/// [`DefaultEncoderArrayLookupTable`]: super::model::lookup::DefaultEncoderArrayLookupTable
/// [`DefaultEncoderHashLookupTable`]: super::model::lookup::DefaultEncoderHashLookupTable
pub type SmallRangeEncoder<Backend = Vec<u16>> = RangeEncoder<u16, u32, Backend>;

impl<Word, State, Backend> Code for RangeEncoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word>,
{
    type State = RangeCoderState<Word, State>;
    type Word = Word;

    fn state(&self) -> Self::State {
        self.state
    }
}

impl<Word, State, Backend> PosSeek for RangeEncoder<Word, State, Backend>
where
    Word: BitArray,
    State: BitArray,
    Backend: WriteWords<Word> + PosSeek,
    Self: Code,
{
    type Position = (Backend::Position, <Self as Code>::State);
}

impl<Word, State, Backend> Pos for RangeEncoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + Pos<Position = usize>,
{
    fn pos(&self) -> Self::Position {
        let num_inverted = if let EncoderSituation::Inverted(num_inverted, _) = self.situation {
            num_inverted.get()
        } else {
            0
        };
        (self.bulk.pos() + num_inverted, self.state())
    }
}

impl<Word, State, Backend> Default for RangeEncoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + Default,
{
    /// This is essentially the same as `#[derive(Default)]`, except for the assertions on
    /// `State::BITS` and `Word::BITS`.
    fn default() -> Self {
        Self::with_backend(Backend::default())
    }
}

impl<Word, State> RangeEncoder<Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    /// Creates an empty encoder for range coding.
    pub fn new() -> Self {
        assert!(State::BITS >= 2 * Word::BITS);
        assert_eq!(State::BITS % Word::BITS, 0);

        Self {
            bulk: Vec::new(),
            state: RangeCoderState::default(),
            situation: EncoderSituation::Normal,
        }
    }
}

impl<Word, State> From<RangeEncoder<Word, State>> for Vec<Word>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn from(val: RangeEncoder<Word, State>) -> Self {
        val.into_compressed().unwrap_infallible()
    }
}

impl<Word, State, Backend> RangeEncoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word>,
{
    /// Assumes that the `backend` is in a state where the encoder can start writing as if
    /// it was an empty backend. If there's already some compressed data on `backend`, then
    /// this method will just concatanate the new sequence of `Word`s to the existing
    /// sequence of `Word`s without gluing them together. This is likely not what you want
    /// since you won't be able to decode the data in one go (however, it is Ok to
    /// concatenate arbitrary data to the output of a `RangeEncoder`; it won't invalidate
    /// the existing data).
    ///
    /// If you need an entropy coder that can be interrupted and serialized/deserialized
    /// (i.e., an encoder that can encode some symbols, return the compressed bit string as
    /// a sequence of `Words`, load the `Words` back in at a later point and then encode
    /// some more symbols), then consider using an [`AnsCoder`].
    ///
    /// TODO: rename to `with_write_backend` and then add the same method to `AnsCoder`
    ///
    /// [`AnsCoder`]: super::stack::AnsCoder
    pub fn with_backend(backend: Backend) -> Self {
        assert!(State::BITS >= 2 * Word::BITS);
        assert_eq!(State::BITS % Word::BITS, 0);

        Self {
            bulk: backend,
            state: RangeCoderState::default(),
            situation: EncoderSituation::Normal,
        }
    }

    /// Check if no data has been encoded yet.
    pub fn is_empty<'a>(&'a self) -> bool
    where
        Backend: AsReadWords<'a, Word, Queue>,
        Backend::AsReadWords: BoundedReadWords<Word, Queue>,
    {
        self.state.range.get() == State::max_value() && self.bulk.as_read_words().is_exhausted()
    }

    /// Same as `Encoder::maybe_full`, but can be called on a concrete type without type
    /// annotations.
    pub fn maybe_full(&self) -> bool {
        self.bulk.maybe_full()
    }

    /// Same as IntoDecoder::into_decoder(self) but can be used for any `PRECISION`
    /// and therefore doesn't require type arguments on the caller side.
    ///
    /// TODO: there should also be a `decoder()` method that takes `&mut self`
    #[allow(clippy::result_unit_err)]
    pub fn into_decoder(self) -> Result<RangeDecoder<Word, State, Backend::IntoReadWords>, ()>
    where
        Backend: IntoReadWords<Word, Queue>,
    {
        // TODO: return proper error (or just box it up).
        RangeDecoder::from_compressed(self.into_compressed().map_err(|_| ())?).map_err(|_| ())
    }

    pub fn into_compressed(mut self) -> Result<Backend, Backend::WriteError> {
        self.seal()?;
        Ok(self.bulk)
    }

    /// Private method; flushes held-back words if in inverted situation and adds one or two
    /// additional words that identify the range regardless of what the compressed data may
    /// be concatenated with (unless no symbols have been encoded yet, in which case this is
    /// a no-op).
    ///
    /// Doesn't change `self.state` or `self.situation` so that this operation can be
    /// reversed if the backend supports removing words (see method `unseal`);
    fn seal(&mut self) -> Result<(), Backend::WriteError> {
        if self.state.range.get() == State::max_value() {
            // This condition only holds upon initialization because encoding a symbol first
            // reduces `range` and then only (possibly) right-shifts it, which introduces
            // some zero bits. We treat this case special and don't emit any words, so that
            // an empty sequence of symbols gets encoded to an empty sequence of words.
            return Ok(());
        }

        let point = self
            .state
            .lower
            .wrapping_add(&((State::one() << (State::BITS - Word::BITS)) - State::one()));

        if let EncoderSituation::Inverted(num_inverted, first_inverted_lower_word) = self.situation
        {
            let (first_word, consecutive_words) = if point < self.state.lower {
                // Unlikely case (addition has wrapped).
                (first_inverted_lower_word + Word::one(), Word::zero())
            } else {
                // Likely case.
                (first_inverted_lower_word, Word::max_value())
            };

            self.bulk.write(first_word)?;
            for _ in 1..num_inverted.get() {
                self.bulk.write(consecutive_words)?;
            }
        }

        let point_word = (point >> (State::BITS - Word::BITS)).as_();
        self.bulk.write(point_word)?;

        let upper_word = (self.state.lower.wrapping_add(&self.state.range.get())
            >> (State::BITS - Word::BITS))
            .as_();
        if upper_word == point_word {
            self.bulk.write(Word::zero())?;
        }

        Ok(())
    }

    fn num_seal_words(&self) -> usize {
        if self.state.range.get() == State::max_value() {
            return 0;
        }

        let point = self
            .state
            .lower
            .wrapping_add(&((State::one() << (State::BITS - Word::BITS)) - State::one()));
        let point_word = (point >> (State::BITS - Word::BITS)).as_();
        let upper_word = (self.state.lower.wrapping_add(&self.state.range.get())
            >> (State::BITS - Word::BITS))
            .as_();
        let mut count = if upper_word == point_word { 2 } else { 1 };

        if let EncoderSituation::Inverted(num_inverted, _) = self.situation {
            count += num_inverted.get();
        }
        count
    }

    /// Returns the number of compressed words on the ans.
    ///
    /// This includes a constant overhead of between one and two words unless the
    /// coder is completely empty.
    ///
    /// This method returns the length of the slice, the `Vec<Word>`, or the iterator
    /// that would be returned by [`get_compressed`], [`into_compressed`], or
    /// [`iter_compressed`], respectively, when called at this time.
    ///
    /// See also [`num_bits`].
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`num_bits`]: #method.num_bits
    pub fn num_words<'a>(&'a self) -> usize
    where
        Backend: AsReadWords<'a, Word, Queue>,
        Backend::AsReadWords: BoundedReadWords<Word, Queue>,
    {
        self.bulk.as_read_words().remaining() + self.num_seal_words()
    }

    /// Returns the size of the current queue of compressed data in bits.
    ///
    /// This includes some constant overhead unless the coder is completely empty
    /// (see [`num_words`](#method.num_words)).
    ///
    /// The returned value is a multiple of the bitlength of the compressed word
    /// type `Word`.
    pub fn num_bits<'a>(&'a self) -> usize
    where
        Backend: AsReadWords<'a, Word, Queue>,
        Backend::AsReadWords: BoundedReadWords<Word, Queue>,
    {
        Word::BITS * self.num_words()
    }

    pub fn bulk(&self) -> &Backend {
        &self.bulk
    }
}

impl<Word, State> RangeEncoder<Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    /// Discards all compressed data and resets the coder to the same state as
    /// [`Coder::new`](#method.new).
    pub fn clear(&mut self) {
        self.bulk.clear();
        self.state = RangeCoderState::default();
    }

    /// Assembles the current compressed data into a single slice.
    ///
    /// This method is only implemented for encoders backed by a `Vec<Word>`
    /// because we have to temporarily seal the encoder and then unseal it when the returned
    /// `EncoderGuard` is dropped, which requires precise knowledge of the backend (and
    /// which is also the reason why this method takes a `&mut self`receiver). If you're
    /// using a different backend than a `Vec`, consider calling [`into_compressed`]
    /// instead.
    ///
    /// TODO: update following documentation.
    ///
    /// This method is similar to [`as_compressed_raw`] with the difference that it
    /// concatenates the `bulk` and `head` before returning them. The concatenation
    /// truncates any trailing zero words, which is compatible with the constructor
    /// [`from_compressed`].
    ///
    /// This method requires a `&mut self` receiver. If you only have a shared reference to
    /// a `Coder`, consider calling [`as_compressed_raw`] or [`iter_compressed`] instead.
    ///
    /// The returned `CoderGuard` dereferences to `&[Word]`, thus providing
    /// read-only access to the compressed data. If you need ownership of the compressed
    /// data, consider calling [`into_compressed`] instead.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{
    ///     model::DefaultContiguousCategoricalEntropyModel, stack::DefaultAnsCoder, Decode
    /// };
    ///
    /// let mut coder = DefaultAnsCoder::new();
    ///
    /// // Push some data on the coder.
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let model = DefaultContiguousCategoricalEntropyModel
    ///     ::from_floating_point_probabilities(&probabilities).unwrap();
    /// coder.encode_iid_symbols_reverse(&symbols, &model).unwrap();
    ///
    /// // Inspect the compressed data.
    /// dbg!(coder.get_compressed());
    ///
    /// // We can still use the coder afterwards.
    /// let reconstructed = coder
    ///     .decode_iid_symbols(4, &model)
    ///     .collect::<Result<Vec<_>, _>>()
    ///     .unwrap();
    /// assert_eq!(reconstructed, symbols);
    /// ```
    ///
    /// TODO: this is currently out of date
    ///
    /// [`as_compressed_raw`]: #method.as_compressed_raw [`from_compressed`]:
    /// #method.from_compressed [`iter_compressed`]: #method.iter_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn get_compressed(&mut self) -> EncoderGuard<'_, Word, State> {
        EncoderGuard::new(self)
    }

    /// A decoder for temporary use.
    ///
    /// Once the returned decoder gets dropped, you can continue using this encoder. If you
    /// don't need this flexibility, call [`into_decoder`] instead.
    ///
    /// This method is only implemented for encoders backed by a `Vec<Word>`
    /// because we have to temporarily seal the encoder and then unseal it when the returned
    /// decoder is dropped, which requires precise knowledge of the backend (and which is
    /// also the reason why this method takes a `&mut self`receiver). If you're using a
    /// different backend than a `Vec`, consider calling [`into_decoder`] instead.
    pub fn decoder(
        &mut self,
    ) -> RangeDecoder<Word, State, Cursor<Word, EncoderGuard<'_, Word, State>>> {
        RangeDecoder::from_compressed(self.get_compressed()).unwrap_infallible()
    }

    fn unseal(&mut self) {
        for _ in 0..self.num_seal_words() {
            let word = self.bulk.pop();
            debug_assert!(word.is_some());
        }
    }
}

impl<Word, State, Backend, const PRECISION: usize> IntoDecoder<PRECISION>
    for RangeEncoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + IntoReadWords<Word, Queue>,
{
    type IntoDecoder = RangeDecoder<Word, State, Backend::IntoReadWords>;

    fn into_decoder(self) -> Self::IntoDecoder {
        self.into()
    }
}

impl<Word, State, Backend, const PRECISION: usize> Encode<PRECISION>
    for RangeEncoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word>,
{
    type FrontendError = DefaultEncoderFrontendError;
    type BackendError = Backend::WriteError;

    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        model: D,
    ) -> Result<(), DefaultEncoderError<Self::BackendError>>
    where
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<D::Probability>,
    {
        // We maintain the following invariant (*):
        //   range >= State::one() << (State::BITS - Word::BITS)

        let (left_sided_cumulative, probability) = model
            .left_cumulative_and_probability(symbol)
            .ok_or_else(|| DefaultEncoderFrontendError::ImpossibleSymbol.into_coder_error())?;

        let scale = self.state.range.get() >> PRECISION;
        // This cannot overflow since `scale * probability <= (range >> PRECISION) << PRECISION`
        self.state.range = (scale * probability.get().into().into())
            .into_nonzero()
            .ok_or_else(|| DefaultEncoderFrontendError::ImpossibleSymbol.into_coder_error())?;
        let new_lower = self
            .state
            .lower
            .wrapping_add(&(scale * left_sided_cumulative.into().into()));

        if let EncoderSituation::Inverted(num_inverted, first_inverted_lower_word) = self.situation
        {
            // unlikely branch
            if new_lower.wrapping_add(&self.state.range.get()) > new_lower {
                // We've transitioned from an inverted to a normal situation.

                let (first_word, consecutive_words) = if new_lower < self.state.lower {
                    (first_inverted_lower_word + Word::one(), Word::zero())
                } else {
                    (first_inverted_lower_word, Word::max_value())
                };

                self.bulk.write(first_word)?;
                for _ in 1..num_inverted.get() {
                    self.bulk.write(consecutive_words)?;
                }

                self.situation = EncoderSituation::Normal;
            }
        }

        self.state.lower = new_lower;

        if self.state.range.get() < State::one() << (State::BITS - Word::BITS) {
            // Invariant `range >= State::one() << (State::BITS - Word::BITS)` is
            // violated. Since `left_cumulative_and_probability` succeeded, we know that
            // `probability != 0` and therefore:
            //   range >= scale * probability = (old_range >> PRECISION) * probability
            //         >= old_range >> PRECISION
            //         >= old_range >> Word::BITS
            // where `old_range` is the `range` at method entry, which satisfied invariant (*)
            // by assumption. Therefore, the following left-shift restores the invariant:
            self.state.range = unsafe {
                // SAFETY:
                // - `range` is nonzero because it is a `State::NonZero`
                // - Shifting `range` left by `Word::BITS` bits doesn't truncate
                //   because we checked that `range < 1 << (State::BITS - Word::Bits)`.
                (self.state.range.get() << Word::BITS).into_nonzero_unchecked()
            };

            let lower_word = (self.state.lower >> (State::BITS - Word::BITS)).as_();
            self.state.lower = self.state.lower << Word::BITS;

            if let EncoderSituation::Inverted(num_inverted, _) = &mut self.situation {
                // Transition from an inverted to an inverted situation (TODO: mark as unlikely branch).
                *num_inverted = NonZeroUsize::new(num_inverted.get().wrapping_add(1))
                    .expect("Cannot encode more symbols than what's addressable with usize.");
            } else if self.state.lower.wrapping_add(&self.state.range.get()) > self.state.lower {
                // Transition from a normal to a normal situation (the most common case).
                self.bulk.write(lower_word)?;
            } else {
                // Transition from a normal to an inverted situation.
                self.situation =
                    EncoderSituation::Inverted(NonZeroUsize::new(1).expect("1 != 0"), lower_word);
            }
        }

        Ok(())
    }

    fn maybe_full(&self) -> bool {
        RangeEncoder::maybe_full(self)
    }
}

#[derive(Debug)]
pub struct RangeDecoder<Word, State, Backend>
where
    Word: BitArray,
    State: BitArray,
    Backend: ReadWords<Word, Queue>,
{
    bulk: Backend,

    state: RangeCoderState<Word, State>,

    /// Invariant: `point.wrapping_sub(&state.lower) < state.range`
    point: State,
}

/// Type alias for a [`RangeDecoder`] with sane parameters for typical use cases.
pub type DefaultRangeDecoder<Backend = Cursor<u32, Vec<u32>>> = RangeDecoder<u32, u64, Backend>;

/// Type alias for a [`RangeDecoder`] for use with [lookup models]
///
/// This encoder has a smaller word size and internal state than [`DefaultRangeDecoder`]. It
/// is optimized for use with lookup entropy models, in particular with a
/// [`DefaultDecoderIndexLookupTable`] or a [`DefaultDecoderGenericLookupTable`].
///
/// # Examples
///
/// See [`DefaultDecoderIndexLookupTable`] and [`DefaultDecoderGenericLookupTable`].
///
/// # See also
///
/// - [`SmallRangeEncoder`]
///
/// [lookup models]: super::model::lookup
/// [`DefaultEncoderArrayLookupTable`]: super::model::lookup::DefaultEncoderArrayLookupTable
/// [`DefaultEncoderHashLookupTable`]: super::model::lookup::DefaultEncoderHashLookupTable
/// [`DefaultDecoderIndexLookupTable`]: super::model::lookup::DefaultDecoderIndexLookupTable
/// [`DefaultDecoderGenericLookupTable`]: super::model::lookup::DefaultDecoderGenericLookupTable
pub type SmallRangeDecoder<Backend> = RangeDecoder<u16, u32, Backend>;

impl<Word, State, Backend> RangeDecoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: ReadWords<Word, Queue>,
{
    pub fn from_compressed<Buf>(compressed: Buf) -> Result<Self, Backend::ReadError>
    where
        Buf: IntoReadWords<Word, Queue, IntoReadWords = Backend>,
    {
        assert!(State::BITS >= 2 * Word::BITS);
        assert_eq!(State::BITS % Word::BITS, 0);

        let mut bulk = compressed.into_read_words();
        let point = Self::read_point(&mut bulk)?;

        Ok(RangeDecoder {
            bulk,
            state: RangeCoderState::default(),
            point,
        })
    }

    pub fn with_backend(backend: Backend) -> Result<Self, Backend::ReadError> {
        assert!(State::BITS >= 2 * Word::BITS);
        assert_eq!(State::BITS % Word::BITS, 0);

        let mut bulk = backend;
        let point = Self::read_point(&mut bulk)?;

        Ok(RangeDecoder {
            bulk,
            state: RangeCoderState::default(),
            point,
        })
    }

    pub fn for_compressed<'a, Buf>(compressed: &'a Buf) -> Result<Self, Backend::ReadError>
    where
        Buf: AsReadWords<'a, Word, Queue, AsReadWords = Backend>,
    {
        assert!(State::BITS >= 2 * Word::BITS);
        assert_eq!(State::BITS % Word::BITS, 0);

        let mut bulk = compressed.as_read_words();
        let point = Self::read_point(&mut bulk)?;

        Ok(RangeDecoder {
            bulk,
            state: RangeCoderState::default(),
            point,
        })
    }

    pub fn from_raw_parts(
        _bulk: Backend,
        _state: State,
    ) -> Result<Self, (Backend, RangeCoderState<Word, State>)> {
        assert!(State::BITS >= 2 * Word::BITS);
        assert_eq!(State::BITS % Word::BITS, 0);

        todo!()
    }

    pub fn into_raw_parts(self) -> (Backend, RangeCoderState<Word, State>) {
        (self.bulk, self.state)
    }

    fn read_point<B: ReadWords<Word, Queue>>(bulk: &mut B) -> Result<State, B::ReadError> {
        let mut num_read = 0;
        let mut point = State::zero();
        while let Some(word) = bulk.read()? {
            point = point << Word::BITS | word.into();
            num_read += 1;
            if num_read == State::BITS / Word::BITS {
                break;
            }
        }

        #[allow(clippy::collapsible_if)]
        if num_read < State::BITS / Word::BITS {
            if num_read != 0 {
                point = point << (State::BITS - num_read * Word::BITS);
            }
            // TODO: do we need to advance the Backend's `pos` beyond the end to make
            // `PosBackend` consistent with its implementation for the encoder?
        }

        Ok(point)
    }

    /// Same as `Decoder::maybe_exhausted`, but can be called on a concrete type without
    /// type annotations.
    pub fn maybe_exhausted(&self) -> bool {
        // The maximum possible difference between `point` and `lower`, even if the
        // compressed data was concatenated with a lot of one bits.
        let max_difference =
            ((State::one() << (State::BITS - Word::BITS)) << 1).wrapping_sub(&State::one());

        // The check for `self.state.range == State::max_value()` is for the special case of
        // an empty buffer.
        self.bulk.maybe_exhausted()
            && (self.state.range.get() == State::max_value()
                || self.point.wrapping_sub(&self.state.lower) < max_difference)
    }
}

impl<Word, State, Backend> Code for RangeDecoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: ReadWords<Word, Queue>,
{
    type State = RangeCoderState<Word, State>;
    type Word = Word;

    fn state(&self) -> Self::State {
        self.state
    }
}

impl<Word, State, Backend> PosSeek for RangeDecoder<Word, State, Backend>
where
    Word: BitArray,
    State: BitArray,
    Backend: ReadWords<Word, Queue>,
    Backend: PosSeek,
    Self: Code,
{
    type Position = (Backend::Position, <Self as Code>::State);
}

impl<Word, State, Backend> Seek for RangeDecoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: ReadWords<Word, Queue> + Seek,
{
    fn seek(&mut self, pos_and_state: Self::Position) -> Result<(), ()> {
        let (pos, state) = pos_and_state;

        self.bulk.seek(pos)?;
        self.point = Self::read_point(&mut self.bulk).map_err(|_| ())?;
        self.state = state;

        // TODO: deal with positions very close to end.

        Ok(())
    }
}

impl<Word, State, Backend> From<RangeEncoder<Word, State, Backend>>
    for RangeDecoder<Word, State, Backend::IntoReadWords>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + IntoReadWords<Word, Queue>,
{
    fn from(encoder: RangeEncoder<Word, State, Backend>) -> Self {
        // TODO: implement a `try_into_decoder` or something instead. Or specialize this
        // method to the case where both read and write error are Infallible, which is
        // probably the only place where this will be used anyway.
        encoder.into_decoder().unwrap()
    }
}

// TODO (implement for infallible case)
// impl<'a, Word, State, Backend> From<&'a mut RangeEncoder<Word, State, Backend>>
//     for RangeDecoder<Word, State, Backend::AsReadWords>
// where
//     Word: BitArray + Into<State>,
//     State: BitArray + AsPrimitive<Word>,
//     Backend: WriteWords<Word> + AsReadWords<'a, Word, Queue>,
// {
//     fn from(encoder: &'a mut RangeEncoder<Word, State, Backend>) -> Self {
//         encoder.as_decoder()
//     }
// }

impl<Word, State, Backend, const PRECISION: usize> Decode<PRECISION>
    for RangeDecoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: ReadWords<Word, Queue>,
{
    type FrontendError = DecoderFrontendError;

    type BackendError = Backend::ReadError;

    /// Decodes a single symbol and pops it off the compressed data.
    ///
    /// This is a low level method. You usually probably want to call a batch method
    /// like [`decode_symbols`](#method.decode_symbols) or
    /// [`decode_iid_symbols`](#method.decode_iid_symbols) instead.
    ///
    /// This method is called `decode_symbol` rather than `decode_symbol` to stress the
    /// fact that the `Coder` is a stack: `decode_symbol` will return the *last* symbol
    /// that was previously encoded via [`encode_symbol`](#method.encode_symbol).
    ///
    /// Note that this method cannot fail. It will still produce symbols in a
    /// deterministic way even if the coder is empty, but such symbols will not
    /// recover any previously encoded data and will generally have low entropy.
    /// Still, being able to pop off an arbitrary number of symbols can sometimes be
    /// useful in edge cases of, e.g., the bits-back algorithm.
    fn decode_symbol<D>(
        &mut self,
        model: D,
    ) -> Result<D::Symbol, CoderError<Self::FrontendError, Self::BackendError>>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<D::Probability>,
    {
        // We maintain the following invariant (*):
        //   point (-) lower < range
        // where (-) denotes wrapping subtraction (in `Self::State`).

        let scale = self.state.range.get() >> PRECISION;
        let quantile = self.point.wrapping_sub(&self.state.lower) / scale;
        if quantile >= State::one() << PRECISION {
            // This can only happen if both of the following conditions apply:
            // (i) we are decoding invalid compressed data; and
            // (ii) we use entropy models with varying `PRECISION`s.
            return Err(CoderError::Frontend(DecoderFrontendError::InvalidData));
        }

        let (symbol, left_sided_cumulative, probability) =
            model.quantile_function(quantile.as_().as_());

        // Update `state` in the same way as we do in `encode_symbol` (see comments there):
        self.state.lower = self
            .state
            .lower
            .wrapping_add(&(scale * left_sided_cumulative.into().into()));
        self.state.range = (scale * probability.get().into().into())
            .into_nonzero()
            .expect("TODO");

        // Invariant (*) is still satisfied at this point because:
        //   (point (-) lower) / scale = (point (-) old_lower) / scale (-) left_sided_cumulative
        //                             = quantile (-) left_sided_cumulative
        //                             < probability
        // Therefore, we have:
        //   point (-) lower < scale * probability <= range

        if self.state.range.get() < State::one() << (State::BITS - Word::BITS) {
            // First update `state` in the same way as we do in `encode_symbol`:
            self.state.lower = self.state.lower << Word::BITS;
            self.state.range = unsafe {
                // SAFETY:
                // - `range` is nonzero because it is a `State::NonZero`
                // - Shifting `range` left by `Word::BITS` bits doesn't truncate
                //   because we checked that `range < 1 << (State::BITS - Word::Bits)`.
                (self.state.range.get() << Word::BITS).into_nonzero_unchecked()
            };

            // Then update `point`, which restores invariant (*):
            self.point = self.point << Word::BITS;
            if let Some(word) = self.bulk.read()? {
                self.point = self.point | word.into();
            }

            // TODO: register reads past end?
        }

        Ok(symbol)
    }

    fn maybe_exhausted(&self) -> bool {
        RangeDecoder::maybe_exhausted(self)
    }
}

/// Provides temporary read-only access to the compressed data wrapped in an
/// [`Encoder`].
///
/// Dereferences to `&[Word]`. See [`Encoder::get_compressed`] for an example.
///
/// [`Coder`]: struct.Coder.html
/// [`Coder::get_compressed`]: struct.Coder.html#method.get_compressed
pub struct EncoderGuard<'a, Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    inner: &'a mut RangeEncoder<Word, State>,
}

impl<Word, State> Debug for EncoderGuard<'_, Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<'a, Word, State> EncoderGuard<'a, Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn new(encoder: &'a mut RangeEncoder<Word, State>) -> Self {
        // Append state. Will be undone in `<Self as Drop>::drop`.
        if !encoder.is_empty() {
            encoder.seal().unwrap_infallible();
        }
        Self { inner: encoder }
    }
}

impl<'a, Word, State> Drop for EncoderGuard<'a, Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn drop(&mut self) {
        self.inner.unseal();
    }
}

impl<'a, Word, State> Deref for EncoderGuard<'a, Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Target = [Word];

    fn deref(&self) -> &Self::Target {
        &self.inner.bulk
    }
}

impl<'a, Word, State> AsRef<[Word]> for EncoderGuard<'a, Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn as_ref(&self) -> &[Word] {
        self
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::dbg;

    use super::super::model::{
        ContiguousCategoricalEntropyModel, IterableEntropyModel, LeakyQuantizer,
    };
    use super::*;

    use probability::distribution::{Gaussian, Inverse};
    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };

    #[test]
    fn compress_none() {
        let encoder = DefaultRangeEncoder::new();
        assert!(encoder.is_empty());
        let compressed = encoder.into_compressed().unwrap();
        assert!(compressed.is_empty());

        let decoder = DefaultRangeDecoder::from_compressed(compressed).unwrap();
        assert!(decoder.maybe_exhausted());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_one() {
        generic_compress_few(core::iter::once(5), 1)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_two() {
        generic_compress_few([2, 8].iter().cloned(), 1)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_ten() {
        generic_compress_few(0..10, 2)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_twenty() {
        generic_compress_few(-10..10, 4)
    }

    fn generic_compress_few<I>(symbols: I, expected_size: usize)
    where
        I: IntoIterator<Item = i32>,
        I::IntoIter: Clone,
    {
        let symbols = symbols.into_iter();

        let mut encoder = DefaultRangeEncoder::new();
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-127..=127);
        let model = quantizer.quantize(Gaussian::new(3.2, 5.1));

        encoder.encode_iid_symbols(symbols.clone(), &model).unwrap();
        let compressed = encoder.into_compressed().unwrap();
        assert_eq!(compressed.len(), expected_size);

        let mut decoder = DefaultRangeDecoder::from_compressed(&compressed).unwrap();
        for symbol in symbols {
            assert_eq!(decoder.decode_symbol(&model).unwrap(), symbol);
        }
        assert!(decoder.maybe_exhausted());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u32_u64_32() {
        generic_compress_many::<u32, u64, u32, 32>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u32_u64_24() {
        generic_compress_many::<u32, u64, u32, 24>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u32_u64_16() {
        generic_compress_many::<u32, u64, u16, 16>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u32_u64_8() {
        generic_compress_many::<u32, u64, u8, 8>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u16_u64_16() {
        generic_compress_many::<u16, u64, u16, 16>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u16_u64_12() {
        generic_compress_many::<u16, u64, u16, 12>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u16_u64_8() {
        generic_compress_many::<u16, u64, u8, 8>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u8_u64_8() {
        generic_compress_many::<u8, u64, u8, 8>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u16_u32_16() {
        generic_compress_many::<u16, u32, u16, 16>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u16_u32_12() {
        generic_compress_many::<u16, u32, u16, 12>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u16_u32_8() {
        generic_compress_many::<u16, u32, u8, 8>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u8_u32_8() {
        generic_compress_many::<u8, u32, u8, 8>();
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn compress_many_u8_u16_8() {
        generic_compress_many::<u8, u16, u8, 8>();
    }

    fn generic_compress_many<Word, State, Probability, const PRECISION: usize>()
    where
        State: BitArray + AsPrimitive<Word>,
        Word: BitArray + Into<State> + AsPrimitive<Probability>,
        Probability: BitArray + Into<Word> + AsPrimitive<usize> + Into<f64>,
        u32: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
        f64: AsPrimitive<Probability>,
        i32: AsPrimitive<Probability>,
    {
        const AMT: usize = 1000;
        let mut symbols_gaussian = Vec::with_capacity(AMT);
        let mut means = Vec::with_capacity(AMT);
        let mut stds = Vec::with_capacity(AMT);

        let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
        for _ in 0..AMT {
            let mean = (200.0 / u32::MAX as f64) * rng.next_u32() as f64 - 100.0;
            let std_dev = (10.0 / u32::MAX as f64) * rng.next_u32() as f64 + 0.001;
            let quantile = (rng.next_u32() as f64 + 0.5) / (1u64 << 32) as f64;
            let dist = Gaussian::new(mean, std_dev);
            let symbol = core::cmp::max(
                -127,
                core::cmp::min(127, dist.inverse(quantile).round() as i32),
            );

            symbols_gaussian.push(symbol);
            means.push(mean);
            stds.push(std_dev);
        }

        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        let categorical_probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();
        let categorical =
            ContiguousCategoricalEntropyModel::<Probability, _, PRECISION>::from_floating_point_probabilities(
                &categorical_probabilities,
            )
            .unwrap();
        let mut symbols_categorical = Vec::with_capacity(AMT);
        let max_probability = Probability::max_value() >> (Probability::BITS - PRECISION);
        for _ in 0..AMT {
            let quantile = rng.next_u32().as_() & max_probability;
            let symbol = categorical.quantile_function(quantile).0;
            symbols_categorical.push(symbol);
        }

        let mut encoder = RangeEncoder::<Word, State>::new();

        encoder
            .encode_iid_symbols(&symbols_categorical, &categorical)
            .unwrap();
        dbg!(
            encoder.num_bits(),
            AMT as f64 * categorical.entropy_base2::<f64>()
        );

        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-127..=127);
        encoder
            .encode_symbols(symbols_gaussian.iter().zip(&means).zip(&stds).map(
                |((&symbol, &mean), &core)| (symbol, quantizer.quantize(Gaussian::new(mean, core))),
            ))
            .unwrap();
        dbg!(encoder.num_bits());

        let mut decoder = encoder.into_decoder().unwrap();

        let reconstructed_categorical = decoder
            .decode_iid_symbols(AMT, &categorical)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let reconstructed_gaussian = decoder
            .decode_symbols(
                means
                    .iter()
                    .zip(&stds)
                    .map(|(&mean, &core)| quantizer.quantize(Gaussian::new(mean, core))),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(decoder.maybe_exhausted());

        assert_eq!(symbols_categorical, reconstructed_categorical);
        assert_eq!(symbols_gaussian, reconstructed_gaussian);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn seek() {
        const NUM_CHUNKS: usize = 100;
        const SYMBOLS_PER_CHUNK: usize = 100;

        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
        let model = quantizer.quantize(Gaussian::new(0.0, 10.0));

        let mut encoder = DefaultRangeEncoder::new();

        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        let mut symbols = Vec::with_capacity(NUM_CHUNKS);
        let mut jump_table = Vec::with_capacity(NUM_CHUNKS);

        for _ in 0..NUM_CHUNKS {
            jump_table.push(encoder.pos());
            let chunk = (0..SYMBOLS_PER_CHUNK)
                .map(|_| model.quantile_function(rng.next_u32() % (1 << 24)).0)
                .collect::<Vec<_>>();
            encoder.encode_iid_symbols(&chunk, &model).unwrap();
            symbols.push(chunk);
        }
        let final_pos_and_state = encoder.pos();

        let mut decoder = encoder.decoder();

        // Verify we can decode the chunks normally (we can't veryify that coding and
        // decoding lead to same `pos_and_state` because the range decoder currently doesn't
        // implement `Pos` due to complications at the stream end.)
        for (chunk, _) in symbols.iter().zip(&jump_table) {
            let decoded = decoder
                .decode_iid_symbols(SYMBOLS_PER_CHUNK, &model)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(&decoded, chunk);
        }
        assert!(decoder.maybe_exhausted());

        // Seek to some random offsets in the jump table and decode one chunk
        for i in 0..100 {
            let chunk_index = if i == 3 {
                // Make sure we test jumping to beginning at least once.
                0
            } else {
                rng.next_u32() as usize % NUM_CHUNKS
            };

            let pos_and_state = jump_table[chunk_index];
            decoder.seek(pos_and_state).unwrap();
            let decoded = decoder
                .decode_iid_symbols(SYMBOLS_PER_CHUNK, &model)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(&decoded, &symbols[chunk_index])
        }

        // Test jumping to end (but first make sure we're not already at the end).
        decoder.seek(jump_table[0]).unwrap();
        assert!(!decoder.maybe_exhausted());
        decoder.seek(final_pos_and_state).unwrap();
        assert!(decoder.maybe_exhausted());
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub enum DecoderFrontendError {
    /// This can only happen if both of the following conditions apply:
    ///
    /// 1. we are decoding invalid compressed data; and
    /// 2. we use entropy models with varying `PRECISION`s.
    ///
    /// Unless you change the `PRECISION` mid-decoding this error cannot occur. However,
    /// note that the encoder is not surjective, i.e., it cannot reach all possible values.
    /// The reason why the decoder still doesn't err (unless varying `PRECISION`s are used)
    /// is that it is not injective, i.e., it maps the bit strings that are unreachable by
    /// the encoder to symbols that could have been encoded into a different bit string.
    ///
    /// The lack of injectivity of the encoder makes the Range Coder implementation in this
    /// library unsuitable for bitsback coding. Even though you can encode an arbitrary bit
    /// string into a sequence of symbols using any entropy model, decoding the sequence of
    /// symbols with the same entropy models won't always give you the same bit string. In
    /// other words,
    ///
    /// - `range_decode(range_encode(sequence_of_symbols)) = sequence_of_symbols` for all
    ///   `sequence_of_symbols`; but
    /// - `range_encode(range_encode(bit_string)) != bit_string` in general.
    ///
    /// If you need equality in the second relation, use an [`AnsCoder`].
    InvalidData,
}

impl Display for DecoderFrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidData => write!(f, "Tried to decode invalid compressed data."),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecoderFrontendError {}
