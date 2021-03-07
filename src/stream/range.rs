use alloc::vec::Vec;
use core::{borrow::Borrow, fmt::Debug, marker::PhantomData, ops::Deref};

use num::cast::AsPrimitive;

use super::{
    models::{DecoderModel, EncoderModel},
    Code, Decode, Encode, IntoDecoder, Pos, Seek,
};
use crate::{bit_array_from_chunks, bit_array_to_chunks_exact, BitArray, EncodingError};

/// Type of the internal state used by [`Encoder<CompressedWord, State>`],
/// [`Decoder<CompressedWord, State>`]. Relevant for [`Seek`]ing.
///
/// [`Seek`]: crate::Seek
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoderState<CompressedWord, State> {
    lower: State,

    /// Invariant: `range >= State::one() << (State::BITS - CompressedWord::BITS)`
    /// Therefore, the highest order `CompressedWord` of `lower` is always sufficient to
    /// identify the current interval, so only it has to be flushed at the end.
    range: State,

    /// We keep track of the `CompressedWord` type so that we can statically enforce
    /// the invariants for `lower` and `range`.
    phantom: PhantomData<CompressedWord>,
}

impl<CompressedWord, State: BitArray> CoderState<CompressedWord, State> {
    /// Get the lower bound of the current range (inclusive)
    pub fn lower(&self) -> State {
        self.lower
    }

    /// Get the size of the current range
    pub fn range(&self) -> State {
        self.range
    }
}

impl<CompressedWord: BitArray, State: BitArray> Default for CoderState<CompressedWord, State> {
    fn default() -> Self {
        Self {
            lower: State::zero(),
            range: State::max_value(),
            phantom: PhantomData,
        }
    }
}

pub struct RangeEncoder<CompressedWord: BitArray, State: BitArray> {
    buf: Vec<CompressedWord>,
    state: CoderState<CompressedWord, State>,
}

/// Type alias for an [`RangeEncoder`] with sane parameters for typical use cases.
pub type DefaultRangeEncoder = RangeEncoder<u32, u64>;

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
/// [lookup models]: super::models::lookup
/// [`DefaultEncoderArrayLookupTable`]: super::models::lookup::DefaultEncoderArrayLookupTable
/// [`DefaultEncoderHashLookupTable`]: super::models::lookup::DefaultEncoderHashLookupTable
pub type SmallRangeEncoder = RangeEncoder<u16, u32>;

impl<CompressedWord, State> Debug for RangeEncoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter_compressed()).finish()
    }
}

impl<CompressedWord, State> Code for RangeEncoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type State = CoderState<CompressedWord, State>;
    type CompressedWord = CompressedWord;

    fn state(&self) -> Self::State {
        self.state
    }

    fn maybe_empty(&self) -> bool {
        self.is_empty()
    }
}

impl<CompressedWord, State> Pos for RangeEncoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn pos(&self) -> usize {
        self.buf.len()
    }
}

impl<CompressedWord, State> Default for RangeEncoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<CompressedWord, State> RangeEncoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    /// Creates an empty encoder for range coding.
    pub fn new() -> Self {
        assert!(State::BITS >= 2 * CompressedWord::BITS);
        assert_eq!(State::BITS % CompressedWord::BITS, 0);

        Self {
            buf: Vec::new(),
            state: CoderState::default(),
        }
    }

    /// Discards all compressed data and resets the coder to the same state as
    /// [`Coder::new`](#method.new).
    pub fn clear(&mut self) {
        self.buf.clear();
        self.state = CoderState::default();
    }

    /// Check if no data has been encoded yet.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty() && self.state.range == State::max_value()
    }

    /// Same as IntoDecoder::into_decoder(self) but can be used for any `PRECISION`
    /// and therefore doesn't require type arguments on the caller side.
    pub fn into_decoder(self) -> RangeDecoder<CompressedWord, State, Vec<CompressedWord>> {
        self.into()
    }

    /// Same as IntoDecoder::into_decoder(&self) but can be used for any `PRECISION`
    /// and therefore doesn't require type arguments on the caller side.
    pub fn decoder(
        &mut self, // TODO: document why we need mutable access (because encoder has to temporary flush its state)
    ) -> RangeDecoder<CompressedWord, State, CoderGuard<'_, CompressedWord, State>> {
        RangeDecoder::from_compressed(self.get_compressed())
    }

    pub fn into_compressed(mut self) -> Vec<CompressedWord> {
        if !self.is_empty() {
            let word = (self.state.lower >> (State::BITS - CompressedWord::BITS)).as_();
            self.buf.push(word);
        }
        self.buf
    }

    /// Returns a view into the full compressed data currently on the ans.
    ///
    /// This is a low level method that provides a view into the current compressed
    /// data at zero cost, but in a somewhat inconvenient representation. In most
    /// cases you will likely want to call one of [`get_compressed`],
    /// [`into_compressed`], or [`iter_compressed`] instead, as these methods
    /// provide the the compressed data in more convenient forms (which is also the
    /// form expected by the constructor [`from_compressed`]).
    ///
    /// The return value of this method is a tuple `(bulk, head)`, where
    /// `bulk: &[CompressedWord]` has variable size (and can be empty) and `head: [CompressedWord; 2]` has a
    /// fixed size. When encoding or decoding data, `head` typically changes with
    /// each encoded or decoded symbol while `bulk` changes only infrequently
    /// (whenever `head` overflows or underflows).
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`from_compressed`]: #method.from_compressed
    pub fn as_compressed_raw(&self) -> (&[CompressedWord], <Self as Code>::State) {
        (&self.buf, self.state)
    }

    /// Assembles the current compressed data into a single slice.
    ///
    /// This method is similar to [`as_compressed_raw`] with the difference that it
    /// concatenates the `bulk` and `head` before returning them. The concatenation
    /// truncates any trailing zero words, which is compatible with the constructor
    /// [`from_compressed`].
    ///
    /// This method requires a `&mut self` receiver. If you only have a shared
    /// reference to a `Coder`, consider calling [`as_compressed_raw`] or
    /// [`iter_compressed`] instead.
    ///
    /// The returned `CoderGuard` dereferences to `&[CompressedWord]`, thus providing read-only
    /// access to the compressed data. If you need ownership of the compressed data,
    /// consider calling [`into_compressed`] instead.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{models::Categorical, ans::DefaultAnsCoder, Decode};
    ///
    /// let mut coder = DefaultAnsCoder::new();
    ///
    /// // Push some data on the coder.
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let model = Categorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
    ///     .unwrap();
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
    /// [`as_compressed_raw`]: #method.as_compressed_raw
    /// [`from_compressed`]: #method.from_compressed
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn get_compressed(&mut self) -> CoderGuard<'_, CompressedWord, State> {
        CoderGuard::new(self)
    }

    pub fn iter_compressed(
        &self,
    ) -> impl Iterator<Item = CompressedWord> + ExactSizeIterator + DoubleEndedIterator + '_ {
        IterCompressed::new(self)
    }

    /// Returns the number of compressed words on the ans.
    ///
    /// This includes a constant overhead of between one and two words unless the
    /// coder is completely empty.
    ///
    /// This method returns the length of the slice, the `Vec<CompressedWord>`, or the iterator
    /// that would be returned by [`get_compressed`], [`into_compressed`], or
    /// [`iter_compressed`], respectively, when called at this time.
    ///
    /// See also [`num_bits`].
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`num_bits`]: #method.num_bits
    pub fn num_words(&self) -> usize {
        if self.is_empty() {
            0
        } else {
            self.buf.len() + 1
        }
    }

    /// Returns the size of the current queue of compressed data in bits.
    ///
    /// This includes some constant overhead unless the coder is completely empty
    /// (see [`num_words`](#method.num_words)).
    ///
    /// The returned value is a multiple of the bitlength of the compressed word
    /// type `CompressedWord`.
    pub fn num_bits(&self) -> usize {
        CompressedWord::BITS * self.num_words()
    }

    pub fn buf(&self) -> &[CompressedWord] {
        &self.buf
    }
}

impl<CompressedWord, State, const PRECISION: usize> IntoDecoder<PRECISION>
    for RangeEncoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type IntoDecoder = RangeDecoder<CompressedWord, State, Vec<CompressedWord>>;
}

impl<CompressedWord, State, const PRECISION: usize> Encode<PRECISION>
    for RangeEncoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        model: D,
    ) -> Result<(), EncodingError>
    where
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        // We maintain the following invariant (*):
        //   range >= State::one() << (State::BITS - CompressedWord::BITS)

        let (left_sided_cumulative, probability) = model
            .left_cumulative_and_probability(symbol)
            .map_err(|()| EncodingError::ImpossibleSymbol)?;

        let scale = self.state.range >> PRECISION;
        let new_lower = self
            .state
            .lower
            .wrapping_add(&(scale * left_sided_cumulative.into().into()));
        if new_lower < self.state.lower {
            // Addition has wrapped around, so we have to propagate back the carry bit.
            for word in self.buf.iter_mut().rev() {
                *word = word.wrapping_add(&CompressedWord::one());
                if *word != CompressedWord::zero() {
                    break;
                }
            }
        }
        self.state.lower = new_lower;

        // This cannot overflow since `scale * probability <= (range >> PRECISION) << PRECISION`
        self.state.range = scale * probability.into().into();

        if self.state.range < State::one() << (State::BITS - CompressedWord::BITS) {
            // Invariant `range >= State::one() << (State::BITS - CompressedWord::BITS)` is
            // violated. Since `left_cumulative_and_probability` succeeded, we know that
            // `probability != 0` and therefore:
            //   range >= scale * probability = (old_range >> PRECISION) * probability
            //         >= old_range >> PRECISION
            //         >= old_range >> CompressedWords::BITS
            // where `old_range` is the `range` at method entry, which satisfied invariant (*)
            // by assumption. Therefore, the following left-shift restores the invariant:
            self.state.range = self.state.range << CompressedWord::BITS;

            let word = (self.state.lower >> (State::BITS - CompressedWord::BITS)).as_();
            self.buf.push(word);
            self.state.lower = self.state.lower << CompressedWord::BITS;
        }

        Ok(())
    }
}

pub struct RangeDecoder<CompressedWord: BitArray, State: BitArray, Buf: AsRef<[CompressedWord]>> {
    buf: Buf,

    /// Points to the next word in `buf` to be read if `state` underflows.
    pos: usize,
    state: CoderState<CompressedWord, State>,

    /// Invariant: `point.wrapping_sub(&state.lower) < state.range`
    point: State,
}

/// Type alias for a [`RangeDecoder`] with sane parameters for typical use cases.
pub type DefaultRangeDecoder<Buf> = RangeDecoder<u32, u64, Buf>;

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
/// [lookup models]: super::models::lookup
/// [`DefaultEncoderArrayLookupTable`]: super::models::lookup::DefaultEncoderArrayLookupTable
/// [`DefaultEncoderHashLookupTable`]: super::models::lookup::DefaultEncoderHashLookupTable
/// [`DefaultDecoderIndexLookupTable`]: super::models::lookup::DefaultDecoderIndexLookupTable
/// [`DefaultDecoderGenericLookupTable`]: super::models::lookup::DefaultDecoderGenericLookupTable
pub type SmallRangeDecoder<Buf> = RangeDecoder<u16, u32, Buf>;

impl<CompressedWord, State, Buf> Debug for RangeDecoder<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: AsRef<[CompressedWord]>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list()
            .entries(
                bit_array_to_chunks_exact(self.state.lower)
                    .chain(self.buf.as_ref().iter().cloned()),
            )
            .finish()
    }
}

impl<CompressedWord, State, Buf> Code for RangeDecoder<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: AsRef<[CompressedWord]>,
{
    type State = CoderState<CompressedWord, State>;
    type CompressedWord = CompressedWord;

    fn state(&self) -> Self::State {
        self.state
    }

    fn maybe_empty(&self) -> bool {
        self.pos >= self.buf.as_ref().len()
            && self.point.wrapping_sub(&self.state.lower)
                < State::one() << (State::BITS - CompressedWord::BITS)
    }
}

impl<CompressedWord, State, Buf> Pos for RangeDecoder<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: AsRef<[CompressedWord]>,
{
    fn pos(&self) -> usize {
        self.pos.saturating_sub(State::BITS / CompressedWord::BITS)
    }
}

impl<CompressedWord, State, Buf> Seek for RangeDecoder<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: AsRef<[CompressedWord]>,
{
    fn seek(&mut self, pos_and_state: (usize, Self::State)) -> Result<(), ()> {
        let (pos, state) = pos_and_state;
        let remainder = self.buf.as_ref().get(pos..).ok_or(())?;
        let mut point = bit_array_from_chunks(remainder.iter().cloned());

        let pos_shift = if remainder.len() < State::BITS / CompressedWord::BITS {
            // A very short amount of compressed data is left, and therefore `point` is still
            // right-aligned. Shift it over so it's left-aligned and fill it up with ones.
            if remainder.is_empty() {
                if self.buf.as_ref().is_empty() && state == Self::State::default() {
                    // Special case: seeking to the beginning of no compressed data. Let's do
                    // the same what `from_compressed` does in this case.
                    point = State::max_value() >> CompressedWord::BITS;
                } else {
                    // Tried to either seek past EOF or to EOF of empty buffer with wrong state.
                    return Err(());
                }
            } else {
                point = point << (State::BITS - remainder.len() * CompressedWord::BITS)
                    | State::max_value() >> remainder.len() * CompressedWord::BITS;
            }

            remainder.len()
        } else {
            State::BITS / CompressedWord::BITS
        };

        self.point = point;
        self.pos = pos + pos_shift;
        self.state = state;

        Ok(())
    }
}

impl<CompressedWord, State, Buf> RangeDecoder<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: AsRef<[CompressedWord]>,
{
    pub fn from_compressed(compressed: Buf) -> Self {
        assert!(State::BITS >= 2 * CompressedWord::BITS);
        assert_eq!(State::BITS % CompressedWord::BITS, 0);

        let mut point = bit_array_from_chunks(compressed.as_ref().iter().cloned());

        let pos = if compressed.as_ref().len() < State::BITS / CompressedWord::BITS {
            // A very short compressed buffer was provided, and therefore `point` is still
            // right-aligned. Shift it over so it's left-aligned and fill it up with ones.
            if compressed.as_ref().len() == 0 {
                // Special case: an empty compressed stream is treated like one with a single
                // `CompressedWord` of value zero.
                point = State::max_value() >> CompressedWord::BITS
            } else {
                point = point << (State::BITS - compressed.as_ref().len() * CompressedWord::BITS)
                    | State::max_value() >> compressed.as_ref().len() * CompressedWord::BITS;
            }
            compressed.as_ref().len()
        } else {
            State::BITS / CompressedWord::BITS
        };

        Self {
            buf: compressed,
            state: CoderState::default(),
            pos,
            point,
        }
    }
}

impl<CompressedWord, State> From<RangeEncoder<CompressedWord, State>>
    for RangeDecoder<CompressedWord, State, Vec<CompressedWord>>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn from(encoder: RangeEncoder<CompressedWord, State>) -> Self {
        Self::from_compressed(encoder.into_compressed())
    }
}

impl<'encoder, CompressedWord, State> From<&'encoder mut RangeEncoder<CompressedWord, State>>
    for RangeDecoder<CompressedWord, State, CoderGuard<'encoder, CompressedWord, State>>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn from(encoder: &'encoder mut RangeEncoder<CompressedWord, State>) -> Self {
        encoder.decoder()
    }
}

impl<CompressedWord, State, Buf, const PRECISION: usize> Decode<PRECISION>
    for RangeDecoder<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: AsRef<[CompressedWord]>,
{
    type DecodingError = DecodingError;

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
    fn decode_symbol<D>(&mut self, model: D) -> Result<D::Symbol, Self::DecodingError>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        // We maintain the following invariant (*):
        //   point (-) lower < range
        // where (-) denotes wrapping subtraction (in `Self::State`).

        let scale = self.state.range >> PRECISION;
        let quantile = self.point.wrapping_sub(&self.state.lower) / scale;
        if quantile >= State::one() << PRECISION {
            // This can only happen if both of the following conditions apply:
            // (i) we are decoding invalid compressed data; and
            // (ii) we use entropy models with varying `PRECISION`s.
            // TODO: Is (ii) necessary? Aren't there always unreachable pockets due to rounding?
            return Err(DecodingError::InvalidCompressedData);
        }

        let (symbol, left_sided_cumulative, probability) =
            model.quantile_function(quantile.as_().as_());

        // Update `state` in the same way as we do in `encode_symbol` (see comments there):
        self.state.lower = self
            .state
            .lower
            .wrapping_add(&(scale * left_sided_cumulative.into().into()));
        self.state.range = scale * probability.into().into();

        // Invariant (*) is still satisfied at this point because:
        //   (point (-) lower) / scale = (point (-) old_lower) / scale (-) left_sided_cumulative
        //                             = quantile (-) left_sided_cumulative
        //                             < probability
        // Therefore, we have:
        //   point (-) lower < scale * probability <= range

        if self.state.range < State::one() << (State::BITS - CompressedWord::BITS) {
            // First update `state` in the same way as we do in `encode_symbol`:
            self.state.lower = self.state.lower << CompressedWord::BITS;
            self.state.range = self.state.range << CompressedWord::BITS;

            // Then update `point`, which restores invariant (*):
            self.point = self.point << CompressedWord::BITS;
            if let Some(&word) = self.buf.as_ref().get(self.pos) {
                self.point = self.point | word.into();
                self.pos += 1;
            } else {
                self.point = self.point | CompressedWord::max_value().into();
                if self.pos < self.buf.as_ref().len() + State::BITS / CompressedWord::BITS - 1 {
                    // We allow `pos` to be up to `State::BITS / CompressedWord::BITS - 1` words
                    // past the end of `buf` so that `point` always contains at least one word from
                    // `buf`. We don't increase `pos` further so that it doesn't wrap around when
                    // the decoder is misused, which would cause `maybe_empty` to misbehave.
                    self.pos += 1;
                }
            }
        }

        Ok(symbol)
    }
}

#[derive(Debug)]
pub enum DecodingError {
    InvalidCompressedData,
}

impl core::fmt::Display for DecodingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodingError::InvalidCompressedData => {
                write!(f, "Tried to decode invalid compressed data.")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodingError {}

/// Provides temporary read-only access to the compressed data wrapped in an
/// [`Encoder`].
///
/// Dereferences to `&[CompressedWord]`. See [`Encoder::get_compressed`] for an example.
///
/// [`Coder`]: struct.Coder.html
/// [`Coder::get_compressed`]: struct.Coder.html#method.get_compressed
pub struct CoderGuard<'a, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    inner: &'a mut RangeEncoder<CompressedWord, State>,
}

impl<CompressedWord, State> Debug for CoderGuard<'_, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<'a, CompressedWord, State> CoderGuard<'a, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn new(encoder: &'a mut RangeEncoder<CompressedWord, State>) -> Self {
        // Append state. Will be undone in `<Self as Drop>::drop`.
        if !encoder.is_empty() {
            let word = (encoder.state.lower >> (State::BITS - CompressedWord::BITS)).as_();
            encoder.buf.push(word);
        }
        Self { inner: encoder }
    }
}

impl<'a, CompressedWord, State> Drop for CoderGuard<'a, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn drop(&mut self) {
        self.inner.buf.pop(); // Does nothing if `coder.is_empty()`, as it should.
    }
}

impl<'a, CompressedWord, State> Deref for CoderGuard<'a, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type Target = [CompressedWord];

    fn deref(&self) -> &Self::Target {
        &self.inner.buf
    }
}

impl<'a, CompressedWord, State> AsRef<[CompressedWord]> for CoderGuard<'a, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn as_ref(&self) -> &[CompressedWord] {
        self
    }
}

struct IterCompressed<'a, CompressedWord> {
    buf: &'a [CompressedWord],
    last: CompressedWord,
    index_front: usize,
    index_back: usize,
}

impl<'a, CompressedWord> IterCompressed<'a, CompressedWord>
where
    CompressedWord: BitArray,
{
    fn new<State>(coder: &'a RangeEncoder<CompressedWord, State>) -> Self
    where
        CompressedWord: Into<State>,
        State: BitArray + AsPrimitive<CompressedWord>,
    {
        let last = (coder.state.lower >> (State::BITS - CompressedWord::BITS)).as_();

        Self {
            buf: &coder.buf,
            last,
            index_front: 0,
            index_back: coder.buf.len() + (!coder.is_empty()) as usize,
        }
    }
}

impl<CompressedWord: BitArray> Iterator for IterCompressed<'_, CompressedWord> {
    type Item = CompressedWord;

    fn next(&mut self) -> Option<Self::Item> {
        let index_front = self.index_front;
        if index_front == self.index_back {
            None
        } else {
            self.index_front += 1;
            let result = self.buf.get(index_front).cloned().unwrap_or(self.last);
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.index_back - self.index_front;
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.index_back - self.index_front
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index_front = core::cmp::min(self.index_back, self.index_front.saturating_add(n));
        self.next()
    }
}

impl<CompressedWord: BitArray> ExactSizeIterator for IterCompressed<'_, CompressedWord> {}

impl<CompressedWord: BitArray> DoubleEndedIterator for IterCompressed<'_, CompressedWord> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index_front == self.index_back {
            None
        } else {
            // We can subtract one because `self.index_back > self.index_front >= 0`.
            self.index_back -= 1;
            let result = self.buf.get(self.index_back).cloned().unwrap_or(self.last);
            Some(result)
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.index_back = core::cmp::max(self.index_front, self.index_back.saturating_sub(n));
        self.next_back()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::dbg;

    use super::super::models::{Categorical, LeakyQuantizer};
    use super::*;

    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };
    use statrs::distribution::{InverseCDF, Normal};

    #[test]
    fn compress_none() {
        let encoder = DefaultRangeEncoder::new();
        assert!(encoder.is_empty());
        let compressed = encoder.into_compressed();
        assert!(compressed.is_empty());

        let decoder = DefaultRangeDecoder::from_compressed(&compressed);
        assert!(decoder.maybe_empty());
    }

    #[test]
    fn compress_one() {
        generic_compress_few(core::iter::once(5), 1)
    }

    #[test]
    fn compress_two() {
        generic_compress_few([2, 8].iter().cloned(), 1)
    }

    #[test]
    fn compress_ten() {
        generic_compress_few(0..10, 2)
    }

    #[test]
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
        let model = quantizer.quantize(Normal::new(3.2, 5.1).unwrap());

        encoder.encode_iid_symbols(symbols.clone(), &model).unwrap();
        let compressed = encoder.into_compressed();
        assert_eq!(compressed.len(), expected_size);

        let mut decoder = DefaultRangeDecoder::from_compressed(&compressed);
        for symbol in symbols {
            assert_eq!(decoder.decode_symbol(&model).unwrap(), symbol);
        }
        assert!(decoder.maybe_empty());
    }

    #[test]
    fn compress_many_u32_u64_32() {
        generic_compress_many::<u32, u64, u32, 32>();
    }

    #[test]
    fn compress_many_u32_u64_24() {
        generic_compress_many::<u32, u64, u32, 24>();
    }

    #[test]
    fn compress_many_u32_u64_16() {
        generic_compress_many::<u32, u64, u16, 16>();
    }

    #[test]
    fn compress_many_u16_u64_16() {
        generic_compress_many::<u16, u64, u16, 16>();
    }

    #[test]
    fn compress_many_u32_u64_8() {
        generic_compress_many::<u32, u64, u8, 8>();
    }

    #[test]
    fn compress_many_u16_u64_8() {
        generic_compress_many::<u16, u64, u8, 8>();
    }

    #[test]
    fn compress_many_u8_u64_8() {
        generic_compress_many::<u8, u64, u8, 8>();
    }

    #[test]
    fn compress_many_u16_u32_16() {
        generic_compress_many::<u16, u32, u16, 16>();
    }

    #[test]
    fn compress_many_u16_u32_8() {
        generic_compress_many::<u16, u32, u8, 8>();
    }

    #[test]
    fn compress_many_u8_u32_8() {
        generic_compress_many::<u8, u32, u8, 8>();
    }

    fn generic_compress_many<CompressedWord, State, Probability, const PRECISION: usize>()
    where
        State: BitArray + AsPrimitive<CompressedWord>,
        CompressedWord: BitArray + Into<State> + AsPrimitive<Probability>,
        Probability: BitArray + Into<CompressedWord> + AsPrimitive<usize> + Into<f64>,
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
            let dist = Normal::new(mean, std_dev).unwrap();
            let symbol = core::cmp::max(
                -127,
                core::cmp::min(127, dist.inverse_cdf(quantile).round() as i32),
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
        let categorical = Categorical::<Probability, PRECISION>::from_floating_point_probabilities(
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

        let mut encoder = RangeEncoder::<CompressedWord, State>::new();

        encoder
            .encode_iid_symbols(&symbols_categorical, &categorical)
            .unwrap();
        dbg!(
            encoder.num_bits(),
            AMT as f64 * categorical.entropy::<f64>()
        );

        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-127..=127);
        encoder
            .encode_symbols(symbols_gaussian.iter().zip(&means).zip(&stds).map(
                |((&symbol, &mean), &core)| {
                    (symbol, quantizer.quantize(Normal::new(mean, core).unwrap()))
                },
            ))
            .unwrap();
        dbg!(encoder.num_bits());

        let mut decoder = encoder.into_decoder();

        let reconstructed_categorical = decoder
            .decode_iid_symbols(AMT, &categorical)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let reconstructed_gaussian = decoder
            .decode_symbols(
                means
                    .iter()
                    .zip(&stds)
                    .map(|(&mean, &core)| quantizer.quantize(Normal::new(mean, core).unwrap())),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(decoder.maybe_empty());

        assert_eq!(symbols_categorical, reconstructed_categorical);
        assert_eq!(symbols_gaussian, reconstructed_gaussian);
    }

    #[test]
    fn seek() {
        const NUM_CHUNKS: usize = 100;
        const SYMBOLS_PER_CHUNK: usize = 100;

        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
        let model = quantizer.quantize(Normal::new(0.0, 10.0).unwrap());

        let mut encoder = DefaultRangeEncoder::new();

        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        let mut symbols = Vec::with_capacity(NUM_CHUNKS);
        let mut jump_table = Vec::with_capacity(NUM_CHUNKS);

        for _ in 0..NUM_CHUNKS {
            jump_table.push(encoder.pos_and_state());
            let chunk = (0..SYMBOLS_PER_CHUNK)
                .map(|_| model.quantile_function(rng.next_u32() % (1 << 24)).0)
                .collect::<Vec<_>>();
            encoder.encode_iid_symbols(&chunk, &model).unwrap();
            symbols.push(chunk);
        }
        let final_pos_and_state = encoder.pos_and_state();

        let mut decoder = encoder.decoder();

        // Verify that decoding leads to the same positions and states.
        for (chunk, &pos_and_state) in symbols.iter().zip(&jump_table) {
            assert_eq!(decoder.pos_and_state(), pos_and_state);
            let decoded = decoder
                .decode_iid_symbols(SYMBOLS_PER_CHUNK, &model)
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert_eq!(&decoded, chunk);
        }
        assert_eq!(decoder.pos_and_state(), final_pos_and_state);
        assert!(decoder.maybe_empty());

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
        assert!(!decoder.maybe_empty());
        decoder.seek(final_pos_and_state).unwrap();
        assert!(decoder.maybe_empty());
    }
}
