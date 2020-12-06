use std::{borrow::Borrow, error::Error, fmt::Debug, marker::PhantomData, ops::Deref};

use num::{cast::AsPrimitive, CheckedDiv, Zero};

use crate::{bit_array_from_chunks, bit_array_to_chunks_exact};

use super::{
    distributions::DiscreteDistribution, BitArray, Code, Decode, Encode, EncodingError,
    TryCodingError,
};

/// Type of the internal state used by [`Encoder<CompressedWord, State>`],
/// [`Decoder<CompressedWord, State>`], and [`EncoderDecoder<CompressedWord, State>`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoderState<CompressedWord, State> {
    lower: State,

    /// Invariant: `range >= State::one() << (State::BITS - CompressedWord::BITS)`
    /// Therefore, the highest order `CompressedWord` of `lower` is always sufficient to
    /// identify the current interval.
    range: State,

    /// We keep track of the `CompressedWord` type so that we can statically enforce
    /// the invariants for `lower` and `range`.
    phantom: PhantomData<CompressedWord>,
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

pub struct Encoder<CompressedWord: BitArray, State: BitArray> {
    buf: Vec<CompressedWord>,
    state: CoderState<CompressedWord, State>,
}

/// Type alias for an [`Encoder`] with sane parameters for typical use cases.
pub type DefaultEncoder = Encoder<u32, u64>;

impl<CompressedWord, State> Debug for Encoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter_compressed()).finish()
    }
}

impl<CompressedWord, State> Code for Encoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type State = CoderState<CompressedWord, State>;
    type CompressedWord = CompressedWord;
}

impl<CompressedWord, State> Encoder<CompressedWord, State>
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

    /// Creates a range coder that appends to *unsealed* compressed data.
    ///
    /// TODO: describe what is meant with "unsealed"
    pub fn with_compressed_data(
        mut compressed: Vec<CompressedWord>,
        state: <Self as Code>::State,
    ) -> Result<Self, ()> {
        assert!(State::BITS >= 2 * CompressedWord::BITS);
        assert_eq!(State::BITS % CompressedWord::BITS, 0);

        // We may assume that `state` satisfies the invariants of a `CoderState` since we
        // don't provide a public API to create an invalid `CoderState`.

        if !(compressed.is_empty() && state.range == State::max_value()) {
            let shift = State::BITS - CompressedWord::BITS;
            let high = (state.lower >> shift).as_();
            if compressed.pop() != Some(high) {
                return Err(());
            }
        }

        Ok(Self {
            buf: compressed,
            state,
        })
    }

    /// Discards all compressed data and resets the coder to the same state as
    /// [`Coder::new`](#method.new).
    pub fn clear(&mut self) {
        self.buf.clear();
        self.state = CoderState::default();
    }

    /// Check if no data for decoding is left.
    ///
    /// This method returns `true` if no data is left for decoding. This means that
    /// the coder is in the same state as it would be after being constructed with
    /// [`Coder::new`](#method.new) or after calling [`clear`](#method.clear).
    ///
    /// Note that you can still pop symbols off an empty coder, but this is only
    /// useful in rare edge cases, see documentation of
    /// [`decode_symbol`](#method.decode_symbol).
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty() && self.state.range == State::max_value()
    }

    /// Consumes the coder and returns the compressed data.
    ///
    /// The returned data can be used to recreate a coder with the same state
    /// (e.g., for decoding) by passing it to
    /// [`with_compressed_data`](#method.with_compressed_data).
    ///
    /// If you don't want to consume the coder, consider calling
    /// [`get_compressed`](#method.get_compressed),
    /// [`as_compressed_raw`](#method.as_compressed_raw), or
    /// [`iter_compressed`](#method.iter_compressed) instead.
    ///
    /// # Example
    ///
    /// ```
    /// use ans::{distributions::Categorical, stack::DefaultCoder, Decode};
    ///
    /// let mut coder = DefaultCoder::new();
    ///
    /// // Push some data on the coder.
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution = Categorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
    ///     .unwrap();
    /// coder.encode_iid_symbols_reverse(&symbols, &distribution).unwrap();
    ///
    /// // Get the compressed data, consuming the coder.
    /// let compressed = coder.into_compressed();
    ///
    /// // ... write `compressed` to a file and then read it back later ...
    ///
    /// // Create a new coder with the same state and use it for decompression.
    /// let mut coder = DefaultCoder::with_compressed_data(compressed);
    /// let reconstructed = coder.decode_iid_symbols(4, &distribution).collect::<Vec<_>>();
    /// assert_eq!(reconstructed, symbols);
    /// assert!(coder.is_empty())
    /// ```
    pub fn into_compressed(mut self) -> Vec<CompressedWord> {
        if !self.is_empty() {
            let high = (self.state.lower >> (State::BITS - CompressedWord::BITS)).as_();
            self.buf.push(high);
        }
        self.buf
    }

    /// Returns a view into the full compressed data currently on the stack.
    ///
    /// This is a low level method that provides a view into the current compressed
    /// data at zero cost, but in a somewhat inconvenient representation. In most
    /// cases you will likely want to call one of [`get_compressed`],
    /// [`into_compressed`], or [`iter_compressed`] instead, as these methods
    /// provide the the compressed data in more convenient forms (which is also the
    /// form expected by the constructor [`with_compressed_data`]).
    ///
    /// The return value of this method is a tuple `(bulk, head)`, where
    /// `bulk: &[CompressedWord]` has variable size (and can be empty) and `head: [CompressedWord; 2]` has a
    /// fixed size. When encoding or decoding data, `head` typically changes with
    /// each encoded or decoded symbol while `bulk` changes only infrequently
    /// (whenever `head` overflows or underflows).
    ///
    /// # TODO
    ///
    /// Should return `(&[CompressedWord], State)` instead, since this will probably be used
    /// by [`SeekableDecoder`].
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`with_compressed_data`]: #method.with_compressed_data
    pub fn as_compressed_raw(&self) -> (&[CompressedWord], <Self as Code>::State) {
        (&self.buf, self.state)
    }

    /// Assembles the current compressed data into a single slice.
    ///
    /// This method is similar to [`as_compressed_raw`] with the difference that it
    /// concatenates the `bulk` and `head` before returning them. The concatenation
    /// truncates any trailing zero words, which is compatible with the constructor
    /// [`with_compressed_data`].
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
    /// use ans::{distributions::Categorical, stack::DefaultCoder, Decode};
    ///
    /// let mut coder = DefaultCoder::new();
    ///
    /// // Push some data on the coder.
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution = Categorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
    ///     .unwrap();
    /// coder.encode_iid_symbols_reverse(&symbols, &distribution).unwrap();
    ///
    /// // Inspect the compressed data.
    /// dbg!(coder.get_compressed());
    ///
    /// // We can still use the coder afterwards.
    /// let reconstructed = coder.decode_iid_symbols(4, &distribution).collect::<Vec<_>>();
    /// assert_eq!(reconstructed, symbols);
    /// ```
    ///
    /// [`as_compressed_raw`]: #method.as_compressed_raw
    /// [`with_compressed_data`]: #method.with_compressed_data
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn get_compressed(&mut self) -> CoderGuard<'_, CompressedWord, State> {
        CoderGuard::new(self)
    }

    /// Iterates over the compressed data currently on the stack.
    ///
    /// In contrast to [`get_compressed`] or [`into_compressed`], this method does
    /// not require mutable access or even ownership of the `Coder`.
    ///
    /// # Example
    ///
    /// ```
    /// use ans::{distributions::{Categorical, LeakyQuantizer}, stack::DefaultCoder, Encode};
    ///
    /// // Create a coder and encode some stuff.
    /// let mut coder = DefaultCoder::new();
    /// let symbols = vec![8, -12, 0, 7];
    /// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let distribution =
    ///     quantizer.quantize(statrs::distribution::Normal::new(0.0, 10.0).unwrap());
    /// coder.encode_iid_symbols(&symbols, &distribution);
    ///
    /// // Iterate over compressed data, collect it into to a vector, and compare to more direct method.
    /// let compressed_iter = coder.iter_compressed();
    /// let compressed_collected = compressed_iter.collect::<Vec<_>>();
    /// assert!(!compressed_collected.is_empty());
    /// assert_eq!(compressed_collected, &*coder.get_compressed());
    ///
    /// // We can also iterate in reverse direction, which is useful for streaming decoding.
    /// let compressed_iter_reverse = coder.iter_compressed().rev();
    /// let compressed_collected_reverse = compressed_iter_reverse.collect::<Vec<_>>();
    /// let mut compressed_direct = coder.into_compressed();
    /// assert!(!compressed_collected_reverse.is_empty());
    /// assert_ne!(compressed_collected_reverse, compressed_direct);
    /// compressed_direct.reverse();
    /// assert_eq!(compressed_collected_reverse, compressed_direct);
    /// ```
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn iter_compressed(
        &self,
    ) -> impl Iterator<Item = CompressedWord> + ExactSizeIterator + DoubleEndedIterator + '_ {
        IterCompressed::new(self)
    }

    /// Returns the number of compressed words on the stack.
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

    /// Returns the size of the current stack of compressed data in bits.
    ///
    /// This includes some constant overhead unless the coder is completely empty
    /// (see [`num_words`](#method.num_words)).
    ///
    /// The returned value is a multiple of the bitlength of the compressed word
    /// type `CompressedWord`.
    pub fn num_bits(&self) -> usize {
        CompressedWord::BITS * self.num_words()
    }
}

impl<CompressedWord, State> Encode for Encoder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    /// To encode a symbol, we do:
    /// - if range <= W::max_value().into():
    ///     - copy the most significant half of `lower` to a buffer
    ///     - shift both `range` and `lower` to the left by `W::num_bits()`
    /// - look up the symbol's `left_marginal` and `probability`
    /// - set `lower = lower + left_marginal * (range >> PRECISION)`
    /// - set `range = probability * (range >> PRECISION)
    fn encode_symbol<S, D>(
        &mut self,
        symbol: impl Borrow<S>,
        distribution: D,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution<Symbol = S>,
        D::Probability: Into<Self::CompressedWord>,
    {
        let (left_sided_cumulative, probability) = distribution
            .left_cumulative_and_probability(symbol)
            .map_err(|()| EncodingError::ImpossibleSymbol)?;

        let scale = self.state.range >> D::PRECISION;
        self.state.lower = self
            .state
            .lower
            .wrapping_add(&(scale * left_sided_cumulative.into().into()));

        // This cannot overflow since `scale * probability <= (range >> PRECISION) << PRECISION`
        self.state.range =
            scale * probability.into().into() + self.state.range % (State::one() << D::PRECISION);
        // TODO: the last part is probably irrelevant since we shift right by PRECISION as soon as we use `range`
        //
        // Another way of thinking about this is that we move the upper bound
        // `upper = lower + range` analogous to how we move `lower`:
        // `new_upper = upper - scale * ((1 << PRECISION) - right_sided_cumulative)`
        // where `right_sided_cumulative = left_sided_cumulative + probability`.

        // TODO: turns out this is wrong; instead:
        // - CHECK AT BEGINNING OF METHOD: if range < 1 << PRECISION:
        //   - set `upper = lower | (State::max_value() >> CompressedWord::BITS)`; this must be:
        //     - `old_lower`
        //   emit a compressed word; unless we would have emitted a word anyway, this must
        //   truncate some set bits of `lower ^ (lower + range)`, so fall through to next
        //   item:
        // - emit words while highest word of `lower + range` is zero (maybe iterate
        //   `(lower ^ (lower + range)).leading_zeros() / CompressedWords::BITS` times);

        if self.state.range < State::one() << (State::BITS - CompressedWord::BITS) {
            // Invariant `range >= State::one() << (State::BITS - CompressedWord::BITS)` is
            // violated. Since `left_cumulative_and_probability` succeeded, we know that
            // `probability != 0` and therefore:
            //   range >= scale * probability = (old_range >> PRECISION) * probability
            //         >= old_range >> PRECISION >= old_range >> CompressedWords::BITS
            // where `old_range` satisfied invariant (1) by assumption. Therefore, the
            // following left-shift restores invariant (1):
            self.state.range = self.state.range << CompressedWord::BITS;

            let high = (self.state.lower >> (State::BITS - CompressedWord::BITS)).as_();
            self.buf.push(high);
            self.state.lower = self.state.lower << CompressedWord::BITS;
        }

        Ok(())
    }

    fn encoder_state(&self) -> &Self::State {
        &self.state
    }
}

pub struct Decoder<'compressed, CompressedWord: BitArray, State: BitArray> {
    buf: &'compressed [CompressedWord],

    /// Points to the next word in `buf` to be read if `state` underflows.
    pos: usize,
    state: CoderState<CompressedWord, State>,

    /// Invariant: `point.wrapping_sub(&state.lower) < state.range`
    point: State,
}

/// Type alias for a [`Decoder`] with sane parameters for typical use cases.
pub type DefaultDecoder<'compressed> = Decoder<'compressed, u32, u64>;

impl<'compressed, CompressedWord, State> Debug for Decoder<'compressed, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(bit_array_to_chunks_exact(self.state.lower).chain(self.buf.iter().cloned()))
            .finish()
    }
}

impl<'compressed, CompressedWord, State> Code for Decoder<'compressed, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type State = CoderState<CompressedWord, State>;
    type CompressedWord = CompressedWord;
}

#[derive(Debug)]
pub enum DecodingError {
    InvalidCompressedData,
}

impl std::fmt::Display for DecodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tried to decode invalid compressed data.")
    }
}

impl Error for DecodingError {}

impl<'compressed, CompressedWord, State> Decode for Decoder<'compressed, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
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
    fn decode_symbol<D>(&mut self, distribution: D) -> Result<D::Symbol, Self::DecodingError>
    where
        D: DiscreteDistribution,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        // We maintain the following invariant (*):
        //   point (-) lower < range
        // where (-) denotes wrapping subtraction (in `Self::State`).

        let scale = self.state.range >> D::PRECISION;
        let quantile = self.point.wrapping_sub(&self.state.lower) / scale;
        if quantile == State::one() << D::PRECISION {
            // This can only happen if both of the following conditions apply:
            // (i) we are decoding invalid compressed data; and
            // (ii) we use entropy models with varying `PRECISION`s.
            return Err(DecodingError::InvalidCompressedData);
        }

        let (symbol, left_sided_cumulative, probability) =
            distribution.quantile_function(quantile.as_().as_());

        // Update `state` in the same way as we do in `encode_symbol` (see comments there):
        self.state.lower = self
            .state
            .lower
            .wrapping_add(&(scale * left_sided_cumulative.into().into()));
        self.state.range =
            scale * probability.into().into() + self.state.range % (State::one() << D::PRECISION);

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
            if let Some(&word) = self.buf.get(self.pos) {
                self.point = self.point | word.into();
                self.pos += 1;
            } else {
                self.point = self.point | CompressedWord::max_value().into()
            }
        }

        Ok(symbol)
    }

    fn decoder_state(&self) -> &Self::State {
        &self.state
    }
}

impl<'compressed, CompressedWord, State> Decoder<'compressed, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    /// Creates an empty encoder for range coding.
    pub fn new(compressed: &'compressed [CompressedWord]) -> Self {
        assert!(State::BITS >= 2 * CompressedWord::BITS);
        assert_eq!(State::BITS % CompressedWord::BITS, 0);

        let mut point = bit_array_from_chunks(compressed.iter().cloned());

        let pos = if compressed.len() < State::BITS / CompressedWord::BITS {
            // A very short compressed buffer was provided, and therefore `point` is still
            // right-aligned. Shift it over so it's left-aligned and fill it up with ones
            if compressed.len() == 0 {
                // Special case: an empty compressed stream is treated like one with a single
                // `CompressedWord` of value zero.
                point = State::max_value() >> CompressedWord::BITS
            } else {
                point = point << (State::BITS - compressed.len() * CompressedWord::BITS)
                    | State::max_value() >> compressed.len() * CompressedWord::BITS;
            }
            compressed.len()
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

    /// Check if all available data might have been decoded.
    ///
    /// It is in general not possible to tell when all compressed data has been decoded.
    /// However, the converse can often (but not always) be detected with certainty.
    ///
    /// If this method returns `false` then there is definitely still data left to be
    /// decoded. If it returns `true` then the situation is unclear: there may or may not
    /// be a few encoded symbols left. In either case, it is always legal to call
    /// [`decode_symbol`]; it may just return a garbage (but deterministically generated)
    /// symbol.
    ///
    /// This method is useful to check for data corruption. When you think you have decoded
    /// all symbols and this method returns `false` then the compressed data must have been
    /// corrupted. If it returns `true` then there is at least no reason to suggest data
    /// corruption (but obviously also no conclusive prove that the data is valid).
    ///
    /// [`decode_symbol`](#method.decode_symbol).
    pub fn maybe_finished(&self) -> bool {
        self.pos == self.buf.len()
            && self.point.wrapping_sub(&self.state.lower)
                < State::one() << (State::BITS - CompressedWord::BITS)
    }
}

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
    inner: &'a mut Encoder<CompressedWord, State>,
}

impl<CompressedWord, State> Debug for CoderGuard<'_, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<'a, CompressedWord, State> CoderGuard<'a, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn new(coder: &'a mut Encoder<CompressedWord, State>) -> Self {
        // Append state. Will be undone in `<Self as Drop>::drop`.
        if !coder.is_empty() {
            let high = (coder.state.lower >> (State::BITS - CompressedWord::BITS)).as_();
            coder.buf.push(high);
        }
        Self { inner: coder }
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
    fn new<State>(coder: &'a Encoder<CompressedWord, State>) -> Self
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
        self.index_front = std::cmp::min(self.index_back, self.index_front.saturating_add(n));
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
        self.index_back = std::cmp::max(self.index_front, self.index_back.saturating_sub(n));
        self.next_back()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Categorical, DiscreteDistribution, LeakyQuantizer};

    use rand_xoshiro::rand_core::{RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;
    use statrs::distribution::{InverseCDF, Normal};

    #[test]
    fn compress_none() {
        let encoder = DefaultEncoder::new();
        assert!(encoder.is_empty());
        let compressed = encoder.into_compressed();
        assert!(compressed.is_empty());

        let decoder = DefaultDecoder::new(&compressed);
        assert!(decoder.maybe_finished());
    }

    #[test]
    fn compress_one() {
        compress_few(std::iter::once(5), 1)
    }

    #[test]
    fn compress_two() {
        compress_few([2, 8].iter().cloned(), 1)
    }

    #[test]
    fn compress_ten() {
        compress_few(0..10, 2)
    }

    #[test]
    fn compress_twenty() {
        compress_few(-10..10, 4)
    }

    fn compress_few(symbols: impl IntoIterator<Item = i32> + Clone, expected_size: usize) {
        let mut encoder = DefaultEncoder::new();
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-127..=127);
        let distribution = quantizer.quantize(Normal::new(3.2, 5.1).unwrap());

        encoder.encode_iid_symbols(symbols.clone(), &distribution);
        let compressed = encoder.into_compressed();
        assert_eq!(compressed.len(), expected_size);

        let mut decoder = DefaultDecoder::new(&compressed);
        for i in symbols {
            assert_eq!(decoder.decode_symbol(&distribution).unwrap(), i);
        }
        assert!(decoder.maybe_finished());
    }

    #[test]
    fn compress_many_u32_32() {
        compress_many::<u32, u64, u32, 32>();
    }

    // #[test]
    // fn compress_many_u32_24() {
    //     compress_many::<u32, u64, u32, 24>();
    // }

    // #[test]
    // fn compress_many_u32_16() {
    //     compress_many::<u32, u64, u32, 16>();
    // }

    // #[test]
    // fn compress_many_u32_8() {
    //     compress_many::<u32, u64, u32, 8>();
    // }

    // #[test]
    // fn compress_many_u16_16() {
    //     compress_many::<u16, u32, u16, 16>();
    // }

    // #[test]
    // fn compress_many_u16_8() {
    //     compress_many::<u16, u32, u16, 8>();
    // }

    fn compress_many<CompressedWord, State, Probability, const PRECISION: usize>()
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
            let symbol = std::cmp::max(
                -127,
                std::cmp::min(127, (dist.inverse_cdf(quantile) + 0.5) as i32),
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

        let mut encoder = Encoder::<CompressedWord, State>::new();

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
                |((&symbol, &mean), &std)| {
                    (symbol, quantizer.quantize(Normal::new(mean, std).unwrap()))
                },
            ))
            .unwrap();
        dbg!(encoder.num_bits());

        // Test if import/export of compressed data works.
        let compressed = encoder.into_compressed();
        let mut decoder = Decoder::new(&compressed);

        let reconstructed_categorical = decoder
            .decode_iid_symbols(AMT, &categorical)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let reconstructed_gaussian = decoder
            .decode_symbols(
                means
                    .iter()
                    .zip(&stds)
                    .map(|(&mean, &std)| quantizer.quantize(Normal::new(mean, std).unwrap())),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        // assert!(decoder.maybe_finished());

        assert_eq!(symbols_categorical, reconstructed_categorical);
        assert_eq!(symbols_gaussian, reconstructed_gaussian);
    }
}
