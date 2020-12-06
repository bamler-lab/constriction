use std::{borrow::Borrow, error::Error, fmt::Debug, ops::Deref};

use num::{cast::AsPrimitive, CheckedDiv, Zero};

use super::{
    bit_array_from_chunks, bit_array_to_chunks_truncated, distributions::DiscreteDistribution,
    BitArray, Code, Decode, Encode, EncodingError, TryCodingError,
};

/// Entropy coder for both encoding and decoding on a stack
///
/// This is is a very general entropy coder that provides both encoding and
/// decoding, and that is generic over a type `CompressedWord` that defines the smallest unit of
/// compressed data, and a constant `PRECISION` that defines the fixed point
/// precision used in entropy models. See [below](
/// #generic-parameters-compressed-word-type-w-and-precision) for details on these
/// parameters. If you're unsure about the choice of `CompressedWord` and `PRECISION` then use
/// the type alias [`DefaultCoder`], which makes sane choices for typical
/// applications.
///
/// The `Coder` uses an entropy coding algorithm called [range Asymmetric
/// Numeral Systems (rANS)]. This means that it operates as a stack, i.e., a "last
/// in first out" data structure: encoding "pushes symbols on" the stack and
/// decoding "pops symbols off" the stack in reverse order. In contrast to
/// [`SeekableDecoder`], decoding with a `Coder` *consumes* the compressed data for
/// the decoded symbols. This means that encoding and decoding can be interleaved
/// arbitrarily, thus growing and shrinking the stack of compressed data as you go.
///
/// # Example
///
/// The following shows a basic usage example. For more examples, see
/// [`encode_symbols`] or [`encode_iid_symbols`].
///
/// ```
/// use ans::{distributions::LeakyQuantizer, stack::DefaultCoder, Decode};
///
/// // `DefaultCoder` is a type alias to `Coder` with sane generic parameters.
/// let mut coder = DefaultCoder::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
/// let entropy_model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 10.0).unwrap());
///
/// let symbols = vec![-10, 4, 0, 3];
/// coder.encode_iid_symbols_reverse(&symbols, &entropy_model);
/// println!("Encoded into {} bits: {:?}", coder.num_bits(), &*coder.get_compressed());
///
/// // The call to `encode_iid_symbols` above encoded the symbols in reverse order (see
/// // documentation). So popping them off now will yield the same symbols in original order.
/// let reconstructed = coder.decode_iid_symbols(4, &entropy_model).collect::<Vec<_>>();
/// assert_eq!(reconstructed, symbols);
/// ```
///
/// # Generic Parameters: Compressed Word Type `CompressedWord` and `PRECISION`
///
/// The `Coder` is generic over a type `CompressedWord`, which is a [`CompressedWord`], and over
/// a constant `PRECISION` of type `usize`. **If you're unsure how to set these
/// parameters, consider using the type alias [`DefaultCoder`], which uses sane
/// default values.**
///
/// ## Meaning of `CompressedWord` and `PRECISION`
///
/// If you need finer control over the entropy coder, and [`DefaultCoder`] does not
/// fit your needs, then here are the details about the parameters `CompressedWord` and
/// `PRECISION`:
///
/// - `CompressedWord` is the smallest "chunk" of compressed data. It is usually a primitive
///   unsigned integer type, such as `u32` or `u16`. The type `CompressedWord` is also used to
///   represent probabilities in fixed-point arithmetic in any
///   [`DiscreteDistribution`] that can be employed as an entropy model for this
///   `Coder` (however, when representing probabilities, only use the lowest
///   `PRECISION` bits of a `CompressedWord` are ever used).
///
///   The `Coder` operates on an internal state whose size is twice as large as `CompressedWord`.
///   When encoding data, the `Coder` keeps filling up this internal state with
///   compressed data until it is about to overflow. Just before the internal state
///   would overflow, the coder chops off half of it and pushes one "compressed
///   word" of type `CompressedWord` onto a dynamically growable and shrinkable buffer of `CompressedWord`s.
///   Once all data is encoded, the encoder chops the final internal state into two
///   `CompressedWord`s, pushes them onto the buffer, and returns the buffer as the compressed
///   data (see method [`into_compressed`]).
///
/// - `PRECISION` defines the number of bits that the entropy models use for
///   fixed-point representation of probabilities. `PRECISION` must be positive and
///   no larger than the bitlength of `CompressedWord` (e.g., if `CompressedWord` is `u32` then we must have
///   `1 <= PRECISION <= 32`).
///
///   Since the smallest representable probability is `(1/2)^PRECISION`, the largest
///   possible (finite) [information content of a single symbol is `PRECISION`
///   bits. Thus, pushing a single symbol onto the `Coder` increases the "filling
///   level" of the `Coder`'s internal state by at most `PRECISION` bits. Since
///   `PRECISION` is at most the bitlength of `CompressedWord`, the procedure of transferring one
///   `CompressedWord` from the internal state to the buffer described in the list item above is
///   guaranteed to free up enough internal state to encode at least one additional
///   symbol.
///
/// ## Guidance for Choosing `CompressedWord` and `PRECISION`
///
/// If you choose `CompressedWord` and `PRECISION` manually (rather than using a
/// [`DefaultCoder`]), then your choice should take into account the following
/// considerations:
///
/// - Set `PRECISION` to a high enough value so that you can approximate your
///   entropy models sufficiently well. If you have a very precise entropy model of
///   your data in your mind, then choosing a low `PRECISION` won't allow you to
///   represent this entropy model very accurately, thus leading to a mismatch
///   between the used entropy model and the true distribution of the encoded data.
///   This will lead to a *linear* overhead in bitrate (i.e., an overhead that is
///   proportional to the number of encoded symbols).
/// - Depending on the entropy models used, a high `PRECISION` may be more expensive
///   in terms of runtime and memory requirements. If this is a concern, then don't
///   set `PRECISION` too high. In particular, a [`LookupDistribution`] allocates  
///   memory on the order of `O(2^PRECISION)`, i.e., *exponential* in `PRECISION`.
/// - Since `CompressedWord` must be at least `PRECISION` bits in size, a high `PRECISION` means
///   that you will have to use a larger `CompressedWord` type. This has several consequences:
///   - it affects the size of the internal state of the coder; this is relevant if
///     you want to store many different internal states, e.g., as a jump table for
///     a [`SeekableDecoder`].
///   - it leads to a small *constant* overhead in bitrate: since the `Coder`
///     operates on an internal  state of two `CompressedWord`s, it has a constant bitrate
///     overhead between zero and two `CompressedWord`s depending on the filling level of the
///     internal state. This constant  overhead is usually negligible unless you
///     want to compress a very small amount of data.
///   - the choice of `CompressedWord` may have some effect on runtime performance since
///     operations on larger types may be more expensive (remember that the `Coder`
///     operates on a state of twice the size as `CompressedWord`, i.e., if `CompressedWord = u64` then the
///     coder will operate on a `u128`, which may be slow on some hardware). On the
///     other hand, this overhead should not be used as an argument for setting `CompressedWord`
///     to a very small type like `u8` or `u16` since common computing architectures
///     are usually not really faster on very small registers, and a very small `CompressedWord`
///     type will lead to more frequent transfers between the internal state and the
///     growable buffer, which requires potentially expensive branching and memory
///     lookups.
/// - Finally, the "slack" between `PRECISION` and the size of `CompressedWord` has an influence
///   on the bitrate. It is usually *not* a good idea to set `PRECISION` to the
///   highest value allowed for a given `CompressedWord` (e.g., setting `CompressedWord = u32` and
///   `PRECISION = 32` is *not* recommended). This is because, when encoding a
///   symbol, the `Coder` expects there to be at least `PRECISION` bits of entropy
///   in its internal state (conceptually, encoding a symbol `s` consists of
///   consuming `PRECISION` bits of entropy followed by pushing
///   `PRECISION + information_content(s)` bits of entropy onto the internal state).
///   If `PRECISION` is set to the full size of `CompressedWord` then there will be relatively
///   frequent situations where the internal state contains less than `PRECISION`
///   bits of entropy, leading to an overhead (this situation will typically arise
///   after the `Coder` transferred a `CompressedWord` from the internal state to the growable
///   buffer).
///
/// The type alias [`DefaultCoder`] was chose with the above considerations in mind.
///
/// # Consistency Between Encoding and Decoding
///
/// As elaborated in the [crate documentation](index.html#streaming-entropy-coding),
/// encoding and decoding operates on a sequence of symbols. Each symbol can be
/// encoded and decoded with its own entropy model (the symbols can even have
/// heterogeneous types). If your goal is to reconstruct the originally encoded
/// symbols during decoding, then you must employ the same sequence of entropy
/// models (in reversed order) during encoding and decoding.
///
/// However, using the same entropy models for encoding and decoding is not a
/// *general* requirement. It is perfectly legal to push (encode) symbols on the
/// `Coder` using some entropy models, and then pop off (decode) symbols using
/// different entropy models. The popped off symbols will then in general be
/// different from the original symbols, but will be generated in a deterministic
/// way. If there is no deterministic relation between the entropy models used for
/// pushing and popping, and if there is still compressed data left at the end
/// (i.e., if [`is_empty`] returns false), then the popped off symbols are
/// approximately distributed as independent samples from the respective entropy
/// models. Such random samples, which consume parts of the compressed data, are
/// useful in the bits-back algorithm.
///
/// [range Asymmetric Numeral Systems (rANS)]:
/// https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#Range_variants_(rANS)_and_streaming
/// [`encode_symbols`]: #method.encode_symbols
/// [`encode_iid_symbols`]: #method.encode_iid_symbols
/// [`distributions`]: distributions/index.html
/// [entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
/// [information content]: https://en.wikipedia.org/wiki/Information_content
/// [`encode_symbols`]: #method.encode_symbols
/// [`is_empty`]: #method.is_empty`
/// [`into_compressed`]: #method.into_compressed
pub struct Coder<CompressedWord: BitArray, State: BitArray> {
    buf: Vec<CompressedWord>,
    state: State,
}

/// Type alias for a [`Coder`] with sane parameters for typical use cases.
///
/// This type alias sets the generic type arguments `CompressedWord` and `State` to
/// sane values for many typical use cases.
pub type DefaultCoder = Coder<u32, u64>;

impl<CompressedWord, State> Debug for Coder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter_compressed()).finish()
    }
}

impl<CompressedWord, State> Coder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    /// Creates an empty ANS entropy coder.
    ///
    /// This is usually the starting point if you want to *compress* data.
    ///
    /// # Example
    ///
    /// ```
    /// let mut coder = ans::stack::DefaultCoder::new();
    ///
    /// // ... push some symbols onto the coder ...
    ///
    /// // Finally, get the compressed data.
    /// let compressed = coder.into_compressed();
    /// ```
    pub fn new() -> Self {
        assert!(State::BITS >= 2 * CompressedWord::BITS);

        Self {
            buf: Vec::new(),
            state: State::zero(),
        }
    }

    /// Creates an ANS coder with some initial compressed data.
    ///
    /// This is usually the starting point if you want to *decompress* data
    /// previously obtained from [`into_compressed`](#method.into_compressed).
    /// However, it can also be used to append more symbols to an existing
    /// compressed buffer of data.
    pub fn with_compressed_data(mut compressed: Vec<CompressedWord>) -> Self {
        assert!(State::BITS >= 2 * CompressedWord::BITS);

        let state = bit_array_from_chunks(
            std::iter::repeat_with(|| compressed.pop()).scan((), |(), chunk| chunk),
        );

        Self {
            buf: compressed,
            state,
        }
    }

    pub fn encode_symbols_reverse<D, S, I>(
        &mut self,
        symbols_and_distributions: I,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution,
        D::Probability: Into<CompressedWord>,
        S: Borrow<D::Symbol>,
        I: IntoIterator<Item = (S, D)>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_symbols(symbols_and_distributions.into_iter().rev())
    }

    pub fn try_encode_symbols_reverse<E, D, S, I>(
        &mut self,
        symbols_and_distributions: I,
    ) -> Result<(), TryCodingError<EncodingError, E>>
    where
        E: Error + 'static,
        D: DiscreteDistribution,
        D::Probability: Into<CompressedWord>,
        S: Borrow<D::Symbol>,
        I: IntoIterator<Item = std::result::Result<(S, D), E>>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.try_encode_symbols(symbols_and_distributions.into_iter().rev())
    }

    pub fn encode_iid_symbols_reverse<D, S, I>(
        &mut self,
        symbols: I,
        distribution: &D,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution,
        D::Probability: Into<CompressedWord>,
        I: IntoIterator<Item = S>,
        S: Borrow<D::Symbol>,
        I::IntoIter: DoubleEndedIterator,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_iid_symbols(symbols.into_iter().rev(), distribution)
    }

    /// Discards all compressed data and resets the coder to the same state as
    /// [`Coder::new`](#method.new).
    pub fn clear(&mut self) {
        self.buf.clear();
        self.state = State::zero();
    }

    /// Check if no data for decoding is left.
    ///
    /// Same as [`Decoder::maybe_finished`], just with a more suitable considering the
    /// fact that this coder operates as a growable and shrinkable stack.
    ///
    /// Note that you can still pop symbols off an empty coder, but this is only
    /// useful in rare edge cases, see documentation of
    /// [`decode_symbol`](#method.decode_symbol).
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty() && self.state == State::zero()
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
    /// // Push some data on the coder:
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution = Categorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
    ///     .unwrap();
    /// coder.encode_iid_symbols_reverse(&symbols, &distribution).unwrap();
    ///
    /// // Get the compressed data, consuming the coder:
    /// let compressed = coder.into_compressed();
    ///
    /// // ... write `compressed` to a file and then read it back later ...
    ///
    /// // Create a new coder with the same state and use it for decompression:
    /// let mut coder = DefaultCoder::with_compressed_data(compressed);
    /// let reconstructed = coder.decode_iid_symbols(4, &distribution).collect::<Vec<_>>();
    /// assert_eq!(reconstructed, symbols);
    /// assert!(coder.is_empty())
    /// ```
    pub fn into_compressed(mut self) -> Vec<CompressedWord> {
        for chunk in bit_array_to_chunks_truncated(self.state).rev() {
            self.buf.push(chunk)
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
    pub fn as_compressed_raw(&self) -> (&[CompressedWord], State) {
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

    /// TODO
    pub fn seekable_decoder(&self) -> SeekableDecoder<'_, CompressedWord, State> {
        SeekableDecoder::from(self)
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
        self.buf.len() + bit_array_to_chunks_truncated::<_, CompressedWord>(self.state).len()
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

impl<CompressedWord, State> Code for Coder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type State = State;
    type CompressedWord = CompressedWord;
}

impl<CompressedWord, State> Encode for Coder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    /// Encodes a single symbol and appends it to the compressed data.
    ///
    /// This is a low level method. You probably usually want to call a batch method
    /// like [`encode_symbols`](#method.encode_symbols) or
    /// [`encode_iid_symbols`](#method.encode_iid_symbols) instead. See examples there.
    ///
    /// The bound `impl Borrow<S>` on argument `symbol` essentially means that you
    /// can provide the symbol either by value or by reference.
    ///
    /// This method is called `encode_symbol` rather than `encode_symbol` to stress
    /// the fact that the `Coder` is a stack: the last symbol *pushed onto* the
    /// stack will be the first symbol that [`decode_symbol`](#method.decode_symbol) will
    /// *pop off* the stack.
    ///
    /// Returns [`Err(ImpossibleSymbol)`] if `symbol` has zero probability under the
    /// entropy model `distribution`. This error can usually be avoided by using a
    /// "leaky" distribution, i.e., a distribution that assigns a nonzero
    /// probability to all symbols within a finite domain. Leaky distributions can
    /// be constructed with, e.g., a
    /// [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_floating_point_probabilities`](
    /// distributions/struct.Categorical.html#method.from_floating_point_probabilities).
    ///
    /// [`Err(ImpossibleSymbol)`]: enum.EncodingError.html#variant.ImpossibleSymbol
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

        if (self.state >> (State::BITS - D::PRECISION)).as_() >= probability.into() {
            self.buf.push(self.state.as_());
            self.state = self.state >> CompressedWord::BITS;
        }

        let prefix = self
            .state
            .checked_div(&probability.into().into())
            .ok_or(EncodingError::ImpossibleSymbol)?;

        let suffix = left_sided_cumulative.into().into() + self.state % probability.into().into();
        self.state = (prefix << D::PRECISION) | suffix.into();

        Ok(())
    }

    fn encoder_state(&self) -> &Self::State {
        &self.state
    }
}

impl<CompressedWord, State> Decode for Coder<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type DecodingError = std::convert::Infallible;

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
        let quantile = (self.state % (State::one() << D::PRECISION)).as_().as_();
        let rest = self.state >> D::PRECISION;
        let (symbol, left_sided_cumulative, probability) = distribution.quantile_function(quantile);
        self.state =
            probability.into().into() * rest + (quantile - left_sided_cumulative).into().into();

        if self.state < State::one() << (State::BITS - CompressedWord::BITS) {
            if let Some(word) = self.buf.pop() {
                self.state = (self.state << CompressedWord::BITS) | word.into();
            }
        }

        Ok(symbol)
    }

    fn decoder_state(&self) -> &Self::State {
        &self.state
    }

    fn maybe_finished(&self) -> bool {
        self.is_empty()
    }
}

/// Provides temporary read-only access to the compressed data wrapped in a
/// [`Coder`].
///
/// Dereferences to `&[CompressedWord]`. See [`Coder::get_compressed`] for an example.
///
/// [`Coder`]: struct.Coder.html
/// [`Coder::get_compressed`]: struct.Coder.html#method.get_compressed
pub struct CoderGuard<'a, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    inner: &'a mut Coder<CompressedWord, State>,
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
    fn new(coder: &'a mut Coder<CompressedWord, State>) -> Self {
        // Append state. Will be undone in `<Self as Drop>::drop`.
        for chunk in bit_array_to_chunks_truncated(coder.state).rev() {
            coder.buf.push(chunk)
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
        // Revert what we did in `Self::new`.
        for _ in bit_array_to_chunks_truncated::<_, CompressedWord>(self.inner.state) {
            self.inner.buf.pop();
        }
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

struct IterCompressed<'a, CompressedWord, State> {
    buf: &'a [CompressedWord],
    state: State,
    index_front: usize,
    index_back: usize,
}

impl<'a, CompressedWord, State> IterCompressed<'a, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn new(coder: &'a Coder<CompressedWord, State>) -> Self {
        let (buf, state) = coder.as_compressed_raw();

        // This can only fail if we wouldn't even be able to allocate space for `state` on the heap.
        let len = buf
            .len()
            .checked_add(bit_array_to_chunks_truncated::<_, CompressedWord>(coder.state).len())
            .expect("Out of memory.");

        Self {
            buf,
            state,
            index_front: 0,
            index_back: len,
        }
    }
}

impl<CompressedWord, State> Iterator for IterCompressed<'_, CompressedWord, State>
where
    CompressedWord: BitArray,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type Item = CompressedWord;

    fn next(&mut self) -> Option<Self::Item> {
        let index_front = self.index_front;
        if index_front == self.index_back {
            None
        } else {
            self.index_front += 1;
            let result = self.buf.get(index_front).cloned().unwrap_or_else(|| {
                (self.state >> (CompressedWord::BITS * (index_front - self.buf.len()))).as_()
            });
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

impl<CompressedWord, State> ExactSizeIterator for IterCompressed<'_, CompressedWord, State>
where
    CompressedWord: BitArray,
    State: BitArray + AsPrimitive<CompressedWord>,
{
}

impl<CompressedWord, State> DoubleEndedIterator for IterCompressed<'_, CompressedWord, State>
where
    CompressedWord: BitArray,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index_front == self.index_back {
            None
        } else {
            // We can subtract one because `self.index_back > self.index_front >= 0`.
            self.index_back -= 1;
            let result = self.buf.get(self.index_back).cloned().unwrap_or_else(|| {
                (self.state >> (CompressedWord::BITS * (self.index_back - self.buf.len()))).as_()
            });
            Some(result)
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.index_back = std::cmp::max(self.index_front, self.index_back.saturating_sub(n));
        self.next_back()
    }
}

/// TODO
///
/// We'll probably need a trait `SeekableDecoder` that works across entropy coding
/// algorithms.
pub struct SeekableDecoder<'data, CompressedWord: BitArray, State: BitArray> {
    // Holds only the bulk of the compressed data, not the initial decoder state.
    data: &'data [CompressedWord],

    // Points one behind the next compressed word that will be read.
    // Thus, `pos == 0` means no more compressed words can be read.
    pos: usize,
    state: State,
}

impl<'data, CompressedWord, State> SeekableDecoder<'data, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    pub fn new(compressed: &'data [CompressedWord]) -> Self {
        assert!(State::BITS >= 2 * CompressedWord::BITS);

        let mut pos = compressed.len();
        let state = bit_array_from_chunks(compressed.iter().rev().map(|word| {
            pos -= 1;
            *word
        }));

        Self {
            data: &compressed[..pos],
            pos,
            state,
        }
    }

    pub fn from_raw(bulk: &'data [CompressedWord], state: State) -> Self {
        assert!(State::BITS >= 2 * CompressedWord::BITS);

        Self {
            data: bulk,
            pos: bulk.len(),
            state,
        }
    }

    pub fn pos(&self) -> (usize, State) {
        (self.pos, self.state)
    }

    pub fn seek(&mut self, pos: usize, state: State) {
        assert!(pos <= self.data.len());
        self.pos = pos;
        self.state = state;
    }
}

impl<'data, CompressedWord, State> From<&'data Coder<CompressedWord, State>>
    for SeekableDecoder<'data, CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn from(coder: &'data Coder<CompressedWord, State>) -> Self {
        SeekableDecoder::from_raw(&coder.buf, coder.state)
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
        let coder1 = DefaultCoder::new();
        assert!(coder1.is_empty());
        let compressed = coder1.into_compressed();
        assert!(compressed.is_empty());

        let coder2 = DefaultCoder::with_compressed_data(compressed);
        assert!(coder2.is_empty());
    }

    #[test]
    fn compress_one() {
        generic_compress_few(std::iter::once(5), 1)
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
        I::IntoIter: Clone + DoubleEndedIterator,
    {
        let symbols = symbols.into_iter();

        let mut encoder = DefaultCoder::new();
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-127..=127);
        let distribution = quantizer.quantize(Normal::new(3.2, 5.1).unwrap());

        // We don't reuse the same encoder for decoding because we want to test
        // if exporting and re-importing of compressed data works.
        encoder
            .encode_iid_symbols(symbols.clone(), &distribution)
            .unwrap();
        let compressed = encoder.into_compressed();
        assert_eq!(compressed.len(), expected_size);

        let mut decoder = DefaultCoder::with_compressed_data(compressed);
        for symbol in symbols.rev() {
            assert_eq!(decoder.decode_symbol(&distribution).unwrap(), symbol);
        }
        assert!(decoder.is_empty());
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

        let mut coder = Coder::<CompressedWord, State>::new();

        coder
            .encode_iid_symbols_reverse(&symbols_categorical, &categorical)
            .unwrap();
        dbg!(coder.num_bits(), AMT as f64 * categorical.entropy::<f64>());

        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-127..=127);
        coder
            .encode_symbols_reverse(symbols_gaussian.iter().zip(&means).zip(&stds).map(
                |((&symbol, &mean), &std)| {
                    (symbol, quantizer.quantize(Normal::new(mean, std).unwrap()))
                },
            ))
            .unwrap();
        dbg!(coder.num_bits());

        // Test if import/export of compressed data works.
        let compressed = coder.into_compressed();
        let mut coder = Coder::with_compressed_data(compressed);

        let reconstructed_gaussian = coder
            .decode_symbols(
                means
                    .iter()
                    .zip(&stds)
                    .map(|(&mean, &std)| quantizer.quantize(Normal::new(mean, std).unwrap())),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let reconstructed_categorical = coder
            .decode_iid_symbols(AMT, &categorical)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(coder.is_empty());

        assert_eq!(symbols_gaussian, reconstructed_gaussian);
        assert_eq!(symbols_categorical, reconstructed_categorical);
    }
}
