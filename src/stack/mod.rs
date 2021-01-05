//! Encoding and infallible decoding on a stack ("last in first out")
//!
//!

pub mod backend;
pub mod stable;

use std::{
    borrow::Borrow, convert::TryInto, error::Error, fmt::Debug, marker::PhantomData, ops::Deref,
};

use num::cast::AsPrimitive;

use crate::{
    bit_array_from_chunks, bit_array_to_chunks_truncated, distributions::DiscreteDistribution,
    BitArray, Code, Decode, Encode, EncodingError, IntoDecoder, TryCodingError,
};

use self::backend::{Backend, ReadItems, ReadLookaheadItems, WriteItems, WriteMutableItems};

/// Entropy coder for both encoding and decoding on a stack
///
/// This is is a very general entropy coder that provides both encoding and
/// decoding, and that is generic over a type `CompressedWord` that defines the smallest unit of
/// compressed data, and a constant `PRECISION` that defines the fixed point
/// precision used in entropy models. See [below](
/// #generic-parameters-compressed-word-type-w-and-precision) for details on these
/// parameters. If you're unsure about the choice of `CompressedWord` and `PRECISION` then use
/// the type alias [`DefaultStack`], which makes sane choices for typical
/// applications.
///
/// The `Stack` uses an entropy coding algorithm called [range Asymmetric
/// Numeral Systems (rANS)]. This means that it operates as a stack, i.e., a "last
/// in first out" data structure: encoding "pushes symbols on" the stack and
/// decoding "pops symbols off" the stack in reverse order. In contrast to
/// [`SeekableDecoder`], decoding with a `Stack` *consumes* the compressed data for
/// the decoded symbols. This means that encoding and decoding can be interleaved
/// arbitrarily, thus growing and shrinking the stack of compressed data as you go.
///
/// # Example
///
/// The following shows a basic usage example. For more examples, see
/// [`encode_symbols`] or [`encode_iid_symbols`].
///
/// ```
/// use constriction::{distributions::LeakyQuantizer, stack::DefaultStack, Decode};
///
/// // `DefaultStack` is a type alias to `Stack` with sane generic parameters.
/// let mut stack = DefaultStack::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
/// let entropy_model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 10.0).unwrap());
///
/// let symbols = vec![-10, 4, 0, 3];
/// stack.encode_iid_symbols_reverse(&symbols, &entropy_model).unwrap();
/// println!("Encoded into {} bits: {:?}", stack.num_bits(), &*stack.get_compressed());
///
/// // The call to `encode_iid_symbols` above encoded the symbols in reverse order (see
/// // documentation). So popping them off now will yield the same symbols in original order.
/// let reconstructed = stack
///     .decode_iid_symbols(4, &entropy_model)
///     .collect::<Result<Vec<_>, std::convert::Infallible>>()
///     .unwrap();
/// assert_eq!(reconstructed, symbols);
/// ```
///
/// # Generic Parameters: Compressed Word Type `CompressedWord` and `PRECISION`
///
/// The `Stack` is generic over a type `CompressedWord`, which is a [`BitArray`],
/// and over a constant `PRECISION` of type `usize`. **If you're unsure how to set
/// these parameters, consider using the type alias [`DefaultStack`], which uses
/// sane default values.**
///
/// ## Meaning of `CompressedWord` and `PRECISION`
///
/// If you need finer control over the entropy coder, and [`DefaultStack`] does not
/// fit your needs, then here are the details about the parameters `CompressedWord`
/// and `PRECISION`:
///
/// - `CompressedWord` is the smallest "chunk" of compressed data. It is usually a
///   primitive unsigned integer type, such as `u32` or `u16`. The type
///   `CompressedWord` is also used to represent probabilities in fixed-point
///   arithmetic in any [`DiscreteDistribution`] that can be employed as an entropy
///   model for this `Stack` (however, when representing probabilities, only use the
///   lowest `PRECISION` bits of a `CompressedWord` are ever used).
///
///   The `Stack` operates on an internal state whose size is twice as large as
///   `CompressedWord`. When encoding data, the `Stack` keeps filling up this
///   internal state with compressed data until it is about to overflow. Just before
///   the internal state would overflow, the stack chops off half of it and pushes
///   one "compressed word" of type `CompressedWord` onto a dynamically growable and
///   shrinkable buffer of `CompressedWord`s. Once all data is encoded, the encoder
///   chops the final internal state into two `CompressedWord`s, pushes them onto
///   the buffer, and returns the buffer as the compressed data (see method
///   [`into_compressed`]).
///
/// - `PRECISION` defines the number of bits that the entropy models use for
///   fixed-point representation of probabilities. `PRECISION` must be positive and
///   no larger than the bitlength of `CompressedWord` (e.g., if `CompressedWord` is
///  `u32` then we must have `1 <= PRECISION <= 32`).
///
///   Since the smallest representable nonzero probability is `(1/2)^PRECISION`, the
///   largest possible (finite) [information content] of a single symbol is
///   `PRECISION` bits. Thus, pushing a single symbol onto the `Stack` increases the
///   "filling level" of the `Stack`'s internal state by at most `PRECISION` bits.
///   Since `PRECISION` is at most the bitlength of `CompressedWord`, the procedure
///   of transferring one `CompressedWord` from the internal state to the buffer
///   described in the list item above is guaranteed to free up enough internal
///   state to encode at least one additional symbol.
///
/// ## Guidance for Choosing `CompressedWord` and `PRECISION`
///
/// If you choose `CompressedWord` and `PRECISION` manually (rather than using a
/// [`DefaultStack`]), then your choice should take into account the following
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
/// - Since `CompressedWord` must be at least `PRECISION` bits in size, a high
///   `PRECISION` means that you will have to use a larger `CompressedWord` type.
///   This has several consequences:
///   - it affects the size of the internal state of the stack; this is relevant if
///     you want to store many different internal states, e.g., as a jump table for
///     a [`SeekableDecoder`].
///   - it leads to a small *constant* overhead in bitrate: since the `Stack`
///     operates on an internal  state of two `CompressedWord`s, it has a constant
///     bitrate  overhead between zero and two `CompressedWord`s depending on the
///     filling level of the internal state. This constant  overhead is usually
///     negligible unless you want to compress a very small amount of data.
///   - the choice of `CompressedWord` may have some effect on runtime performance
///     since operations on larger types may be more expensive (remember that the x
///     since operates on a state of twice the size as `CompressedWord`, i.e., if
///     `CompressedWord = u64` then the stack will operate on a `u128`, which may be
///     slow on some hardware). On the other hand, this overhead should not be used
///     as an argument for setting `CompressedWord` to a very small type like `u8`
///     or `u16` since common computing architectures are usually not really faster
///     on very small registers, and a very small `CompressedWord` type will lead to
///     more frequent transfers between the internal state and the growable buffer,
///     which requires potentially expensive branching and memory lookups.
/// - Finally, the "slack" between `PRECISION` and the size of `CompressedWord` has
///   an influence on the bitrate. It is usually *not* a good idea to set
///   `PRECISION` to the highest value allowed for a given `CompressedWord` (e.g.,
///   setting `CompressedWord = u32` and `PRECISION = 32` is *not* recommended).
///   This is because, when encoding a symbol, the `Stack` expects there to be at
///   least `PRECISION` bits of entropy in its internal state (conceptually,
///   encoding a symbol `s` consists of consuming `PRECISION` bits of entropy
///   followed by pushing `PRECISION + information_content(s)` bits of entropy onto
///   the internal state).  If `PRECISION` is set to the full size of
///   `CompressedWord` then there will be relatively frequent situations where the
///   internal state contains less than `PRECISION` bits of entropy, leading to an
///   overhead (this situation will typically arise after the `Stack` transferred a
///   `CompressedWord` from the internal state to the growable buffer).
///
/// The type alias [`DefaultStack`] was chose with the above considerations in mind.
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
/// `Stack` using some entropy models, and then pop off (decode) symbols using
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
#[derive(Clone)]
pub struct Stack<CompressedWord, State, Buf = Vec<CompressedWord>>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    buf: Buf,

    /// Invariant: `state >= State::one() << (State::BITS - CompressedWord::BITS)`
    /// unless `buf.is_empty()` or this is the `waste` part of an `stable::Coder`.
    state: State,

    /// We keep track of the `CompressedWord` type so that we can statically enforce
    /// the invariant for `state`.
    phantom: PhantomData<CompressedWord>,
}

/// Type alias for a [`Stack`] with sane parameters for typical use cases.
///
/// This type alias sets the generic type arguments `CompressedWord` and `State` to
/// sane values for many typical use cases.
pub type DefaultStack = Stack<u32, u64>;

impl<CompressedWord, State, Buf> Debug for Stack<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: ReadItems<CompressedWord>,
    for<'a> &'a Buf: IntoIterator<Item = &'a CompressedWord>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter_compressed()).finish()
    }
}

impl<CompressedWord, State, Buf, const PRECISION: usize> IntoDecoder<PRECISION>
    for Stack<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: ReadItems<CompressedWord>,
{
    type Decoder = Self;
}

impl<CompressedWord, State, Buf> Default for Stack<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: Default,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<CompressedWord, State, Buf> Stack<CompressedWord, State, Buf>
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
    /// let mut stack = constriction::stack::DefaultStack::new();
    ///
    /// // ... push some symbols onto the stack ...
    ///
    /// // Finally, get the compressed data.
    /// let compressed = stack.into_compressed();
    /// ```
    pub fn new() -> Self
    where
        Buf: Default,
    {
        assert!(State::BITS >= 2 * CompressedWord::BITS);

        Self {
            state: State::zero(),
            buf: Default::default(),
            phantom: PhantomData,
        }
    }

    pub fn with_state_and_empty_buf(state: State) -> Self
    where
        Buf: Default,
    {
        Self {
            state,
            buf: Default::default(),
            phantom: PhantomData,
        }
    }

    pub fn with_buf_and_state(buf: Buf, state: State) -> Result<Self, ()>
    where
        Buf: ReadLookaheadItems<CompressedWord>,
    {
        if buf.is_at_end() || state >= State::one() << (State::BITS - CompressedWord::BITS) {
            Ok(Self {
                buf,
                state,
                phantom: PhantomData,
            })
        } else {
            Err(())
        }
    }

    /// Creates an ANS stack with some initial compressed data.
    ///
    /// This is usually the starting point if you want to *decompress* data previously
    /// obtained from [`into_compressed`].  However, it can also be used to append more
    /// symbols to an existing compressed buffer of data.
    ///
    /// Returns `Err(compressed)` if `compressed` is not empty and its last entry is
    /// zero, since a `Stack` cannot represent trailing zero words. This error cannot
    /// occur if `compressed` was obtained from [`into_compressed`], which never returns
    /// data with a trailing zero word. If you want to construct a `Stack` from an
    /// unknown source of binary data (e.g., to decode some side information into latent
    /// variables) then call [`from_binary`] instead.
    ///
    /// [`into_compressed`]: #method.into_compressed
    /// [`from_binary`]: #method.from_binary
    pub fn from_compressed(mut compressed: Buf) -> Result<Self, Buf>
    where
        Buf: ReadItems<CompressedWord>,
    {
        assert!(State::BITS >= 2 * CompressedWord::BITS);

        if compressed.peek() == Some(&CompressedWord::zero()) {
            return Err(compressed);
        }

        let state = bit_array_from_chunks(
            std::iter::repeat_with(|| compressed.pop()).scan((), |(), chunk| chunk),
        );

        Ok(Self {
            buf: compressed,
            state,
            phantom: PhantomData,
        })
    }

    /// Like [`from_compressed`] but works on any binary data.
    ///
    /// This method is meant for rather advanced use cases. For most common use cases,
    /// you probably want to call [`from_compressed`] instead.
    ///
    /// Different to `from_compressed`, this method also works if `data` ends in a zero
    /// word. Calling this method is equivalent to (but likely more efficient than)
    /// appending a `1` word to `data` and then calling `from_compressed`. Note that
    /// therefore, this method always constructs a non-empty `Stack` (even if `data` is
    /// empty):
    ///
    /// ```
    /// use constriction::stack::DefaultStack;
    ///
    /// let stack1 = DefaultStack::from_binary(Vec::new());
    /// assert!(!stack1.is_empty()); // <-- stack1 is *not* empty.
    ///
    /// let stack2 = DefaultStack::from_compressed(Vec::new()).unwrap();
    /// assert!(stack2.is_empty()); // <-- stack2 is empty.
    /// ```
    /// [`from_compressed`]: #method.from_compressed
    pub fn from_binary(mut data: Buf) -> Self
    where
        Buf: ReadItems<CompressedWord>,
    {
        // We only simulate the effect of appending a `1` to `data` because actually
        // appending a word may cause an expensive resizing of the vector `data`. This
        // resizing is both likely to happen and likely to be unnecessary: `data` may be
        // explicitly copied out from some larger buffer, in which case its capacity will
        // likely match its size, so appending a single word would cause a resize. Further,
        // the `Stack` may be intended for decoding, and so the resizing is completely
        // avoidable.
        let state = bit_array_from_chunks(
            std::iter::once(CompressedWord::one())
                .chain(std::iter::repeat_with(|| data.pop()).scan((), |(), chunk| chunk)),
        );

        Self {
            buf: data,
            state,
            phantom: PhantomData,
        }
    }

    /// Discards all compressed data and resets the stack to the same state as
    /// [`Coder::new`](#method.new).
    pub fn clear(&mut self)
    where
        Buf: WriteMutableItems<CompressedWord>,
    {
        self.buf.clear();
        self.state = State::zero();
    }

    pub fn buf(&self) -> &Buf {
        &self.buf
    }

    pub fn into_buf_and_state(self) -> (Buf, State) {
        (self.buf, self.state)
    }

    #[inline(always)]
    pub(crate) fn decode_remainder_off_state<D, const PRECISION: usize>(
        &mut self,
        probability: D::Probability,
    ) -> Result<D::Probability, EncodingError>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
    {
        let remainder = (self.state % probability.into().into()).as_().as_();
        self.state = self
            .state
            .checked_div(&probability.into().into())
            .ok_or(EncodingError::ImpossibleSymbol)?;

        Ok(remainder)
    }

    #[inline(always)]
    pub(crate) fn append_quantile_to_state<D, const PRECISION: usize>(
        &mut self,
        quantile: D::Probability,
    ) where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<CompressedWord>,
    {
        self.state = (self.state << PRECISION) | quantile.into().into();
    }
}

impl<CompressedWord, State, Buf> Stack<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: WriteItems<CompressedWord>,
{
    #[inline(always)]
    pub(crate) fn flush_state(&mut self) {
        self.buf.push(self.state.as_());
        self.state = self.state >> CompressedWord::BITS;
    }

    pub fn encode_symbols_reverse<S, D, I, const PRECISION: usize>(
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

    pub fn try_encode_symbols_reverse<S, D, E, I, const PRECISION: usize>(
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

    pub fn encode_iid_symbols_reverse<S, D, I, const PRECISION: usize>(
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

    /// Consumes the stack and returns the compressed data.
    ///
    /// The returned data can be used to recreate a stack with the same state
    /// (e.g., for decoding) by passing it to
    /// [`from_compressed`](#method.from_compressed).
    ///
    /// If you don't want to consume the stack, consider calling
    /// [`get_compressed`](#method.get_compressed),
    /// [`iter_compressed`](#method.iter_compressed) instead.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{distributions::Categorical, stack::DefaultStack, Decode};
    ///
    /// let mut stack = DefaultStack::new();
    ///
    /// // Push some data onto the stack:
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution = Categorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
    ///     .unwrap();
    /// stack.encode_iid_symbols_reverse(&symbols, &distribution).unwrap();
    ///
    /// // Get the compressed data, consuming the stack:
    /// let compressed = stack.into_compressed();
    ///
    /// // ... write `compressed` to a file and then read it back later ...
    ///
    /// // Create a new stack with the same state and use it for decompression:
    /// let mut stack = DefaultStack::from_compressed(compressed).expect("Corrupted compressed file.");
    /// let reconstructed = stack
    ///     .decode_iid_symbols(4, &distribution)
    ///     .collect::<Result<Vec<_>, std::convert::Infallible>>()
    ///     .unwrap();
    /// assert_eq!(reconstructed, symbols);
    /// assert!(stack.is_empty())
    /// ```
    pub fn into_compressed(mut self) -> Buf {
        self.buf
            .extend(bit_array_to_chunks_truncated(self.state).rev());
        self.buf
    }

    /// Returns the binary data if it fits precisely into an integer number of
    /// `CompressedWord`s
    ///
    /// This method is meant for rather advanced use cases. For most common use cases,
    /// you probably want to call [`into_compressed`] instead.
    ///
    /// This method is the inverse of [`from_binary`]. It is equivalent to calling
    /// [`into_compressed`], verifying that the returned vector ends in a `1` word, and
    /// popping off that trailing `1` word.
    ///
    /// Returns `Err(())` if the compressed data (excluding an obligatory trailing
    /// `1` bit) does not fit into an integer number of `CompressedWord`s. This error
    /// case includes the case of an empty `Stack` (since an empty `Stack` lacks the
    /// obligatory trailing one-bit).
    ///
    /// # Example
    ///
    /// ```
    /// // Some binary data we want to represent on a `Stack`.
    /// let data = vec![0x89ab_cdef, 0x0123_4567];
    ///
    /// // Constructing a `Stack` with `from_binary` indicates that all bits of `data` are
    /// // considered part of the information-carrying payload.
    /// let stack1 = constriction::stack::DefaultStack::from_binary(data.clone());
    /// assert_eq!(stack1.clone().into_binary().unwrap(), data); // <-- Retrieves the original `data`.
    ///
    /// // By contrast, if we construct a `Stack` with `from_compressed`, we indicate that
    /// // - any leading `0` bits of the last entry of `data` are not considered part of
    /// //   the information-carrying payload; and
    /// // - the (obligatory) first `1` bit of the last entry of `data` defines the
    /// //   boundary between unused bits and information-carrying bits; it is therefore
    /// //   also not considered part of the payload.
    /// // Therefore, `stack2` below only contains `32 * 2 - 7 - 1 = 56` bits of payload,
    /// // which cannot be exported into an integer number of `u32` words:
    /// let stack2 = constriction::stack::DefaultStack::from_compressed(data.clone()).unwrap();
    /// assert!(stack2.clone().into_binary().is_err()); // <-- Returns an error.
    ///
    /// // Use `into_compressed` to retrieve the data in this case:
    /// assert_eq!(stack2.into_compressed(), data);
    ///
    /// // Calling `into_compressed` on `stack1` would append an extra `1` bit to indicate
    /// // the boundary between information-carrying bits and padding `0` bits:
    /// assert_eq!(stack1.into_compressed(), vec![0x89ab_cdef, 0x0123_4567, 0x0000_0001]);
    /// ```
    ///
    /// [`from_binary`]: #method.from_binary
    /// [`into_compressed`]: #method.into_compressed
    pub fn into_binary(mut self) -> Result<Buf, ()> {
        let valid_bits = (State::BITS - 1).wrapping_sub(self.state.leading_zeros() as usize);

        if valid_bits % CompressedWord::BITS != 0 || valid_bits == usize::max_value() {
            Err(())
        } else {
            let truncated_state = self.state ^ (State::one() << valid_bits);
            self.buf
                .extend(bit_array_to_chunks_truncated(truncated_state).rev());
            Ok(self.buf)
        }
    }
}

impl<CompressedWord, State, Buf> Stack<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    /// Check if no data for decoding is left.
    ///
    /// Same as [`Code::maybe_empty`], just with a more suitable name considering the
    /// fact that this particular implementation of `Code` can decide with certainty
    /// whether or not it is empty.
    ///
    /// Note that you can still pop symbols off an empty stack, but this is only
    /// useful in rare edge cases, see documentation of
    /// [`decode_symbol`](#method.decode_symbol).
    pub fn is_empty(&self) -> bool
    where
        Buf: ReadLookaheadItems<CompressedWord>,
    {
        self.buf.is_at_end() && self.state == State::zero()
    }

    /// Assembles the current compressed data into a single slice.
    ///
    /// Returns the concatenation of [`buf`] and [`state`]. The concatenation truncates
    /// any trailing zero words, which is compatible with the constructor
    /// [`from_compressed`].
    ///
    /// This method requires a `&mut self` receiver to temporarily append `state` to
    /// [`buf`] (this mutationwill be reversed to recreate the original `buf` as soon as
    /// the caller drops the returned value). If you don't have mutable access to the
    /// `Stack`, consider calling [`iter_compressed`] instead, or get the `buf` and
    /// `state` separately by calling [`buf`] and [`state`], respectively.
    ///
    /// The return type dereferences to `&[CompressedWord]`, thus providing read-only
    /// access to the compressed data. If you need ownership of the compressed data,
    /// consider calling [`into_compressed`] instead.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{distributions::Categorical, stack::DefaultStack, Decode};
    ///
    /// let mut stack = DefaultStack::new();
    ///
    /// // Push some data on the stack.
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution = Categorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
    ///     .unwrap();
    /// stack.encode_iid_symbols_reverse(&symbols, &distribution).unwrap();
    ///
    /// // Inspect the compressed data.
    /// dbg!(stack.get_compressed());
    ///
    /// // We can still use the stack afterwards.
    /// let reconstructed = stack
    ///     .decode_iid_symbols(4, &distribution)
    ///     .collect::<Result<Vec<_>, std::convert::Infallible>>()
    ///     .unwrap();
    /// assert_eq!(reconstructed, symbols);
    /// ```
    ///
    /// [`buf`]: #method.buf
    /// [`state`]: #method.state
    /// [`from_compressed`]: #method.from_compressed
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn get_compressed<'a>(&'a mut self) -> impl Deref<Target = Buf> + Debug + Drop + 'a
    where
        Buf: ReadItems<CompressedWord> + WriteItems<CompressedWord> + Debug,
    {
        CoderGuard::new(self)
    }

    /// Iterates over the compressed data currently on the stack.
    ///
    /// In contrast to [`get_compressed`] or [`into_compressed`], this method does
    /// not require mutable access or even ownership of the `Stack`.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{distributions::{Categorical, LeakyQuantizer}, stack::DefaultStack, Encode};
    ///
    /// // Create a stack and encode some stuff.
    /// let mut stack = DefaultStack::new();
    /// let symbols = vec![8, -12, 0, 7];
    /// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let distribution =
    ///     quantizer.quantize(statrs::distribution::Normal::new(0.0, 10.0).unwrap());
    /// stack.encode_iid_symbols(&symbols, &distribution).unwrap();
    ///
    /// // Iterate over compressed data, collect it into to a vector, and compare to more direct method.
    /// let compressed_iter = stack.iter_compressed();
    /// let compressed_collected = compressed_iter.collect::<Vec<_>>();
    /// assert!(!compressed_collected.is_empty());
    /// assert_eq!(compressed_collected, *stack.get_compressed());
    /// ```
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn iter_compressed<'a>(&'a self) -> impl Iterator<Item = CompressedWord> + '_
    where
        &'a Buf: IntoIterator<Item = &'a CompressedWord>,
    {
        let buf_iter = self.buf.into_iter().cloned();
        let state_iter = bit_array_to_chunks_truncated(self.state).rev();
        buf_iter.chain(state_iter)
    }

    /// Returns the number of compressed words on the stack.
    ///
    /// This includes a constant overhead of between one and two words unless the
    /// stack is completely empty.
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
    pub fn num_words(&self) -> usize
    where
        Buf: ReadLookaheadItems<CompressedWord>,
    {
        self.buf.amt_left() + bit_array_to_chunks_truncated::<_, CompressedWord>(self.state).len()
    }

    pub fn num_bits(&self) -> usize
    where
        Buf: ReadLookaheadItems<CompressedWord>,
    {
        CompressedWord::BITS * self.num_words()
    }

    pub fn num_valid_bits(&self) -> usize
    where
        Buf: ReadLookaheadItems<CompressedWord>,
    {
        CompressedWord::BITS * self.buf.amt_left()
            + std::cmp::max(State::BITS - self.state.leading_zeros() as usize, 1)
            - 1
    }

    #[inline(always)]
    pub(crate) fn chop_quantile_off_state<D, const PRECISION: usize>(&mut self) -> D::Probability
    where
        CompressedWord: AsPrimitive<D::Probability>,
        D: DiscreteDistribution<PRECISION>,
    {
        let quantile = (self.state % (State::one() << PRECISION)).as_().as_();
        self.state = self.state >> PRECISION;
        quantile
    }

    #[inline(always)]
    pub(crate) fn encode_remainder_onto_state<D, const PRECISION: usize>(
        &mut self,
        remainder: D::Probability,
        probability: D::Probability,
    ) where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<CompressedWord>,
    {
        self.state = self.state * probability.into().into() + remainder.into().into();
    }
}

impl<CompressedWord, State> Stack<CompressedWord, State, Vec<CompressedWord>>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    pub fn into_stable_decoder<const PRECISION: usize>(
        self,
    ) -> Result<stable::Decoder<CompressedWord, State, PRECISION>, Self> {
        self.try_into()
    }

    pub fn into_stable_encoder<const PRECISION: usize>(
        self,
    ) -> Result<stable::Encoder<CompressedWord, State, PRECISION>, Self> {
        self.try_into()
    }
}

impl<CompressedWord, State, Buf> Code for Stack<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: Backend<CompressedWord>,
{
    type State = State;
    type CompressedWord = CompressedWord;

    fn state(&self) -> Self::State {
        self.state
    }

    fn maybe_empty(&self) -> bool {
        true
        // TODO: the following only works if `Buf: ReadLookaheadItems<CompressedWord>.
        // This would need specialization.
        // self.is_at_read_end()
    }
}

impl<CompressedWord, State, Buf, const PRECISION: usize> Encode<PRECISION>
    for Stack<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: WriteItems<CompressedWord>,
{
    /// Encodes a single symbol and appends it to the compressed data.
    ///
    /// This is a low level method. You probably usually want to call a batch method
    /// like [`encode_symbols`](#method.encode_symbols) or
    /// [`encode_iid_symbols`](#method.encode_iid_symbols) instead. See examples there.
    ///
    /// The bound `impl Borrow<D::Symbol>` on argument `symbol` essentially means that
    /// you can provide the symbol either by value or by reference, at your choice.
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
    /// TODO: move this and similar doc comments to the trait definition.
    ///
    /// [`Err(ImpossibleSymbol)`]: enum.EncodingError.html#variant.ImpossibleSymbol
    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        distribution: D,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        let (left_sided_cumulative, probability) = distribution
            .left_cumulative_and_probability(symbol)
            .map_err(|()| EncodingError::ImpossibleSymbol)?;

        if (self.state >> (State::BITS - PRECISION)) >= probability.into().into() {
            self.flush_state();
            // At this point, the invariant on `self.state` (see its doc comment) is
            // temporarily violated, but it will be restored below.
        }

        let remainder = self.decode_remainder_off_state::<D, PRECISION>(probability)?;
        self.append_quantile_to_state::<D, PRECISION>(left_sided_cumulative + remainder);

        Ok(())
    }
}

impl<CompressedWord, State, Buf> Stack<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: ReadItems<CompressedWord>,
{
    /// Checks the invariant on `self.state` and restores it if necessary and possible.
    ///
    /// Returns `Err(())` if the `state` should have been refilled but there are no
    /// compressed words left. Returns `Ok(())` in all other cases.
    #[inline(always)]
    pub(crate) fn try_refill_state_if_necessary(&mut self) -> Result<(), ()> {
        if self.state < State::one() << (State::BITS - CompressedWord::BITS) {
            // Invariant on `self.state` (see its doc comment) is violated. Restore it by
            // refilling with a compressed word from `self.buf` if available.
            self.try_refill_state()
        } else {
            Ok(())
        }
    }

    /// Tries to push a compressed word onto `state` without checking for overflow, thus
    /// potentially truncating `state`. If you're not sure if the operation might
    /// overflow, call [`try_refill_state_if_necessary`] instead.
    ///
    /// This method is *not* declared `unsafe` because the potential truncation is
    /// well-defined and we consider truncating `state` a logic error but not a
    /// memory/safety violation.
    ///
    /// Returns `Ok(())` if a compressed word to refill the state was available and
    /// `Err(())` if no compressed word was available.
    #[inline(always)]
    pub(crate) fn try_refill_state(&mut self) -> Result<(), ()> {
        self.state = (self.state << CompressedWord::BITS) | self.buf.pop().ok_or(())?.into();
        Ok(())
    }
}

impl<CompressedWord, State, Buf, const PRECISION: usize> Decode<PRECISION>
    for Stack<CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: ReadItems<CompressedWord>,
{
    type DecodingError = std::convert::Infallible;

    /// Decodes a single symbol and pops it off the compressed data.
    ///
    /// This is a low level method. You usually probably want to call a batch method
    /// like [`decode_symbols`](#method.decode_symbols) or
    /// [`decode_iid_symbols`](#method.decode_iid_symbols) instead.
    ///
    /// This method is called `decode_symbol` rather than `decode_symbol` to stress the
    /// fact that the `Stack` is a stack: `decode_symbol` will return the *last* symbol
    /// that was previously encoded via [`encode_symbol`](#method.encode_symbol).
    ///
    /// Note that this method cannot fail. It will still produce symbols in a
    /// deterministic way even if the stack is empty, but such symbols will not
    /// recover any previously encoded data and will generally have low entropy.
    /// Still, being able to pop off an arbitrary number of symbols can sometimes be
    /// useful in edge cases of, e.g., the bits-back algorithm.
    fn decode_symbol<D>(&mut self, distribution: D) -> Result<D::Symbol, Self::DecodingError>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        let quantile = self.chop_quantile_off_state::<D, PRECISION>();
        let (symbol, left_sided_cumulative, probability) = distribution.quantile_function(quantile);
        let remainder = quantile - left_sided_cumulative;
        self.encode_remainder_onto_state::<D, PRECISION>(remainder, probability);
        let _ = self.try_refill_state_if_necessary();

        Ok(symbol)
    }
}

/// Provides temporary read-only access to the compressed data wrapped in a
/// [`Stack`].
///
/// Dereferences to `&[CompressedWord]`. See [`Coder::get_compressed`] for an example.
///
/// [`Stack`]: struct.Coder.html
/// [`Coder::get_compressed`]: struct.Coder.html#method.get_compressed
struct CoderGuard<'a, CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: WriteItems<CompressedWord> + ReadItems<CompressedWord>,
{
    inner: &'a mut Stack<CompressedWord, State, Buf>,
}

impl<'a, CompressedWord, State, Buf> CoderGuard<'a, CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: WriteItems<CompressedWord> + ReadItems<CompressedWord>,
{
    fn new(stack: &'a mut Stack<CompressedWord, State, Buf>) -> Self {
        // Append state. Will be undone in `<Self as Drop>::drop`.
        for chunk in bit_array_to_chunks_truncated(stack.state).rev() {
            stack.buf.push(chunk)
        }

        Self { inner: stack }
    }
}

impl<'a, CompressedWord, State, Buf> Drop for CoderGuard<'a, CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: WriteItems<CompressedWord> + ReadItems<CompressedWord>,
{
    fn drop(&mut self) {
        // Revert what we did in `Self::new`.
        for _ in bit_array_to_chunks_truncated::<_, CompressedWord>(self.inner.state) {
            self.inner.buf.pop();
        }
    }
}

impl<'a, CompressedWord, State, Buf> Deref for CoderGuard<'a, CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: WriteItems<CompressedWord> + ReadItems<CompressedWord>,
{
    type Target = Buf;

    fn deref(&self) -> &Self::Target {
        &self.inner.buf
    }
}

impl<CompressedWord, State, Buf> Debug for CoderGuard<'_, CompressedWord, State, Buf>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
    Buf: WriteItems<CompressedWord> + ReadItems<CompressedWord> + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Categorical, DiscreteDistribution, LeakyQuantizer};

    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };
    use statrs::distribution::{InverseCDF, Normal};

    #[test]
    fn compress_none() {
        let coder1 = DefaultStack::new();
        assert!(coder1.is_empty());
        let compressed = coder1.into_compressed();
        assert!(compressed.is_empty());

        let coder2 = DefaultStack::from_compressed(compressed).unwrap();
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

        let mut encoder = DefaultStack::new();
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-127..=127);
        let distribution = quantizer.quantize(Normal::new(3.2, 5.1).unwrap());

        // We don't reuse the same encoder for decoding because we want to test
        // if exporting and re-importing of compressed data works.
        encoder
            .encode_iid_symbols(symbols.clone(), &distribution)
            .unwrap();
        let compressed = encoder.into_compressed();
        assert_eq!(compressed.len(), expected_size);

        let mut decoder = DefaultStack::from_compressed(compressed).unwrap();
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
                std::cmp::min(127, dist.inverse_cdf(quantile).round() as i32),
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

        let mut stack = Stack::<CompressedWord, State>::new();

        stack
            .encode_iid_symbols_reverse(&symbols_categorical, &categorical)
            .unwrap();
        dbg!(
            stack.num_valid_bits(),
            AMT as f64 * categorical.entropy::<f64>()
        );

        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-127..=127);
        stack
            .encode_symbols_reverse(symbols_gaussian.iter().zip(&means).zip(&stds).map(
                |((&symbol, &mean), &std)| {
                    (symbol, quantizer.quantize(Normal::new(mean, std).unwrap()))
                },
            ))
            .unwrap();
        dbg!(stack.num_valid_bits());

        // Test if import/export of compressed data works.
        let compressed = stack.into_compressed();
        let mut stack = Stack::from_compressed(compressed).unwrap();

        let reconstructed_gaussian = stack
            .decode_symbols(
                means
                    .iter()
                    .zip(&stds)
                    .map(|(&mean, &std)| quantizer.quantize(Normal::new(mean, std).unwrap())),
            )
            .collect::<Result<Vec<_>, std::convert::Infallible>>()
            .unwrap();
        let reconstructed_categorical = stack
            .decode_iid_symbols(AMT, &categorical)
            .collect::<Result<Vec<_>, std::convert::Infallible>>()
            .unwrap();

        assert!(stack.is_empty());

        assert_eq!(symbols_gaussian, reconstructed_gaussian);
        assert_eq!(symbols_categorical, reconstructed_categorical);
    }
}
