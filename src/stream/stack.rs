//! Encoding and infallible decoding on a stack ("last in first out")
//!
//!

use alloc::vec::Vec;
use core::{
    borrow::Borrow, convert::Infallible, fmt::Debug, iter::Fuse, marker::PhantomData, ops::Deref,
};
use num::cast::AsPrimitive;

use super::{
    models::{DecoderModel, EncoderModel},
    AsDecoder, Code, Decode, Encode, IntoDecoder, TryCodingError,
};
use crate::{
    backends::{
        self, AsReadWords, AsSeekReadWords, BoundedReadWords, Cursor, FallibleIteratorReadWords,
        IntoReadWords, IntoSeekReadWords, ReadWords, Reverse, WriteWords,
    },
    bit_array_to_chunks_truncated, BitArray, CoderError, DefaultEncoderError,
    DefaultEncoderFrontendError, NonZeroBitArray, Pos, PosSeek, Seek, Stack, UnwrapInfallible,
};

/// Entropy coder for both encoding and decoding on a stack
///
/// This is is a very general entropy coder that provides both encoding and
/// decoding, and that is generic over a type `Word` that defines the smallest unit of
/// compressed data, and a constant `PRECISION` that defines the fixed point
/// precision used in entropy models. See [below](
/// #generic-parameters-compressed-word-type-w-and-precision) for details on these
/// parameters. If you're unsure about the choice of `Word` and `PRECISION` then use
/// the type alias [`DefaultAnsCoder`], which makes sane choices for typical
/// applications.
///
/// The `AnsCoder` uses an entropy coding algorithm called [range Asymmetric
/// Numeral Systems (rANS)]. This means that it operates as a stack, i.e., a "last
/// in first out" data structure: encoding "pushes symbols on" the stack and
/// decoding "pops symbols off" the stack in reverse order. In contrast to
/// [`SeekableDecoder`], decoding with a `AnsCoder` *consumes* the compressed data for
/// the decoded symbols. This means that encoding and decoding can be interleaved
/// arbitrarily, thus growing and shrinking the stack of compressed data as you go.
///
/// # Example
///
/// The following shows a basic usage example. For more examples, see
/// [`encode_symbols`] or [`encode_iid_symbols`].
///
/// ```
/// use constriction::stream::{models::LeakyQuantizer, stack::DefaultAnsCoder, Decode};
///
/// // `DefaultAnsCoder` is a type alias to `AnsCoder` with sane generic parameters.
/// let mut ans = DefaultAnsCoder::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
/// let entropy_model = quantizer.quantize(probability::distribution::Gaussian::new(0.0, 10.0));
///
/// let symbols = vec![-10, 4, 0, 3];
/// ans.encode_iid_symbols_reverse(&symbols, &entropy_model).unwrap();
/// println!("Encoded into {} bits: {:?}", ans.num_bits(), &*ans.get_compressed().unwrap());
///
/// // The call to `encode_iid_symbols` above encoded the symbols in reverse order (see
/// // documentation). So popping them off now will yield the same symbols in original order.
/// let reconstructed = ans
///     .decode_iid_symbols(4, &entropy_model)
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
/// assert_eq!(reconstructed, symbols);
/// ```
///
/// # Generic Parameters: Compressed Word Type `Word` and `PRECISION`
///
/// The `AnsCoder` is generic over a type `Word`, which is a [`BitArray`],
/// and over a constant `PRECISION` of type `usize`. **If you're unsure how to set
/// these parameters, consider using the type alias [`DefaultAnsCoder`], which uses
/// sane default values.**
///
/// ## Meaning of `Word` and `PRECISION`
///
/// If you need finer control over the entropy coder, and [`DefaultAnsCoder`] does not
/// fit your needs, then here are the details about the parameters `Word`
/// and `PRECISION`:
///
/// - `Word` is the smallest "chunk" of compressed data. It is usually a
///   primitive unsigned integer type, such as `u32` or `u16`. The type
///   `Word` is also used to represent probabilities in fixed-point
///   arithmetic in any [`EntropyModel`] that can be employed as an entropy
///   model for this `AnsCoder` (however, when representing probabilities, only use the
///   lowest `PRECISION` bits of a `Word` are ever used).
///
///   The `AnsCoder` operates on an internal state whose size is twice as large as
///   `Word`. When encoding data, the `AnsCoder` keeps filling up this
///   internal state with compressed data until it is about to overflow. Just before
///   the internal state would overflow, the stack chops off half of it and pushes
///   one "compressed word" of type `Word` onto a dynamically growable and
///   shrinkable buffer of `Word`s. Once all data is encoded, the encoder
///   chops the final internal state into two `Word`s, pushes them onto
///   the buffer, and returns the buffer as the compressed data (see method
///   [`into_compressed`]).
///
/// - `PRECISION` defines the number of bits that the entropy models use for
///   fixed-point representation of probabilities. `PRECISION` must be positive and
///   no larger than the bitlength of `Word` (e.g., if `Word` is
///  `u32` then we must have `1 <= PRECISION <= 32`).
///
///   Since the smallest representable nonzero probability is `(1/2)^PRECISION`, the
///   largest possible (finite) [information content] of a single symbol is
///   `PRECISION` bits. Thus, pushing a single symbol onto the `AnsCoder` increases the
///   "filling level" of the `AnsCoder`'s internal state by at most `PRECISION` bits.
///   Since `PRECISION` is at most the bitlength of `Word`, the procedure
///   of transferring one `Word` from the internal state to the buffer
///   described in the list item above is guaranteed to free up enough internal
///   state to encode at least one additional symbol.
///
/// ## Guidance for Choosing `Word` and `PRECISION`
///
/// If you choose `Word` and `PRECISION` manually (rather than using a
/// [`DefaultAnsCoder`]), then your choice should take into account the following
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
/// - Since `Word` must be at least `PRECISION` bits in size, a high
///   `PRECISION` means that you will have to use a larger `Word` type.
///   This has several consequences:
///   - it affects the size of the internal state of the stack; this is relevant if
///     you want to store many different internal states, e.g., as a jump table for
///     a [`SeekableDecoder`].
///   - it leads to a small *constant* overhead in bitrate: since the `AnsCoder`
///     operates on an internal  state of two `Word`s, it has a constant
///     bitrate  overhead between zero and two `Word`s depending on the
///     filling level of the internal state. This constant  overhead is usually
///     negligible unless you want to compress a very small amount of data.
///   - the choice of `Word` may have some effect on runtime performance
///     since operations on larger types may be more expensive (remember that the x
///     since operates on a state of twice the size as `Word`, i.e., if
///     `Word = u64` then the stack will operate on a `u128`, which may be
///     slow on some hardware). On the other hand, this overhead should not be used
///     as an argument for setting `Word` to a very small type like `u8`
///     or `u16` since common computing architectures are usually not really faster
///     on very small registers, and a very small `Word` type will lead to
///     more frequent transfers between the internal state and the growable buffer,
///     which requires potentially expensive branching and memory lookups.
/// - Finally, the "slack" between `PRECISION` and the size of `Word` has
///   an influence on the bitrate. It is usually *not* a good idea to set
///   `PRECISION` to the highest value allowed for a given `Word` (e.g.,
///   setting `Word = u32` and `PRECISION = 32` is *not* recommended).
///   This is because, when encoding a symbol, the `AnsCoder` expects there to be at
///   least `PRECISION` bits of entropy in its internal state (conceptually,
///   encoding a symbol `s` consists of consuming `PRECISION` bits of entropy
///   followed by pushing `PRECISION + information_content(s)` bits of entropy onto
///   the internal state).  If `PRECISION` is set to the full size of
///   `Word` then there will be relatively frequent situations where the
///   internal state contains less than `PRECISION` bits of entropy, leading to an
///   overhead (this situation will typically arise after the `AnsCoder` transferred a
///   `Word` from the internal state to the growable buffer).
///
/// The type alias [`DefaultAnsCoder`] was chose with the above considerations in mind.
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
/// `AnsCoder` using some entropy models, and then pop off (decode) symbols using
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
/// [`models`]: models/index.html
/// [entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
/// [information content]: https://en.wikipedia.org/wiki/Information_content
/// [`encode_symbols`]: #method.encode_symbols
/// [`is_empty`]: #method.is_empty`
/// [`into_compressed`]: #method.into_compressed
#[derive(Clone)]
pub struct AnsCoder<Word, State, Backend = Vec<Word>>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    bulk: Backend,

    /// Invariant: `state >= State::one() << (State::BITS - Word::BITS)`
    /// unless `bulk.is_empty()` or this is the `waste` part of an `stable::Coder`.
    state: State,

    /// We keep track of the `Word` type so that we can statically enforce
    /// the invariant `Word: Into<State>`.
    phantom: PhantomData<Word>,
}

/// Type alias for an [`AnsCoder`] with sane parameters for typical use cases.
///
/// This type alias sets the generic type arguments `Word` and `State` to
/// sane values for many typical use cases.
pub type DefaultAnsCoder<Backend = Vec<u32>> = AnsCoder<u32, u64, Backend>;

/// Type alias for an [`AnsCoder`] for use with [lookup models]
///
/// This encoder has a smaller word size and internal state than [`AnsCoder`]. It is
/// optimized for use with lookup entropy models, in particular with a
/// [`DefaultEncoderArrayLookupTable`] or [`DefaultEncoderHashLookupTable`] for encoding, or
/// with a [`DefaultDecoderIndexLookupTable`] or [`DefaultDecoderGenericLookupTable`] for
/// decoding.
///
/// # Examples
///
/// See [`DefaultEncoderArrayLookupTable`], [`DefaultEncoderHashLookupTable`],
/// [`DefaultDecoderIndexLookupTable`], and [`DefaultDecoderGenericLookupTable`].
///
/// [lookup models]: super::models::lookup
/// [`DefaultEncoderArrayLookupTable`]: super::models::lookup::DefaultEncoderArrayLookupTable
/// [`DefaultEncoderHashLookupTable`]: super::models::lookup::DefaultEncoderHashLookupTable
/// [`DefaultDecoderIndexLookupTable`]: super::models::lookup::DefaultDecoderIndexLookupTable
/// [`DefaultDecoderGenericLookupTable`]: super::models::lookup::DefaultDecoderGenericLookupTable
pub type SmallAnsCoder<Backend = Vec<u16>> = AnsCoder<u16, u32, Backend>;

impl<Word, State, Backend> Debug for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    for<'a> &'a Backend: IntoIterator<Item = &'a Word>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter_compressed()).finish()
    }
}

impl<Word, State, Backend, const PRECISION: usize> IntoDecoder<PRECISION>
    for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + IntoReadWords<Word, Stack>,
{
    type IntoDecoder = AnsCoder<Word, State, Backend::IntoReadWords>;

    fn into_decoder(self) -> Self::IntoDecoder {
        AnsCoder {
            bulk: self.bulk.into_read_words(),
            state: self.state,
            phantom: PhantomData,
        }
    }
}

impl<'a, Word, State, Backend> From<&'a AnsCoder<Word, State, Backend>>
    for AnsCoder<Word, State, <Backend as AsReadWords<'a, Word, Stack>>::AsReadWords>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: AsReadWords<'a, Word, Stack>,
{
    fn from(ans: &'a AnsCoder<Word, State, Backend>) -> Self {
        AnsCoder {
            bulk: ans.bulk().as_read_words(),
            state: ans.state(),
            phantom: PhantomData,
        }
    }
}

impl<'a, Word, State, Backend, const PRECISION: usize> AsDecoder<'a, PRECISION>
    for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + AsReadWords<'a, Word, Stack>,
{
    type AsDecoder = AnsCoder<Word, State, Backend::AsReadWords>;

    fn as_decoder(&'a self) -> Self::AsDecoder {
        self.into()
    }
}

impl<Word, State> AnsCoder<Word, State, Vec<Word>>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    /// Creates an empty ANS entropy coder.
    ///
    /// This is usually the starting point if you want to *compress* data.
    ///
    /// # Example
    ///
    /// ```
    /// let mut ans = constriction::stream::stack::DefaultAnsCoder::new();
    ///
    /// // ... push some symbols onto the ANS coder's stack ...
    ///
    /// // Finally, get the compressed data.
    /// let compressed = ans.into_compressed();
    /// ```
    ///
    /// # Generality
    ///
    /// To avoid type parameters in common use cases, `new` is only implemented for
    /// `AnsCoder`s with a `Vec` backend. To create an empty coder with a different backend,
    /// call [`Default::default`] instead.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<Word, State, Backend> Default for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: Default,
{
    fn default() -> Self {
        assert!(State::BITS >= 2 * Word::BITS);

        Self {
            state: State::zero(),
            bulk: Default::default(),
            phantom: PhantomData,
        }
    }
}

impl<Word, State, Backend> AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    pub fn with_state_and_empty_bulk(state: State) -> Self
    where
        Backend: Default,
    {
        Self {
            state,
            bulk: Default::default(),
            phantom: PhantomData,
        }
    }

    pub unsafe fn from_raw_parts(bulk: Backend, state: State) -> Self {
        Self {
            bulk,
            state,
            phantom: PhantomData,
        }
    }

    /// Creates an ANS stack with some initial compressed data.
    ///
    /// This is usually the starting point if you want to *decompress* data previously
    /// obtained from [`into_compressed`].  However, it can also be used to append more
    /// symbols to an existing compressed buffer of data.
    ///
    /// Returns `Err(compressed)` if `compressed` is not empty and its last entry is
    /// zero, since an `AnsCoder` cannot represent trailing zero words. This error cannot
    /// occur if `compressed` was obtained from [`into_compressed`], which never returns
    /// data with a trailing zero word. If you want to construct a `AnsCoder` from an
    /// unknown source of binary data (e.g., to decode some side information into latent
    /// variables) then call [`from_binary`] instead.
    ///
    /// [`into_compressed`]: #method.into_compressed
    /// [`from_binary`]: #method.from_binary
    pub fn from_compressed(mut compressed: Backend) -> Result<Self, Backend>
    where
        Backend: ReadWords<Word, Stack>,
    {
        assert!(State::BITS >= 2 * Word::BITS);

        let state = match Self::read_initial_state(|| compressed.read()) {
            Ok(state) => state,
            Err(_) => return Err(compressed),
        };

        Ok(Self {
            bulk: compressed,
            state,
            phantom: PhantomData,
        })
    }

    fn read_initial_state<Error>(
        mut read_word: impl FnMut() -> Result<Option<Word>, Error>,
    ) -> Result<State, ()>
    where
        Backend: ReadWords<Word, Stack>,
    {
        if let Some(first_word) = read_word().map_err(|_| ())? {
            if first_word == Word::zero() {
                return Err(());
            }

            let mut state = first_word.into();
            while let Some(word) = read_word().map_err(|_| ())? {
                state = state << Word::BITS | word.into();
                if state >= State::one() << (State::BITS - Word::BITS) {
                    break;
                }
            }
            Ok(state)
        } else {
            Ok(State::zero())
        }
    }

    /// Like [`from_compressed`] but works on any binary data.
    ///
    /// This method is meant for rather advanced use cases. For most common use cases,
    /// you probably want to call [`from_compressed`] instead.
    ///
    /// Different to `from_compressed`, this method also works if `data` ends in a zero
    /// word. Calling this method is equivalent to (but likely more efficient than)
    /// appending a `1` word to `data` and then calling `from_compressed`. Note that
    /// therefore, this method always constructs a non-empty `AnsCoder` (even if `data` is
    /// empty):
    ///
    /// ```
    /// use constriction::stream::stack::DefaultAnsCoder;
    ///
    /// let stack1 = DefaultAnsCoder::from_binary(Vec::new()).unwrap();
    /// assert!(!stack1.is_empty()); // <-- stack1 is *not* empty.
    ///
    /// let stack2 = DefaultAnsCoder::from_compressed(Vec::new()).unwrap();
    /// assert!(stack2.is_empty()); // <-- stack2 is empty.
    /// ```
    /// [`from_compressed`]: #method.from_compressed
    pub fn from_binary(mut data: Backend) -> Result<Self, Backend::ReadError>
    where
        Backend: ReadWords<Word, Stack>,
    {
        let mut state = State::one();

        while state < State::one() << (State::BITS - Word::BITS) {
            if let Some(word) = data.read()? {
                state = state << Word::BITS | word.into();
            } else {
                break;
            }
        }

        Ok(Self {
            bulk: data,
            state,
            phantom: PhantomData,
        })
    }

    #[inline(always)]
    pub fn bulk(&self) -> &Backend {
        &self.bulk
    }

    pub fn into_raw_parts(self) -> (Backend, State) {
        (self.bulk, self.state)
    }

    /// Check if no data for decoding is left.
    ///
    /// Same as [`Code::maybe_empty`], just with a more suitable name considering the
    /// fact that this particular implementation of `Code` can decide with certainty
    /// whether or not it is empty.
    ///
    /// Note that you can still pop symbols off an empty stack, but this is only
    /// useful in rare edge cases, see documentation of
    /// [`decode_symbol`](#method.decode_symbol).
    pub fn is_empty(&self) -> bool {
        // We don't need to check if `bulk` is empty (which would require an additional type
        // type bound `Backend: ReadLookaheadItems<Word>` because we keep up the
        // invariant that `state >= State::one() << (State::BITS - Word::BITS))`
        // when `bulk` is not empty.
        self.state == State::zero()
    }

    /// Assembles the current compressed data into a single slice.
    ///
    /// Returns the concatenation of [`bulk`] and [`state`]. The concatenation truncates
    /// any trailing zero words, which is compatible with the constructor
    /// [`from_compressed`].
    ///
    /// This method requires a `&mut self` receiver to temporarily append `state` to
    /// [`bulk`] (this mutationwill be reversed to recreate the original `bulk` as soon as
    /// the caller drops the returned value). If you don't have mutable access to the
    /// `AnsCoder`, consider calling [`iter_compressed`] instead, or get the `bulk` and
    /// `state` separately by calling [`bulk`] and [`state`], respectively.
    ///
    /// The return type dereferences to `&[Word]`, thus providing read-only
    /// access to the compressed data. If you need ownership of the compressed data,
    /// consider calling [`into_compressed`] instead.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{models::LeakyCategorical, stack::DefaultAnsCoder, Decode};
    ///
    /// let mut ans = DefaultAnsCoder::new();
    ///
    /// // Push some data on the ans.
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let model = LeakyCategorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
    ///     .unwrap();
    /// ans.encode_iid_symbols_reverse(&symbols, &model).unwrap();
    ///
    /// // Inspect the compressed data.
    /// dbg!(ans.get_compressed());
    ///
    /// // We can still use the ANS coder afterwards.
    /// let reconstructed = ans
    ///     .decode_iid_symbols(4, &model)
    ///     .collect::<Result<Vec<_>, _>>()
    ///     .unwrap();
    /// assert_eq!(reconstructed, symbols);
    /// ```
    ///
    /// [`bulk`]: #method.bulk
    /// [`state`]: #method.state
    /// [`from_compressed`]: #method.from_compressed
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn get_compressed<'a>(
        &'a mut self,
    ) -> Result<impl Deref<Target = Backend> + Debug + Drop + 'a, Backend::WriteError>
    where
        Backend: ReadWords<Word, Stack> + WriteWords<Word> + Debug,
    {
        CoderGuard::new(self)
    }

    /// Iterates over the compressed data currently on the ans.
    ///
    /// In contrast to [`get_compressed`] or [`into_compressed`], this method does
    /// not require mutable access or even ownership of the `AnsCoder`.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{models::{LeakyCategorical, LeakyQuantizer}, stack::DefaultAnsCoder, Encode};
    ///
    /// // Create a stack and encode some stuff.
    /// let mut ans = DefaultAnsCoder::new();
    /// let symbols = vec![8, -12, 0, 7];
    /// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let model =
    ///     quantizer.quantize(probability::distribution::Gaussian::new(0.0, 10.0));
    /// ans.encode_iid_symbols(&symbols, &model).unwrap();
    ///
    /// // Iterate over compressed data, collect it into to a vector, and compare to more direct method.
    /// let compressed_iter = ans.iter_compressed();
    /// let compressed_collected = compressed_iter.collect::<Vec<_>>();
    /// assert!(!compressed_collected.is_empty());
    /// assert_eq!(compressed_collected, *ans.get_compressed().unwrap());
    /// ```
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn iter_compressed<'a>(&'a self) -> impl Iterator<Item = Word> + '_
    where
        &'a Backend: IntoIterator<Item = &'a Word>,
    {
        let bulk_iter = self.bulk.into_iter().cloned();
        let state_iter = bit_array_to_chunks_truncated(self.state).rev();
        bulk_iter.chain(state_iter)
    }

    /// Returns the number of compressed words on the ANS coder's stack.
    ///
    /// This includes a constant overhead of between one and two words unless the
    /// stack is completely empty.
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
    pub fn num_words(&self) -> usize
    where
        Backend: BoundedReadWords<Word, Stack>,
    {
        self.bulk.remaining() + bit_array_to_chunks_truncated::<_, Word>(self.state).len()
    }

    pub fn num_bits(&self) -> usize
    where
        Backend: BoundedReadWords<Word, Stack>,
    {
        Word::BITS * self.num_words()
    }

    pub fn num_valid_bits(&self) -> usize
    where
        Backend: BoundedReadWords<Word, Stack>,
    {
        Word::BITS * self.bulk.remaining()
            + core::cmp::max(State::BITS - self.state.leading_zeros() as usize, 1)
            - 1
    }

    pub fn into_decoder(self) -> AnsCoder<Word, State, Backend::IntoReadWords>
    where
        Backend: IntoReadWords<Word, Stack>,
    {
        AnsCoder {
            bulk: self.bulk.into_read_words(),
            state: self.state,
            phantom: PhantomData,
        }
    }

    /// Consumes the `AnsCoder` and returns a decoder that implements [`Seek`].
    ///
    /// This method is similar to [`seekable_decoder`] except that it takes ownership of
    /// the original `AnsCoder`, so the returned seekable decoder can typically be returned
    /// from the calling function or put on the heap.
    ///
    /// [`seekable_decoder`]: Self::seekable_decoder
    pub fn into_seekable_decoder(self) -> AnsCoder<Word, State, Backend::IntoSeekReadWords>
    where
        Backend: IntoSeekReadWords<Word, Stack>,
    {
        AnsCoder {
            bulk: self.bulk.into_seek_read_words(),
            state: self.state,
            phantom: PhantomData,
        }
    }

    pub fn as_decoder<'a>(&'a self) -> AnsCoder<Word, State, Backend::AsReadWords>
    where
        Backend: AsReadWords<'a, Word, Stack>,
    {
        AnsCoder {
            bulk: self.bulk.as_read_words(),
            state: self.state,
            phantom: PhantomData,
        }
    }

    /// Returns a decoder that implements [`Seek`].
    ///
    /// The returned decoder shares access to the compressed data with the original
    /// `AnsCoder` (i.e., `self`). This means that:
    /// - you can call this method several times to create several seekable decoders
    ///   with independent views into the same compressed data;
    /// - once the lifetime of all handed out seekable decoders ends, the original
    ///   `AnsCoder` can be used again; and
    /// - the constructed seekable decoder cannot outlive the original `AnsCoder`; for
    ///   example, if the original `AnsCoder` lives on the calling function's call stack
    ///   frame then you cannot return the constructed seekable decoder from the calling
    ///   function. If this is a problem then call [`into_seekable_decoder`] instead.
    ///
    /// # Limitations
    ///
    /// TODO: this text is outdated.
    ///
    /// This method is only implemented for `AnsCoder`s whose backing store of compressed
    /// data (`Backend`) implements `AsRef<[Word]>`. This includes the default
    /// backing data store `Backend = Vec<Word>`.
    ///
    /// [`into_seekable_decoder`]: Self::into_seekable_decoder
    pub fn as_seekable_decoder<'a>(&'a self) -> AnsCoder<Word, State, Backend::AsSeekReadWords>
    where
        Backend: AsSeekReadWords<'a, Word, Stack>,
    {
        AnsCoder {
            bulk: self.bulk.as_seek_read_words(),
            state: self.state,
            phantom: PhantomData,
        }
    }
}

impl<Word, State> AnsCoder<Word, State>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    /// Discards all compressed data and resets the coder to the same state as
    /// [`Coder::new`](#method.new).
    pub fn clear(&mut self) {
        self.bulk.clear();
        self.state = State::zero();
    }
}

impl<'bulk, Word, State> AnsCoder<Word, State, Cursor<Word, &'bulk [Word]>>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    // TODO: proper error type (also for `from_compressed`)
    pub fn from_compressed_slice(compressed: &'bulk [Word]) -> Result<Self, ()> {
        Self::from_compressed(backends::Cursor::new_at_write_end(compressed)).map_err(|_| ())
    }

    pub fn from_binary_slice(data: &'bulk [Word]) -> Self {
        Self::from_binary(backends::Cursor::new_at_write_end(data)).unwrap_infallible()
    }
}

impl<Word, State, Buf> AnsCoder<Word, State, Reverse<Cursor<Word, Buf>>>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Buf: AsRef<[Word]>,
{
    pub fn from_reversed_compressed(compressed: Buf) -> Result<Self, Buf> {
        Self::from_compressed(Reverse(Cursor::new_at_write_beginning(compressed)))
            .map_err(|Reverse(cursor)| cursor.into_buf_and_pos().0)
    }

    pub fn from_reversed_binary(data: Buf) -> Self {
        Self::from_binary(Reverse(Cursor::new_at_write_beginning(data))).unwrap_infallible()
    }
}

impl<Word, State, Iter, ReadError> AnsCoder<Word, State, FallibleIteratorReadWords<Iter>>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Iter: Iterator<Item = Result<Word, ReadError>>,
    FallibleIteratorReadWords<Iter>: ReadWords<Word, Stack, ReadError = ReadError>,
{
    pub fn from_reversed_compressed_iter(compressed: Iter) -> Result<Self, Fuse<Iter>> {
        Self::from_compressed(FallibleIteratorReadWords::new(compressed))
            .map_err(|iterator_backend| iterator_backend.into_iter())
    }

    pub fn from_reversed_binary_iter(data: Iter) -> Result<Self, ReadError> {
        Self::from_binary(FallibleIteratorReadWords::new(data))
    }
}

impl<Word, State, Backend> AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word>,
{
    pub fn encode_symbols_reverse<S, D, I, const PRECISION: usize>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<(), DefaultEncoderError<Backend::WriteError>>
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

    pub fn try_encode_symbols_reverse<S, D, E, I, const PRECISION: usize>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<(), TryCodingError<DefaultEncoderError<Backend::WriteError>, E>>
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

    pub fn encode_iid_symbols_reverse<S, D, I, const PRECISION: usize>(
        &mut self,
        symbols: I,
        model: &D,
    ) -> Result<(), DefaultEncoderError<Backend::WriteError>>
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

    /// Consumes the ANS coder and returns the compressed data.
    ///
    /// The returned data can be used to recreate an ANS coder with the same state
    /// (e.g., for decoding) by passing it to
    /// [`from_compressed`](#method.from_compressed).
    ///
    /// If you don't want to consume the ANS coder, consider calling
    /// [`get_compressed`](#method.get_compressed),
    /// [`iter_compressed`](#method.iter_compressed) instead.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{models::LeakyCategorical, stack::DefaultAnsCoder, Decode};
    ///
    /// let mut ans = DefaultAnsCoder::new();
    ///
    /// // Push some data onto the ANS coder's stack:
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let model = LeakyCategorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
    ///     .unwrap();
    /// ans.encode_iid_symbols_reverse(&symbols, &model).unwrap();
    ///
    /// // Get the compressed data, consuming the ANS coder:
    /// let compressed = ans.into_compressed().unwrap();
    ///
    /// // ... write `compressed` to a file and then read it back later ...
    ///
    /// // Create a new ANS coder with the same state and use it for decompression:
    /// let mut ans = DefaultAnsCoder::from_compressed(compressed).expect("Corrupted compressed file.");
    /// let reconstructed = ans
    ///     .decode_iid_symbols(4, &model)
    ///     .collect::<Result<Vec<_>, _>>()
    ///     .unwrap();
    /// assert_eq!(reconstructed, symbols);
    /// assert!(ans.is_empty())
    /// ```
    pub fn into_compressed(mut self) -> Result<Backend, Backend::WriteError> {
        self.bulk
            .extend_from_iter(bit_array_to_chunks_truncated(self.state).rev())?;
        Ok(self.bulk)
    }

    /// Returns the binary data if it fits precisely into an integer number of
    /// `Word`s
    ///
    /// This method is meant for rather advanced use cases. For most common use cases,
    /// you probably want to call [`into_compressed`] instead.
    ///
    /// This method is the inverse of [`from_binary`]. It is equivalent to calling
    /// [`into_compressed`], verifying that the returned vector ends in a `1` word, and
    /// popping off that trailing `1` word.
    ///
    /// Returns `Err(())` if the compressed data (excluding an obligatory trailing
    /// `1` bit) does not fit into an integer number of `Word`s. This error
    /// case includes the case of an empty `AnsCoder` (since an empty `AnsCoder` lacks the
    /// obligatory trailing one-bit).
    ///
    /// # Example
    ///
    /// ```
    /// // Some binary data we want to represent on a `AnsCoder`.
    /// let data = vec![0x89ab_cdef, 0x0123_4567];
    ///
    /// // Constructing a `AnsCoder` with `from_binary` indicates that all bits of `data` are
    /// // considered part of the information-carrying payload.
    /// let stack1 = constriction::stream::stack::DefaultAnsCoder::from_binary(data.clone()).unwrap();
    /// assert_eq!(stack1.clone().into_binary().unwrap(), data); // <-- Retrieves the original `data`.
    ///
    /// // By contrast, if we construct a `AnsCoder` with `from_compressed`, we indicate that
    /// // - any leading `0` bits of the last entry of `data` are not considered part of
    /// //   the information-carrying payload; and
    /// // - the (obligatory) first `1` bit of the last entry of `data` defines the
    /// //   boundary between unused bits and information-carrying bits; it is therefore
    /// //   also not considered part of the payload.
    /// // Therefore, `stack2` below only contains `32 * 2 - 7 - 1 = 56` bits of payload,
    /// // which cannot be exported into an integer number of `u32` words:
    /// let stack2 = constriction::stream::stack::DefaultAnsCoder::from_compressed(data.clone()).unwrap();
    /// assert!(stack2.clone().into_binary().is_err()); // <-- Returns an error.
    ///
    /// // Use `into_compressed` to retrieve the data in this case:
    /// assert_eq!(stack2.into_compressed().unwrap(), data);
    ///
    /// // Calling `into_compressed` on `stack1` would append an extra `1` bit to indicate
    /// // the boundary between information-carrying bits and padding `0` bits:
    /// assert_eq!(stack1.into_compressed().unwrap(), vec![0x89ab_cdef, 0x0123_4567, 0x0000_0001]);
    /// ```
    ///
    /// [`from_binary`]: #method.from_binary
    /// [`into_compressed`]: #method.into_compressed
    pub fn into_binary(mut self) -> Result<Backend, Option<Backend::WriteError>> {
        let valid_bits = (State::BITS - 1).wrapping_sub(self.state.leading_zeros() as usize);

        if valid_bits % Word::BITS != 0 || valid_bits == usize::max_value() {
            Err(None)
        } else {
            let truncated_state = self.state ^ (State::one() << valid_bits);
            self.bulk
                .extend_from_iter(bit_array_to_chunks_truncated(truncated_state).rev())?;
            Ok(self.bulk)
        }
    }
}

impl<Word, State, Buf> AnsCoder<Word, State, Cursor<Word, Buf>>
where
    Word: BitArray,
    State: BitArray + AsPrimitive<Word> + From<Word>,
    Buf: AsRef<[Word]> + AsMut<[Word]>,
{
    pub fn into_reversed(self) -> AnsCoder<Word, State, Reverse<Cursor<Word, Buf>>> {
        let (bulk, state) = self.into_raw_parts();
        AnsCoder {
            bulk: bulk.into_reversed(),
            state,
            phantom: PhantomData,
        }
    }
}

impl<Word, State, Buf> AnsCoder<Word, State, Reverse<Cursor<Word, Buf>>>
where
    Word: BitArray,
    State: BitArray + AsPrimitive<Word> + From<Word>,
    Buf: AsRef<[Word]> + AsMut<[Word]>,
{
    pub fn into_reversed(self) -> AnsCoder<Word, State, Cursor<Word, Buf>> {
        let (bulk, state) = self.into_raw_parts();
        AnsCoder {
            bulk: bulk.into_reversed(),
            state,
            phantom: PhantomData,
        }
    }
}

impl<Word, State, Backend> Code for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    type Word = Word;
    type State = State;

    #[inline(always)]
    fn state(&self) -> Self::State {
        self.state
    }
}

impl<Word, State, Backend, const PRECISION: usize> Encode<PRECISION>
    for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word>,
{
    type FrontendError = DefaultEncoderFrontendError;
    type BackendError = Backend::WriteError;

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
    /// entropy model `model`. This error can usually be avoided by using a
    /// "leaky" distribution as the entropy model, i.e., a distribution that assigns a
    /// nonzero probability to all symbols within a finite domain. Leaky distributions
    /// can be constructed with, e.g., a
    /// [`LeakyQuantizer`](models/struct.LeakyQuantizer.html) or with
    /// [`LeakyCategorical::from_floating_point_probabilities`](
    /// models/struct.LeakyCategorical.html#method.from_floating_point_probabilities).
    ///
    /// TODO: move this and similar doc comments to the trait definition.
    ///
    /// [`Err(ImpossibleSymbol)`]: enum.EncodingError.html#variant.ImpossibleSymbol
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
        let (left_sided_cumulative, probability) = model
            .left_cumulative_and_probability(symbol)
            .ok_or(DefaultEncoderFrontendError::ImpossibleSymbol.into_coder_error())?;

        if (self.state >> (State::BITS - PRECISION)) >= probability.get().into().into() {
            self.bulk.write(self.state.as_())?;
            self.state = self.state >> Word::BITS;
            // At this point, the invariant on `self.state` (see its doc comment) is
            // temporarily violated, but it will be restored below.
        }

        let remainder = (self.state % probability.get().into().into()).as_().as_();
        let prefix = self.state / probability.get().into().into();
        let quantile = left_sided_cumulative + remainder;
        self.state = prefix << PRECISION | quantile.into().into();

        Ok(())
    }

    fn maybe_full(&self) -> bool {
        self.bulk.maybe_full()
    }
}

impl<Word, State, Backend, const PRECISION: usize> Decode<PRECISION>
    for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: ReadWords<Word, Stack>,
{
    /// ANS coding is surjective, and we (deliberately) allow decoding past EOF (in a
    /// deterministic way) for consistency. Therefore, decoding cannot fail.    
    type FrontendError = Infallible;

    type BackendError = Backend::ReadError;

    /// Decodes a single symbol and pops it off the compressed data.
    ///
    /// This is a low level method. You usually probably want to call a batch method
    /// like [`decode_symbols`](#method.decode_symbols) or
    /// [`decode_iid_symbols`](#method.decode_iid_symbols) instead.
    ///
    /// This method is called `decode_symbol` rather than `decode_symbol` to stress the
    /// fact that the `AnsCoder` is a stack: `decode_symbol` will return the *last* symbol
    /// that was previously encoded via [`encode_symbol`](#method.encode_symbol).
    ///
    /// Note that this method cannot fail. It will still produce symbols in a
    /// deterministic way even if the stack is empty, but such symbols will not
    /// recover any previously encoded data and will generally have low entropy.
    /// Still, being able to pop off an arbitrary number of symbols can sometimes be
    /// useful in edge cases of, e.g., the bits-back algorithm.
    #[inline(always)]
    fn decode_symbol<D>(
        &mut self,
        model: D,
    ) -> Result<D::Symbol, CoderError<Self::FrontendError, Self::BackendError>>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<D::Probability>,
    {
        let quantile = (self.state % (State::one() << PRECISION)).as_().as_();
        let (symbol, left_sided_cumulative, probability) = model.quantile_function(quantile);
        let remainder = quantile - left_sided_cumulative;
        self.state =
            (self.state >> PRECISION) * probability.get().into().into() + remainder.into().into();
        if self.state < State::one() << (State::BITS - Word::BITS) {
            // Invariant on `self.state` (see its doc comment) is violated. Restore it by
            // refilling with a compressed word from `self.bulk` if available.
            if let Some(word) = self.bulk.read()? {
                self.state = (self.state << Word::BITS) | word.into();
            }
        }

        Ok(symbol)
    }

    fn maybe_exhausted(&self) -> bool {
        self.is_empty()
    }
}

impl<Word, State, Backend> PosSeek for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: PosSeek,
    Self: Code,
{
    type Position = (Backend::Position, <Self as Code>::State);
}

impl<Word, State, Backend> Seek for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: Seek,
{
    fn seek(&mut self, (pos, state): Self::Position) -> Result<(), ()> {
        self.bulk.seek(pos)?;
        self.state = state;
        Ok(())
    }
}

impl<Word, State, Backend> Pos for AnsCoder<Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: Pos,
{
    fn pos(&self) -> Self::Position {
        (self.bulk.pos(), self.state())
    }
}

/// Provides temporary read-only access to the compressed data wrapped in a
/// [`AnsCoder`].
///
/// Dereferences to `&[Word]`. See [`Coder::get_compressed`] for an example.
///
/// [`AnsCoder`]: struct.Coder.html
/// [`Coder::get_compressed`]: struct.Coder.html#method.get_compressed
struct CoderGuard<'a, Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + ReadWords<Word, Stack>,
{
    inner: &'a mut AnsCoder<Word, State, Backend>,
}

impl<'a, Word, State, Backend> CoderGuard<'a, Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + ReadWords<Word, Stack>,
{
    fn new(ans: &'a mut AnsCoder<Word, State, Backend>) -> Result<Self, Backend::WriteError> {
        // Append state. Will be undone in `<Self as Drop>::drop`.
        for chunk in bit_array_to_chunks_truncated(ans.state).rev() {
            ans.bulk.write(chunk)?
        }

        Ok(Self { inner: ans })
    }
}

impl<'a, Word, State, Backend> Drop for CoderGuard<'a, Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + ReadWords<Word, Stack>,
{
    fn drop(&mut self) {
        // Revert what we did in `Self::new`.
        for _ in bit_array_to_chunks_truncated::<_, Word>(self.inner.state) {
            core::mem::drop(self.inner.bulk.read());
        }
    }
}

impl<'a, Word, State, Backend> Deref for CoderGuard<'a, Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + ReadWords<Word, Stack>,
{
    type Target = Backend;

    fn deref(&self) -> &Self::Target {
        &self.inner.bulk
    }
}

impl<Word, State, Backend> Debug for CoderGuard<'_, Word, State, Backend>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + ReadWords<Word, Stack> + Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::super::models::{ContiguousCategorical, IterableEntropyModel, LeakyQuantizer};
    use super::*;
    extern crate std;
    use std::dbg;

    use probability::distribution::{Gaussian, Inverse};
    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };

    #[test]
    fn compress_none() {
        let coder1 = DefaultAnsCoder::new();
        assert!(coder1.is_empty());
        let compressed = coder1.into_compressed().unwrap();
        assert!(compressed.is_empty());

        let coder2 = DefaultAnsCoder::from_compressed(compressed).unwrap();
        assert!(coder2.is_empty());
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
        I::IntoIter: Clone + DoubleEndedIterator,
    {
        let symbols = symbols.into_iter();

        let mut encoder = DefaultAnsCoder::new();
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-127..=127);
        let model = quantizer.quantize(Gaussian::new(3.2, 5.1));

        // We don't reuse the same encoder for decoding because we want to test
        // if exporting and re-importing of compressed data works.
        encoder.encode_iid_symbols(symbols.clone(), &model).unwrap();
        let compressed = encoder.into_compressed().unwrap();
        assert_eq!(compressed.len(), expected_size);

        let mut decoder = DefaultAnsCoder::from_compressed(compressed).unwrap();
        for symbol in symbols.rev() {
            assert_eq!(decoder.decode_symbol(&model).unwrap(), symbol);
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
    fn compress_many_u32_u64_8() {
        generic_compress_many::<u32, u64, u8, 8>();
    }

    #[test]
    fn compress_many_u16_u64_16() {
        generic_compress_many::<u16, u64, u16, 16>();
    }

    #[test]
    fn compress_many_u16_u64_12() {
        generic_compress_many::<u16, u64, u16, 12>();
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
    fn compress_many_u16_u32_12() {
        generic_compress_many::<u16, u32, u16, 12>();
    }

    #[test]
    fn compress_many_u16_u32_8() {
        generic_compress_many::<u16, u32, u8, 8>();
    }

    #[test]
    fn compress_many_u8_u32_8() {
        generic_compress_many::<u8, u32, u8, 8>();
    }

    #[test]
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

        let mut rng = Xoshiro256StarStar::seed_from_u64(
            (Word::BITS as u64).rotate_left(3 * 16)
                ^ (State::BITS as u64).rotate_left(2 * 16)
                ^ (Probability::BITS as u64).rotate_left(1 * 16)
                ^ PRECISION as u64,
        );

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
            ContiguousCategorical::<Probability, PRECISION>::from_floating_point_probabilities(
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

        let mut ans = AnsCoder::<Word, State>::new();

        ans.encode_iid_symbols_reverse(&symbols_categorical, &categorical)
            .unwrap();
        dbg!(
            ans.num_valid_bits(),
            AMT as f64 * categorical.entropy_base2::<f64>()
        );

        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-127..=127);
        ans.encode_symbols_reverse(symbols_gaussian.iter().zip(&means).zip(&stds).map(
            |((&symbol, &mean), &core)| (symbol, quantizer.quantize(Gaussian::new(mean, core))),
        ))
        .unwrap();
        dbg!(ans.num_valid_bits());

        // Test if import/export of compressed data works.
        let compressed = ans.into_compressed().unwrap();
        let mut ans = AnsCoder::from_compressed(compressed).unwrap();

        let reconstructed_gaussian = ans
            .decode_symbols(
                means
                    .iter()
                    .zip(&stds)
                    .map(|(&mean, &core)| quantizer.quantize(Gaussian::new(mean, core))),
            )
            .collect::<Result<Vec<_>, CoderError<Infallible, Infallible>>>()
            .unwrap();
        let reconstructed_categorical = ans
            .decode_iid_symbols(AMT, &categorical)
            .collect::<Result<Vec<_>, CoderError<Infallible, Infallible>>>()
            .unwrap();

        assert!(ans.is_empty());

        assert_eq!(symbols_gaussian, reconstructed_gaussian);
        assert_eq!(symbols_categorical, reconstructed_categorical);
    }

    #[test]
    fn seek() {
        const NUM_CHUNKS: usize = 100;
        const SYMBOLS_PER_CHUNK: usize = 100;

        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
        let model = quantizer.quantize(Gaussian::new(0.0, 10.0));

        let mut encoder = DefaultAnsCoder::new();

        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        let mut symbols = Vec::with_capacity(NUM_CHUNKS);
        let mut jump_table = Vec::with_capacity(NUM_CHUNKS);
        let (initial_pos, initial_state) = encoder.pos();

        for _ in 0..NUM_CHUNKS {
            let chunk = (0..SYMBOLS_PER_CHUNK)
                .map(|_| model.quantile_function(rng.next_u32() % (1 << 24)).0)
                .collect::<Vec<_>>();
            encoder.encode_iid_symbols_reverse(&chunk, &model).unwrap();
            symbols.push(chunk);
            jump_table.push(encoder.pos());
        }

        // Test decoding from back to front.
        {
            let mut seekable_decoder = encoder.as_seekable_decoder();

            // Verify that decoding leads to the same positions and states.
            for (chunk, &(pos, state)) in symbols.iter().zip(&jump_table).rev() {
                assert_eq!(seekable_decoder.pos(), (pos, state));
                let decoded = seekable_decoder
                    .decode_iid_symbols(SYMBOLS_PER_CHUNK, &model)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(&decoded, chunk)
            }
            assert_eq!(seekable_decoder.pos(), (initial_pos, initial_state));
            assert!(seekable_decoder.is_empty());

            // Seek to some random offsets in the jump table and decode one chunk
            for _ in 0..100 {
                let chunk_index = rng.next_u32() as usize % NUM_CHUNKS;
                let (pos, state) = jump_table[chunk_index];
                seekable_decoder.seek((pos, state)).unwrap();
                let decoded = seekable_decoder
                    .decode_iid_symbols(SYMBOLS_PER_CHUNK, &model)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(&decoded, &symbols[chunk_index])
            }
        }

        // Reverse compressed data, map positions in jump table to reversed positions,
        // and test decoding from front to back.
        let mut compressed = encoder.into_compressed().unwrap();
        compressed.reverse();
        for (pos, _state) in jump_table.iter_mut() {
            *pos = compressed.len() - *pos;
        }
        let initial_pos = compressed.len() - initial_pos;

        {
            let mut seekable_decoder = AnsCoder::from_reversed_compressed(compressed).unwrap();

            // Verify that decoding leads to the expected positions and states.
            for (chunk, &(pos, state)) in symbols.iter().zip(&jump_table).rev() {
                assert_eq!(seekable_decoder.pos(), (pos, state));
                let decoded = seekable_decoder
                    .decode_iid_symbols(SYMBOLS_PER_CHUNK, &model)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(&decoded, chunk)
            }
            assert_eq!(seekable_decoder.pos(), (initial_pos, initial_state));
            assert!(seekable_decoder.is_empty());

            // Seek to some random offsets in the jump table and decode one chunk each time.
            for _ in 0..100 {
                let chunk_index = rng.next_u32() as usize % NUM_CHUNKS;
                let (pos, state) = jump_table[chunk_index];
                seekable_decoder.seek((pos, state)).unwrap();
                let decoded = seekable_decoder
                    .decode_iid_symbols(SYMBOLS_PER_CHUNK, &model)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(&decoded, &symbols[chunk_index])
            }
        }
    }
}
