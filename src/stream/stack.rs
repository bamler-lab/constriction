//! Fast and Near-optimal compression on a stack ("last in first out")
//!
//! This module provides the [`AnsCoder`], a highly efficient entropy coder with
//! near-optimal compression effectiveness that operates as a *stack* data structure. It
//! implements the Asymmetric Numeral Systems (ANS) compression algorithm \[1].
//!
//! # Comparison to sister module `queue`
//!
//! ANS Coding operates as a stack, which means that encoding and decoding operate in
//! reverse direction with respect to each other. The provided implementation of ANS Coding
//! uses a single data structure, the [`AnsCoder`], for both encoding and decoding. It
//! allows you to interleave encoding and decoding operations arbitrarily, which is in
//! contrast to the situation in the sister module [`queue`] and important for advanced
//! compression techniques such as bits-back coding in hierarchical probabilistic models.
//!
//! The parent module contains a more detailed discussion of the [differences between ANS
//! Coding and Range Coding](super#which-stream-code-should-i-use) .
//!
//! # References
//!
//! \[1] Duda, Jarek, et al. "The use of asymmetric numeral systems as an accurate
//! replacement for Huffman coding." 2015 Picture Coding Symposium (PCS). IEEE, 2015.
//!
//! [`queue`]: super::queue

use alloc::vec::Vec;
use core::{
    borrow::Borrow, convert::Infallible, fmt::Debug, iter::Fuse, marker::PhantomData, ops::Deref,
};
use num_traits::AsPrimitive;

use super::{
    model::{DecoderModel, EncoderModel},
    AsDecoder, Code, Decode, Encode, IntoDecoder, TryCodingError,
};
use crate::{
    backends::{
        self, AsReadWords, AsSeekReadWords, BoundedReadWords, Cursor, FallibleIteratorReadWords,
        IntoReadWords, IntoSeekReadWords, ReadWords, Reverse, WriteWords,
    },
    bit_array_to_chunks_truncated, generic_static_asserts, BitArray, CoderError,
    DefaultEncoderError, DefaultEncoderFrontendError, NonZeroBitArray, Pos, PosSeek, Seek, Stack,
    UnwrapInfallible,
};

/// Entropy coder for both encoding and decoding on a stack.
///
/// This is the generic struct for an ANS coder. It provides fine-tuned control over type
/// parameters (see [discussion in parent
/// module](super#highly-customizable-implementations-with-sane-presets)). You'll usually
/// want to use this type through the type alias [`DefaultAnsCoder`], which provides sane
/// default settings for the type parameters.
///
/// The `AnsCoder` uses an entropy coding algorithm called [range Asymmetric
/// Numeral Systems (rANS)]. This means that it operates as a stack, i.e., a "last
/// in first out" data structure: encoding "pushes symbols on" the stack and
/// decoding "pops symbols off" the stack in reverse order. In default operation, decoding
/// with an `AnsCoder` *consumes* the compressed data for the decoded symbols (however, you
/// can also decode immutable data by using a [`Cursor`]). This means
/// that encoding and decoding can be interleaved arbitrarily, thus growing and shrinking
/// the stack of compressed data as you go.
///
/// # Example
///
/// Basic usage example:
///
/// ```
/// use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode};
///
/// // `DefaultAnsCoder` is a type alias to `AnsCoder` with sane generic parameters.
/// let mut ans = DefaultAnsCoder::new();
///
/// // Create an entropy model based on a quantized Gaussian distribution. You can use `AnsCoder`
/// // with any entropy model defined in the `models` module.
/// let quantizer = DefaultLeakyQuantizer::new(-100..=100);
/// let entropy_model = quantizer.quantize(probability::distribution::Gaussian::new(0.0, 10.0));
///
/// let symbols = vec![-10, 4, 0, 3];
/// // Encode symbols in *reverse* order, so that we can decode them in forward order.
/// ans.encode_iid_symbols_reverse(&symbols, &entropy_model).unwrap();
///
/// // Obtain temporary shared access to the compressed bit string. If you want ownership of the
/// // compressed bit string, call `.into_compressed()` instead of `.get_compressed()`.
/// println!("Encoded into {} bits: {:?}", ans.num_bits(), &*ans.get_compressed().unwrap());
///
/// // Decode the symbols and verify correctness.
/// let reconstructed = ans
///     .decode_iid_symbols(4, &entropy_model)
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
/// assert_eq!(reconstructed, symbols);
/// ```
///
/// # Consistency Between Encoding and Decoding
///
/// As elaborated in the [parent module's documentation](super#whats-a-stream-code),
/// encoding and decoding operates on a sequence of symbols. Each symbol can be encoded and
/// decoded with its own entropy model (the symbols can even have heterogeneous types). If
/// your goal is to reconstruct the originally encoded symbols during decoding, then you
/// must employ the same sequence of entropy models (in reversed order) during encoding and
/// decoding.
///
/// However, using the same entropy models for encoding and decoding is not a *general*
/// requirement. It is perfectly legal to push (encode) symbols on the `AnsCoder` using some
/// entropy models, and then pop off (decode) symbols using different entropy models. The
/// popped off symbols will then in general be different from the original symbols, but will
/// be generated in a deterministic way. If there is no deterministic relation between the
/// entropy models used for pushing and popping, and if there is still compressed data left
/// at the end (i.e., if [`is_empty`] returns false), then the popped off symbols are, to a
/// very good approximation, distributed as independent samples from the respective entropy
/// models. Such random samples, which consume parts of the compressed data, are useful in
/// the bits-back algorithm.
///
/// [range Asymmetric Numeral Systems (rANS)]:
/// https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#Range_variants_(rANS)_and_streaming
/// [`is_empty`]: #method.is_empty`
/// [`Cursor`]: crate::backends::Cursor
#[derive(Clone)]
pub struct AnsCoder<Word, State, Backend = Vec<Word>>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    bulk: Backend,

    /// Invariant: `state >= State::one() << (State::BITS - Word::BITS)` unless
    /// `bulk.is_empty()`.
    state: State,

    /// We keep track of the `Word` type so that we can statically enforce the invariant
    /// `Word: Into<State>`.
    phantom: PhantomData<Word>,
}

/// Type alias for an [`AnsCoder`] with sane parameters for typical use cases.
///
/// This type alias sets the generic type arguments `Word` and `State` to sane values for
/// many typical use cases.
pub type DefaultAnsCoder<Backend = Vec<u32>> = AnsCoder<u32, u64, Backend>;

/// Type alias for an [`AnsCoder`] for use with a [`LookupDecoderModel`]
///
/// This encoder has a smaller word size and internal state than [`AnsCoder`]. It is
/// optimized for use with a [`LookupDecoderModel`].
///
/// # Examples
///
/// See [`SmallContiguousLookupDecoderModel`].
///
/// [`LookupDecoderModel`]: super::model::LookupDecoderModel
/// [`SmallContiguousLookupDecoderModel`]: super::model::SmallContiguousLookupDecoderModel
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

impl<Word, State> From<AnsCoder<Word, State, Vec<Word>>> for Vec<Word>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
{
    fn from(val: AnsCoder<Word, State, Vec<Word>>) -> Self {
        val.into_compressed().unwrap_infallible()
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
        generic_static_asserts!(
            (Word: BitArray, State:BitArray);
            STATE_SUPPORTS_AT_LEAST_TWO_WORDS: State::BITS >= 2 * Word::BITS;
        );

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
    /// Low-level constructor that assembles an `AnsCoder` from its internal components.
    ///
    /// The arguments `bulk` and `state` correspond to the two return values of the method
    /// [`into_raw_parts`](Self::into_raw_parts).
    ///
    /// The caller must ensure that `state >= State::one() << (State::BITS - Word::BITS)`
    /// unless `bulk` is empty. This cannot be checked by the method since not all
    /// `Backend`s have an `is_empty` method. Violating this invariant is not a memory
    /// safety issue but it will lead to incorrect behavior.
    pub fn from_raw_parts(bulk: Backend, state: State) -> Self {
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
        generic_static_asserts!(
            (Word: BitArray, State:BitArray);
            STATE_SUPPORTS_AT_LEAST_TWO_WORDS: State::BITS >= 2 * Word::BITS;
        );

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

    /// Low-level method that disassembles the `AnsCoder` into its internal components.
    ///
    /// Can be used together with [`from_raw_parts`](Self::from_raw_parts).
    pub fn into_raw_parts(self) -> (Backend, State) {
        (self.bulk, self.state)
    }

    /// Check if no data for decoding is left.
    ///
    /// Note that you can still pop symbols off an empty stack, but this is only
    /// useful in rare edge cases, see documentation of
    /// [`decode_symbol`](#method.decode_symbol).
    pub fn is_empty(&self) -> bool {
        // We don't need to check if `bulk` is empty (which would require an additional
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
    /// use constriction::stream::{
    ///     model::DefaultContiguousCategoricalEntropyModel, stack::DefaultAnsCoder, Decode
    /// };
    ///
    /// let mut ans = DefaultAnsCoder::new();
    ///
    /// // Push some data on the ans.
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let model = DefaultContiguousCategoricalEntropyModel
    ///     ::from_floating_point_probabilities_fast(&probabilities, None).unwrap();
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
    pub fn get_compressed(
        &mut self,
    ) -> Result<impl Deref<Target = Backend> + Debug + Drop + '_, Backend::WriteError>
    where
        Backend: ReadWords<Word, Stack> + WriteWords<Word> + Debug,
    {
        CoderGuard::<'_, _, _, _, false>::new(self).map_err(|err| match err {
            CoderError::Frontend(()) => unreachable!("Can't happen for SEALED==false."),
            CoderError::Backend(err) => err,
        })
    }

    pub fn get_binary(
        &mut self,
    ) -> Result<impl Deref<Target = Backend> + Debug + Drop + '_, CoderError<(), Backend::WriteError>>
    where
        Backend: ReadWords<Word, Stack> + WriteWords<Word> + Debug,
    {
        CoderGuard::<'_, _, _, _, true>::new(self)
    }

    /// Iterates over the compressed data currently on the ans.
    ///
    /// In contrast to [`get_compressed`] or [`into_compressed`], this method does
    /// not require mutable access or even ownership of the `AnsCoder`.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode};
    ///
    /// // Create a stack and encode some stuff.
    /// let mut ans = DefaultAnsCoder::new();
    /// let symbols = vec![8, -12, 0, 7];
    /// let quantizer = DefaultLeakyQuantizer::new(-100..=100);
    /// let model =
    ///     quantizer.quantize(probability::distribution::Gaussian::new(0.0, 10.0));
    /// ans.encode_iid_symbols_reverse(&symbols, &model).unwrap();
    ///
    /// // Iterate over compressed data, collect it into to a Vec``, and compare to direct method.
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
    /// This method is similar to [`as_seekable_decoder`] except that it takes ownership of
    /// the original `AnsCoder`, so the returned seekable decoder can typically be returned
    /// from the calling function or put on the heap.
    ///
    /// [`as_seekable_decoder`]: Self::as_seekable_decoder
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
    #[allow(clippy::result_unit_err)]
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
    pub fn encode_symbols_reverse<S, M, I, const PRECISION: usize>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<(), DefaultEncoderError<Backend::WriteError>>
    where
        S: Borrow<M::Symbol>,
        M: EncoderModel<PRECISION>,
        M::Probability: Into<Word>,
        Word: AsPrimitive<M::Probability>,
        I: IntoIterator<Item = (S, M)>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_symbols(symbols_and_models.into_iter().rev())
    }

    pub fn try_encode_symbols_reverse<S, M, E, I, const PRECISION: usize>(
        &mut self,
        symbols_and_models: I,
    ) -> Result<(), TryCodingError<DefaultEncoderError<Backend::WriteError>, E>>
    where
        S: Borrow<M::Symbol>,
        M: EncoderModel<PRECISION>,
        M::Probability: Into<Word>,
        Word: AsPrimitive<M::Probability>,
        I: IntoIterator<Item = core::result::Result<(S, M), E>>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.try_encode_symbols(symbols_and_models.into_iter().rev())
    }

    pub fn encode_iid_symbols_reverse<S, M, I, const PRECISION: usize>(
        &mut self,
        symbols: I,
        model: M,
    ) -> Result<(), DefaultEncoderError<Backend::WriteError>>
    where
        S: Borrow<M::Symbol>,
        M: EncoderModel<PRECISION> + Copy,
        M::Probability: Into<Word>,
        Word: AsPrimitive<M::Probability>,
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
    /// use constriction::stream::{
    ///     model::DefaultContiguousCategoricalEntropyModel, stack::DefaultAnsCoder, Decode
    /// };
    ///
    /// let mut ans = DefaultAnsCoder::new();
    ///
    /// // Push some data onto the ANS coder's stack:
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let model = DefaultContiguousCategoricalEntropyModel
    ///     ::from_floating_point_probabilities_fast(&probabilities, None).unwrap();
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

        if valid_bits % Word::BITS != 0 || valid_bits == usize::MAX {
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
    /// The bound `impl Borrow<M::Symbol>` on argument `symbol` essentially means that
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
    fn encode_symbol<M>(
        &mut self,
        symbol: impl Borrow<M::Symbol>,
        model: M,
    ) -> Result<(), DefaultEncoderError<Self::BackendError>>
    where
        M: EncoderModel<PRECISION>,
        M::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<M::Probability>,
    {
        generic_static_asserts!(
            (Word: BitArray, State:BitArray; const PRECISION: usize);
            PROBABILITY_SUPPORTS_PRECISION: State::BITS >= Word::BITS + PRECISION;
            NON_ZERO_PRECISION: PRECISION > 0;
            STATE_SUPPORTS_AT_LEAST_TWO_WORDS: State::BITS >= 2 * Word::BITS;
        );

        let (left_sided_cumulative, probability) = model
            .left_cumulative_and_probability(symbol)
            .ok_or_else(|| DefaultEncoderFrontendError::ImpossibleSymbol.into_coder_error())?;

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
    fn decode_symbol<M>(
        &mut self,
        model: M,
    ) -> Result<M::Symbol, CoderError<Self::FrontendError, Self::BackendError>>
    where
        M: DecoderModel<PRECISION>,
        M::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<M::Probability>,
    {
        generic_static_asserts!(
            (Word: BitArray, State:BitArray; const PRECISION: usize);
            PROBABILITY_SUPPORTS_PRECISION: State::BITS >= Word::BITS + PRECISION;
            NON_ZERO_PRECISION: PRECISION > 0;
            STATE_SUPPORTS_AT_LEAST_TWO_WORDS: State::BITS >= 2 * Word::BITS;
        );

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
struct CoderGuard<'a, Word, State, Backend, const SEALED: bool>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + ReadWords<Word, Stack>,
{
    inner: &'a mut AnsCoder<Word, State, Backend>,
}

impl<'a, Word, State, Backend, const SEALED: bool> CoderGuard<'a, Word, State, Backend, SEALED>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + ReadWords<Word, Stack>,
{
    #[inline(always)]
    fn new(
        ans: &'a mut AnsCoder<Word, State, Backend>,
    ) -> Result<Self, CoderError<(), Backend::WriteError>> {
        // Append state. Will be undone in `<Self as Drop>::drop`.
        let mut chunks_rev = bit_array_to_chunks_truncated(ans.state);
        if SEALED && chunks_rev.next() != Some(Word::one()) {
            return Err(CoderError::Frontend(()));
        }
        for chunk in chunks_rev.rev() {
            ans.bulk.write(chunk)?
        }

        Ok(Self { inner: ans })
    }
}

impl<'a, Word, State, Backend, const SEALED: bool> Drop
    for CoderGuard<'a, Word, State, Backend, SEALED>
where
    Word: BitArray + Into<State>,
    State: BitArray + AsPrimitive<Word>,
    Backend: WriteWords<Word> + ReadWords<Word, Stack>,
{
    fn drop(&mut self) {
        // Revert what we did in `Self::new`.
        let mut chunks_rev = bit_array_to_chunks_truncated(self.inner.state);
        if SEALED {
            chunks_rev.next();
        }
        for _ in chunks_rev {
            core::mem::drop(self.inner.bulk.read());
        }
    }
}

impl<'a, Word, State, Backend, const SEALED: bool> Deref
    for CoderGuard<'a, Word, State, Backend, SEALED>
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

impl<Word, State, Backend, const SEALED: bool> Debug
    for CoderGuard<'_, Word, State, Backend, SEALED>
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
    use super::super::model::{
        ContiguousCategoricalEntropyModel, DefaultLeakyQuantizer, IterableEntropyModel,
        LeakyQuantizer,
    };
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
        let quantizer = DefaultLeakyQuantizer::new(-127..=127);
        let model = quantizer.quantize(Gaussian::new(3.2, 5.1));

        // We don't reuse the same encoder for decoding because we want to test
        // if exporting and re-importing of compressed data works.
        encoder.encode_iid_symbols(symbols.clone(), model).unwrap();
        let compressed = encoder.into_compressed().unwrap();
        assert_eq!(compressed.len(), expected_size);

        let mut decoder = DefaultAnsCoder::from_compressed(compressed).unwrap();
        for symbol in symbols.rev() {
            assert_eq!(decoder.decode_symbol(model).unwrap(), symbol);
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
        #[cfg(not(miri))]
        const AMT: usize = 1000;

        #[cfg(miri)]
        const AMT: usize = 100;

        let mut symbols_gaussian = Vec::with_capacity(AMT);
        let mut means = Vec::with_capacity(AMT);
        let mut stds = Vec::with_capacity(AMT);

        let mut rng = Xoshiro256StarStar::seed_from_u64(
            (Word::BITS as u64).rotate_left(3 * 16)
                ^ (State::BITS as u64).rotate_left(2 * 16)
                ^ (Probability::BITS as u64).rotate_left(16)
                ^ PRECISION as u64,
        );

        for _ in 0..AMT {
            let mean = (200.0 / u32::MAX as f64) * rng.next_u32() as f64 - 100.0;
            let std_dev = (10.0 / u32::MAX as f64) * rng.next_u32() as f64 + 0.001;
            let quantile = (rng.next_u32() as f64 + 0.5) / (1u64 << 32) as f64;
            let dist = Gaussian::new(mean, std_dev);
            let symbol = (dist.inverse(quantile).round() as i32).clamp(-127, 127);

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
            ContiguousCategoricalEntropyModel::<Probability, _, PRECISION>::from_floating_point_probabilities_fast::<f64>(
                &categorical_probabilities,None
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
        #[cfg(not(miri))]
        let (num_chunks, symbols_per_chunk) = (100, 100);

        #[cfg(miri)]
        let (num_chunks, symbols_per_chunk) = (10, 10);

        let quantizer = DefaultLeakyQuantizer::new(-100..=100);
        let model = quantizer.quantize(Gaussian::new(0.0, 10.0));

        let mut encoder = DefaultAnsCoder::new();

        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        let mut symbols = Vec::with_capacity(num_chunks);
        let mut jump_table = Vec::with_capacity(num_chunks);
        let (initial_pos, initial_state) = encoder.pos();

        for _ in 0..num_chunks {
            let chunk = (0..symbols_per_chunk)
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
                    .decode_iid_symbols(symbols_per_chunk, &model)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(&decoded, chunk)
            }
            assert_eq!(seekable_decoder.pos(), (initial_pos, initial_state));
            assert!(seekable_decoder.is_empty());

            // Seek to some random offsets in the jump table and decode one chunk
            for _ in 0..100 {
                let chunk_index = rng.next_u32() as usize % num_chunks;
                let (pos, state) = jump_table[chunk_index];
                seekable_decoder.seek((pos, state)).unwrap();
                let decoded = seekable_decoder
                    .decode_iid_symbols(symbols_per_chunk, &model)
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
                    .decode_iid_symbols(symbols_per_chunk, &model)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(&decoded, chunk)
            }
            assert_eq!(seekable_decoder.pos(), (initial_pos, initial_state));
            assert!(seekable_decoder.is_empty());

            // Seek to some random offsets in the jump table and decode one chunk each time.
            for _ in 0..100 {
                let chunk_index = rng.next_u32() as usize % num_chunks;
                let (pos, state) = jump_table[chunk_index];
                seekable_decoder.seek((pos, state)).unwrap();
                let decoded = seekable_decoder
                    .decode_iid_symbols(symbols_per_chunk, &model)
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                assert_eq!(&decoded, &symbols[chunk_index])
            }
        }
    }
}
