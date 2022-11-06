//! Symbol Codes (mainly provided for teaching purpose; typically inferior to stream codes)
//!
//! # TODO
//!
//! - implement `Pos` and `Seek` for `SymbolCoder` and for `QueueDecoder`.

#![allow(clippy::type_complexity)]

pub mod exp_golomb;
pub mod huffman;

use alloc::vec::Vec;
use core::{
    borrow::Borrow,
    convert::Infallible,
    fmt::{Debug, Display},
    iter::{Repeat, Take},
    marker::PhantomData,
    ops::Deref,
};

use smallvec::SmallVec;

use crate::{
    backends::{AsReadWords, BoundedReadWords, Cursor, IntoReadWords, ReadWords, WriteWords},
    BitArray, CoderError, DefaultEncoderError, Queue, Semantics, Stack, UnwrapInfallible,
};

// TRAITS FOR READING AND WRITNIG STREAMS OF BITS =============================

pub trait ReadBitStream<S: Semantics> {
    type ReadError;

    fn read_bit(&mut self) -> Result<Option<bool>, Self::ReadError>;

    fn decode_symbol<C: DecoderCodebook>(
        &mut self,
        codebook: C,
    ) -> Result<C::Symbol, CoderError<SymbolCodeError<C::InvalidCodeword>, Self::ReadError>>;

    fn decode_symbols<'s, I, C>(&'s mut self, codebooks: I) -> DecodeSymbols<'s, Self, I, S>
    where
        I: IntoIterator<Item = C> + 's,
        C: DecoderCodebook,
    {
        DecodeSymbols {
            bit_stream: self,
            codebooks,
            semantics: PhantomData,
        }
    }

    fn decode_iid_symbols<'a, C>(
        &'a mut self,
        amt: usize,
        codebook: &'a C,
    ) -> DecodeSymbols<'a, Self, Take<Repeat<&'a C>>, S>
    where
        C: DecoderCodebook,
    {
        self.decode_symbols(core::iter::repeat(codebook).take(amt))
    }
}

pub trait WriteBitStream<S: Semantics> {
    type WriteError;

    fn write_bit(&mut self, bit: bool) -> Result<(), Self::WriteError>;

    fn encode_symbol<Symbol, C>(
        &mut self,
        symbol: Symbol,
        codebook: C,
    ) -> Result<(), DefaultEncoderError<Self::WriteError>>
    where
        C: EncoderCodebook,
        Symbol: Borrow<C::Symbol>;

    fn encode_symbols<Symbol, C>(
        &mut self,
        symbols_and_codebooks: impl IntoIterator<Item = (Symbol, C)>,
    ) -> Result<(), DefaultEncoderError<Self::WriteError>>
    where
        C: EncoderCodebook,
        Symbol: Borrow<C::Symbol>,
    {
        for (symbol, codebook) in symbols_and_codebooks.into_iter() {
            self.encode_symbol(symbol, codebook)?;
        }

        Ok(())
    }

    fn encode_iid_symbols<Symbol, C>(
        &mut self,
        symbols: impl IntoIterator<Item = Symbol>,
        codebook: &C,
    ) -> Result<(), DefaultEncoderError<Self::WriteError>>
    where
        C: EncoderCodebook,
        Symbol: Borrow<C::Symbol>,
    {
        self.encode_symbols(symbols.into_iter().map(|symbol| (symbol, codebook)))
    }
}

#[derive(Debug)]
pub struct DecodeSymbols<'a, Stream: ?Sized, I, S: Semantics> {
    bit_stream: &'a mut Stream,
    codebooks: I,
    semantics: PhantomData<S>,
}

impl<'a, Stream, I, C, S> Iterator for DecodeSymbols<'a, Stream, I, S>
where
    S: Semantics,
    Stream: ReadBitStream<S>,
    C: DecoderCodebook,
    I: Iterator<Item = C>,
{
    type Item =
        Result<C::Symbol, CoderError<SymbolCodeError<C::InvalidCodeword>, Stream::ReadError>>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.bit_stream.decode_symbol(self.codebooks.next()?))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.codebooks.size_hint()
    }
}

impl<'a, Stream, I, C, S> ExactSizeIterator for DecodeSymbols<'a, Stream, I, S>
where
    S: Semantics,
    Stream: ReadBitStream<S>,
    C: DecoderCodebook,
    I: ExactSizeIterator<Item = C>,
{
}

// ADAPTER THAT TURNS A BACKEND INTO A BIT STREAM =============================

/// Generic symbol coder for 3 out of 4 possible cases
///
/// You likely won't spell out this type explicitly. It's more convenient to use one of the
/// type aliases [`QueueEncoder`] or [`StackCoder`] (or the more opinionated aliases
/// [`DefaultQueueEncoder`] and [`DefaultStackCoder`]).
///
/// Depending on the type parameter `S`, this type supports:
/// - encoding on a queue (i.e., writing prefix codes); this is type aliased as
///   [`QueueEncoder`].
/// - encoding and decoding on a stack (i.e., writing suffix codes and reading them back in
///   reoverse order). This is type aliased as [`StackCoder`].
///
/// This type does not support decoding from a queue. Use a [`QueueDecoder`] for this
/// purpose.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct SymbolCoder<Word: BitArray, S: Semantics, B = Vec<Word>> {
    backend: B,
    current_word: Word,

    /// A `BitArray` with at most one set bit:
    /// - `mask_last_written == 0` if all bits written so far have already been flushed to
    ///   the backend (in case of a write backend) and/or if reading the next bit would
    ///   require obtaining a new word from the backend (in case of a read backend); This
    ///   includes the case of an empty `QueueEncoder`. In all of these cases, `current_word`
    ///  as to be zero.
    /// - otherwise, `mask_last_written` has a single set bit which marks the position in
    ///   `current_word` where the next bit should be written if any.
    mask_last_written: Word,

    semantics: PhantomData<S>,
}

pub type QueueEncoder<Word, B = Vec<Word>> = SymbolCoder<Word, Queue, B>;
pub type StackCoder<Word, B = Vec<Word>> = SymbolCoder<Word, Stack, B>;

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct QueueDecoder<Word: BitArray, B> {
    backend: B,
    current_word: Word,

    /// If zero then `current_word` is meaningless and has to be read in from `backend`.
    mask_next_to_read: Word,
}

pub type DefaultQueueEncoder = QueueEncoder<u32, Vec<u32>>;
pub type DefaultQueueDecoder = QueueDecoder<u32, Cursor<u32, Vec<u32>>>;
pub type DefaultStackCoder = StackCoder<u32, Vec<u32>>;

// GENERIC IMPLEMENTATIONS ====================================================

impl<Word: BitArray, S: Semantics, B> SymbolCoder<Word, S, B> {
    pub fn new() -> Self
    where
        B: Default,
    {
        Default::default()
    }

    /// Returns the correct len for, e.g., `B = Vec<Word>` regardless of whether `S = Queue`
    /// or `S = Stack`.
    pub fn len(&self) -> usize
    where
        B: BoundedReadWords<Word, Stack>,
    {
        self.backend
            .remaining()
            .checked_mul(Word::BITS)
            .expect("len overflows addressable space")
            .checked_add(if self.mask_last_written == Word::zero() {
                0
            } else {
                (self.mask_last_written.trailing_zeros() + 1) as usize
            })
            .expect("len overflows addressable space")
    }

    /// Returns `true` if no bits are on the `SymbolCoder`.
    pub fn is_empty(&self) -> bool
    where
        B: BoundedReadWords<Word, Stack>,
    {
        self.mask_last_written == Word::zero() && self.backend.is_exhausted()
    }
}

// SPECIAL IMPLEMENTATIONS FOR VEC ============================================

impl<Word: BitArray> StackCoder<Word, Vec<Word>> {
    pub fn with_bit_capacity(bit_capacity: usize) -> Self {
        Self {
            // Reserve capacity for one additional bit for sealing.
            backend: Vec::with_capacity(bit_capacity / Word::BITS + 1),
            ..Default::default()
        }
    }

    pub fn get_compressed(&mut self) -> StackCoderGuard<'_, Word> {
        StackCoderGuard::new(self)
    }
}

impl<Word: BitArray> QueueEncoder<Word, Vec<Word>> {
    pub fn with_bit_capacity(bit_capacity: usize) -> Self {
        Self {
            backend: Vec::with_capacity((bit_capacity + Word::BITS - 1) / Word::BITS),
            ..Default::default()
        }
    }

    pub fn get_compressed(&mut self) -> QueueEncoderGuard<'_, Word> {
        QueueEncoderGuard::new(self)
    }
}

#[derive(Debug)]
pub struct StackCoderGuard<'a, Word: BitArray> {
    inner: &'a mut StackCoder<Word, Vec<Word>>,
}

impl<'a, Word: BitArray> StackCoderGuard<'a, Word> {
    fn new(stack_coder: &'a mut StackCoder<Word, Vec<Word>>) -> Self {
        // Stacks need to be sealed by one additional bit so that the end can be discovered.
        stack_coder.write_bit(true).unwrap_infallible();
        if stack_coder.mask_last_written != Word::zero() {
            stack_coder.backend.push(stack_coder.current_word);
        }
        Self { inner: stack_coder }
    }
}

impl<'a, Word: BitArray> Drop for StackCoderGuard<'a, Word> {
    fn drop(&mut self) {
        if self.inner.mask_last_written != Word::zero() {
            self.inner.backend.pop();
        }
        self.inner.read_bit().expect("The constructor wrote a bit.");
    }
}

impl<'a, Word: BitArray> Deref for StackCoderGuard<'a, Word> {
    type Target = [Word];

    fn deref(&self) -> &Self::Target {
        &self.inner.backend
    }
}

#[derive(Debug)]
pub struct QueueEncoderGuard<'a, Word: BitArray> {
    inner: &'a mut QueueEncoder<Word, Vec<Word>>,
}

impl<'a, Word: BitArray> QueueEncoderGuard<'a, Word> {
    fn new(queue_encoder: &'a mut QueueEncoder<Word, Vec<Word>>) -> Self {
        // Queues don't need to be sealed, so just flush the remaining word if any.
        if queue_encoder.mask_last_written != Word::zero() {
            queue_encoder.backend.push(queue_encoder.current_word);
        }
        Self {
            inner: queue_encoder,
        }
    }
}

impl<'a, Word: BitArray> Drop for QueueEncoderGuard<'a, Word> {
    fn drop(&mut self) {
        if self.inner.mask_last_written != Word::zero() {
            self.inner.backend.pop();
        }
    }
}

impl<'a, Word: BitArray> Deref for QueueEncoderGuard<'a, Word> {
    type Target = [Word];

    fn deref(&self) -> &Self::Target {
        &self.inner.backend
    }
}

// IMPLEMENTATIONS FOR A QUEUE ================================================

impl<Word: BitArray, B> QueueEncoder<Word, B> {
    pub fn from_compressed(compressed: B) -> Self
    where
        B: Default,
    {
        Self {
            backend: compressed,
            ..Default::default()
        }
    }

    pub fn into_decoder(self) -> Result<QueueDecoder<Word, B::IntoReadWords>, B::WriteError>
    where
        B: WriteWords<Word> + IntoReadWords<Word, Queue>,
    {
        Ok(QueueDecoder::from_compressed(
            self.into_compressed()?.into_read_words(),
        ))
    }

    pub fn into_compressed(mut self) -> Result<B, B::WriteError>
    where
        B: WriteWords<Word>,
    {
        // Queues don't need to be sealed, so just flush the remaining word if any.
        if self.mask_last_written != Word::zero() {
            self.backend.write(self.current_word)?;
        }
        Ok(self.backend)
    }

    pub fn into_overshooting_iter(
        self,
    ) -> Result<
        impl Iterator<Item = Result<bool, <B::IntoReadWords as ReadWords<Word, Queue>>::ReadError>>,
        B::WriteError,
    >
    where
        B: WriteWords<Word> + IntoReadWords<Word, Queue>,
    {
        // TODO: return `impl ExactSizeIterator` for `B: BoundedReadWords` once
        // specialization is stable
        self.into_decoder()
    }
}

impl<Word: BitArray, B: WriteWords<Word>> WriteBitStream<Queue> for QueueEncoder<Word, B> {
    type WriteError = B::WriteError;

    fn write_bit(&mut self, bit: bool) -> Result<(), Self::WriteError> {
        let write_mask = self.mask_last_written << 1;
        self.mask_last_written = if write_mask != Word::zero() {
            let new_bit = if bit { write_mask } else { Word::zero() };
            self.current_word = self.current_word | new_bit;
            write_mask
        } else {
            if self.mask_last_written != Word::zero() {
                self.backend.write(self.current_word)?;
            }
            self.current_word = if bit { Word::one() } else { Word::zero() };
            Word::one()
        };

        Ok(())
    }

    #[inline(always)]
    fn encode_symbol<Symbol, C>(
        &mut self,
        symbol: Symbol,
        codebook: C,
    ) -> Result<(), DefaultEncoderError<Self::WriteError>>
    where
        C: EncoderCodebook,
        Symbol: Borrow<C::Symbol>,
    {
        codebook.encode_symbol_prefix(symbol, |bit| self.write_bit(bit))
    }
}

impl<Word: BitArray, B> QueueDecoder<Word, B> {
    pub fn from_compressed(compressed: B) -> Self {
        Self {
            backend: compressed,
            current_word: Word::zero(),
            mask_next_to_read: Word::zero(),
        }
    }

    /// We don't keep track of the exact length of a queue, so we can only say with
    /// certainty if we can detect that there's something left.
    pub fn maybe_exhausted(&self) -> bool
    where
        B: BoundedReadWords<Word, Queue>,
    {
        let mask_remaining_bits = !self.mask_next_to_read.wrapping_sub(&Word::one());
        self.current_word & mask_remaining_bits == Word::zero() && self.backend.is_exhausted()
    }
}

impl<Word: BitArray, B: ReadWords<Word, Queue>> ReadBitStream<Queue> for QueueDecoder<Word, B> {
    type ReadError = B::ReadError;

    #[inline(always)]
    fn decode_symbol<C: DecoderCodebook>(
        &mut self,
        codebook: C,
    ) -> Result<C::Symbol, CoderError<SymbolCodeError<C::InvalidCodeword>, Self::ReadError>> {
        codebook.decode_symbol(self)
    }

    fn read_bit(&mut self) -> Result<Option<bool>, Self::ReadError> {
        if self.mask_next_to_read == Word::zero() {
            match self.backend.read() {
                Ok(Some(next_word)) => {
                    self.current_word = next_word;
                    self.mask_next_to_read = Word::one();
                }
                Ok(None) => return Ok(None),
                Err(err) => return Err(err),
            }
        }

        let bit = self.current_word & self.mask_next_to_read != Word::zero();
        // No need to unset the bit in `current_word` since we're only reading, never writing.
        self.mask_next_to_read = self.mask_next_to_read << 1;
        Ok(Some(bit))
    }
}

impl<Word: BitArray, B: ReadWords<Word, Queue>> Iterator for QueueDecoder<Word, B> {
    type Item = Result<bool, B::ReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_bit().transpose()
    }
}

// IMPLEMENTATIONS FOR A STACK ================================================

impl<Word: BitArray, B: WriteWords<Word>> StackCoder<Word, B> {
    /// # Errors
    ///
    /// - Returns `Err(CoderError::FrontendError(compressed))` if `compressed` ends in a
    ///   zero which is not allowed for stacks (because we need a terminal "one" bit to
    ///   identify the head position). Note that "EOF" is not considered an error, and the
    ///   method will return with a success if called with an empty backend.
    /// - Returns `Err(CoderError::BackendError(err))` if reading the last word from the
    ///   backend resulted in `Err(err)`.
    /// - Returns `Ok(stack_coder)` in all other cases, including the case where the backend
    ///   is empty.
    pub fn from_compressed(mut compressed: B) -> Result<Self, CoderError<B, B::ReadError>>
    where
        B: ReadWords<Word, Stack>,
    {
        let (current_word, mask_last_written) = if let Some(last_word) = compressed.read()? {
            if last_word == Word::zero() {
                // A stack of compressed data must not end in a zero word.
                return Err(CoderError::Frontend(compressed));
            }
            let mask_end_bit = Word::one() << last_word.trailing_zeros() as usize;
            (last_word ^ mask_end_bit, mask_end_bit >> 1)
        } else {
            (Word::zero(), Word::zero())
        };

        Ok(Self {
            backend: compressed,
            current_word,
            mask_last_written,
            semantics: PhantomData,
        })
    }

    pub fn into_compressed(mut self) -> Result<B, B::WriteError> {
        // Stacks need to be sealed by one additional bit so that the end can be discovered.
        self.write_bit(true)?;
        if self.mask_last_written != Word::zero() {
            self.backend.write(self.current_word)?;
        }
        Ok(self.backend)
    }

    #[inline(always)]
    pub fn encode_symbols_reverse<Symbol, C, I>(
        &mut self,
        symbols_and_codebooks: I,
    ) -> Result<(), DefaultEncoderError<B::WriteError>>
    where
        Symbol: Borrow<C::Symbol>,
        C: EncoderCodebook,
        I: IntoIterator<Item = (Symbol, C)>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_symbols(symbols_and_codebooks.into_iter().rev())
    }

    #[inline(always)]
    pub fn encode_iid_symbols_reverse<Symbol, C, I>(
        &mut self,
        symbols: I,
        codebook: &C,
    ) -> Result<(), DefaultEncoderError<B::WriteError>>
    where
        Symbol: Borrow<C::Symbol>,
        C: EncoderCodebook,
        I: IntoIterator<Item = Symbol>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_iid_symbols(symbols.into_iter().rev(), codebook)
    }

    pub fn into_decoder(self) -> SymbolCoder<Word, Stack, B::IntoReadWords>
    where
        B: IntoReadWords<Word, Stack>,
    {
        SymbolCoder {
            backend: self.backend.into_read_words(),
            current_word: self.current_word,
            mask_last_written: self.mask_last_written,
            semantics: PhantomData,
        }
    }

    pub fn as_decoder<'a>(&'a self) -> SymbolCoder<Word, Stack, B::AsReadWords>
    where
        B: AsReadWords<'a, Word, Stack>,
    {
        SymbolCoder {
            backend: self.backend.as_read_words(),
            current_word: self.current_word,
            mask_last_written: self.mask_last_written,
            semantics: PhantomData,
        }
    }

    /// Consumes the coder and returns an iterator over bits (in reverse direction)
    ///
    /// You often don't need to call this method since a `StackCoder` is already an iterator
    /// if the backend implements `ReadWords<Word, Stack>` (as the default backend
    /// `Vec<Word>` does).
    pub fn into_iterator(
        self,
    ) -> impl Iterator<Item = Result<bool, <B::IntoReadWords as ReadWords<Word, Stack>>::ReadError>>
    where
        B: IntoReadWords<Word, Stack>,
    {
        self.into_decoder()
    }

    /// Returns an iterator over bits (in reverse direction) that leaves the current coder untouched.
    ///
    /// You often don't need to call this method since a `StackCoder` is already an iterator
    /// if the backend implements `ReadWords<Word, Stack>` (as the default backend
    /// `Vec<Word>` does).
    pub fn iter<'a>(
        &'a self,
    ) -> impl Iterator<
        Item = Result<
            bool,
            <<B as AsReadWords<'a, Word, Stack>>::AsReadWords as ReadWords<Word, Stack>>::ReadError,
        >,
    > + 'a
    where
        B: AsReadWords<'a, Word, Stack>,
    {
        self.as_decoder()
    }
}

impl<Word: BitArray, B: WriteWords<Word>> WriteBitStream<Stack> for StackCoder<Word, B> {
    type WriteError = B::WriteError;

    fn write_bit(&mut self, bit: bool) -> Result<(), Self::WriteError> {
        let write_mask = self.mask_last_written << 1;
        self.mask_last_written = if write_mask != Word::zero() {
            let new_bit = if bit { write_mask } else { Word::zero() };
            self.current_word = self.current_word | new_bit;
            write_mask
        } else {
            if self.mask_last_written != Word::zero() {
                self.backend.write(self.current_word)?;
            }
            self.current_word = if bit { Word::one() } else { Word::zero() };
            Word::one()
        };

        Ok(())
    }

    #[inline(always)]
    fn encode_symbol<Symbol, C>(
        &mut self,
        symbol: Symbol,
        codebook: C,
    ) -> Result<(), DefaultEncoderError<Self::WriteError>>
    where
        Symbol: Borrow<C::Symbol>,
        C: EncoderCodebook,
    {
        codebook.encode_symbol_suffix(symbol, |bit| self.write_bit(bit))
    }
}

impl<Word: BitArray, B: ReadWords<Word, Stack>> ReadBitStream<Stack> for StackCoder<Word, B> {
    type ReadError = B::ReadError;

    #[inline(always)]
    fn decode_symbol<C: DecoderCodebook>(
        &mut self,
        codebook: C,
    ) -> Result<C::Symbol, CoderError<SymbolCodeError<C::InvalidCodeword>, Self::ReadError>> {
        codebook.decode_symbol(self)
    }

    fn read_bit(&mut self) -> Result<Option<bool>, Self::ReadError> {
        if self.mask_last_written == Word::zero() {
            self.current_word = if let Some(next_word) = self.backend.read()? {
                next_word
            } else {
                // Reached end of stream (this is considered `Ok`).
                return Ok(None);
            };
            self.mask_last_written = Word::one() << (Word::BITS - 1);
        }

        let bit = self.current_word & self.mask_last_written;
        self.current_word = self.current_word ^ bit;
        self.mask_last_written = self.mask_last_written >> 1;
        Ok(Some(bit != Word::zero()))
    }
}

impl<Word: BitArray, B: ReadWords<Word, Stack>> Iterator for StackCoder<Word, B> {
    type Item = Result<bool, B::ReadError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.read_bit().transpose()
    }

    // TODO: override `size_hint` for `B: BoundedReadWords` when specialization is stable.
}

impl<Word: BitArray, B: BoundedReadWords<Word, Stack>> ExactSizeIterator for StackCoder<Word, B> {
    fn len(&self) -> usize {
        StackCoder::len(self)
    }
}

type SmallBitStack = StackCoder<usize, SmallVec<[usize; 1]>>;

#[derive(Debug)]
pub enum SymbolCodeError<InvalidCodeword = Infallible> {
    /// The compressed data ended before the current codeword was complete.
    OutOfCompressedData,

    /// Found a code word that does not map to any symbol.
    InvalidCodeword(InvalidCodeword),
}

impl<InvalidCodeword> SymbolCodeError<InvalidCodeword> {
    pub fn into_coder_error<BackendError>(self) -> CoderError<Self, BackendError> {
        CoderError::Frontend(self)
    }
}

impl<InvalidCodeword: Display> Display for SymbolCodeError<InvalidCodeword> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfCompressedData => write!(
                f,
                "The compressed data ended before the current codeword was complete."
            ),
            Self::InvalidCodeword(err) => write!(f, "Invalid codeword for this codebook: {}", err),
        }
    }
}

#[cfg(feature = "std")]
impl<InvalidCodeword: std::error::Error + 'static> std::error::Error
    for SymbolCodeError<InvalidCodeword>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OutOfCompressedData => None,
            Self::InvalidCodeword(source) => Some(source),
        }
    }
}

pub trait Codebook {
    type Symbol;
}

pub trait EncoderCodebook: Codebook {
    fn encode_symbol_prefix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        mut emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), DefaultEncoderError<BackendError>> {
        let mut reverse_codeword = SmallBitStack::new();
        self.encode_symbol_suffix(symbol, |bit| reverse_codeword.write_bit(bit))
            .map_err(|err| CoderError::Frontend(err.into_frontend_error()))?;

        for bit in reverse_codeword {
            emit(bit.unwrap_infallible())?;
        }
        Ok(())
    }

    fn encode_symbol_suffix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        mut emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), DefaultEncoderError<BackendError>> {
        let mut reverse_codeword = SmallBitStack::new();
        self.encode_symbol_prefix(symbol, |bit| reverse_codeword.write_bit(bit))
            .map_err(|err| CoderError::Frontend(err.into_frontend_error()))?;

        for bit in reverse_codeword {
            emit(bit.unwrap_infallible())?;
        }
        Ok(())
    }
}

pub trait DecoderCodebook: Codebook {
    type InvalidCodeword;

    fn decode_symbol<BackendError>(
        &self,
        source: impl Iterator<Item = Result<bool, BackendError>>,
    ) -> Result<Self::Symbol, CoderError<SymbolCodeError<Self::InvalidCodeword>, BackendError>>;
}

impl<C: Codebook> Codebook for &C {
    type Symbol = C::Symbol;
}

impl<C: EncoderCodebook> EncoderCodebook for &C {
    #[inline(always)]
    fn encode_symbol_prefix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), DefaultEncoderError<BackendError>> {
        (*self).encode_symbol_prefix(symbol, emit)
    }

    #[inline(always)]
    fn encode_symbol_suffix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), DefaultEncoderError<BackendError>> {
        (*self).encode_symbol_suffix(symbol, emit)
    }
}

impl<C: DecoderCodebook> DecoderCodebook for &C {
    type InvalidCodeword = C::InvalidCodeword;

    fn decode_symbol<BackendError>(
        &self,
        source: impl Iterator<Item = Result<bool, BackendError>>,
    ) -> Result<Self::Symbol, CoderError<SymbolCodeError<Self::InvalidCodeword>, BackendError>>
    {
        (*self).decode_symbol(source)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        huffman::{DecoderHuffmanTree, EncoderHuffmanTree},
        *,
    };

    use crate::UnwrapInfallible;

    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };

    #[test]
    fn bit_queue() {
        let mut bit_queue = DefaultQueueEncoder::new();
        assert_eq!(bit_queue.len(), 0);
        assert!(bit_queue.is_empty());

        let amt = 150;
        let mut bool_vec = Vec::with_capacity(amt);
        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        for _ in 0..amt {
            let bit = rng.next_u32() % 2 != 0;
            bit_queue.write_bit(bit).unwrap();
            bool_vec.push(bit);
        }

        assert_eq!(bit_queue.len(), amt);
        assert!(!bit_queue.is_empty());

        let mut queue_iter = bit_queue.into_overshooting_iter().unwrap_infallible();
        for expected in bool_vec {
            assert_eq!(queue_iter.next().unwrap().unwrap_infallible(), expected);
        }

        for remaining in queue_iter {
            assert!(!remaining.unwrap_infallible());
        }
    }

    #[test]
    fn bit_stack() {
        let mut bit_stack = DefaultStackCoder::new();
        assert_eq!(bit_stack.len(), 0);
        assert!(bit_stack.is_empty());

        let amt = 150;
        let mut bool_vec = Vec::with_capacity(amt);
        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        for _ in 0..amt {
            let bit = rng.next_u32() % 2 != 0;
            bit_stack.write_bit(bit).unwrap();
            bool_vec.push(bit);
        }

        assert_eq!(bit_stack.len(), amt);
        assert!(!bit_stack.is_empty());
        assert!(bool_vec.into_iter().rev().eq(bit_stack.map(Result::unwrap)));
    }

    #[test]
    fn encode_decode_iid_queue() {
        let amt = 1000;
        let mut rng = Xoshiro256StarStar::seed_from_u64(12345);
        let symbols = (0..amt)
            .map(|_| (rng.next_u32() % 5) as usize)
            .collect::<Vec<_>>();

        let probabilities = [2, 2, 4, 1, 1];
        let encoder_codebook = EncoderHuffmanTree::from_probabilities::<u32, _>(&probabilities);
        let decoder_codebook = DecoderHuffmanTree::from_probabilities::<u32, _>(&probabilities);

        let mut encoder = DefaultQueueEncoder::new();

        assert_eq!(encoder.len(), 0);
        encoder
            .encode_iid_symbols(&symbols, &encoder_codebook)
            .unwrap();
        assert!(encoder.len() > amt);

        let mut decoder = encoder.into_decoder().unwrap_infallible();
        let reconstructed = decoder
            .decode_iid_symbols(amt, &decoder_codebook)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(reconstructed, symbols);
        assert!(decoder.maybe_exhausted());
    }

    #[test]
    fn encode_decode_iid_stack() {
        let amt = 1000;
        let mut rng = Xoshiro256StarStar::seed_from_u64(12345);
        let symbols = (0..amt)
            .map(|_| (rng.next_u32() % 5) as usize)
            .collect::<Vec<_>>();

        let probabilities = [2, 2, 4, 1, 1];
        let encoder_codebook = EncoderHuffmanTree::from_probabilities::<u32, _>(&probabilities);
        let decoder_codebook = DecoderHuffmanTree::from_probabilities::<u32, _>(&probabilities);

        let mut coder = DefaultStackCoder::new();

        assert_eq!(coder.len(), 0);
        coder
            .encode_iid_symbols_reverse(&symbols, &encoder_codebook)
            .unwrap();
        assert!(coder.len() > amt);

        let reconstructed = coder
            .decode_iid_symbols(amt, &decoder_codebook)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(reconstructed, symbols);
        assert!(coder.is_empty());
    }

    #[test]
    fn encode_decode_non_iid() {
        fn iter_probs_and_symbols(amt: usize) -> impl Iterator<Item = (Vec<u32>, usize)> {
            let mut rng = Xoshiro256StarStar::seed_from_u64(123456);
            (0..amt).map(move |_| {
                let num_symbols = 1 + rng.next_u32() % 10;
                let probs = (0..num_symbols).map(|_| rng.next_u32() >> 16).collect();
                let symbol = rng.next_u32() % num_symbols;
                (probs, symbol as usize)
            })
        }

        #[cfg(not(miri))]
        let amt = 1000;

        // We use different settings when testing on miri so that the test time stays reasonable.
        #[cfg(miri)]
        let amt = 100;

        let mut compressed = DefaultQueueEncoder::new();

        assert_eq!(compressed.len(), 0);
        compressed
            .encode_symbols(iter_probs_and_symbols(amt).map(|(probs, symbol)| {
                (
                    symbol,
                    EncoderHuffmanTree::from_probabilities::<u32, _>(&probs),
                )
            }))
            .unwrap();
        assert!(compressed.len() > amt);

        let mut decoder = compressed.into_decoder().unwrap_infallible();
        let reconstructed = decoder
            .decode_symbols(
                iter_probs_and_symbols(amt)
                    .map(|(probs, _)| DecoderHuffmanTree::from_probabilities::<u32, _>(&probs)),
            )
            .map(Result::unwrap);

        assert!(reconstructed.eq(iter_probs_and_symbols(amt).map(|(_, symbol)| symbol)));
        assert!(decoder.maybe_exhausted());
    }
}
