//! Sources and sinks of compressed data
//!
//! # TODO:
//!
//! - generizise `Range{En,De}Coder`.
//!
//! # Example
//!
//! TODO: turn this into a Range Coding example where both the encoder and the decoder
//! operate on the fly (that actually requires changing how the range coder deals with carry
//! bits, it has to become lazy).
//!
//! ```
//! use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
//! use constriction::stream::{stack::DefaultAnsCoder, models::DefaultLeakyQuantizer, Code, Decode};
//! use statrs::distribution::Normal;
//! use std::{fs::File, io::{BufReader, BufWriter}};
//!
//! fn encode_to_file(amt: u32) {
//!     // Some simple entropy model, just for demonstration purpose.
//!     let quantizer = DefaultLeakyQuantizer::new(-256..=255);
//!     let model = quantizer.quantize(Normal::new(0.0, 100.0).unwrap());
//!
//!     // Some long-ish sequence of test symbols, made up in a reproducible way.
//!     let symbols = (0..amt).map(|i| {
//!         let cheap_hash = i.wrapping_mul(0x6979_E2F3).wrapping_add(0x0059_0E91);
//!         (cheap_hash >> (32 - 9)) as i32 - 256
//!     });
//!
//!     // Encode (compress) the symbols.
//!     let mut encoder = DefaultAnsCoder::new();
//!     encoder.encode_iid_symbols_reverse(symbols, &model).unwrap();
//!     let compressed = encoder.into_compressed();
//!
//!     // Write the compressed words to a file *in reverse order* (using `.rev()`), so
//!     // that the resulting file can be decoded in normal order from front to back.
//!     let mut file = BufWriter::new(File::create("backend_example.tmp").unwrap());
//!     for &word in compressed.iter().rev() {
//!         file.write_u32::<LittleEndian>(word).unwrap();
//!     }
//! }
//!
//! fn decode_from_file_on_the_fly(amt: u32) {
//!     // Same toy entropy model that we used for encoding.
//!     let quantizer = DefaultLeakyQuantizer::new(-256..=255);
//!     let model = quantizer.quantize(Normal::new(0.0, 100.0).unwrap());
//!
//!     // Open the file and iterate over its contents in `u32` words (wrapping it in a `BufReader`
//!     // here isn't strictly necessary, it's just good practice when reading from a file).
//!     let mut file = BufReader::new(File::open("backend_example.tmp").unwrap());
//!     let word_iterator = std::iter::from_fn(move || file.read_u32::<LittleEndian>().ok());
//!
//!     // Create a decoder that consumes an iterator over compressed words. This `decoder` will
//!     // never keep the full compressed data in memory, it will just consume one compressed word
//!     // at a time whenever its small internal state underflows. Note that an `AnsCoder` that is
//!     // backed by an iterator only implements the `Decode` trait but not the `Encode` trait
//!     // because encoding additional data would require modifying the compressed data over which
//!     // the iterator iterates.
//!     let mut decoder = DefaultAnsCoder::from_reversed_compressed_iter(word_iterator).unwrap();
//!
//!     // Decode the symbols and verify their correctness.
//!     for (i, symbol) in decoder.decode_iid_symbols(amt as usize, &model).enumerate() {
//!         let cheap_hash = (i as u32).wrapping_mul(0x6979_E2F3).wrapping_add(0x0059_0E91);
//!         let expected = (cheap_hash >> (32 - 9)) as i32 - 256;
//!         assert_eq!(symbol.unwrap(), expected);
//!     }
//!     assert!(decoder.is_empty());
//!
//!     // Recover the original iterator over compressed words and verify that it's been exhausted.
//!     let mut word_iterator = decoder.into_raw_parts().0.into_iter();
//!     assert!(word_iterator.next().is_none());
//!
//!     // `word_iterator` owns the file since we used a `move` clausure above to construct it.
//!     // So dropping it calls `std::fs::File`'s destructor, which releases the file handle.
//!     std::mem::drop(word_iterator);
//!     std::fs::remove_file("backend_example.tmp").unwrap();
//! }
//!
//! encode_to_file(1000);
//! decode_from_file_on_the_fly(1000);
//! ```

use alloc::vec::Vec;
use core::{fmt::Debug, hint::unreachable_unchecked, marker::PhantomData};

// READ WRITE LOGICS ==========================================================

/// TODO: rename to `ReadWriteSemantics` or just `Semantics`.
pub trait Semantics {}

pub struct Stack;
impl Semantics for Stack {}

pub struct Queue;
impl Semantics for Queue {}

// MAIN TRAITS FOR CAPABILITIES OF BACKENDS ===================================

/// A trait for backends that write compressed words (used by encoders)
pub trait WriteBackend<Word> {
    /// TODO:
    /// - make this return a Result with associated error type (which may be infallible)
    /// - also introduce a `BoundedWrite` trait (and implement it for mut cursors, which
    ///   always write from left to right).
    fn write(&mut self, word: Word);

    fn extend(&mut self, iter: impl Iterator<Item = Word>) {
        for word in iter {
            self.write(word);
        }
    }
}

/// A trait for backends that read compressed words (used by decoders)
pub trait ReadBackend<Word, S: Semantics> {
    fn read(&mut self) -> Option<Word>;

    fn maybe_exhausted(&self) -> bool {
        true
    }
}

// A trait for read backends that know how much data is left.
pub trait BoundedReadBackend<Word, S: Semantics>: ReadBackend<Word, S> {
    // Returns the amount of data that's left for reading.
    fn remaining(&self) -> usize;

    /// TODO: don't forget to overwrite the default implementation of
    /// `Backend::maybe_empty`.
    #[inline(always)]
    fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }
}

// A trait for backends that keep track of their current position in the compressed data.
pub trait PosBackend<Word> {
    fn pos(&self) -> usize;
}

// A trait for backends that allow random access.
pub trait SeekBackend<Word> {
    fn seek(&mut self, pos: usize) -> Result<(), ()>;
}

// TRAITS FOR CONVERSIONS BETWEEN BACKENDS WITH DIFFERENT CAPABILITIES ========

pub trait IntoReadBackend<Word, S: Semantics> {
    type IntoReadBackend: ReadBackend<Word, S>;
    fn into_read_backend(self) -> Self::IntoReadBackend;
}

pub trait AsReadBackend<'a, Word, S: Semantics>: 'a {
    type AsReadBackend: ReadBackend<Word, S>;
    fn as_read_backend(&'a self) -> Self::AsReadBackend;
}

pub trait IntoSeekReadBackend<Word, S: Semantics> {
    type IntoSeekReadBackend: SeekBackend<Word> + ReadBackend<Word, S>;
    fn into_seek_read_backend(self) -> Self::IntoSeekReadBackend;
}

pub trait AsSeekReadBackend<'a, Word, S: Semantics>: 'a {
    type AsSeekReadBackend: SeekBackend<Word> + ReadBackend<Word, S>;
    fn as_seek_read_backend(&'a self) -> Self::AsSeekReadBackend;
}

// While neither `SeekBackend` nor `WriteBackend` are parameterized by a `ReadWriteLogic`,
// we do need a `ReadWriteLogic` type parameter here because we need to initialize the
// resulting backend correctly.
pub trait IntoSeekWriteBackend<Word, S: Semantics> {
    type IntoSeekWriteBackend: SeekBackend<Word> + WriteBackend<Word>;
    fn into_seek_write_backend(self) -> Self::IntoSeekWriteBackend;
}

// While neither `SeekBackend` nor `WriteBackend` are parameterized by a `ReadWriteLogic`,
// we do need a `ReadWriteLogic` type parameter here because we need to initialize the
// resulting backend correctly.
pub trait AsSeekWriteBackend<'a, Word, S: Semantics>: 'a {
    type AsSeekWriteBackend: SeekBackend<Word> + WriteBackend<Word>;
    fn as_seek_write_backend(&'a mut self) -> Self::AsSeekWriteBackend;
}

// IMPLEMENTATIONS FOR `Vec<Word>` ============================================

impl<Word> WriteBackend<Word> for Vec<Word> {
    #[inline(always)]
    fn write(&mut self, word: Word) {
        self.push(word)
    }
}

impl<Word> ReadBackend<Word, Stack> for Vec<Word> {
    #[inline(always)]
    fn read(&mut self) -> Option<Word> {
        self.pop()
    }

    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        self.is_empty()
    }
}

impl<Word> BoundedReadBackend<Word, Stack> for Vec<Word> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.len()
    }
}

impl<Word> PosBackend<Word> for Vec<Word> {
    fn pos(&self) -> usize {
        self.len()
    }
}

// ADAPTER FOR (SEMANTIC) REVERSING OF READING DIRECTION ======================

pub struct ReverseReads<Backend>(pub Backend);

impl<Word, B: WriteBackend<Word>> WriteBackend<Word> for ReverseReads<B> {
    #[inline(always)]
    fn write(&mut self, word: Word) {
        self.0.write(word)
    }
}

impl<Word, B: ReadBackend<Word, Stack>> ReadBackend<Word, Queue> for ReverseReads<B> {
    #[inline(always)]
    fn read(&mut self) -> Option<Word> {
        self.0.read()
    }

    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        self.0.maybe_exhausted()
    }
}

impl<Word, B: ReadBackend<Word, Queue>> ReadBackend<Word, Stack> for ReverseReads<B> {
    #[inline(always)]
    fn read(&mut self) -> Option<Word> {
        self.0.read()
    }

    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        self.0.maybe_exhausted()
    }
}

impl<Word, B: BoundedReadBackend<Word, Stack>> BoundedReadBackend<Word, Queue> for ReverseReads<B> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.0.remaining()
    }

    #[inline(always)]
    fn is_exhausted(&self) -> bool {
        self.0.is_exhausted()
    }
}

impl<Word, B: BoundedReadBackend<Word, Queue>> BoundedReadBackend<Word, Stack> for ReverseReads<B> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.0.remaining()
    }

    #[inline(always)]
    fn is_exhausted(&self) -> bool {
        self.0.is_exhausted()
    }
}

impl<Word, B: PosBackend<Word>> PosBackend<Word> for ReverseReads<B> {
    #[inline(always)]
    fn pos(&self) -> usize {
        self.0.pos()
    }
}

impl<Word, B: SeekBackend<Word>> SeekBackend<Word> for ReverseReads<B> {
    fn seek(&mut self, pos: usize) -> Result<(), ()> {
        self.0.seek(pos)
    }
}

// ADAPTER FOR IN-MEMORY BUFFERS ==============================================

#[derive(Clone, Debug)]
pub struct Cursor<Buf> {
    buf: Buf,

    /// The index of the next word to be read with a `ReadBackend<Word, Queue>` or written
    /// with a `WriteBackend<Word>, and one plus the index of the next word to read with
    /// `ReadBackend<Word, Stack>.
    ///
    /// Satisfies the invariant `pos <= buf.as_ref().len()` if `Buf: AsRef<[Word]>`.
    pos: usize,
}

impl<Buf> Cursor<Buf> {
    /// TODO: rename into `new_at_buf_start`
    #[inline(always)]
    pub fn new_at_write_beginning(buf: Buf) -> Self {
        Self { buf, pos: 0 }
    }

    /// TODO: rename into `new_at_buf_end`
    #[inline(always)]
    pub fn new_at_write_end<Word>(buf: Buf) -> Self
    where
        Buf: AsRef<[Word]>,
    {
        let pos = buf.as_ref().len();
        Self { buf, pos }
    }

    #[inline(always)]
    pub fn new_at_write_end_mut<Word>(mut buf: Buf) -> Self
    where
        Buf: AsMut<[Word]>,
    {
        let pos = buf.as_mut().len();
        Self { buf, pos }
    }

    pub fn with_buf_and_pos<Word>(buf: Buf, pos: usize) -> Result<Self, ()>
    where
        Buf: AsRef<[Word]>,
    {
        if pos > buf.as_ref().len() {
            Err(())
        } else {
            Ok(Self { buf, pos })
        }
    }

    /// Same as `with_buf_and_pos` except for trait bound. For `Buf`s that implement `AsMut`
    /// but not `AsRef`.
    pub fn with_buf_and_pos_mut<Word>(mut buf: Buf, pos: usize) -> Result<Self, ()>
    where
        Buf: AsMut<[Word]>,
    {
        if pos > buf.as_mut().len() {
            Err(())
        } else {
            Ok(Self { buf, pos })
        }
    }

    pub fn as_view<Word>(&self) -> Cursor<&[Word]>
    where
        Buf: AsRef<[Word]>,
    {
        Cursor {
            buf: self.buf.as_ref(),
            pos: self.pos,
        }
    }

    pub fn as_mut_view<Word>(&mut self) -> Cursor<&mut [Word]>
    where
        Buf: AsMut<[Word]>,
    {
        Cursor {
            buf: self.buf.as_mut(),
            pos: self.pos,
        }
    }

    pub fn cloned<Word: Clone>(&self) -> Cursor<Vec<Word>>
    where
        Buf: AsRef<[Word]>,
    {
        Cursor {
            buf: self.buf.as_ref().to_vec(),
            pos: self.pos,
        }
    }

    pub fn buf(&self) -> &Buf {
        &self.buf
    }

    pub fn into_buf_and_pos(self) -> (Buf, usize) {
        (self.buf, self.pos)
    }

    /// Reverses both the data and the reading direction.
    ///
    /// This method consumes the original `ReadCursor`, reverses the order of the
    /// `Word`s in-place, updates the cursor position accordingly, and returns a
    /// `ReadCursor` that progresses in the opposite direction. Reading from the
    /// returned `ReadCursor` will yield the same `Word`s as continued reading from the
    /// original one would, but the changed direction will be observable via different
    /// behavior of [`Pos::pos`], [`Seek::seek`], and [`Self::buf`].
    pub fn into_reversed<Word>(mut self) -> ReverseReads<Self>
    where
        Buf: AsMut<[Word]>,
    {
        self.buf.as_mut().reverse();
        self.pos = self.buf.as_mut().len() - self.pos;
        ReverseReads(self)
    }
}

impl<Buf> ReverseReads<Cursor<Buf>> {
    pub fn into_reversed<Word>(self) -> Cursor<Buf>
    where
        Buf: AsMut<[Word]>,
    {
        // Accessing `.0` twice removes *two* `ReverseReads`, resulting in no semantic change.
        self.0.into_reversed().0
    }
}

impl<Word, Buf: AsMut<[Word]>> WriteBackend<Word> for Cursor<Buf> {
    #[inline(always)]
    fn write(&mut self, word: Word) {
        todo!()
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>> ReadBackend<Word, Stack> for Cursor<Buf> {
    #[inline(always)]
    fn read(&mut self) -> Option<Word> {
        if self.pos == 0 {
            None
        } else {
            self.pos -= 1;
            unsafe {
                // SAFETY: We maintain the invariant `self.pos <= self.buf.as_ref().len()`
                // and we just decreased `self.pos` (and made sure that didn't wrap around),
                // so we now have `self.pos < self.buf.as_ref().len()`.
                Some(self.buf.as_ref().get_unchecked(self.pos).clone())
            }
        }
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>> ReadBackend<Word, Queue> for Cursor<Buf> {
    #[inline(always)]
    fn read(&mut self) -> Option<Word> {
        let word = self.buf.as_ref().get(self.pos)?.clone();
        self.pos += 1;
        Some(word)
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>> BoundedReadBackend<Word, Stack> for Cursor<Buf> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.pos
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>> BoundedReadBackend<Word, Queue> for Cursor<Buf> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.buf.as_ref().len() - self.pos
    }
}

impl<Word, Buf: AsRef<[Word]>> PosBackend<Word> for Cursor<Buf> {
    #[inline(always)]
    fn pos(&self) -> usize {
        self.pos
    }
}

impl<Word, Buf: AsRef<[Word]>> SeekBackend<Word> for Cursor<Buf> {
    #[inline(always)]
    fn seek(&mut self, pos: usize) -> Result<(), ()> {
        if pos > self.buf.as_ref().len() {
            // Note that `pos == buf.len()` is still a valid position (EOF for queues and
            // beginning for stacks).
            Err(())
        } else {
            self.pos = pos;
            Ok(())
        }
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>> IntoReadBackend<Word, Stack> for Buf {
    type IntoReadBackend = Cursor<Buf>;

    fn into_read_backend(self) -> Self::IntoReadBackend {
        Cursor::new_at_write_end(self)
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>> IntoReadBackend<Word, Queue> for Buf {
    type IntoReadBackend = Cursor<Buf>;

    fn into_read_backend(self) -> Self::IntoReadBackend {
        Cursor::new_at_write_beginning(self)
    }
}

impl<'a, Word: Clone + 'a, Buf: AsRef<[Word]> + 'a> AsReadBackend<'a, Word, Stack> for Buf {
    type AsReadBackend = Cursor<&'a [Word]>;

    fn as_read_backend(&'a self) -> Self::AsReadBackend {
        Cursor::new_at_write_end(self.as_ref())
    }
}

impl<'a, Word: Clone + 'a, Buf: AsRef<[Word]> + 'a> AsReadBackend<'a, Word, Queue> for Buf {
    type AsReadBackend = Cursor<&'a [Word]>;

    fn as_read_backend(&'a self) -> Self::AsReadBackend {
        Cursor::new_at_write_beginning(self.as_ref())
    }
}

impl<Word, Buf, S: Semantics> IntoSeekReadBackend<Word, S> for Buf
where
    Buf: AsRef<[Word]> + IntoReadBackend<Word, S, IntoReadBackend = Cursor<Buf>>,
    Cursor<Buf>: ReadBackend<Word, S>,
{
    type IntoSeekReadBackend = Cursor<Buf>;

    fn into_seek_read_backend(self) -> Self::IntoSeekReadBackend {
        self.into_read_backend()
    }
}

impl<'a, Word: 'a, Buf, S: Semantics> AsSeekReadBackend<'a, Word, S> for Buf
where
    Buf: AsReadBackend<'a, Word, S, AsReadBackend = Cursor<&'a [Word]>>,
    Cursor<&'a [Word]>: ReadBackend<Word, S>,
{
    type AsSeekReadBackend = Cursor<&'a [Word]>;

    fn as_seek_read_backend(&'a self) -> Self::AsSeekReadBackend {
        self.as_read_backend()
    }
}

impl<Word: Clone, Buf: AsRef<[Word]> + AsMut<[Word]>> IntoSeekWriteBackend<Word, Stack> for Buf {
    type IntoSeekWriteBackend = Cursor<Buf>;

    fn into_seek_write_backend(self) -> Self::IntoSeekWriteBackend {
        Cursor::new_at_write_end_mut(self)
    }
}

impl<Word: Clone, Buf: AsRef<[Word]> + AsMut<[Word]>> IntoSeekWriteBackend<Word, Queue> for Buf {
    type IntoSeekWriteBackend = Cursor<Buf>;

    fn into_seek_write_backend(self) -> Self::IntoSeekWriteBackend {
        Cursor::new_at_write_beginning(self)
    }
}

impl<'a, Word: Clone + 'a, Buf: AsMut<[Word]> + 'a> AsSeekWriteBackend<'a, Word, Stack> for Buf {
    type AsSeekWriteBackend = Cursor<&'a mut [Word]>;

    fn as_seek_write_backend(&'a mut self) -> Self::AsSeekWriteBackend {
        Cursor::new_at_write_end_mut(self.as_mut())
    }
}

impl<'a, Word: Clone + 'a, Buf: AsMut<[Word]> + 'a> AsSeekWriteBackend<'a, Word, Queue> for Buf {
    type AsSeekWriteBackend = Cursor<&'a mut [Word]>;

    fn as_seek_write_backend(&'a mut self) -> Self::AsSeekWriteBackend {
        Cursor::new_at_write_beginning(self.as_mut())
    }
}

// ADAPTER FOR ITERATORS ======================================================

#[derive(Clone, Debug)]
pub struct IteratorBackend<Iter: Iterator> {
    inner: core::iter::Fuse<Iter>,
}

impl<Iter: Iterator> IteratorBackend<Iter> {
    pub fn new(iter: Iter) -> Self {
        Self { inner: iter.fuse() }
    }
}

impl<Iter: Iterator> IntoIterator for IteratorBackend<Iter> {
    type Item = Iter::Item;
    type IntoIter = core::iter::Fuse<Iter>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }
}

/// Since `IteratorBackend` doesn't implement `WriteBackend`, it is allowed to implement
/// `ReadBackend` for all `ReadWriteLogic`s
impl<Iter: Iterator, S: Semantics> ReadBackend<Iter::Item, S> for IteratorBackend<Iter> {
    #[inline(always)]
    fn read(&mut self) -> Option<Iter::Item> {
        self.inner.next()
    }
}

impl<Iter: ExactSizeIterator, S: Semantics> BoundedReadBackend<Iter::Item, S>
    for IteratorBackend<Iter>
{
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.inner.len()
    }
}
