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
use core::{fmt::Debug, marker::PhantomData};

pub trait Backend<Word> {
    fn maybe_empty(&self) -> bool {
        true
    }
}

/// A trait for backends that read compressed words (used by decoders)
///
/// Encoders should not use this trait directly to bound type parameters. Instead, they
/// should bound type arguments by one of the more specific traits [`ReadStackBackend`] or
/// [`ReadQueueBackend`], or add their own more specific trait if they operate neither as a
/// stack nor as a queue.
pub trait ReadBackend<Word>: Backend<Word> {
    fn read(&mut self) -> Option<Word>;
}

pub trait LookaheadBackend<Word>: ReadBackend<Word> + Backend<Word> {
    fn amt_left(&self) -> usize;

    /// TODO: don't forget to overwrite the default implementation of
    /// `Backend::maybe_empty`.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.amt_left() == 0
    }
}

pub trait SeekBackend<Word>: Backend<Word> {
    fn seek(&mut self, pos: usize, must_be_end: bool) -> Result<(), ()>;
}

pub trait PosBackend<Word>: Backend<Word> {
    fn pos(&self) -> usize;
}

/// A trait for backends that write compressed words (used by encoders)
///
/// # Consistency Rule
///
/// No type may implement more than two of the following three traits:
///
/// - [`ReadStackBackend`]
/// - ReadQueueBackend (this trait)
/// - [`WriteBackend`]
///
/// In other words, a backend may either support both reading and writing, in which case
/// interleaved reads and writes cannot have both stack and queue semantics at the same time
/// (thus implementing all three of the above traits would be illogical); or the backend may
/// be read-only or write-only, in which case the distinction between stack and queue
/// semantics is moot. (One might argue that stack and queue semantics are consistent with
/// each other if `Word` is a zero-sized type but the use case for zero-sized `Word`s is
/// unclear.)
pub trait WriteBackend<Word>: core::iter::Extend<Word> + Backend<Word> {
    fn write(&mut self, item: Word);
}

/// Backends that can mutate already existing words
///
/// Actually, we only require the possibility to clear everything.
pub trait MutBackend<Word>: WriteBackend<Word> {
    fn clear(&mut self);
}

/// Marker trait for a [`ReadBackend`] that either doesn't implement [`WriteBackend`] or
/// that satisfies stack semantics.
///
/// # Consistency Rule
///
/// No type may implement more than two of the following three traits:
///
/// - ReadStackBackend (this trait)
/// - [`ReadQueueBackend`]
/// - [`WriteBackend`]
///
/// See [`WriteBackend`] for an explanation of this rule.
pub trait ReadStackBackend<Word>: ReadBackend<Word> {}

/// Marker trait for a [`ReadBackend`] that either doesn't implement [`WriteBackend`] or
/// that satisfies queue semantics.
///
/// # Consistency Rule
///
/// No type may implement more than two of the following three traits:
///
/// - [`ReadStackBackend`]
/// - ReadQueueBackend (this trait)
/// - [`WriteBackend`]
///
/// See [`WriteBackend`] for an explanation of this rule.
pub trait ReadQueueBackend<Word>: ReadBackend<Word> {}

pub trait AsReadStackBackend<'a, Word>: 'a {
    type AsReadStackBackend: ReadStackBackend<Word> + From<&'a Self>;

    fn as_read_stack_backend(&'a self) -> Self::AsReadStackBackend {
        self.into()
    }
}

impl<'a, Word: Clone + 'a> AsReadStackBackend<'a, Word> for Vec<Word> {
    type AsReadStackBackend = ReadCursorBackward<Word, &'a [Word]>;
}

impl<'a, Word: 'a> From<&'a Vec<Word>> for ReadCursorBackward<Word, &'a [Word]> {
    fn from(buf: &'a Vec<Word>) -> Self {
        Self::new(buf)
    }
}

impl<Word> Backend<Word> for Vec<Word> {
    fn maybe_empty(&self) -> bool {
        self.is_empty()
    }
}

impl<Word> ReadBackend<Word> for Vec<Word> {
    #[inline(always)]
    fn read(&mut self) -> Option<Word> {
        self.pop()
    }
}

/// `Vec` implements `ReadStackBackend` and `WriteBackend`, therefore it is not allowed to
/// implement `ReadQueueBackend` (which it can't anyway). If you want to read from a `Vec`
/// as a queue, wrap either the `Vec` itself or the slice that it dereferences to in a
/// `ReadCursorForward`.
impl<Word> ReadStackBackend<Word> for Vec<Word> {}

impl<Word> LookaheadBackend<Word> for Vec<Word> {
    #[inline(always)]
    fn amt_left(&self) -> usize {
        self.len()
    }
}

impl<Word> WriteBackend<Word> for Vec<Word> {
    #[inline(always)]
    fn write(&mut self, item: Word) {
        self.push(item)
    }
}

impl<Word> MutBackend<Word> for Vec<Word> {
    fn clear(&mut self) {
        self.clear()
    }
}

impl<Word> PosBackend<Word> for Vec<Word> {
    fn pos(&self) -> usize {
        self.len()
    }
}

pub trait Direction: 'static {
    const FORWARD: bool;
    type Reverse: Direction;
}

#[derive(Debug, Clone)]
pub struct Forward;

#[derive(Debug, Clone)]
pub struct Backward;

impl Direction for Forward {
    const FORWARD: bool = true;
    type Reverse = Backward;
}

impl Direction for Backward {
    const FORWARD: bool = false;
    type Reverse = Forward;
}

#[derive(Clone, Debug)]
pub struct ReadFromIterBackend<Iter: Iterator> {
    inner: Iter,
}

impl<Iter: Iterator> ReadFromIterBackend<Iter> {
    pub fn new(inner: Iter) -> Self {
        Self { inner }
    }
}

impl<Iter: Iterator> IntoIterator for ReadFromIterBackend<Iter> {
    type Item = Iter::Item;
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }
}

impl<Iter: Iterator> Backend<Iter::Item> for ReadFromIterBackend<Iter> {}

impl<Iter: Iterator> ReadBackend<Iter::Item> for ReadFromIterBackend<Iter> {
    fn read(&mut self) -> Option<Iter::Item> {
        self.inner.next()
    }
}

impl<Iter: Iterator> ReadStackBackend<Iter::Item> for ReadFromIterBackend<Iter> {}
impl<Iter: Iterator> ReadQueueBackend<Iter::Item> for ReadFromIterBackend<Iter> {}

impl<Iter: ExactSizeIterator> LookaheadBackend<Iter::Item> for ReadFromIterBackend<Iter> {
    fn amt_left(&self) -> usize {
        self.inner.len()
    }
}

#[derive(Clone)]
pub struct ReadCursor<Word, Buf: AsRef<[Word]>, Dir: Direction> {
    buf: Buf,

    /// If `Dir::FORWARD`: the index of the next item to be read.
    /// else: one plus the index of the next item to read.
    ///
    /// In both cases: satisfies invariant `pos <= buf.as_ref().len()`.
    pos: usize,

    phantom: PhantomData<(Word, Dir)>,
}

pub type ReadCursorForward<Word, Buf> = ReadCursor<Word, Buf, Forward>;
pub type ReadCursorBackward<Word, Buf> = ReadCursor<Word, Buf, Backward>;

impl<Word, Buf: AsRef<[Word]>, Dir: Direction> ReadCursor<Word, Buf, Dir> {
    #[inline(always)]
    pub fn new(buf: Buf) -> Self {
        let pos = if Dir::FORWARD { 0 } else { buf.as_ref().len() };
        Self {
            buf,
            pos,
            phantom: PhantomData,
        }
    }

    pub fn with_buf_and_pos(buf: Buf, pos: usize) -> Result<Self, ()> {
        if pos > buf.as_ref().len() {
            Err(())
        } else {
            Ok(Self {
                buf,
                pos,
                phantom: PhantomData,
            })
        }
    }

    pub fn as_view(&self) -> ReadCursor<Word, &[Word], Dir> {
        ReadCursor {
            buf: self.buf.as_ref(),
            pos: self.pos,
            phantom: PhantomData,
        }
    }

    pub fn cloned(&self) -> ReadCursor<Word, Vec<Word>, Dir>
    where
        Word: Clone,
    {
        ReadCursor {
            buf: self.buf.as_ref().to_vec(),
            pos: self.pos,
            phantom: PhantomData,
        }
    }

    pub fn buf(&self) -> &[Word] {
        self.buf.as_ref()
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
    pub fn into_reversed(self) -> ReadCursor<Word, Buf, Dir::Reverse>
    where
        Buf: AsMut<[Word]>,
    {
        let ReadCursor {
            mut buf, mut pos, ..
        } = self;

        buf.as_mut().reverse();
        pos = buf.as_ref().len() - pos;

        ReadCursor {
            buf,
            pos,
            phantom: PhantomData,
        }
    }
}

impl<'a, Word: Clone, Buf: AsRef<[Word]>, Dir: Direction> IntoIterator
    for &'a ReadCursor<Word, Buf, Dir>
{
    type Item = Word;
    type IntoIter = core::iter::Cloned<core::slice::Iter<'a, Word>>;

    fn into_iter(self) -> Self::IntoIter {
        let slice = unsafe {
            // SAFETY: We maintain the invariant `self.pos <= self.buf.len()`.
            if Dir::FORWARD {
                self.buf.as_ref().get_unchecked(self.pos..)
            } else {
                self.buf.as_ref().get_unchecked(..self.pos)
            }
        };

        slice.iter().cloned()
    }
}

impl<Word: Clone + Debug, Buf: AsRef<[Word]>, Dir: Direction> Debug for ReadCursor<Word, Buf, Dir> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>, Dir: Direction> Backend<Word> for ReadCursor<Word, Buf, Dir> {
    fn maybe_empty(&self) -> bool {
        self.is_empty()
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>, Dir: Direction> ReadBackend<Word>
    for ReadCursor<Word, Buf, Dir>
{
    #[inline(always)]
    fn read(&mut self) -> Option<Word> {
        if Dir::FORWARD {
            let item = self.buf.as_ref().get(self.pos)?.clone();
            self.pos += 1;
            Some(item)
        } else {
            if self.pos == 0 {
                None
            } else {
                self.pos -= 1;
                unsafe {
                    // SAFETY: We maintain the invariant `self.pos <=self.buf.as_ref().len()` and we
                    // just decreased `self.pos` (making sure it doesn't wrap around), so we now have
                    // `self.pos < self.buf.as_ref().len()`.
                    Some(self.buf.as_ref().get_unchecked(self.pos).clone())
                }
            }
        }
    }
}

/// Both `ReadStackBackend` and `ReadQueueBacken` are implemented for `ReadCursor`s of both
/// directions because the reading direction is not tied to the stack vs. queue nature.
/// What's important is that reading and writing directions are compatible, which is
/// trivially given since `ReadCursor`s don't implement `WriteBackend`.
impl<Word: Clone, Buf: AsRef<[Word]>, Dir: Direction> ReadStackBackend<Word>
    for ReadCursor<Word, Buf, Dir>
{
}

impl<Word: Clone, Buf: AsRef<[Word]>, Dir: Direction> ReadQueueBackend<Word>
    for ReadCursor<Word, Buf, Dir>
{
}

impl<'a, Word: Clone + 'a, Buf: AsRef<[Word]> + 'a, Dir: Direction> AsReadStackBackend<'a, Word>
    for ReadCursor<Word, Buf, Dir>
{
    type AsReadStackBackend = ReadCursor<Word, &'a [Word], Dir>;
}

impl<'a, Word: 'a, Buf: AsRef<[Word]> + 'a, Dir: Direction> From<&'a ReadCursor<Word, Buf, Dir>>
    for ReadCursor<Word, &'a [Word], Dir>
{
    fn from(cursor: &'a ReadCursor<Word, Buf, Dir>) -> Self {
        cursor.as_view()
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>, Dir: Direction> LookaheadBackend<Word>
    for ReadCursor<Word, Buf, Dir>
{
    fn amt_left(&self) -> usize {
        if Dir::FORWARD {
            // This cannot underflow since we maintain the invariant `pos >= buf.as_ref().len()`.
            self.buf.as_ref().len() - self.pos
        } else {
            self.pos
        }
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>, Dir: Direction> SeekBackend<Word>
    for ReadCursor<Word, Buf, Dir>
{
    fn seek(&mut self, pos: usize, must_be_end: bool) -> Result<(), ()> {
        let end_pos = if Dir::FORWARD {
            self.buf.as_ref().len()
        } else {
            0
        };

        if pos > self.buf.as_ref().len() || (must_be_end && pos != end_pos) {
            Err(())
        } else {
            self.pos = pos;
            Ok(())
        }
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>, Dir: Direction> PosBackend<Word>
    for ReadCursor<Word, Buf, Dir>
{
    fn pos(&self) -> usize {
        self.pos
    }
}
