//! Sources and sinks of compressed data
//!
//! # TODO:
//!
//! - move up one level in the module hierarchy
//! - rename `pop` to `read` and `push` to `write`
//! - rename `ReadItems` and `WriteItems` to `ReadWords` and `WriteWords`.
//! - add empty marker traits `ReadWordsStack: ReadWords` and `ReadWordsQueue: ReadWords`
//!   that are implemented for types that either implement only reading, or that implement
//!   both in reading and writing in a way that's consistent with a stack or queue,
//!   respectively. All decoder operations then depend either on `StackRead` or on
//!   `QueueRead`, never directly on `Read`.
//! - reading always goes in just one direction, even if a type implements both `StackRead`
//!   and `QueueRead`
//! - generizise `Range{En,De}Coder`.
//!
//! Note: `Vec` should not implement `Queue`.
//!
//! # Example
//!
//! TODO: turn this into a Range Coding example where both the encoder and the decoder
//! operate on the fly.
//!
//! ```
//! use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
//! use constriction::stream::{ans::DefaultAnsCoder, models::DefaultLeakyQuantizer, Code, Decode};
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
//!     let mut word_iterator = decoder.into_buf_and_state().0.into_iter();
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

/// TODO: document that, if a `Backend<Item>` implements `AsRef<[Item]>` then the
/// head of the stack must be at the end of the slice returned by `as_ref`.
/// (this condition will probably only apply to `ReadWordsStack`)
pub trait Backend<Item> {}

pub trait ReadItems<Item>: Backend<Item> {
    fn pop(&mut self) -> Option<Item>;
}

pub trait ReadLookaheadItems<Item>: Backend<Item> {
    fn amt_left(&self) -> usize;

    #[inline(always)]
    fn is_at_end(&self) -> bool {
        self.amt_left() == 0
    }
}

pub trait Seek<Item>: Backend<Item> {
    fn seek(&mut self, pos: usize, must_be_end: bool) -> Result<(), ()>;
}

pub trait Pos<Item>: Backend<Item> {
    fn pos(&self) -> usize;
}

pub trait WriteItems<Item>: Backend<Item> + core::iter::Extend<Item> {
    fn push(&mut self, item: Item);
}

pub trait WriteMutableItems<Item>: WriteItems<Item> {
    fn clear(&mut self);
}

impl<Item> Backend<Item> for Vec<Item> {}

impl<Item> ReadItems<Item> for Vec<Item> {
    #[inline(always)]
    fn pop(&mut self) -> Option<Item> {
        self.pop()
    }
}

impl<Item> ReadLookaheadItems<Item> for Vec<Item> {
    #[inline(always)]
    fn amt_left(&self) -> usize {
        self.len()
    }
}

impl<Item> WriteItems<Item> for Vec<Item> {
    #[inline(always)]
    fn push(&mut self, item: Item) {
        self.push(item)
    }
}

impl<Item> WriteMutableItems<Item> for Vec<Item> {
    fn clear(&mut self) {
        self.clear()
    }
}

impl<Item> Pos<Item> for Vec<Item> {
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
pub struct ReadFromIter<Iter: Iterator> {
    inner: Iter,
}

impl<Iter: Iterator> ReadFromIter<Iter> {
    pub fn new(inner: Iter) -> Self {
        Self { inner }
    }
}

impl<Iter: Iterator> IntoIterator for ReadFromIter<Iter> {
    type Item = Iter::Item;
    type IntoIter = Iter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }
}

impl<Iter: Iterator> Backend<Iter::Item> for ReadFromIter<Iter> {}

impl<Iter: Iterator> ReadItems<Iter::Item> for ReadFromIter<Iter> {
    fn pop(&mut self) -> Option<Iter::Item> {
        self.inner.next()
    }
}

impl<Iter: ExactSizeIterator> ReadLookaheadItems<Iter::Item> for ReadFromIter<Iter> {
    fn amt_left(&self) -> usize {
        self.inner.len()
    }
}

#[derive(Clone)]
pub struct ReadCursor<Item, Buf: AsRef<[Item]>, Dir: Direction> {
    buf: Buf,

    /// If `Dir::FORWARD`: the index of the next item to be read.
    /// else: one plus the index of the next item to read.
    ///
    /// In both cases: satisfies invariant `pos <= buf.as_ref().len()`.
    pos: usize,

    phantom: PhantomData<(Item, Dir)>,
}

pub type ReadCursorForward<Item, Buf> = ReadCursor<Item, Buf, Forward>;
pub type ReadCursorBackward<Item, Buf> = ReadCursor<Item, Buf, Backward>;

impl<Item, Buf: AsRef<[Item]>, Dir: Direction> ReadCursor<Item, Buf, Dir> {
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

    pub fn as_view(&self) -> ReadCursor<Item, &[Item], Dir> {
        ReadCursor {
            buf: self.buf.as_ref(),
            pos: self.pos,
            phantom: PhantomData,
        }
    }

    pub fn cloned(&self) -> ReadCursor<Item, Vec<Item>, Dir>
    where
        Item: Clone,
    {
        ReadCursor {
            buf: self.buf.as_ref().to_vec(),
            pos: self.pos,
            phantom: PhantomData,
        }
    }

    pub fn buf(&self) -> &[Item] {
        self.buf.as_ref()
    }

    pub fn into_buf_and_pos(self) -> (Buf, usize) {
        (self.buf, self.pos)
    }

    /// Reverses both the data and the reading direction.
    ///
    /// This method consumes the original `ReadCursor`, reverses the order of the
    /// `Item`s in-place, updates the cursor position accordingly, and returns a
    /// `ReadCursor` that progresses in the opposite direction. Reading from the
    /// returned `ReadCursor` will yield the same `Item`s as continued reading from the
    /// original one would, but the changed direction will be observable via different
    /// behavior of [`Pos::pos`], [`Seek::seek`], and [`Self::buf`].
    pub fn into_reversed(self) -> ReadCursor<Item, Buf, Dir::Reverse>
    where
        Buf: AsMut<[Item]>,
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

impl<'a, Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> IntoIterator
    for &'a ReadCursor<Item, Buf, Dir>
{
    type Item = Item;
    type IntoIter = core::iter::Cloned<core::slice::Iter<'a, Item>>;

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

impl<Item: Clone + Debug, Buf: AsRef<[Item]>, Dir: Direction> Debug for ReadCursor<Item, Buf, Dir> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<Item, Buf: AsRef<[Item]>, Dir: Direction> Backend<Item> for ReadCursor<Item, Buf, Dir> {}

impl<Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> ReadItems<Item>
    for ReadCursor<Item, Buf, Dir>
{
    #[inline(always)]
    fn pop(&mut self) -> Option<Item> {
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

impl<Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> ReadLookaheadItems<Item>
    for ReadCursor<Item, Buf, Dir>
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

impl<Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> Seek<Item> for ReadCursor<Item, Buf, Dir> {
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

impl<Item: Clone, Buf: AsRef<[Item]>, Dir: Direction> Pos<Item> for ReadCursor<Item, Buf, Dir> {
    fn pos(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod test {
    use std::{
        fs::File,
        io::{BufReader, BufWriter},
    };

    use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

    use super::super::{
        super::{models::DefaultLeakyQuantizer, Code, Decode},
        DefaultAnsCoder,
    };
    use statrs::distribution::Normal;

    #[test]
    fn decode_on_the_fly() {
        fn encode_to_file(amt: u32) {
            // Some simple entropy model, just for demonstration purpose:
            let quantizer = DefaultLeakyQuantizer::new(-256..=255);
            let model = quantizer.quantize(Normal::new(0.0, 10.0).unwrap());

            // Some long-ish sequence of test symbols, made up in a reproducible way:
            let symbols = (0..amt).map(|i| {
                let cheap_hash = i.wrapping_mul(0x6979_E2F3).wrapping_add(0x0059_0E91);
                (cheap_hash >> (32 - 9)) as i32 - 256
            });

            // Compress the data:
            let mut encoder = DefaultAnsCoder::new();
            encoder.encode_iid_symbols_reverse(symbols, &model).unwrap();
            let compressed = encoder.into_compressed();

            // Write the compressed words to a file *in reverse order* (using `.rev()`), so
            // that the resulting file can be decoded from front to back:
            let mut file = BufWriter::new(File::create("backend_example.tmp").unwrap());
            for &word in compressed.iter().rev() {
                file.write_u32::<LittleEndian>(word).unwrap();
            }
        }

        fn decode_from_file_on_the_fly(amt: u32) {
            // Same toy entropy model as we used for encoding:
            let quantizer = DefaultLeakyQuantizer::new(-256..=255);
            let model = quantizer.quantize(Normal::new(0.0, 10.0).unwrap());

            // Open the file and iterate over its contents in `u32` words.
            let mut file = BufReader::new(File::open("backend_example.tmp").unwrap());
            let word_iterator = std::iter::from_fn(move || file.read_u32::<LittleEndian>().ok());

            // Create a decoder that consumes an iterator over compressed words. This
            // decoder will never keep the full compressed data in memory, it will just
            // consume one compressed word at a time whenever its small internal state
            // underflows. Note that a `AnsCoder` that is backed by an iterator only
            // implements the `Decode` trait but not the `Encode` trait because encoding new
            // data would require modifying the data over which the iterator iterates. as
            // long as it iterates in the correct order for *decoding* (i.e., in opposite
            // direction compared to the encoding direction). The resulting `AnsCoder` can
            // only be used for *decoding*, it doesn't implement the `Encode` trait.
            let mut decoder =
                DefaultAnsCoder::from_reversed_compressed_iter(word_iterator).unwrap();

            for (i, symbol) in decoder.decode_iid_symbols(amt as usize, &model).enumerate() {
                let cheap_hash = (i as u32)
                    .wrapping_mul(0x6979_E2F3)
                    .wrapping_add(0x0059_0E91);
                let expected = (cheap_hash >> (32 - 9)) as i32 - 256;
                assert_eq!(symbol.unwrap(), expected);
            }

            assert!(decoder.maybe_empty());
            // Recover the original iterator over compressed words and verify that it has been exhausted.
            let (word_iterator, _) = decoder.into_buf_and_state();
            assert!(word_iterator.into_iter().next().is_none());
        }

        encode_to_file(1000);
        decode_from_file_on_the_fly(1000);
        std::fs::remove_file("backend_example.tmp").unwrap();
    }
}
