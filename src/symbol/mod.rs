pub mod huffman;

use smallvec::SmallVec;

use alloc::vec::Vec;
use core::{iter::FromIterator, ops::DerefMut};

use crate::{BitArray, EncodingError};

#[derive(Debug)]
pub enum DecodingError {
    OutOfCompressedData,
}

trait EncoderCodebook {
    type BitIterator: Iterator<Item = bool>;

    fn encode_symbol(&self, symbol: usize) -> Result<Self::BitIterator, EncodingError>;

    fn num_symbols(&self) -> usize;
}

trait DecoderCodebook {
    fn decode_symbol(&self, source: impl Iterator<Item = bool>) -> Result<usize, DecodingError>;

    fn num_symbols(&self) -> usize;
}

pub trait GenericVec<T>: Default + DerefMut<Target = [T]> {
    fn with_capacity(capacity: usize) -> Self;
    fn push(&mut self, x: T);
    fn pop(&mut self) -> Option<T>;
    fn clear(&mut self);
    fn resize_with(&mut self, new_len: usize, f: impl FnMut() -> T);

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct BitVec<W: BitArray, V: GenericVec<W> = Vec<W>> {
    buf: V,

    /// A `BitArray` with at most one set bit, satisfying the following invariant:
    /// - if the `BitVec` is empty then `mask_last_written == W::zero()` and `buf` is
    ///   empty;
    /// - otherwise, `mask_last_written` has a single set bit, `buf` is not empty, and
    ///   the last `push`ed bit is equal to
    ///   `*buf.last().unwrap() & mask_last_written != W::zero()`.
    mask_last_written: W,
}

#[derive(Debug)]
pub struct BitVecReverseIterator<W: BitArray, V: GenericVec<W> = Vec<W>> {
    inner: BitVec<W, V>,
}

type SmallBitVec<W> = BitVec<W, SmallVec<[W; 1]>>;
type SmallBitVecReverseIterator<W> = BitVecReverseIterator<W, SmallVec<[W; 1]>>;

impl<W: BitArray, V: GenericVec<W>> BitVec<W, V> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: V::with_capacity((capacity + W::BITS - 1) / W::BITS),
            mask_last_written: W::zero(),
        }
    }

    pub fn push(&mut self, bit: bool) {
        let write_mask = self.mask_last_written << 1;
        self.mask_last_written = if write_mask != W::zero() {
            if bit {
                let last_word = self
                    .buf
                    .last_mut()
                    .expect("buf is not empty since mask_last_written != 0.");
                *last_word = *last_word | write_mask;
            };
            write_mask
        } else {
            self.buf.push(if bit { W::one() } else { W::zero() });
            W::one()
        };
    }

    pub fn pop(&mut self) -> Option<bool> {
        let old_mask = self.mask_last_written;
        let new_mask = old_mask >> 1;

        let bit = if new_mask != W::zero() {
            // Most common case, therefore reachable with only a single branch.
            self.mask_last_written = new_mask;
            let last_word = self
                .buf
                .last_mut()
                .expect("`old_mask != 0`, so `buf` is not empty.");
            let bit = *last_word & old_mask;
            *last_word = *last_word ^ bit;
            bit
        } else if old_mask != W::zero() {
            let last_word = self
                .buf
                .pop()
                .expect("`old_mask != 0`, so `buf` is not empty.");
            self.mask_last_written = if self.buf.is_empty() {
                W::zero()
            } else {
                W::one() << (W::BITS - 1)
            };
            last_word & old_mask
        } else {
            return None;
        };

        Some(bit != W::zero())
    }

    pub fn is_empty(&self) -> bool {
        self.mask_last_written == W::zero()
    }

    pub fn len(&self) -> usize {
        let capacity = self.buf.len() * W::BITS;
        let unused = if self.mask_last_written == W::zero() {
            0
        } else {
            self.mask_last_written.leading_zeros()
        };

        capacity - unused as usize
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.mask_last_written = W::zero();
    }

    pub fn into_iter_reverse(self) -> BitVecReverseIterator<W, V> {
        BitVecReverseIterator { inner: self }
    }

    /// TODO: test
    pub fn discard(&mut self, amt: usize) -> Result<(), ()> {
        let mut num_words = amt / W::BITS;
        let remainder = amt % W::BITS;

        let old_mask = self.mask_last_written;
        let mut new_mask = old_mask >> remainder;
        if new_mask == W::zero() {
            num_words += 1;
            new_mask = old_mask << (W::BITS - remainder);
        }

        if let Some(new_len) = self.buf.len().checked_sub(num_words) {
            self.buf.resize_with(new_len, W::zero);
            self.mask_last_written = new_mask;

            if let Some(last_word) = self.buf.last_mut() {
                let mask = (new_mask - W::one()) << 1 | W::one();
                *last_word = *last_word & mask;
            } else {
                self.mask_last_written = W::zero();
            }
        } else if amt != 0 {
            // The test for `amt != 0` is for case where `BitVec` was originally empty, in
            // which case calling `discard(0)` leads to `num_words == 1`, which is larger
            // than `buf.len()` but the call should still be allowed.
            self.clear();
            return Err(());
        }

        Ok(())
    }
}

impl<W: BitArray, V: GenericVec<W>> FromIterator<bool> for BitVec<W, V> {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut bit_vec = Self::with_capacity(iter.size_hint().0);
        for bit in iter {
            bit_vec.push(bit)
        }
        bit_vec
    }
}

impl<W: BitArray, V: GenericVec<W>> Iterator for BitVecReverseIterator<W, V> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.inner.len();
        (len, Some(len))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.inner.buf.first().map(|&x| x & W::one() != W::zero())
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.discard(n).ok()?;
        self.next()
    }
}

impl<W: BitArray, V: GenericVec<W>> ExactSizeIterator for BitVecReverseIterator<W, V> {}

impl<T> GenericVec<T> for Vec<T> {
    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }

    fn push(&mut self, x: T) {
        self.push(x)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn clear(&mut self) {
        self.clear()
    }

    fn resize_with(&mut self, new_len: usize, f: impl FnMut() -> T) {
        self.resize_with(new_len, f)
    }
}

impl<T> GenericVec<T> for SmallVec<[T; 1]> {
    fn with_capacity(capacity: usize) -> Self {
        SmallVec::with_capacity(capacity)
    }

    fn push(&mut self, x: T) {
        self.push(x)
    }

    fn pop(&mut self) -> Option<T> {
        self.pop()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn clear(&mut self) {
        self.clear()
    }

    fn resize_with(&mut self, new_len: usize, f: impl FnMut() -> T) {
        self.resize_with(new_len, f)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    extern crate std;

    #[test]
    fn bit_vec() {
        let mut bit_vec = BitVec::<u32>::new();
        assert_eq!(bit_vec.len(), 0);
        assert!(bit_vec.is_empty());

        let amt = 150usize;
        for i in 0..amt {
            bit_vec.push(i.count_ones() % 2 != 0);
        }

        assert_eq!(bit_vec.len(), amt);
        assert!(!bit_vec.is_empty());

        let mut count_down = amt;
        for bit in bit_vec.into_iter_reverse() {
            count_down -= 1;
            assert_eq!(bit, count_down.count_ones() % 2 != 0);
        }

        assert_eq!(count_down, 0);
    }
}
