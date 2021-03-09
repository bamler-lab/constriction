//! Symbol Codes (mainly provided for teaching purpose; typically inferior to stream codes)

pub mod huffman;

use smallvec::SmallVec;

use alloc::vec::Vec;
use core::{borrow::Borrow, convert::Infallible, iter::FromIterator};

use crate::{BitArray, EncoderError};

#[derive(Debug)]
pub enum DecodingError {
    OutOfCompressedData,
}

pub trait Codebook {
    fn num_symbols(&self) -> usize;
}
pub trait EncoderCodebook: Codebook {
    type BitIterator: Iterator<Item = bool>;

    fn encode_symbol(&self, symbol: usize) -> Result<Self::BitIterator, EncoderError<Infallible>>;
}

pub trait DecoderCodebook: Codebook {
    fn decode_symbol(&self, source: impl Iterator<Item = bool>) -> Result<usize, DecodingError>;
}

impl<C: Codebook> Codebook for &C {
    fn num_symbols(&self) -> usize {
        (*self).num_symbols()
    }
}

impl<C: EncoderCodebook> EncoderCodebook for &C {
    type BitIterator = C::BitIterator;

    fn encode_symbol(&self, symbol: usize) -> Result<Self::BitIterator, EncoderError<Infallible>> {
        (*self).encode_symbol(symbol)
    }
}

impl<C: DecoderCodebook> DecoderCodebook for &C {
    fn decode_symbol(&self, source: impl Iterator<Item = bool>) -> Result<usize, DecodingError> {
        (*self).decode_symbol(source)
    }
}

pub trait GenericVec<T>: Default + AsRef<[T]> + AsMut<[T]> {
    fn with_capacity(capacity: usize) -> Self;
    fn push(&mut self, x: T);
    fn pop(&mut self) -> Option<T>;
    fn clear(&mut self);
    fn resize_with(&mut self, new_len: usize, f: impl FnMut() -> T);

    fn is_empty(&self) -> bool {
        self.as_ref().len() == 0
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

pub type DefaultBitVec = BitVec<u32>;

#[derive(Debug)]
pub struct BitVecIterRev<W: BitArray, V: GenericVec<W> = Vec<W>> {
    inner: BitVec<W, V>,
}

type SmallBitVec<W> = BitVec<W, SmallVec<[W; 1]>>;
type SmallBitVecReverseIterator<W> = BitVecIterRev<W, SmallVec<[W; 1]>>;

impl<W: BitArray, V: GenericVec<W>> BitVec<W, V> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_bit_capacity(bit_capacity: usize) -> Self {
        Self {
            buf: V::with_capacity((bit_capacity + W::BITS - 1) / W::BITS),
            mask_last_written: W::zero(),
        }
    }

    pub fn from_buf_and_bit_len(buf: V, bit_len: usize) -> Result<Self, ()> {
        let num_words = (bit_len + W::BITS - 1) / W::BITS;
        if buf.as_ref().len() != num_words {
            Err(())
        } else {
            let mask_last_written = if num_words == 0 {
                W::zero()
            } else {
                W::one() << ((bit_len - 1) % W::BITS)
            };

            Ok(Self {
                buf,
                mask_last_written,
            })
        }
    }

    pub fn push(&mut self, bit: bool) {
        let write_mask = self.mask_last_written << 1;
        self.mask_last_written = if write_mask != W::zero() {
            if bit {
                let last_word = self
                    .buf
                    .as_mut()
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
                .as_mut()
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
        let capacity = self.buf.as_ref().len() * W::BITS;
        let unused = if self.mask_last_written == W::zero() {
            0
        } else {
            self.mask_last_written.leading_zeros()
        };

        capacity - unused as usize
    }

    pub fn buf(&self) -> &[W] {
        self.buf.as_ref()
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.mask_last_written = W::zero();
    }

    pub fn into_iter_reverse(self) -> BitVecIterRev<W, V> {
        BitVecIterRev { inner: self }
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

        if let Some(new_len) = self.buf.as_ref().len().checked_sub(num_words) {
            self.buf.resize_with(new_len, W::zero);
            self.mask_last_written = new_mask;

            if let Some(last_word) = self.buf.as_mut().last_mut() {
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

    pub fn encode_symbol(
        &mut self,
        symbol: usize,
        codebook: impl EncoderCodebook,
    ) -> Result<(), EncoderError<Infallible>> {
        Ok(self.extend(codebook.encode_symbol(symbol)?))
    }

    pub fn encode_symbols<S, C>(
        &mut self,
        symbols_and_codebooks: impl IntoIterator<Item = (S, C)>,
    ) -> Result<(), EncoderError<Infallible>>
    where
        S: Borrow<usize>,
        C: EncoderCodebook,
    {
        for (symbol, codebook) in symbols_and_codebooks.into_iter() {
            self.encode_symbol(*symbol.borrow(), codebook)?;
        }

        Ok(())
    }

    pub fn encode_iid_symbols<S>(
        &mut self,
        symbols: impl IntoIterator<Item = S>,
        codebook: &impl EncoderCodebook,
    ) -> Result<(), EncoderError<Infallible>>
    where
        S: Borrow<usize>,
    {
        self.encode_symbols(symbols.into_iter().map(|symbol| (symbol, codebook)))
    }

    pub fn iter(&self) -> BitVecIter<W, &[W]> {
        self.into()
    }

    pub fn into_iter(self) -> BitVecIter<W, V> {
        self.into()
    }
}

impl<W: BitArray, V: GenericVec<W>> FromIterator<bool> for BitVec<W, V> {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut bit_vec = Self::with_bit_capacity(iter.size_hint().0);
        for bit in iter {
            bit_vec.push(bit)
        }
        bit_vec
    }
}

impl<W: BitArray, V: GenericVec<W>> Extend<bool> for BitVec<W, V> {
    fn extend<T: IntoIterator<Item = bool>>(&mut self, iter: T) {
        // TODO: when specialization becomes stable, distinguish between extending from a
        // `SmallBitVecReverseIterator` and other iterators. For extending from
        // `SmallBitVecReverseIterator`, leave it as is. For other iterators, calculate
        // the number of added words upfront and insert a `reserve` call here.
        for bit in iter {
            self.push(bit);
        }
    }
}

impl<W: BitArray, V: GenericVec<W>> Iterator for BitVecIterRev<W, V> {
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
        self.inner
            .buf
            .as_ref()
            .first()
            .map(|&x| x & W::one() != W::zero())
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.inner.discard(n).ok()?;
        self.next()
    }
}

impl<W: BitArray, V: GenericVec<W>> ExactSizeIterator for BitVecIterRev<W, V> {}

#[derive(Debug)]
pub struct BitVecIter<W: BitArray, B: AsRef<[W]>> {
    buf: B,
    next_pos: usize,
    last_word_allowed_bits: W,
    current_word: W,
    current_allowed_bits: W,
    mask_next_to_read: W,
}

impl<'buf, W: BitArray, V: GenericVec<W>> From<&'buf BitVec<W, V>> for BitVecIter<W, &'buf [W]> {
    fn from(bit_vec: &'buf BitVec<W, V>) -> Self {
        Self {
            buf: bit_vec.buf.as_ref(),
            next_pos: 0,
            last_word_allowed_bits: (bit_vec.mask_last_written << 1).wrapping_sub(&W::one()),
            current_word: W::zero(),
            current_allowed_bits: W::max_value(),
            mask_next_to_read: W::zero(),
        }
    }
}

impl<W: BitArray, V: GenericVec<W>> From<BitVec<W, V>> for BitVecIter<W, V> {
    fn from(bit_vec: BitVec<W, V>) -> Self {
        Self {
            buf: bit_vec.buf,
            next_pos: 0,
            last_word_allowed_bits: (bit_vec.mask_last_written << 1).wrapping_sub(&W::one()),
            current_word: W::zero(),
            current_allowed_bits: W::max_value(),
            mask_next_to_read: W::zero(),
        }
    }
}

impl<W: BitArray, B: AsRef<[W]>> Iterator for BitVecIter<W, B> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.mask_next_to_read & self.current_allowed_bits != W::zero() {
            // Most common case, therefore reachable with only a single branch.
            let bit = self.current_word & self.mask_next_to_read != W::zero();
            self.mask_next_to_read = self.mask_next_to_read << 1;
            Some(bit)
        } else if self.mask_next_to_read == W::zero() {
            if self.next_pos >= self.buf.as_ref().len() {
                None
            } else {
                self.current_word = self.buf.as_ref()[self.next_pos]; // TODO: use get_unchecked
                self.next_pos += 1;
                self.mask_next_to_read = W::one() << 1;
                if self.next_pos == self.buf.as_ref().len() {
                    self.current_allowed_bits = self.last_word_allowed_bits
                }
                Some(self.current_word & W::one() != W::zero())
            }
        } else {
            None
        }
    }
}

impl<W: BitArray, B: AsRef<[W]>> BitVecIter<W, B> {
    pub fn decode_symbol(
        &mut self,
        codebook: impl DecoderCodebook,
    ) -> Result<usize, DecodingError> {
        codebook.decode_symbol(self)
    }

    pub fn decode_symbols<'s, I, C>(
        &'s mut self,
        codebooks: I,
    ) -> DecodeSymbols<'s, Self, I::IntoIter>
    where
        I: IntoIterator<Item = C> + 's,
        C: DecoderCodebook,
    {
        // TODO: It would be much nicer to implement this just as a `map`, but it doesn't
        // currently seem to work with lifetimes (because of 'buf).
        DecodeSymbols {
            bit_iterator: self,
            codebooks: codebooks.into_iter(),
        }
    }

    pub fn decode_iid_symbols<'s, 'c, C>(
        &'s mut self,
        amt: usize,
        codebook: &'c C,
    ) -> DecodeIidSymbols<'s, 'c, Self, C>
    where
        C: DecoderCodebook,
    {
        DecodeIidSymbols {
            bit_iterator: self,
            codebook,
            amt,
        }
    }
}

#[derive(Debug)]
pub struct DecodeSymbols<'a, BI, CI> {
    bit_iterator: &'a mut BI,
    codebooks: CI,
}

impl<'bi, BI, CI> Iterator for DecodeSymbols<'bi, BI, CI>
where
    BI: Iterator<Item = bool>,
    CI: Iterator,
    CI::Item: DecoderCodebook,
{
    type Item = Result<usize, DecodingError>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(
            self.codebooks
                .next()?
                .decode_symbol(&mut *self.bit_iterator),
        )
    }
}

#[derive(Debug)]
pub struct DecodeIidSymbols<'bi, 'c, BI, C> {
    bit_iterator: &'bi mut BI,
    codebook: &'c C,
    amt: usize,
}

impl<'bi, 'c, BI, C> Iterator for DecodeIidSymbols<'bi, 'c, BI, C>
where
    BI: Iterator<Item = bool>,
    C: DecoderCodebook,
{
    type Item = Result<usize, DecodingError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.amt != 0 {
            self.amt -= 1;
            Some(self.codebook.decode_symbol(&mut *self.bit_iterator))
        } else {
            None
        }
    }
}

// fn f<'buf>(iter: BitVecIter<'buf, u32>, codebooks: Vec<DecoderHuffmanTree>) {
//     let codebook_slice = &codebooks[..];
//     let new_iter = iter.decode_symbols(codebook_slice);

//     // return new_iter;  <-- forbidden since `iter` and `codebook` will be dropped
// }

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
    use super::{
        huffman::{DecoderHuffmanTree, EncoderHuffmanTree},
        *,
    };

    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };

    extern crate std;

    #[test]
    fn bit_vec() {
        let mut bit_vec = BitVec::<u32>::new();
        assert_eq!(bit_vec.len(), 0);
        assert!(bit_vec.is_empty());

        let amt = 150;
        let mut bool_vec = Vec::with_capacity(amt);
        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        for _ in 0..amt {
            let bit = rng.next_u32() % 2 != 0;
            bit_vec.push(bit);
            bool_vec.push(bit);
        }

        assert_eq!(bit_vec.len(), amt);
        assert!(!bit_vec.is_empty());

        for bit in bit_vec.into_iter_reverse() {
            assert_eq!(bit, bool_vec.pop().unwrap());
        }

        assert!(bool_vec.is_empty());
    }

    #[test]
    fn bit_vec_iter() {
        let amt = 100;
        let mut bit_vec = BitVec::<u16>::new();
        let mut bool_vec = Vec::with_capacity(amt);
        let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
        for _ in 0..amt {
            let bit = rng.next_u32() % 2 != 0;
            bit_vec.push(bit);
            bool_vec.push(bit);
        }
        assert_eq!(bit_vec.len(), amt);
        assert!(!bit_vec.is_empty());

        let mut count = 0;
        for bit in bit_vec.iter() {
            assert_eq!(bit, bool_vec[count]);
            count += 1;
        }
        assert_eq!(count, amt);
    }

    #[test]
    fn encode_decode_iid() {
        let amt = 1000;
        let mut rng = Xoshiro256StarStar::seed_from_u64(12345);
        let symbols = (0..amt)
            .map(|_| (rng.next_u32() % 5) as usize)
            .collect::<Vec<_>>();

        let probabilities = [2, 2, 4, 1, 1];
        let encoder = EncoderHuffmanTree::from_probabilities::<u32, _>(&probabilities);
        let decoder = DecoderHuffmanTree::from_probabilities::<u32, _>(&probabilities);

        let mut compressed = BitVec::<u32>::new();

        assert_eq!(compressed.len(), 0);
        compressed.encode_iid_symbols(&symbols, &encoder).unwrap();
        assert!(compressed.len() > amt);

        let mut bit_iter = compressed.iter();
        let reconstructed = bit_iter
            .decode_iid_symbols(amt, &decoder)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(reconstructed, symbols);
        assert!(bit_iter.next().is_none());
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

        let amt = 1000;
        let mut compressed = BitVec::<u32>::new();

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

        let mut bit_iter = compressed.iter();
        let reconstructed = bit_iter
            .decode_symbols(
                iter_probs_and_symbols(amt)
                    .map(|(probs, _)| DecoderHuffmanTree::from_probabilities::<u32, _>(&probs)),
            )
            .map(Result::unwrap);

        assert!(reconstructed.eq(iter_probs_and_symbols(amt).map(|(_, symbol)| symbol)));
        assert!(bit_iter.next().is_none());
    }
}
