pub mod huffman;

use smallvec::SmallVec;

use alloc::collections::VecDeque;

use crate::{BitArray, EncodingError};

#[derive(Clone, Debug)]
pub struct BitDeque<W: BitArray> {
    buf: VecDeque<W>,
    next_mask_back: W,
    next_mask_front: W,
}

impl<W: BitArray> BitDeque<W> {
    pub fn new() -> Self {
        todo!()
    }

    pub fn pop_back(&mut self) -> Option<bool> {
        todo!()
    }

    pub fn push_back(&mut self, bit: bool) {
        todo!()
    }

    pub fn pop_front(&mut self) -> Option<bool> {
        todo!()
    }

    pub fn push_front(&mut self, bit: bool) {
        todo!()
    }

    fn discard_front(&mut self, n: usize) -> Result<(), ()> {
        todo!()
    }

    fn discard_back(&mut self, n: usize) -> Result<(), ()> {
        todo!()
    }
}

#[derive(Debug)]
pub enum DecodingError {
    OutOfCompressedData,
}

trait EncoderCodebook<W: BitArray> {
    /// TODO: separate codebooks from BitDeque: they should only operate
    /// on iterators over bits. Then, VecDeque should implement generic
    /// methods that can encode one or more symbols from arbitrary kinds
    /// of EncoderCodebooks. Same for decoding.
    fn encode_symbol(
        &self,
        symbol: usize,
        destination: &mut BitDeque<W>,
    ) -> Result<(), EncodingError>;
}

trait DecoderCodebook<W: BitArray> {
    fn decode_symbol(&self, source: &mut BitDeque<W>) -> Result<usize, DecodingError>;

    fn decode_iid_symbols(&self, source: &mut BitDeque<W>, amt: usize) {
        todo!() // (need to define return type)
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct SmallBitVec<W: BitArray> {
    buf: SmallVec<[W; 1]>,

    /// A `BitArray` that is either zero or has a single "one" bit exactly where the the
    /// next bit should be inserted within a `W` word when [`push`]ing to the
    /// `SmallBitVec`. The special value of zero is equivalent to a value of one (which
    /// is also allowed), and indicates that [`push`]ing a bit will require appending a
    /// new word to `buf`. Allowing `next_mask` to be either zero or one in this case
    /// makes the code slightly simpler.
    next_mask: W,
}

impl<W: BitArray> SmallBitVec<W> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn push(&mut self, bit: bool) {
        if self.next_mask > W::one() {
            let added_bit = if bit { self.next_mask } else { W::zero() };
            let last_word = self.buf.last_mut().expect("next_mask > 1.");
            *last_word = *last_word | added_bit;
            self.next_mask = self.next_mask << 1;
        } else {
            self.buf.push(if bit { W::one() } else { W::zero() });
            self.next_mask = W::one() << 1;
        }
    }

    pub fn pop(&mut self) -> Option<bool> {
        self.next_mask = self.next_mask >> 1;
        let masked_bit = if self.next_mask != W::zero() {
            let last_word = self
                .buf
                .last_mut()
                .expect("next_mask was > 1 before right-shift.");
            let masked_bit = *last_word & self.next_mask;
            *last_word = *last_word ^ masked_bit;
            masked_bit
        } else {
            self.next_mask = W::one() << (W::BITS - 1);
            let last_word = self.buf.pop()?;
            last_word & self.next_mask
        };

        Some(masked_bit != W::zero())
    }
}
