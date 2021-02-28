use smallvec::SmallVec;

use alloc::{
    collections::{BinaryHeap, VecDeque},
    vec,
    vec::Vec,
};
use core::{borrow::Borrow, cmp::Reverse, ops::Add};

use crate::{BitArray, EncodingError};

#[derive(Clone, Debug)]
struct BitDeque<W: BitArray> {
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

#[derive(Debug, Clone)]
pub struct EncoderHuffmanTree {
    /// A `Vec` of size `num_symbols * 2 - 1`, where the first `num_symbol` items
    /// correspond to the symbols, i.e., leaf nodes of the Huffman tree, and the
    /// remaining items are ancestors. An entry with value `x: usize` represents a node
    /// with the following properties:
    /// - root node if `x == 0`;
    /// - otherwise, the lowest significant bit distinguishes left vs right children,
    ///   and the parent node is at index `x >> 1`.
    /// (This works the node with index 0, if it exists, is always a leaf node, i.e., it
    /// cannot be any other node's parent node.)
    ///
    /// It is guaranteed that `num_symbols != 0` i.e., `nodes` is not empty.
    nodes: Vec<usize>,
}

impl EncoderHuffmanTree {
    pub fn from_probabilities<P: Ord + Clone + Add<Output = P>>(
        probabilities: impl IntoIterator<Item = impl Borrow<P>>,
    ) -> Self {
        let mut heap = probabilities
            .into_iter()
            .enumerate()
            .map(|(i, s)| Reverse((s.borrow().clone(), i)))
            .collect::<BinaryHeap<_>>();

        if heap.is_empty() || heap.len() > usize::max_value() / 4 {
            panic!();
        }

        let mut nodes = vec![0; heap.len() * 2 - 1];
        let mut next_node_index = heap.len();

        while let (Some(Reverse((prob0, index0))), Some(Reverse((prob1, index1)))) =
            (heap.pop(), heap.pop())
        {
            // TODO: turn into `get_unchecked`
            heap.push(Reverse((prob0 + prob1, next_node_index)));
            nodes[index0] = next_node_index << 1;
            nodes[index1] = (next_node_index << 1) | 1;
            next_node_index += 1;
        }

        Self { nodes }
    }
}

impl<W: BitArray> EncoderCodebook<W> for EncoderHuffmanTree {
    fn encode_symbol(
        &self,
        symbol: usize,
        destination: &mut BitDeque<W>,
    ) -> Result<(), EncodingError> {
        if symbol > self.nodes.len() / 2 {
            return Err(EncodingError::ImpossibleSymbol);
        }

        let mut reverse_codeword = SmallBitVec::<W>::new();
        let mut node_index = symbol;
        loop {
            let node = self.nodes[node_index];
            if node == 0 {
                break;
            }
            reverse_codeword.push(node & 1 != 0);
            node_index = node >> 1;
        }

        while let Some(bit) = reverse_codeword.pop() {
            destination.push_back(bit);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DecoderHuffmanTree {
    /// A `Vec` of size `num_symbols - 1`, containing only the non-leaf nodes of the
    /// Huffman tree. The root node is at the end. An entry with value
    /// `[x, y]: [usize; 2]` represents a with children `x` and `y`, each represented
    /// either by the associated symbol (if the respective child is a leaf node), or by
    /// `num_symbols + index` where `index` is the index into `nodes` where the
    /// respective child node can be found.
    ///
    /// It is guaranteed that `num_symbols != 0`, but `nodes` can still be empty if
    /// `num_symbols == 1`.
    nodes: Vec<[usize; 2]>,
}

impl DecoderHuffmanTree {
    pub fn from_probabilities<P: Ord + Clone + Add<Output = P>>(
        probabilities: impl IntoIterator<Item = impl Borrow<P>>,
    ) -> Self {
        let mut heap = probabilities
            .into_iter()
            .enumerate()
            .map(|(i, s)| Reverse((s.borrow().clone(), i)))
            .collect::<BinaryHeap<_>>();

        if heap.is_empty() || heap.len() > usize::max_value() / 2 {
            panic!();
        }

        let mut nodes = Vec::with_capacity(heap.len() - 1);
        let mut next_node_index = heap.len();

        while let (Some(Reverse((prob0, index0))), Some(Reverse((prob1, index1)))) =
            (heap.pop(), heap.pop())
        {
            heap.push(Reverse((prob0 + prob1, next_node_index)));
            nodes.push([index0, index1]);
            next_node_index += 1;
        }

        Self { nodes }
    }
}

impl<W: BitArray> DecoderCodebook<W> for DecoderHuffmanTree {
    fn decode_symbol(&self, source: &mut BitDeque<W>) -> Result<usize, DecodingError> {
        let num_nodes = self.nodes.len();
        let num_symbols = num_nodes + 1;
        let mut node_index = 2 * num_nodes;

        while node_index >= num_symbols {
            let bit = source
                .pop_front()
                .ok_or(DecodingError::OutOfCompressedData)?;
            // TODO: turn into `get_unchecked`
            node_index = self.nodes[node_index - num_symbols][bit as usize];
        }

        Ok(node_index)
    }
}

#[derive(Debug)]
pub enum DecodingError {
    OutOfCompressedData,
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
