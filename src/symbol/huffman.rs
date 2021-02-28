use alloc::{collections::BinaryHeap, vec, vec::Vec};
use core::{borrow::Borrow, cmp::Reverse, ops::Add};

use super::{BitDeque, DecoderCodebook, DecodingError, EncoderCodebook, SmallBitVec};
use crate::{BitArray, EncodingError};

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
