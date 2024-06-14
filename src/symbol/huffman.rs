//! Codebooks for Huffman Coding
//!
//! # References
//!
//! Huffman, David A. "A method for the construction of minimum-redundancy codes."
//! Proceedings of the IRE 40.9 (1952): 1098-1101.

use num_traits::float::FloatCore;

use alloc::{collections::BinaryHeap, vec, vec::Vec};
use core::{
    borrow::Borrow,
    cmp::Reverse,
    convert::Infallible,
    fmt::{Debug, Display},
    ops::Add,
};

use super::{Codebook, DecoderCodebook, EncoderCodebook, SymbolCodeError};
use crate::{CoderError, DefaultEncoderError, DefaultEncoderFrontendError, UnwrapInfallible};

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
    pub fn from_probabilities<P, I>(probabilities: I) -> Self
    where
        P: Ord + Clone + Add<Output = P>,
        I: IntoIterator,
        I::Item: Borrow<P>,
    {
        Self::try_from_probabilities::<_, Infallible, _>(
            probabilities.into_iter().map(|p| Ok(p.borrow().clone())),
        )
        .unwrap_infallible()
    }

    pub fn from_float_probabilities<P, I>(probabilities: I) -> Result<Self, NanError>
    where
        P: FloatCore + Clone + Add<Output = P>,
        I: IntoIterator,
        I::Item: Borrow<P>,
    {
        Self::try_from_probabilities(
            probabilities
                .into_iter()
                .map(|p| NonNanFloatCore::new(*p.borrow())),
        )
    }

    pub fn try_from_probabilities<P, E, I>(probabilities: I) -> Result<Self, E>
    where
        P: Ord + Clone + Add<Output = P>,
        I: IntoIterator<Item = Result<P, E>>,
    {
        // Collecting into a Vec first and then creating a binary heap is O(n)
        // whereas collecting directly into a binary heap would be O(n log(n)).
        let heap = probabilities
            .into_iter()
            .enumerate()
            .map(|(i, s)| s.map(|s| (Reverse((s, i)))))
            .collect::<Result<Vec<_>, E>>()?;
        let mut heap = BinaryHeap::from(heap);

        if heap.is_empty() || heap.len() > usize::MAX / 4 {
            panic!();
        }

        let mut nodes = vec![0; heap.len() * 2 - 1];
        let mut next_node_index = heap.len();

        while let (Some(Reverse((prob0, index0))), Some(Reverse((prob1, index1)))) =
            (heap.pop(), heap.pop())
        {
            heap.push(Reverse((prob0 + prob1, next_node_index)));
            unsafe {
                // SAFETY:
                // - `nodes.len() == original_heap_len * 2 - 1` (which we made sure doesn't wrap),
                //   where `original_heap_len` is the value of `heap.len()` before entering this
                //   `while` loop, which we checked is nonzero.
                // - We access `nodes` and indices found in `heap`. These have to be either the
                //   indices `0..original_heap_len` that we wrote into it initially (which are all
                //   smaller than `original_heap_len * 2 - 1` since we checked that
                //   `!heap.is_empty()`, i.e., that `original_heap_len != 0`); or they have to be
                //   the indices we write to the heap in this `while` loop, which come from
                //   `next_node_index`.
                // - `next_node_index` starts at `original_heap_len` and increases by one in each
                //   iteration of this `while` loop.
                // - Each iteration of this `while` loop removes two elements from `heap` and
                //   pushes one element back onto `heap`; so each iteration reduces the number of
                //   elements on `heap` by one; since we terminate as soon as there are fewer than
                //   2 elements on `heap`, this `while` loop iterates `original_heap_len - 1` times
                //   (which is nonnegative since `original_heap_len != 0`).
                // - Thus, the largest value that `next_node_index` can take is
                //   `original_heap_len * 2 - 1`; but since we access `next_node_index` before
                //   incrementing it, all values we ever push on the heap are strictly smaller than
                //   `original_heap_len * 2 - 1`, and thus are valid indices.
                *nodes.get_unchecked_mut(index0) = next_node_index << 1;
                *nodes.get_unchecked_mut(index1) = (next_node_index << 1) | 1;
            }
            next_node_index += 1;
        }

        Ok(Self { nodes })
    }

    pub fn num_symbols(&self) -> usize {
        self.nodes.len() / 2 + 1
    }
}

impl Codebook for EncoderHuffmanTree {
    type Symbol = usize;
}

impl EncoderCodebook for EncoderHuffmanTree {
    fn encode_symbol_suffix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        mut emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), DefaultEncoderError<BackendError>> {
        let symbol = *symbol.borrow();
        if symbol > self.nodes.len() / 2 {
            return Err(DefaultEncoderFrontendError::ImpossibleSymbol.into_coder_error());
        }

        let mut node_index = symbol;
        loop {
            let node = unsafe {
                // SAFETY: `node_index` is
                // - either its initial value of `symbol`, which is `<= num_symbols`, and
                //   `nodes.len() = 2 * num_symbols - 1 > num_symbols` since `num_symbols != 0`;
                // - or `node_index` is `node >> 1` where `node` is the value of a parent node; in
                //   this case it is guaranteed to be a valid index.
                *self.nodes.get_unchecked(node_index)
            };
            if node == 0 {
                break;
            }
            emit(node & 1 != 0)?;
            node_index = node >> 1;
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
    /// # Invariants
    /// - `num_symbols != 0` (but `nodes` can still be empty if `num_symbols == 1`.
    /// - All entries of `nodes` are strictly smaller than `2 * nodes.len()`.
    nodes: Vec<[usize; 2]>,
}

impl DecoderHuffmanTree {
    pub fn from_probabilities<P, I>(probabilities: I) -> Self
    where
        P: Ord + Clone + Add<Output = P>,
        I: IntoIterator,
        I::Item: Borrow<P>,
    {
        Self::try_from_probabilities::<_, Infallible, _>(
            probabilities.into_iter().map(|p| Ok(p.borrow().clone())),
        )
        .unwrap_infallible()
    }

    pub fn from_float_probabilities<P, I>(probabilities: I) -> Result<Self, NanError>
    where
        P: FloatCore + Clone + Add<Output = P>,
        I: IntoIterator,
        I::Item: Borrow<P>,
    {
        Self::try_from_probabilities(
            probabilities
                .into_iter()
                .map(|p| NonNanFloatCore::new(*p.borrow())),
        )
    }

    pub fn try_from_probabilities<P, E, I>(probabilities: I) -> Result<Self, E>
    where
        P: Ord + Clone + Add<Output = P>,
        I: IntoIterator<Item = Result<P, E>>,
    {
        // Collecting into a Vec first and then creating a binary heap is O(n)
        // whereas collecting directly into a binary heap would be O(n log(n)).
        let heap = probabilities
            .into_iter()
            .enumerate()
            .map(|(i, s)| s.map(|s| (Reverse((s, i)))))
            .collect::<Result<Vec<_>, E>>()?;
        let mut heap = BinaryHeap::from(heap);

        if heap.is_empty() || heap.len() > usize::MAX / 2 {
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

        Ok(Self { nodes })
    }

    pub fn num_symbols(&self) -> usize {
        self.nodes.len() + 1
    }
}

impl Codebook for DecoderHuffmanTree {
    type Symbol = usize;
}

impl DecoderCodebook for DecoderHuffmanTree {
    type InvalidCodeword = Infallible;

    fn decode_symbol<BackendError>(
        &self,
        mut source: impl Iterator<Item = Result<bool, BackendError>>,
    ) -> Result<Self::Symbol, CoderError<SymbolCodeError<Self::InvalidCodeword>, BackendError>>
    {
        let num_nodes = self.nodes.len();
        let num_symbols = num_nodes + 1;
        let mut node_index = 2 * num_nodes; // Start at root node.

        while node_index >= num_symbols {
            let bit = source
                .next()
                .ok_or_else(|| SymbolCodeError::OutOfCompressedData.into_coder_error())??;
            unsafe {
                // SAFETY:
                // - `node_index >= num_symbols` within this loop, so `node_index - num_symbols`
                //   does not wrap.
                // - `node_index is either the initial value `2 * num_nodes` or it comes from an
                //   entry of `nodes`, which are all strictly smaller than `2 * num_nodes`.
                // - Thus, `node_index - num_symbols = node_index - num_nodes - 1 <= num_nodes - 1`,
                //   which is a valid index into `nodes`.
                //
                // NOTE: No need to use `get_unchecked(bit as usize)` since the compiler is smart
                //       enough to optimize away the bounds check in this case on its own.
                node_index = self.nodes.get_unchecked(node_index - num_symbols)[bit as usize];
            }
        }

        Ok(node_index)
    }
}

#[derive(PartialOrd, Clone, Copy)]
struct NonNanFloatCore<F: FloatCore> {
    inner: F,
}

impl<F: FloatCore> NonNanFloatCore<F> {
    fn new(x: F) -> Result<Self, NanError> {
        if x.is_nan() {
            Err(NanError::NaN)
        } else {
            Ok(Self { inner: x })
        }
    }
}

impl<F: FloatCore> PartialEq for NonNanFloatCore<F> {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

impl<F: FloatCore> Eq for NonNanFloatCore<F> {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl<F: FloatCore> Ord for NonNanFloatCore<F> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.inner
            .partial_cmp(&other.inner)
            .expect("NonNanFloatCore::inner is not NaN.")
    }
}

impl<F: FloatCore> Add for NonNanFloatCore<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        NonNanFloatCore {
            inner: self.inner + rhs.inner,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum NanError {
    NaN,
}

impl Display for NanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NaN => write!(f, "NaN Encountered."),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NanError {}

#[cfg(test)]
mod tests {
    use super::{
        super::{SmallBitStack, WriteBitStream},
        *,
    };
    extern crate std;
    use std::string::String;

    #[test]
    fn encoder_huffman_tree() {
        fn encode_all_symbols(tree: &EncoderHuffmanTree) -> Vec<String> {
            (0..tree.num_symbols())
                .map(|symbol| {
                    let mut codeword = String::new();
                    tree.encode_symbol_prefix(symbol, |bit| {
                        codeword.push(if bit { '1' } else { '0' });
                        Result::<_, Infallible>::Ok(())
                    })
                    .unwrap();
                    codeword
                })
                .collect()
        }

        let tree = EncoderHuffmanTree::from_probabilities::<u32, _>(&[1]);
        assert_eq!(tree.nodes, [0]);
        assert_eq!(encode_all_symbols(&tree), [""]);

        let tree = EncoderHuffmanTree::from_probabilities::<u32, _>(&[1, 2]);
        assert_eq!(tree.nodes, [4, 5, 0]);
        assert_eq!(encode_all_symbols(&tree), ["0", "1"]);

        let tree = EncoderHuffmanTree::from_probabilities::<u32, _>(&[2, 1]);
        assert_eq!(tree.nodes, [5, 4, 0]);
        assert_eq!(encode_all_symbols(&tree), ["1", "0"]);

        // Ties are broken by index.
        let tree = EncoderHuffmanTree::from_probabilities::<u32, _>(&[1, 1]);
        assert_eq!(tree.nodes, [4, 5, 0]);
        assert_eq!(encode_all_symbols(&tree), ["0", "1"]);

        let tree = EncoderHuffmanTree::from_probabilities::<u32, _>(&[2, 2, 4, 1, 1]);
        assert_eq!(tree.nodes, [12, 13, 15, 10, 11, 14, 16, 17, 0]);
        assert_eq!(encode_all_symbols(&tree), ["00", "01", "11", "100", "101"]);

        // Let's not test ties of sums in floatCoreing point probabilities since they'll depend
        // on rounding errors (but should still be deterministic).
        let tree =
            EncoderHuffmanTree::from_float_probabilities::<f32, _>(&[0.19, 0.2, 0.41, 0.1, 0.1])
                .unwrap();
        assert_eq!(tree.nodes, [12, 13, 16, 10, 11, 14, 15, 17, 0,]);
        assert_eq!(
            encode_all_symbols(&tree),
            ["110", "111", "0", "100", "101",]
        );
    }

    #[test]
    fn decoder_huffman_tree() {
        fn test_decoding_all_symbols(
            decoder_tree: &DecoderHuffmanTree,
            encoder_tree: &EncoderHuffmanTree,
        ) {
            for symbol in 0..encoder_tree.num_symbols() {
                let mut codeword = SmallBitStack::new();
                encoder_tree
                    .encode_symbol_suffix(symbol, |bit| codeword.write_bit(bit))
                    .unwrap();
                let decoded = decoder_tree.decode_symbol(&mut codeword).unwrap();
                assert_eq!(symbol, decoded);
                assert!(codeword.next().is_none());
            }
        }

        let tree = DecoderHuffmanTree::from_probabilities::<u32, _>(&[1]);
        assert!(tree.nodes.is_empty());
        test_decoding_all_symbols(
            &tree,
            &EncoderHuffmanTree::from_probabilities::<u32, _>(&[1]),
        );

        let tree = DecoderHuffmanTree::from_probabilities::<u32, _>(&[1, 2]);
        assert_eq!(tree.nodes, [[0, 1]]);
        test_decoding_all_symbols(
            &tree,
            &EncoderHuffmanTree::from_probabilities::<u32, _>(&[0, 1]),
        );

        let tree = DecoderHuffmanTree::from_probabilities::<u32, _>(&[2, 1]);
        assert_eq!(tree.nodes, [[1, 0]]);
        test_decoding_all_symbols(
            &tree,
            &EncoderHuffmanTree::from_probabilities::<u32, _>(&[2, 1]),
        );

        // Ties are broken by index.
        let tree = DecoderHuffmanTree::from_probabilities::<u32, _>(&[1u32, 1]);
        assert_eq!(tree.nodes, [[0, 1]]);
        test_decoding_all_symbols(
            &tree,
            &EncoderHuffmanTree::from_probabilities::<u32, _>(&[1, 1]),
        );

        let tree = DecoderHuffmanTree::from_probabilities::<u32, _>(&[2, 2, 4, 1, 1]);
        assert_eq!(tree.nodes, [[3, 4], [0, 1], [5, 2], [6, 7]]);
        test_decoding_all_symbols(
            &tree,
            &EncoderHuffmanTree::from_probabilities::<u32, _>(&[2, 2, 4, 1, 1]),
        );

        // Let's not test ties of sums in floatCoreing point probabilities since they'll depend
        // on rounding errors (but should still be deterministic).
        let tree =
            DecoderHuffmanTree::from_float_probabilities::<f32, _>(&[0.19, 0.2, 0.41, 0.1, 0.1])
                .unwrap();
        assert_eq!(tree.nodes, [[3, 4], [0, 1], [5, 6], [2, 7]]);
        test_decoding_all_symbols(
            &tree,
            &EncoderHuffmanTree::from_float_probabilities::<f32, _>(&[0.19, 0.2, 0.41, 0.1, 0.1])
                .unwrap(),
        );
    }
}
