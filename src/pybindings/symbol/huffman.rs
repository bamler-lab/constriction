use std::prelude::v1::*;

use pyo3::prelude::*;

use crate::{
    pybindings::{PyReadonlyFloatArray, PyReadonlyFloatArray1},
    symbol::huffman,
};

/// Codebooks for Huffman coding [1].
///
/// The Huffman algorithm constructs an optimal code book for a given categorical
/// probability distribution. Note however, that even an optimal code book is limited to
/// assigning an *integer* number of bits to each symbol, which results in an overhead of up
/// to 1 bit per symbol in the regime of low entropy per symbol. Stream codes, as provided
/// by the module [`constriction.stream`](../stream.html) remove most of this overhead by
/// amortizing compressed bits over several symbols.
///
/// Our implementation of Huffman trees uses the order in of provided symbols to break ties
/// if two subtrees have the same weight. Thus, reordering probabilities can affect the
/// shape of a Huffman tree in a nontrivial way in edge cases.
///
/// ## References
///
/// [1] Huffman, David A. "A method for the construction of minimum-redundancy codes."
/// Proceedings of the IRE 40.9 (1952): 1098-1101.
#[pymodule]
#[pyo3(name = "huffman")]
pub fn init_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<EncoderHuffmanTree>()?;
    module.add_class::<DecoderHuffmanTree>()?;
    Ok(())
}

/// A Huffman tree that can be used for encoding data.
///
/// Expects a single argument `probabilities`, which is a rank-1 numpy array with float
/// `dtype` that specifies the probabilities of each one of the symbols in the range
/// `{0, 1, ..., len(probabilities)-1}`. All probabilities must be nonnegative and
/// finite, but probabilities do not need to add up to one since only the ratios of
/// probabilities will affect the shape of the constructed Huffman tree (note, however, that
/// rescaling probabilities can, in edge cases, affect the shape of the Huffman tree due
/// to rounding errors, so be consistent with how you scale probabilities).
///
/// # Examples
///
/// See [examples](../symbol.html#examples) in parent module.
#[pyclass]
#[derive(Debug)]
pub struct EncoderHuffmanTree {
    pub(crate) inner: huffman::EncoderHuffmanTree,
}

#[pymethods]
impl EncoderHuffmanTree {
    #[new]
    #[pyo3(signature = (probabilities))]
    pub fn new(probabilities: PyReadonlyFloatArray1<'_>) -> PyResult<Self> {
        let inner = match probabilities {
            PyReadonlyFloatArray::F32(probabilities) => {
                huffman::EncoderHuffmanTree::from_float_probabilities::<f32, _>(
                    probabilities.as_array(),
                )
            }
            PyReadonlyFloatArray::F64(probabilities) => {
                huffman::EncoderHuffmanTree::from_float_probabilities::<f64, _>(
                    probabilities.as_array(),
                )
            }
        }?;

        Ok(Self { inner })
    }
}

/// A Huffman tree that can be used for decoding data.
///
/// Expects a single argument `probabilities`, which is a rank-1 numpy array with float
/// `dtype` that specifies the probabilities of each one of the symbols in the range
/// `{0, 1, ..., len(probabilities)-1}`. All probabilities must be nonnegative and
/// finite, but probabilities do not need to add up to one since only the ratios of
/// probabilities will affect the shape of the constructed Huffman tree (note, however, that
/// rescaling probabilities can, in edge cases, affect the shape of the Huffman tree due
/// to rounding errors, so be consistent with how you scale probabilities).
///
/// # Examples
///
/// See [examples](../symbol.html#examples) in parent module.
#[pyclass]
#[derive(Debug)]
pub struct DecoderHuffmanTree {
    pub(crate) inner: huffman::DecoderHuffmanTree,
}

#[pymethods]
impl DecoderHuffmanTree {
    #[new]
    #[pyo3(signature = (probabilities))]
    pub fn new(probabilities: PyReadonlyFloatArray1<'_>) -> PyResult<Self> {
        let inner = match probabilities {
            PyReadonlyFloatArray::F32(probabilities) => {
                huffman::DecoderHuffmanTree::from_float_probabilities::<f32, _>(
                    probabilities.as_array(),
                )
            }
            PyReadonlyFloatArray::F64(probabilities) => {
                huffman::DecoderHuffmanTree::from_float_probabilities::<f64, _>(
                    probabilities.as_array(),
                )
            }
        }?;

        Ok(Self { inner })
    }
}
