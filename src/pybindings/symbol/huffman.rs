use std::prelude::v1::*;

use pyo3::prelude::*;

use crate::{
    pybindings::{PyReadonlyFloatArray, PyReadonlyFloatArray1},
    symbol::huffman,
};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
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
    #[pyo3(text_signature = "(self, probabilities)")]
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
    #[pyo3(text_signature = "(self, probabilities)")]
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
