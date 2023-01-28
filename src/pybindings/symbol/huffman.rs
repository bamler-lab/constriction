use std::prelude::v1::*;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::symbol::huffman::{self, NanError};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<EncoderHuffmanTree>()?;
    module.add_class::<DecoderHuffmanTree>()?;
    Ok(())
}

/// A Huffman tree that can be used for encoding data.
///
/// Expects a single argument `probabilities`, which is a rank-1 numpy array with
/// `dtype=np.float64` that specifies the probabilities of each one of the symbols in the
/// range `{0, 1, ..., len(probabilities)-1}`. All probabilities must be nonnegative and
/// finite, but probabilities do not need to add up to one since only the ratios of
/// probabilities will affect the shape of the constructed Huffman tree (note, however, that
/// rescaling probabilities can, in edge cases, affect the shape of the Huffman tree due
/// to rounding errors, so be consistent with how you scale probabilities).
///
/// # Examples
///
/// See [examples](../symbol.html#examples) in parent module.
#[pyclass]
#[pyo3(text_signature = "(self, probabilities)")]
#[derive(Debug)]
pub struct EncoderHuffmanTree {
    pub(crate) inner: huffman::EncoderHuffmanTree,
}

#[pymethods]
impl EncoderHuffmanTree {
    #[new]
    pub fn new(probabilities: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let inner = huffman::EncoderHuffmanTree::from_float_probabilities::<f64, _>(
            probabilities.as_array(),
        )?;

        Ok(Self { inner })
    }
}

/// A Huffman tree that can be used for decoding data.
///
/// Expects a single argument `probabilities`, which is a rank-1 numpy array with
/// `dtype=np.float64` that specifies the probabilities of each one of the symbols in the
/// range `{0, 1, ..., len(probabilities)-1}`. All probabilities must be nonnegative and
/// finite, but probabilities do not need to add up to one since only the ratios of
/// probabilities will affect the shape of the constructed Huffman tree (note, however, that
/// rescaling probabilities can, in edge cases, affect the shape of the Huffman tree due
/// to rounding errors, so be consistent with how you scale probabilities).
///
/// # Examples
///
/// See [examples](../symbol.html#examples) in parent module.
#[pyclass]
#[pyo3(text_signature = "(self, probabilities)")]
#[derive(Debug)]
pub struct DecoderHuffmanTree {
    pub(crate) inner: huffman::DecoderHuffmanTree,
}

#[pymethods]
impl DecoderHuffmanTree {
    #[new]
    pub fn new(probabilities: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let inner = huffman::DecoderHuffmanTree::from_float_probabilities::<f64, _>(
            probabilities.as_array(),
        )?;

        Ok(Self { inner })
    }
}

impl From<NanError> for PyErr {
    fn from(err: NanError) -> Self {
        match err {
            NanError::NaN => pyo3::exceptions::PyValueError::new_err("NaN probability provided."),
        }
    }
}
