use std::prelude::v1::*;

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::symbol::huffman::{self, NanError};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<EncoderHuffmanTree>()?;
    module.add_class::<DecoderHuffmanTree>()?;
    Ok(())
}

#[pyclass]
#[pyo3(text_signature = "(probabilities)")]
#[derive(Debug)]
pub struct EncoderHuffmanTree {
    pub(crate) inner: huffman::EncoderHuffmanTree,
}

#[pymethods]
impl EncoderHuffmanTree {
    #[new]
    pub fn new(probabilities: PyReadonlyArray1<'_, f32>) -> PyResult<Self> {
        let inner = huffman::EncoderHuffmanTree::from_float_probabilities::<f32, _>(
            probabilities.iter().unwrap(),
        )?;

        Ok(Self { inner })
    }
}

#[pyclass]
#[pyo3(text_signature = "(probabilities)")]
#[derive(Debug)]
pub struct DecoderHuffmanTree {
    pub(crate) inner: huffman::DecoderHuffmanTree,
}

#[pymethods]
impl DecoderHuffmanTree {
    #[new]
    pub fn new(probabilities: PyReadonlyArray1<'_, f32>) -> PyResult<Self> {
        let inner = huffman::DecoderHuffmanTree::from_float_probabilities::<f32, _>(
            probabilities.iter().unwrap(),
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
