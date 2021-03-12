use std::{prelude::v1::*, vec};

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::symbol::codebooks::huffman;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<EncoderHuffmanTree>()?;
    module.add_class::<DecoderHuffmanTree>()?;
    Ok(())
}

#[pyclass]
#[text_signature = "(probabilities)"]
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
        )
        .map_err(|()| pyo3::exceptions::PyValueError::new_err("NaN probability provided."))?;

        Ok(Self { inner })
    }
}

#[pyclass]
#[text_signature = "(probabilities)"]
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
        )
        .map_err(|()| pyo3::exceptions::PyValueError::new_err("NaN probability provided."))?;

        Ok(Self { inner })
    }
}
