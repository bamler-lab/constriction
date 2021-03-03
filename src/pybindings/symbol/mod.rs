pub mod huffman;

use std::{prelude::v1::*, vec};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, wrap_pymodule};

use crate::symbol::DecodingError;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(huffman))?;
    module.add_class::<BitVec>()?;
    Ok(())
}

/// Docstring of huffman module
#[pymodule]
fn huffman(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    huffman::init_module(py, module)
}

#[pyclass]
#[text_signature = "(compressed, bit_len)"]
#[derive(Debug)]
pub struct BitVec {
    inner: crate::symbol::DefaultBitVec,
}

#[pymethods]
impl BitVec {
    #[new]
    pub fn new(buf: Option<PyReadonlyArray1<'_, u32>>, bit_len: Option<usize>) -> PyResult<Self> {
        let inner = match (buf, bit_len) {
            (None, None) | (None, Some(0)) => crate::symbol::BitVec::new(),
            (Some(buf), Some(bit_len)) => {
                crate::symbol::BitVec::from_buf_and_bit_len(buf.to_vec()?, bit_len).map_err(
                    |_| {
                        pyo3::exceptions::PyAttributeError::new_err(
                    "Invalid length: at least one bit of the last word (if present) must be valid."
                )
                    },
                )?
            }
            _ => {
                todo!()
            }
        };

        Ok(Self { inner })
    }

    pub fn encode_symbol(
        &mut self,
        symbol: usize,
        codebook: &huffman::EncoderHuffmanTree,
    ) -> PyResult<()> {
        Ok(self.inner.encode_symbol(symbol, &codebook.inner)?)
    }

    pub fn get_decoder(&self) -> BitVecDecoder {
        BitVecDecoder {
            inner: self.inner.clone().into_iter(),
        }
    }

    pub fn numpy<'p>(&self, py: Python<'p>) -> (&'p PyArray1<u32>, usize) {
        (
            PyArray1::from_slice(py, &self.inner.buf()),
            self.inner.len(),
        )
    }
}

#[pyclass]
#[text_signature = "(compressed, bit_len)"]
#[derive(Debug)]
pub struct BitVecDecoder {
    inner: crate::symbol::BitVecIter<u32, Vec<u32>>,
}

#[pymethods]
impl BitVecDecoder {
    #[new]
    pub fn new(buf: Option<PyReadonlyArray1<'_, u32>>, bit_len: Option<usize>) -> PyResult<Self> {
        let bit_vec = BitVec::new(buf, bit_len)?;
        Ok(Self {
            inner: bit_vec.inner.into_iter(),
        })
    }

    pub fn decode_symbol(&mut self, codebook: &huffman::DecoderHuffmanTree) -> PyResult<usize> {
        Ok(self.inner.decode_symbol(&codebook.inner)?)
    }
}

impl From<DecodingError> for PyErr {
    fn from(err: DecodingError) -> Self {
        match err {
            DecodingError::OutOfCompressedData => {
                pyo3::exceptions::PyValueError::new_err("Ran out of bits in compressed data.")
            }
        }
    }
}
