pub mod huffman;

use core::convert::Infallible;
use std::{prelude::v1::*, vec};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, wrap_pymodule};

use crate::symbol::{codebooks::SymbolCodeError, ReadBitStream, WriteBitStream};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(huffman))?;
    module.add_class::<StackCoder>()?;
    Ok(())
}

/// Docstring of huffman module
#[pymodule]
fn huffman(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    huffman::init_module(py, module)
}

#[pyclass]
#[text_signature = "(compressed)"]
#[derive(Debug)]
pub struct StackCoder {
    inner: crate::symbol::DefaultStackCoder,
}

#[pymethods]
impl StackCoder {
    #[new]
    pub fn new(compressed: Option<PyReadonlyArray1<'_, u32>>) -> PyResult<Self> {
        let inner = match compressed {
            None => crate::symbol::DefaultStackCoder::new(),
            Some(compressed) => crate::symbol::DefaultStackCoder::from_compressed(
                compressed.to_vec()?,
            )
            .map_err(|_| {
                pyo3::exceptions::PyAttributeError::new_err(
                    "Compressed data for a stack must not end in a zero word.",
                )
            })?,
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

    pub fn decode_symbol(&mut self, codebook: &huffman::DecoderHuffmanTree) -> PyResult<usize> {
        Ok(self.inner.decode_symbol(&codebook.inner)?)
    }

    // TODO
    // pub fn get_decoder(&self) -> BitVecDecoder {
    //     BitVecDecoder {
    //         inner: self.inner.clone().into_iter(),
    //     }
    // }

    // TODO
    // pub fn get_compressed<'p>(&self, py: Python<'p>) -> (&'p PyArray1<u32>, usize) {
    //     (
    //         PyArray1::from_slice(py, &self.inner.buf()),
    //         self.inner.len(),
    //     )
    // }
}

impl From<SymbolCodeError<Infallible>> for PyErr {
    fn from(err: SymbolCodeError<Infallible>) -> Self {
        match err {
            SymbolCodeError::OutOfCompressedData => {
                pyo3::exceptions::PyValueError::new_err("Ran out of bits in compressed data.")
            }
            SymbolCodeError::InvalidCodeword(infallible) => match infallible {},
        }
    }
}
