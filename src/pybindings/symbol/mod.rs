pub mod huffman;

use core::convert::Infallible;
use std::{prelude::v1::*, vec};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, wrap_pymodule};

use crate::{
    backends::Cursor,
    symbol::{
        codebooks::SymbolCodeError, DefaultQueueDecoder, DefaultQueueEncoder, DefaultStackCoder,
        ReadBitStream, WriteBitStream,
    },
};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(huffman))?;
    module.add_class::<StackCoder>()?;
    module.add_class::<QueueEncoder>()?;
    module.add_class::<QueueDecoder>()?;
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
    inner: DefaultStackCoder,
}

#[pymethods]
impl StackCoder {
    #[new]
    pub fn new(compressed: Option<PyReadonlyArray1<'_, u32>>) -> PyResult<Self> {
        let inner = match compressed {
            None => DefaultStackCoder::new(),
            Some(compressed) => {
                DefaultStackCoder::from_compressed(compressed.to_vec()?).map_err(|_| {
                    pyo3::exceptions::PyAttributeError::new_err(
                        "Compressed data for a stack must not end in a zero word.",
                    )
                })?
            }
        };

        Ok(Self { inner })
    }

    #[text_signature = "(symbol, codebook)"]
    pub fn encode_symbol(
        &mut self,
        symbol: usize,
        codebook: &huffman::EncoderHuffmanTree,
    ) -> PyResult<()> {
        Ok(self.inner.encode_symbol(symbol, &codebook.inner)?)
    }

    #[text_signature = "(codebook)"]
    pub fn decode_symbol(&mut self, codebook: &huffman::DecoderHuffmanTree) -> PyResult<usize> {
        Ok(self.inner.decode_symbol(&codebook.inner)?)
    }

    /// Returns a tuple of the compressed data (filled with zero bits to a multiple of 32
    /// bits) and the number of valid bits.
    #[text_signature = "()"]
    pub fn get_compressed<'p>(&mut self, py: Python<'p>) -> (&'p PyArray1<u32>, usize) {
        let len = self.inner.len();
        (PyArray1::from_slice(py, &self.inner.get_compressed()), len)
    }
}

#[pyclass]
#[text_signature = "(compressed)"]
#[derive(Debug)]
pub struct QueueEncoder {
    inner: DefaultQueueEncoder,
}

#[pymethods]
impl QueueEncoder {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: DefaultQueueEncoder::new(),
        }
    }

    #[text_signature = "(symbol, codebook)"]
    pub fn encode_symbol(
        &mut self,
        symbol: usize,
        codebook: &huffman::EncoderHuffmanTree,
    ) -> PyResult<()> {
        Ok(self.inner.encode_symbol(symbol, &codebook.inner)?)
    }

    /// Returns a tuple of the compressed data (filled with zero bits to a multiple of 32
    /// bits) and the number of valid bits.
    #[text_signature = "()"]
    pub fn get_compressed<'p>(&mut self, py: Python<'p>) -> (&'p PyArray1<u32>, usize) {
        let len = self.inner.len();
        (PyArray1::from_slice(py, &self.inner.get_compressed()), len)
    }

    #[text_signature = "()"]
    pub fn get_decoder<'p>(&mut self) -> QueueDecoder {
        let compressed = self.inner.get_compressed().to_vec();
        QueueDecoder::from_vec(compressed)
    }
}

#[pyclass]
#[text_signature = "(compressed)"]
#[derive(Debug)]
pub struct QueueDecoder {
    inner: DefaultQueueDecoder,
}

#[pymethods]
impl QueueDecoder {
    #[new]
    pub fn new(compressed: PyReadonlyArray1<'_, u32>) -> PyResult<Self> {
        Ok(Self::from_vec(compressed.to_vec()?))
    }

    #[text_signature = "(codebook)"]
    pub fn decode_symbol(&mut self, codebook: &huffman::DecoderHuffmanTree) -> PyResult<usize> {
        Ok(self.inner.decode_symbol(&codebook.inner)?)
    }
}

impl QueueDecoder {
    fn from_vec(compressed: Vec<u32>) -> Self {
        let compressed = Cursor::new_at_write_beginning(compressed);
        Self {
            inner: DefaultQueueDecoder::from_compressed(compressed),
        }
    }
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
