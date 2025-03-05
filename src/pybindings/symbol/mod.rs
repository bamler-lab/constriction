pub mod huffman;

use core::{
    convert::Infallible,
    sync::atomic::{AtomicBool, Ordering},
};
use std::prelude::v1::*;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, wrap_pymodule};

use crate::{
    backends::Cursor,
    symbol::{
        DefaultQueueDecoder, DefaultQueueEncoder, DefaultStackCoder, ReadBitStream,
        SymbolCodeError, WriteBitStream,
    },
};

use super::array1_to_vec;

/// Symbol codes. Mainly provided for teaching purpose. You'll probably want to use a
/// [stream code](stream.html) instead.
///
/// Symbol codes encode messages (= sequences of symbols) by compressing each symbol
/// individually and then concatenating the compressed representations of the symbols
/// (called code words). This has the advantage of being conceptually simple, but the
/// disadvantage of incurring an overhead of up to almost 1 bit *per symbol*. The overhead
/// is most significant in the regime of low entropy per symbol (often the regime of novel
/// machine-learning based compression methods), since each (nontrivial) symbol in a symbol
/// code contributes at least 1 bit to the total bitrate of the message, even if its
/// information content is only marginally above zero.
///
/// This module defines the types `QueueEncoder`, `QueueDecoder`, and `StackCoder`, which
/// are essentially just efficient containers for of growable bit strings. The interesting
/// part happens in the so-called code book, which defines how symbols map to code words. We
/// provide the Huffman Coding algorithm in the submodule [`huffman`](symbol/huffman.html),
/// which calculates an optimal code book for a given probability distribution.
///
/// ## Examples
///
/// The following two examples both encode and then decode a sequence of symbols using the
/// same entropy model for each symbol. We could as well use a different entropy model for
/// each symbol, but it would make the examples more lengthy.
///
/// - Example 1: Encoding and decoding with "queue" semantics ("first in first out", i.e.,
///   we'll decode symbols in the same order in which we encode them):
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Define an entropy model over the (implied) alphabet {0, 1, 2, 3}:
/// probabils = np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float32)
///
/// # Encode some example message, using the same model for each symbol here:
/// message = [1, 3, 2, 3, 0, 1, 3, 0, 2, 1, 1, 3, 3, 1, 2, 0, 1, 3, 1]
/// encoder = constriction.symbol.QueueEncoder()
/// encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(probabils)
/// for symbol in message:
///     encoder.encode_symbol(symbol, encoder_codebook)
///
/// # Obtain the compressed representation and the bitrate:
/// compressed, bitrate = encoder.get_compressed()
/// print(compressed, bitrate) # (prints: [3756389791, 61358], 48)
/// print(f"(in binary: {[bin(word) for word in compressed]}")
///
/// # Decode the message
/// decoder = constriction.symbol.QueueDecoder(compressed)
/// decoded = []
/// decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(probabils)
/// for symbol in range(19):
///     decoded.append(decoder.decode_symbol(decoder_codebook))
///
/// assert decoded == message # (verifies correctness)
/// ```
///
/// - Example 2: Encoding and decoding with "stack" semantics ("last in first out", i.e.,
///   we'll encode symbols from back to front and then decode them from front to back):
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Define an entropy model over the (implied) alphabet {0, 1, 2, 3}:
/// probabils = np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float32)
///
/// # Encode some example message, using the same model for each symbol here:
/// message = [1, 3, 2, 3, 0, 1, 3, 0, 2, 1, 1, 3, 3, 1, 2, 0, 1, 3, 1]
/// coder = constriction.symbol.StackCoder()
/// encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(probabils)
/// for symbol in reversed(message): # Note: reversed
///     coder.encode_symbol(symbol, encoder_codebook)
///
/// # Obtain the compressed representation and the bitrate:
/// compressed, bitrate = coder.get_compressed()
/// print(compressed, bitrate) # (prints: [[2818274807, 129455] 48)
/// print(f"(in binary: {[bin(word) for word in compressed]}")
///
/// # Decode the message (we could explicitly construct a decoder:
/// # `decoder = constriction.symbol.StackCoder(compressed)`
/// # but we can also also reuse our existing `coder` for decoding):
/// decoded = []
/// decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(probabils)
/// for symbol in range(19):
///     decoded.append(coder.decode_symbol(decoder_codebook))
///
/// assert decoded == message # (verifies correctness)
/// ```
#[pymodule]
#[pyo3(name = "symbol")]
pub fn init_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(huffman::init_module))?;
    module.add_class::<StackCoder>()?;
    module.add_class::<QueueEncoder>()?;
    module.add_class::<QueueDecoder>()?;
    Ok(())
}

/// A container of compressed bits that allows appending and consuming bits from the same
/// end.
///
/// When encoding onto a `StackCoder`, the bits that comprise each code word are
/// automatically appended in reverse order so that a prefix-free code becomes a suffix-free
/// code which can easily be decoded from the end. For Huffman Coding, this is actually the
/// natural way to generate code words (by walking the tree from a leaf to the root).
///
/// A `StackCoder` does not distinguish between an encoder and a decoder. It supports both
/// encoding and decoding with a single data structure and even allows you to arbitrarily
/// switch back and forth between encoding and decoding operations (e.g., for bits-back
/// coding).
///
/// The constructor takes existing compressed data as an optional argument. If it is
/// provided, it must must be a rank-1 numpy array with, as in the first return value of the
/// method `get_compressed`. If no argument is provided, then the `StackCoder` is
/// initialized empty (useful for encoding).
///
/// ## Example:
///
/// See second [module level example](#examples).
#[pyclass]
#[derive(Debug)]
pub struct StackCoder {
    inner: DefaultStackCoder,
}

#[pymethods]
impl StackCoder {
    #[new]
    #[pyo3(signature = (compressed=None))]
    pub fn new(compressed: Option<PyReadonlyArray1<'_, u32>>) -> PyResult<Self> {
        let inner = match compressed {
            None => DefaultStackCoder::new(),
            Some(compressed) => DefaultStackCoder::from_compressed(array1_to_vec(compressed))
                .map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Compressed data for a stack must not end in a zero word.",
                    )
                })?,
        };

        Ok(Self { inner })
    }

    /// Looks up the provided symbol in the provided codebook and appends its bits to the
    /// compressed data such that they form a suffix-free code (i.e., a code that can easily
    /// be read from back to front).
    ///
    /// ## Arguments
    ///
    /// - **symbol** --- an integer in the range `{0, 1, ..., n-1}` where `n` is the size of
    ///   the `codebook` provided in the second argument.
    /// - **codebook** --- an encoder code book for a symbol code. Currently,
    ///   [`EncoderHuffmanTree`](symbol/huffman.html#constriction.symbol.huffman.EncoderHuffmanTree)
    ///   is the only implemented encoder code book.
    #[pyo3(signature = (symbol, codebook))]
    pub fn encode_symbol(
        &mut self,
        symbol: usize,
        codebook: &huffman::EncoderHuffmanTree,
    ) -> PyResult<()> {
        Ok(self.inner.encode_symbol(symbol, &codebook.inner)?)
    }

    /// Consumes bits from the end of the encapsulated compressed data and tries to match
    /// them up with one of the code words in the provided code book, returning the
    /// corresponding symbol on success.
    ///
    /// ## Arguments
    ///
    /// - **codebook** --- a decoder code book for a symbol code. Currently,
    ///   [`DecoderHuffmanTree`](symbol/huffman.html#constriction.symbol.huffman.DecoderHuffmanTree)
    ///   is the only implemented decoder code book.
    #[pyo3(signature = (codebook))]
    pub fn decode_symbol(&mut self, codebook: &huffman::DecoderHuffmanTree) -> PyResult<usize> {
        Ok(self.inner.decode_symbol(&codebook.inner)?)
    }

    /// Returns a tuple `(compressed, bitrate)`, where `compressed` is a copy of the
    /// compressed representation that is currently on the coder (filled with zero bits to a
    /// multiple of 32 bits), and `bitrate` is the number of information-carrying bits.
    ///
    /// You can write the compressed data to a file (by calling
    /// `compressed.tofile("filename")`), read it back in at a later point (with
    /// `compressed = np.fromfile("filename")`) and then reconstruct a coder (for decoding)
    /// py passing `compressed` to the constructor of `StackCoder`.
    #[pyo3(signature = ())]
    pub fn get_compressed_and_bitrate<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<u32>>, usize) {
        let len = self.inner.len();
        (PyArray1::from_slice(py, &self.inner.get_compressed()), len)
    }

    /// Deprecated method. Please use `get_compressed_and_bitrate` instead.
    ///
    /// (The method was renamed to `get_compressed_and_bitrate` in `constriction` version
    /// 0.4.0 to avoid confusion about the return type.)
    #[pyo3(signature = ())]
    pub fn get_compressed<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyArray1<u32>>, usize) {
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::AcqRel) {
            let _ = py.run(
                pyo3::ffi::c_str!(
                    "print('WARNING: `StackCoder.get_compressed` has been renamed to\\n\
                     \x20        `StackCoder.get_compressed_and_bitrate` to avoid confusion."
                ),
                None,
                None,
            );
        }

        self.get_compressed_and_bitrate(py)
    }
}

#[pyclass]
#[derive(Debug, Default)]
pub struct QueueEncoder {
    inner: DefaultQueueEncoder,
}

/// A container of compressed bits that allows appending at the end, and that can be turned
/// into a `QueueDecoder`.
///
/// When encoding onto a `QueueEncoder`, the bits that comprise each code word are appended
/// in such a way that they form a prefix-free code. Thus, the symbols can be decoded in the
/// same order in which they were encoded.
///
/// A `QueueEncoder` can only *encode* symbols. To decode symbols again, turn the
/// `QueueEncoder` into a `QueueDecoder`, either directly by calling `.get_decoder()` on it
/// or by obtaining the compressed representation and then manually constructing a
/// `QueueDecoder` from it, as in the example.
///
/// ## Example:
///
/// See first [module level example](#examples).
#[pymethods]
impl QueueEncoder {
    #[new]
    #[pyo3(signature = ())]
    pub fn new() -> Self {
        Self {
            inner: DefaultQueueEncoder::new(),
        }
    }

    /// Looks up the provided symbol in the provided codebook and appends its bits to the
    /// compressed data such that they form a prefix-free code (i.e., a code that can easily
    /// be read from front to back).
    ///
    /// ## Arguments
    ///
    /// - **symbol** --- an integer in the range `{0, 1, ..., n-1}` where `n` is the size of
    ///   the `codebook` provided in the second argument.
    /// - **codebook** --- an encoder code book for a symbol code. Currently,
    ///   [`EncoderHuffmanTree`](symbol/huffman.html#constriction.symbol.huffman.EncoderHuffmanTree)
    ///   is the only implemented encoder code book.
    #[pyo3(signature = (symbol, codebook))]
    pub fn encode_symbol(
        &mut self,
        symbol: usize,
        codebook: &huffman::EncoderHuffmanTree,
    ) -> PyResult<()> {
        Ok(self.inner.encode_symbol(symbol, &codebook.inner)?)
    }

    /// Returns a tuple `(compressed, bitrate)`, where `compressed` is a copy of the
    /// compressed representation that is currently on the coder (filled with zero bits to a
    /// multiple of 32 bits), and `bitrate` is the number of information-carrying bits.
    ///
    /// You can write the compressed data to a file (by calling
    /// `compressed.tofile("filename")`), read it back in at a later point (with
    /// `compressed = np.fromfile("filename")`) and then construct a `QueueDecoder` from.
    #[pyo3(signature = ())]
    pub fn get_compressed_and_bitrate<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray1<u32>>, usize) {
        let len = self.inner.len();
        (PyArray1::from_slice(py, &self.inner.get_compressed()), len)
    }

    /// Deprecated method. Please use `get_compressed_and_bitrate` instead.
    ///
    /// (The method was renamed to `get_compressed_and_bitrate` in `constriction` version
    /// 0.4.0 to avoid confusion about the return type.)
    #[pyo3(signature = ())]
    pub fn get_compressed<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyArray1<u32>>, usize) {
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::AcqRel) {
            let _ = py.run(
                pyo3::ffi::c_str!(
                    "print('WARNING: `QueueEncoder.get_compressed` has been renamed to\\n\
                     \x20        `QueueEncoder.get_compressed_and_bitrate` to avoid confusion."
                ),
                None,
                None,
            );
        }

        self.get_compressed_and_bitrate(py)
    }

    /// Shortcut for `QueueDecoder(encoder.get_compressed())` where `encoder` is a
    /// `QueueEncoder`.
    ///
    /// Copies the compressed data out of the encoder. Thus, if you continue to encode
    /// symbols on the encoder, those won't affect what you will be able to decode on the
    /// decoder.
    #[pyo3(signature = ())]
    pub fn get_decoder(&mut self) -> QueueDecoder {
        let compressed = self.inner.get_compressed().to_vec();
        QueueDecoder::from_vec(compressed)
    }
}

/// A container of compressed bits that can be read from front to back for decoding.
///
/// This is the counterpart to `QueueEncoder`. The constructor takes a single argument,
/// which must be a rank-1 numpy array with `dtype=np.uint32`, as in the first return value
/// of  is returned by `QueueEncoder.get_compressed()
///
/// ## Example:
///
/// See first [module level example](#examples).
#[pyclass]
#[derive(Debug)]
pub struct QueueDecoder {
    inner: DefaultQueueDecoder,
}

#[pymethods]
impl QueueDecoder {
    #[new]
    #[pyo3(signature = (compressed))]
    pub fn new(compressed: PyReadonlyArray1<'_, u32>) -> PyResult<Self> {
        Ok(Self::from_vec(array1_to_vec(compressed)))
    }

    /// Reads more bits from the current encapsulated compressed data and tries to match
    /// them up with one of the code words in the provided code book, returning the
    /// corresponding symbol on success.
    ///
    /// ## Arguments
    ///
    /// - **codebook** --- a decoder code book for a symbol code. Currently,
    ///   [`DecoderHuffmanTree`](symbol/huffman.html#constriction.symbol.huffman.DecoderHuffmanTree)
    ///   is the only implemented decoder code book.
    #[pyo3(signature = (codebook))]
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
