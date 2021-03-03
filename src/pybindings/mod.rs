//! Wrapper types that expose functionality to Python<'_> 3 code
//!
//! This module is only compiled if the feature `pybindings` is turned on, which is
//! turned off by default.
//!
//! # Compiling the Python Extension Module
//!
//! 1. If you haven't already: [install the Rust toolchain](https://rustup.rs/).
//!     If Rust is already installed on your system then make sure you have the
//!     latest version:
//!
//!     ```bash
//!     rustup update
//!     ```
//!
//! 2. Build an optimized library *with the `pybindings` feature flag*:
//!
//!     ```bash
//!     cd constriction
//!     cargo build --release --features pybindings
//!     ```
//!
//! 3. Check if the file `constriction.so` exists in the top level directory. The git
//!     repository should contain this file, and it should be a symlink that points
//!     to the library you just compiled:
//!
//!     ```bash
//!     $ ls -l constriction.so
//!     lrwxrwxrwx 1 user group Date Time constriction.so -> target/release/libconstriction.so
//!     ```
//!
//! # Example
//!
//! After compiling the python extension module as described above, `cd` to the
//! directory that contains the symlink `constriction.so`, open a python REPL, and try it
//! out:
//!
//! ```bash
//! $ ipython3
//!
//! In [1]: import constriction
//!    ...: import numpy as np
//!
//! In [2]: coder = constriction.Ans()
//!
//! In [3]: symbols = np.array([2, -1, 0, 2, 3], dtype=np.int32)
//!    ...: min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
//!    ...: means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype=np.float64)
//!    ...: stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype=np.float64)
//!
//! In [4]: coder.encode_gaussian_symbols_reverse(
//!    ...:     symbols, min_supported_symbol, max_supported_symbol, means, stds)
//!
//! In [5]: print(f"Compressed size: {coder.num_valid_bits()} bits")
//! Compressed size: 34 bits
//!
//! In [6]: coder.get_compressed()
//! Out[6]: array([746415963,         5], dtype=uint32)
//!
//! In [7]: coder.decode_gaussian_symbols(min_supported_symbol, max_supported_symbol, means, stds)
//! Out[7]: array([ 2, -1,  0,  2,  3], dtype=int32)
//!
//! In [8]: assert coder.is_empty()
//! ```

pub mod stream;
pub mod symbol;

use std::prelude::v1::*;

use pyo3::{prelude::*, wrap_pymodule};

/// ## Entropy Coding Primitives for Research and Production
///
/// The `constriction` Python and Rust libraries provide a wide range of efficient entropy
/// coding primitives that can be combined in flexible ways. Its goal is to facilitate
/// research on novel lossless and lossy compression methods by providing a simple API for
/// common algorithms like range coding or asymmetric numeral systems while, at the same,
/// also enabling specialized use cases like changing the fixed-point precision of entropy
/// models within a stream of symbols or bits-back coding with cyclic dependencies (TODO:
/// links).
///
/// ## Quick Start
///
/// If you already have an entropy model and you just want to encode and decode some
/// sequence of symbols then you probably want to have a look at the [`stream`](stream.html)
/// submodule, in particular the [Asymmetric Numeral Systems Coder](ans.html#ans.Ans) or the
/// [Range coder](range.html#range.Range). Or check out the example below.
///
/// ### Example
///
/// Let's encode a sequence of symbols and write the compressed data to a binary file. We'll
/// use a quantized Gaussian distribution as entropy model, with a different mean and
/// standard deviation of the Gaussian for each symbol so that the example is not too
/// simplistic.
///
/// We'll use an Asymmetric Numeral Systems (ANS) coder here for its speed and compression
/// performance, but we could as well have used a range coder by replacing, in the example
/// below, `constriction.stream.ans.Ans` with `constriction.stream.range.Range` and
/// `coder.encode_gaussian_symbols_reverse` by `coder.encode_gaussian_symbols` (since ANS is
/// a stack while range coding is a queue). We could also use a symbol code like Huffman
/// coding (using `constriction.symbol.EncoderHuffmanTree`) but that would have considerably
/// worse compression performance on larger files.
///
/// ```python
/// import constriction
/// import numpy as np
/// import sys
///
/// # Create an empty Asymmetric Numeral Systems (ANS) Coder:
/// coder = constriction.stream.ans.Ans()
///
/// # Some made up data with some made up entropy models:
/// min_supported_symbol, max_supported_symbol = -100, 100  # both inclusively
/// symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
/// means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
/// stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)
///
/// # Encode the data (in reverse order, since ANS is a stack):
/// coder.encode_gaussian_symbols_reverse(
///     symbols, min_supported_symbol, max_supported_symbol, means, stds)
///
/// print(f"Compressed size: {coder.num_bits()} bits")
/// print(f"(without unnecessary trailing zeros: {coder.num_valid_bits()} bits)")
///
/// # Get the compressed bit string, convert it into an architecture-independent
/// # byte order, and write it to a binary file:
/// compressed = coder.get_compressed()
/// if sys.byteorder == "big":
///     compressed.byteswap(inplace=True)
/// compressed.tofile("compressed.bin")
/// ```
///
/// Now let's read the compressed bit string back in and decode it.
///
/// ```python
/// import constriction
/// import numpy as np
/// import sys
///
/// # Read the compressed bit string from the file we created above and convert
/// # it into the right byte order for the current computer architecture:
/// compressed = np.fromfile("compressed.bin", dtype=np.uint32)
/// if sys.byteorder == "big":
///     compressed.byteswap(inplace=True)
///
/// # Initialize an ANS coder from the compressed bit string:
/// coder = constriction.stream.ans.Ans(compressed)
///
/// # Use the same entropy models that we used for encoding:
/// min_supported_symbol, max_supported_symbol = -100, 100  # both inclusively
/// means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
/// stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)
///
/// # Decode and print the data:
/// reconstructed = coder.decode_gaussian_symbols(
///     min_supported_symbol, max_supported_symbol, means, stds)
/// assert coder.is_empty()
/// print(reconstructed)  # Should print [23, -15, 78, 43, -69]
/// ```
///
/// ## Python API vs Rust API
///
/// This package provides the Python API for many typical use cases of the `constriction`
/// library. The Python API is meant for rapid experimentation with new entropy models,
/// e.g., for machine learning based compression methods. Internally, this Python module
/// runs native code that uses `constriction`'s API for the Rust programming language.
/// Having both a Python and a Rust API to the same implementation of common entropy coding
/// primitives provides the following advantages to users:
///
/// - Encoding and decoding run at native speed, much faster than any pure Python
///   implementation of the same algorithms could perform.
/// - Research code that uses `constriction`'s Python API can easily be turned into a
///   production-ready compression codec. Only the entropy model and any orchestration code
///   have to be ported to a more efficient and embeddable language (like Rust or to any
///   other compiled language with a compatible ABI like C or C++). The ported code can then
///   still use the same `constriction` library for entropy coding, thus maintaining
///   compatibility with the research code for reading and writing compressed data. This
///   makes it easy to turn experimental compression codecs into standalone compression
///   programs or libraries, or to use a specialized compression codec in a web app by
///   leveraging Rust's excellent tooling for WebAssembly.
/// - Since the Rust API provides much more fine grained control over details of the
///   provided entropy coding primitives due to Rust's powerful type system, Python users
///   with very specific needs can easily extend the Python API to their specialized use
///   case by copying existing Python wrappers and adjusting type parameters.
#[pymodule(constriction)]
fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(stream))?;
    module.add_wrapped(wrap_pymodule!(symbol))?;
    Ok(())
}

/// Stream Codes
///
/// TODO
#[pymodule]
fn stream(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    stream::init_module(py, module)
}

/// Symbol Codes
///
/// TODO
#[pymodule]
fn symbol(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    symbol::init_module(py, module)
}
