//! Wrapper types that expose functionality to Python<'_> 3 code
//!
//! This module is only compiled if the feature `pybindings` is turned on, which is
//! turned off by default.
//!
//! # Compiling the Python<'_> Extension Module
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
//!     cd ans
//!     cargo build --release --features pybindings
//!     ```
//!
//! 3. Check if the file `ans.so` exists in the top level directory. The git
//!     repository should contain this file, and it should be a symlink that points
//!     to the library you just compiled:
//!
//!     ```bash
//!     $ ls -l ans.so
//!     lrwxrwxrwx 1 user group Date Time ans.so -> target/release/libans.so
//!     ```
//!
//! # Example
//!
//! After compiling the python extension module as described above, `cd` to the
//! directory that contains the symlink `ans.so`, open a python REPL, and try it
//! out:
//!
//! ```bash
//! $ ipython3
//!
//! In [1]: import ans
//!    ...: import numpy as np
//!
//! In [2]: coder = ans.Coder()
//!
//! In [3]: symbols = np.array([2, -1, 0, 2], dtype = np.int32)
//!    ...: min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
//!    ...: means = np.array([2.3, -1.7, 0.1, 2.2], dtype = np.float64)
//!    ...: stds = np.array([1.1, 5.3, 3.8, 1.4], dtype = np.float64)
//!
//! In [4]: coder.push_gaussian_symbols(
//!    ...:     symbols, min_supported_symbol, max_supported_symbol, means, stds)
//!
//! In [5]: print(f"Compressed size (including constant overhead): {coder.num_bits()} bits")
//! Compressed size (including constant overhead): 64 bits
//!
//! In [6]: coder.get_compressed()
//! Out[6]: array([2372401017,        101], dtype=uint32)
//!
//! In [7]: coder.pop_gaussian_symbols(min_supported_symbol, max_supported_symbol, means, stds)
//! Out[7]: array([ 2, -1,  0,  2], dtype=int32)
//!
//! In [8]: assert coder.is_empty()
//! ```

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::distributions::{Categorical, LeakyQuantizer};
use statrs::distribution::Normal;

#[pymodule]
fn ans(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Coder>()?;
    Ok(())
}

/// An entropy coder based on [Asymmetric Numeral Systems (ANS)].
///
/// This is a wrapper around the Rust type [`ans::Coder<u32>`](crate::Coder)
/// with python bindings.
///
/// Note that this entropy coder is a stack (a "last in first out" data
/// structure). You can push symbols on the stack using the methods
/// `push_gaussian_symbols` or `push_iid_categorical_symbols`, and then pop
/// them off *in reverse order* using the methods `pop_gaussian_symbols` or
/// `pop_iid_categorical_symbols`, respectively.
///
/// To copy out the compressed data that is currently on the stack, call
/// `get_compressed`. You would typically want write this to a binary file in some
/// well-documented byte order. After reading it back in at a later time, you can
/// decompress it by constructing an `ans.Coder` where you pass in the compressed
/// data as an argument to the constructor.
///
/// If you're only interested in the compressed file size, calling `num_bits` will
/// be cheaper as it won't actually copy out the compressed data.
///
/// # Examples
///
/// ## Compression:
///
/// ```python
/// import sys
/// import ans
/// import numpy as np
///
/// coder = ans.Coder()
///
/// symbols = np.array([2, -1, 0, 2], dtype = np.int32)
/// min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
/// means = np.array([2.3, -1.7, 0.1, 2.2], dtype = np.float64)
/// stds = np.array([1.1, 5.3, 3.8, 1.4], dtype = np.float64)
///
/// coder.push_gaussian_symbols(
///     symbols, min_supported_symbol, max_supported_symbol, means, stds)
///
/// print(f"Compressed size (including constant overhead): {coder.num_bits()} bits")
///
/// compressed = coder.get_compressed()
/// if sys.byteorder == "big":
///     # Convert native byte order to a consistent one (here: little endian).
///     compressed.byteswap(inplace=True)
/// compressed.tofile("compressed.bin")
/// ```
///
/// ## Decompression:
///
/// ```python
/// import sys
/// import ans
/// import numpy as np
///
/// compressed = np.fromfile("compressed.bin")
/// if sys.byteorder == "big":
///     # Convert little endian byte order to native byte order.
///     compressed.byteswap(inplace=True)
///
/// coder = ans.Coder(compressed)
///
/// min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
/// means = np.array([2.3, -1.7, 0.1, 2.2], dtype = np.float64)
/// stds = np.array([1.1, 5.3, 3.8, 1.4], dtype = np.float64)
///
/// reconstructed = coder.pop_gaussian_symbols(
///     min_supported_symbol, max_supported_symbol, means, stds)
/// assert coder.is_empty()
/// ```
///
/// # Constructor
///
/// Coder(compressed)
///
/// Arguments:
/// compressed (optional) -- initial compressed data, as a numpy array with
///     dtype `uint32`.
///
/// [Asymmetric Numeral Systems (ANS)]: https://en.wikipedia.org/wiki/Asymmetric_numeral_systems
#[pyclass]
#[text_signature = "(compressed)"]
#[derive(Debug)]
pub struct Coder {
    inner: super::DefaultCoder,
}

#[pymethods]
impl Coder {
    /// Constructs a new entropy coder, optionally passing initial compressed data.
    #[new]
    pub fn new(compressed: Option<PyReadonlyArray1<'_, u32>>) -> PyResult<Self> {
        let inner = if let Some(compressed) = compressed {
            super::Coder::with_compressed_data(compressed.to_vec()?)
        } else {
            super::Coder::new()
        };

        Ok(Self { inner })
    }

    /// Resets the coder for compression.
    ///
    /// After calling this method, the method `is_empty` will return `True`.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// The current size of the compressed data, in `np.uint32` words.
    pub fn num_words(&self) -> usize {
        self.inner.num_words()
    }

    /// The current size of the compressed data, in bits.
    pub fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    /// Returns `True` iff the coder is in its default initial state.
    ///
    /// The default initial state is the state returned by the constructor when
    /// called without arguments, or the state to which the coder is set when
    /// calling `clear`.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Copies the compressed data to the provided numpy array.
    ///
    /// The argument `destination` must by a one-dimensional numpy array with
    /// dtype `uint32` and with the exact correct size. Use the method `num_words`
    /// to find out the correct size.
    ///
    /// Example:
    ///
    /// ```python
    /// coder = ans.Coder()
    /// # ... push some symbols on coder ...
    /// compressed_len = coder.num_words()
    /// compressed = np.empty((compressed_len,), dtype=np.uint32)
    /// coder.copy_compressed(compressed)
    ///
    /// # Optional: write the compressed data to a file in
    /// #           platform-independent byte ordering.
    /// if sys.byteorder == "big":
    ///     compressed.byteswap()
    /// with open("path/to/file", "wb") as file:
    ///     compressed.tofile(file)
    /// ```
    pub fn get_compressed<'p>(&mut self, py: Python<'p>) -> &'p PyArray1<u32> {
        PyArray1::from_slice(py, &*self.inner.get_compressed())
    }

    /// Encodes a sequence of symbols using (leaky) Gaussian entropy models.
    ///
    /// The provided numpy arrays `symbols`, `means`, and `stds` must all have the
    /// same size.
    ///
    /// See method `pop_gaussian_symbols` for a usage example.
    ///
    /// Arguments:
    /// min_supported_symbol -- lower bound of the domain for argument `symbols`
    ///     (inclusively).
    /// max_supported_symbol -- upper bound of the domain for argument `symbols`
    ///     (inclusively).
    /// symbols -- the symbols to be encoded. Must be a contiguous one-dimensional
    ///     numpy array (call `.copy()` on it if it is not contiguous) with dtype
    ///     `np.int32`. Each value in the array must be no smaller than
    ///     `min_supported_symbol` and no larger than `max_supported_symbol`.
    /// means -- the mean values of the Gaussian entropy models for each symbol.
    ///     Must be a contiguous one-dimensional numpy array with dtype `np.float64`
    ///     and with the exact same length as the argument `symbols`.
    /// stds -- the standard deviations of the Gaussian entropy models for each
    ///     symbol. Must be a contiguous one-dimensional numpy array with dtype
    ///     `np.float64` and with the exact same length as the argument `symbols`.
    ///     All entries must be strictly positive (i.e., nonzero and nonnegative)
    ///     and finite.
    #[text_signature = "(symbols, min_supported_symbol, max_supported_symbol, means, stds)"]
    pub fn push_gaussian_symbols(
        &mut self,
        symbols: PyReadonlyArray1<'_, i32>,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        means: PyReadonlyArray1<'_, f64>,
        stds: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let (symbols, means, stds) = (symbols.as_slice()?, means.as_slice()?, stds.as_slice()?);
        if symbols.len() != means.len() || symbols.len() != stds.len() {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "`symbols`, `means`, and `stds` must all have the same length.",
            ));
        }

        let quantizer = LeakyQuantizer::new(min_supported_symbol..=max_supported_symbol);
        self.inner
            .try_push_symbols(symbols.iter().zip(means.iter()).zip(stds.iter()).map(
                |((&symbol, &mean), &std)| {
                    Normal::new(mean, std)
                        .map(|distribution| (symbol, quantizer.quantize(distribution)))
                },
            ))?;

        Ok(())
    }

    /// Decodes a sequence of symbols *in reverse order* using (leaky) Gaussian entropy
    /// models.
    ///
    /// The provided numpy arrays `means`, `stds`, and `symbols_out` must all have
    /// the same size. The provided `means`, `stds`, `min_supported_symbol`,
    /// `max_supported_symbol`, and `leaky` must be the exact same values that were
    /// used for encoding. Even a tiny modification of these arguments can cause the
    /// coder to decode *completely* different symbols.
    ///
    /// The symbols will be popped off the stack and written to the target array in
    /// reverseorder so as to simplify usage, e.g.:
    ///
    /// ```python
    /// coder = ans.Coder()
    /// symbols = np.array([2, 8, -5], dtype=np.int32)
    /// decoded = np.empty((3,), dtype=np.int32)
    /// means = np.array([0.1, 10.3, -3.2], dtype=np.float64)
    /// stds = np.array([3.2, 1.3, 1.9], dtype=np.float64)
    ///
    /// # Push symbols on the stack:
    /// coder.push_gaussian_symbols(symbols, -10, 10, means, stds, True)
    ///
    /// # Pop symbols off the stack in reverse order:
    /// coder.pop_gaussian_symbols(-10, 10, means, stds, decoded, True)
    ///
    /// # Verify that the decoded symbols match the encoded ones.
    /// assert np.all(symbols == decoded)
    /// assert coder.is_empty()
    /// ```
    ///
    /// Arguments:
    /// min_supported_symbol -- lower bound of the domain supported by the entropy
    ///     model (inclusively). Must be the same value that was used for encoding.
    /// max_supported_symbol -- upper bound of the domain supported by the entropy
    ///     model (inclusively). Must be the same value that was used for encoding.
    /// means -- the mean values of the Gaussian entropy models for each symbol.
    ///     Must be a contiguous one-dimensional numpy array with dtype `float64`
    ///     and with the exact same length as the argument `symbols_out`.
    /// stds -- the standard deviations of the Gaussian entropy models for each
    ///     symbol. Must be a contiguous one-dimensional numpy array with dtype
    ///     `float64` and with the exact same length as the argument `symbols_out`.
    pub fn pop_gaussian_symbols<'p>(
        &mut self,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        means: PyReadonlyArray1<'_, f64>,
        stds: PyReadonlyArray1<'_, f64>,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray1<i32>> {
        if means.len() != stds.len() {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "`means`, and `stds` must have the same length.",
            ));
        }

        let quantizer = LeakyQuantizer::new(min_supported_symbol..=max_supported_symbol);
        let symbols = self
            .inner
            .try_pop_symbols(means.iter()?.zip(stds.iter()?).map(|(&mean, &std)| {
                Normal::new(mean, std).map(|distribution| quantizer.quantize(distribution))
            }))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(PyArray1::from_vec(py, symbols))
    }

    /// Encodes a sequence of symbols using a fixed categorical distribution.
    ///
    /// This method is analogous to the method `push_gaussian_symbols` except that
    /// - all symbols are encoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian distribution.
    ///
    /// In detail, the categorical entropy model is constructed as follows:
    /// - each symbol from `min_supported_symbol` to `max_supported_symbol`
    ///   (inclusively) gets assigned at least the smallest nonzero probability
    ///   that is representable within the internally used precision.
    /// - the remaining probability mass is distributed among the symbols from
    ///   `min_provided_symbol` to `min_provided_symbol + len(probabilities) - 1`
    ///   (inclusively), in the proportions specified by the provided probabilities
    ///   (as far as this is possible within the internally used fixed point
    ///   accuracy). The provided probabilities do not need to be normalized (i.e.,
    ///   the do not need to add up to one) but they must all be nonnegative.
    pub fn push_iid_categorical_symbols(
        &mut self,
        symbols: PyReadonlyArray1<'_, i32>,
        min_supported_symbol: i32,
        probabilities: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let distribution = Categorical::from_floating_point_probabilities(
            probabilities.as_slice()?,
        )
        .map_err(|()| {
            pyo3::exceptions::PyValueError::new_err(
                "Probability distribution is either degenerate or not normalizable.",
            )
        })?;

        self.inner.push_iid_symbols(
            symbols
                .as_slice()?
                .iter()
                .map(|s| s.wrapping_sub(min_supported_symbol) as usize),
            &distribution,
        )?;

        Ok(())
    }

    /// Decodes a sequence of categorically distributed symbols *in reverse order*.
    ///
    /// This method is analogous to the method `pop_gaussian_symbols` except that
    /// - all symbols are decoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian distribution.
    ///
    /// See documentation of `push_iid_categorical_symbols` for details of the
    /// categorical entropy model. See documentation of `pop_gaussian_symbols` for a
    /// discussion of the reverse order of decoding, and for a related usage
    /// example.
    pub fn pop_iid_categorical_symbols<'p>(
        &mut self,
        amt: usize,
        min_supported_symbol: i32,
        probabilities: PyReadonlyArray1<'_, f64>,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray1<i32>> {
        let distribution = Categorical::from_floating_point_probabilities(
            probabilities.as_slice()?,
        )
        .map_err(|()| {
            pyo3::exceptions::PyValueError::new_err(
                "Probability distribution is either degenerate or not normalizable.",
            )
        })?;

        Ok(PyArray1::from_iter(
            py,
            self.inner
                .pop_iid_symbols(amt, &distribution)
                .map(|s| (s as i32).wrapping_add(min_supported_symbol)),
        ))
    }
}

impl From<super::CoderError> for PyErr {
    fn from(err: super::CoderError) -> Self {
        match err {
            crate::CoderError::ImpossibleSymbol => pyo3::exceptions::PyKeyError::new_err(
                "Tried to encode symbol that has zero probability under entropy model.",
            ),
            crate::CoderError::IterationError(_) => pyo3::exceptions::PyValueError::new_err(
                "Invalid parameters for probability distribution.",
            ),
        }
    }
}
