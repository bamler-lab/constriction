use core::convert::Infallible;
use std::{prelude::v1::*, vec};

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::{
    stream::{
        models::{LeakyCategorical, LeakyQuantizer},
        Decode, TryCodingError,
    },
    CoderError, DefaultEncoderFrontendError, UnwrapInfallible,
};

use probability::distribution::Gaussian;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<AnsCoder>()?;
    Ok(())
}

/// An entropy coder based on [Asymmetric Numeral Systems (ANS)].
///
/// This is a wrapper around the Rust type [`constriction::stream::stack::DefaultAnsCoder`]
/// with python bindings.
///
/// Note that this entropy coder is a stack (a "last in first out" data
/// structure). You can push symbols on the stack using the methods
/// `encode_leaky_gaussian_symbols_reverse` or `encode_iid_categorical_symbols_reverse`, and then pop
/// them off *in reverse order* using the methods `decode_leaky_gaussian_symbols` or
/// `decode_iid_categorical_symbols`, respectively.
///
/// To copy out the compressed data that is currently on the stack, call
/// `get_compressed`. You would typically want write this to a binary file in some
/// well-documented byte order. After reading it back in at a later time, you can
/// decompress it by constructing an `constriction.AnsCoder` where you pass in the compressed
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
/// import constriction
/// import numpy as np
///
/// ans = constriction.AnsCoder()  # Creates an empty ANS coder when called with no arguments.
///
/// symbols = np.array([2, -1, 0, 2, 3], dtype = np.int32)
/// min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
/// means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype = np.float64)
/// stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype = np.float64)
///
/// ans.encode_leaky_gaussian_symbols_reverse(
///     symbols, min_supported_symbol, max_supported_symbol, means, stds)
///
/// print(f"Compressed size: {ans.num_valid_bits()} bits")
///
/// compressed = ans.get_compressed()
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
/// import constriction
/// import numpy as np
///
/// compressed = np.fromfile("compressed.bin", dtype=np.uint32)
/// if sys.byteorder == "big":
///     # Convert little endian byte order to native byte order.
///     compressed.byteswap(inplace=True)
///
/// ans = constriction.AnsCoder(compressed)
///
/// min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
/// means = np.array([2.3, -1.7, 0.1, 2.2], -5.1, dtype = np.float64)
/// stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype = np.float64)
///
/// reconstructed = ans.decode_leaky_gaussian_symbols(
///     min_supported_symbol, max_supported_symbol, means, stds)
/// assert ans.is_empty()
/// ```
///
/// # Constructor
///
/// AnsCoder(compressed)
///
/// Arguments:
/// compressed (optional) -- initial compressed data, as a numpy array with
///     dtype `uint32`.
///
/// [Asymmetric Numeral Systems (ANS)]: https://en.wikipedia.org/wiki/Asymmetric_numeral_systems
/// [`constriction::stream::ans::DefaultAnsCoder`]: crate::stream::stack::DefaultAnsCoder
#[pyclass]
#[text_signature = "(compressed)"]
#[derive(Debug)]
pub struct AnsCoder {
    inner: crate::stream::stack::DefaultAnsCoder,
}

#[pymethods]
impl AnsCoder {
    /// Constructs a new entropy coder, optionally passing initial compressed data.
    #[new]
    pub fn new(compressed: Option<PyReadonlyArray1<'_, u32>>) -> PyResult<Self> {
        let inner = if let Some(compressed) = compressed {
            crate::stream::stack::AnsCoder::from_compressed(compressed.to_vec()?).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "Invalid compressed data: ANS compressed data never ends in a zero word.",
                )
            })?
        } else {
            crate::stream::stack::AnsCoder::new()
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

    /// The current size of the compressed data, in bits, rounded up to full words.
    pub fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    /// The current size of the compressed data, in bits, not rounded up to full words.
    pub fn num_valid_bits(&self) -> usize {
        self.inner.num_valid_bits()
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
    /// coder = constriction.AnsCoder()
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
        PyArray1::from_slice(py, &*self.inner.get_compressed().unwrap())
    }

    /// Encodes a sequence of symbols using (leaky) Gaussian entropy models.
    ///
    /// The provided numpy arrays `symbols`, `means`, and `stds` must all have the
    /// same size.
    ///
    /// See method `decode_leaky_gaussian_symbols` for a usage example.
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
    pub fn encode_leaky_gaussian_symbols_reverse(
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

        let quantizer =
            LeakyQuantizer::<_, _, u32, 24>::new(min_supported_symbol..=max_supported_symbol);
        self.inner.try_encode_symbols_reverse(
            symbols
                .iter()
                .zip(means.iter())
                .zip(stds.iter())
                .map(|((&symbol, &mean), &std)| {
                    if std > 0.0 && std.is_finite() && mean.is_finite() {
                        Ok((symbol, quantizer.quantize(Gaussian::new(mean, std))))
                    } else {
                        Err(())
                    }
                }),
        )?;

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
    /// coder = constriction.AnsCoder()
    /// symbols = np.array([2, 8, -5], dtype=np.int32)
    /// decoded = np.empty((3,), dtype=np.int32)
    /// means = np.array([0.1, 10.3, -3.2], dtype=np.float64)
    /// stds = np.array([3.2, 1.3, 1.9], dtype=np.float64)
    ///
    /// # Push symbols on the stack:
    /// coder.encode_leaky_gaussian_symbols_reverse(symbols, -10, 10, means, stds, True)
    ///
    /// # Pop symbols off the stack in reverse order:
    /// coder.decode_leaky_gaussian_symbols(-10, 10, means, stds, decoded, True)
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
    pub fn decode_leaky_gaussian_symbols<'p>(
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

        let quantizer =
            LeakyQuantizer::<_, _, u32, 24>::new(min_supported_symbol..=max_supported_symbol);
        let symbols = self
            .inner
            .try_decode_symbols(means.iter()?.zip(stds.iter()?).map(|(&mean, &std)| {
                if std > 0.0 && std.is_finite() && mean.is_finite() {
                    Ok(quantizer.quantize(Gaussian::new(mean, std)))
                } else {
                    Err(())
                }
            }))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(PyArray1::from_vec(py, symbols))
    }

    /// Encodes a sequence of symbols using a fixed categorical entropy model.
    ///
    /// This method is analogous to the method `encode_leaky_gaussian_symbols_reverse` except that
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
    pub fn encode_iid_categorical_symbols_reverse(
        &mut self,
        symbols: PyReadonlyArray1<'_, i32>,
        min_supported_symbol: i32,
        probabilities: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let model = LeakyCategorical::<u32, 24>::from_floating_point_probabilities(
            probabilities.as_slice()?,
        )
        .map_err(|()| {
            pyo3::exceptions::PyValueError::new_err(
                "Probability model is either degenerate or not normalizable.",
            )
        })?;

        self.inner.encode_iid_symbols_reverse(
            symbols
                .as_slice()?
                .iter()
                .map(|s| s.wrapping_sub(min_supported_symbol) as usize),
            &model,
        )?;

        Ok(())
    }

    /// Decodes a sequence of categorically distributed symbols *in reverse order*.
    ///
    /// This method is analogous to the method `decode_leaky_gaussian_symbols` except that
    /// - all symbols are decoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian model.
    ///
    /// See documentation of `encode_iid_categorical_symbols_reverse` for details of the
    /// categorical entropy model. See documentation of `decode_leaky_gaussian_symbols` for a
    /// discussion of the reverse order of decoding, and for a related usage
    /// example.
    pub fn decode_iid_categorical_symbols<'p>(
        &mut self,
        amt: usize,
        min_supported_symbol: i32,
        probabilities: PyReadonlyArray1<'_, f64>,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray1<i32>> {
        let model = LeakyCategorical::<u32, 24>::from_floating_point_probabilities(
            probabilities.as_slice()?,
        )
        .map_err(|()| {
            pyo3::exceptions::PyValueError::new_err(
                "Probability distribution is either degenerate or not normalizable.",
            )
        })?;

        Ok(PyArray1::from_iter(
            py,
            self.inner.decode_iid_symbols(amt, &model).map(|symbol| {
                (symbol.unwrap_infallible() as i32).wrapping_add(min_supported_symbol)
            }),
        ))
    }
}

impl<CodingError: Into<PyErr>, ModelError> From<TryCodingError<CodingError, ModelError>> for PyErr {
    fn from(err: TryCodingError<CodingError, ModelError>) -> Self {
        match err {
            crate::stream::TryCodingError::CodingError(err) => err.into(),
            crate::stream::TryCodingError::InvalidEntropyModel(_) => {
                pyo3::exceptions::PyValueError::new_err("Invalid parameters for entropy model")
            }
        }
    }
}

impl<FrontendError: Into<PyErr>> From<CoderError<FrontendError, Infallible>> for PyErr {
    fn from(err: CoderError<FrontendError, Infallible>) -> Self {
        match err {
            CoderError::Frontend(err) => err.into(),
            CoderError::Backend(infallible) => Err(infallible).unwrap_infallible(),
        }
    }
}

impl From<DefaultEncoderFrontendError> for PyErr {
    fn from(err: DefaultEncoderFrontendError) -> Self {
        match err {
            DefaultEncoderFrontendError::ImpossibleSymbol => {
                pyo3::exceptions::PyKeyError::new_err(err.to_string())
            }
        }
    }
}
