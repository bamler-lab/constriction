use std::{prelude::v1::*, vec};

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use probability::distribution::Gaussian;
use pyo3::prelude::*;

use crate::{
    stream::{
        chain::{BackendError, DecoderFrontendError, EncoderFrontendError},
        model::{DefaultContiguousCategoricalEntropyModel, DefaultLeakyQuantizer},
        Decode,
    },
    UnwrapInfallible,
};

use super::model::CustomModel;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<ChainCoder>()?;
    Ok(())
}

#[pyclass]
#[text_signature = "(compressed, is_remaining=False, seal=False)"]
#[derive(Debug)]
pub struct ChainCoder {
    inner: crate::stream::chain::DefaultChainCoder,
}

#[pymethods]
impl ChainCoder {
    /// Constructs a new chain coder, optionally passing initial compressed data.
    #[new]
    pub fn new(
        data: PyReadonlyArray1<'_, u32>,
        is_remaining: Option<bool>,
        seal: Option<bool>,
    ) -> PyResult<Self> {
        let data = data.to_vec()?;
        let inner = if is_remaining == Some(true) {
            if seal == Some(true) {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "Cannot seal remaining data.",
                ));
            } else {
                crate::stream::chain::ChainCoder::from_remaining(data).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Too little data provided, or provided data ends in zero word and `is_remaining==True`.",
                    )
                })?
            }
        } else if seal == Some(true) {
            crate::stream::chain::ChainCoder::from_binary(data)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Too little data provided."))?
        } else {
            crate::stream::chain::ChainCoder::from_compressed(data).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Too little data provided, or provided data ends in zero word and `seal==False`.",
                    )
                })?
        };

        Ok(Self { inner })
    }

    /// Returns a copy of the compressed data and the remainders.
    pub fn get_data<'p>(
        &self,
        unseal: Option<bool>,
        py: Python<'p>,
    ) -> PyResult<(&'p PyArray1<u32>, &'p PyArray1<u32>)> {
        let cloned = self.inner.clone();
        let data = if unseal == Some(true) {
            cloned.into_binary()
        } else {
            cloned.into_compressed()
        };
        let (remaining, compressed) = data.map_err(|_| {
            pyo3::exceptions::PyAssertionError::new_err(
                "Fractional number of words in compressed or remaining data.",
            )
        })?;

        let remaining = PyArray1::from_vec(py, remaining);
        let compressed = PyArray1::from_vec(py, compressed);
        Ok((remaining, compressed))
    }

    pub fn get_remaining<'p>(
        &self,
        py: Python<'p>,
    ) -> PyResult<(&'p PyArray1<u32>, &'p PyArray1<u32>)> {
        let (compressed, remaining) = self.inner.clone().into_remaining().unwrap_infallible();
        let remaining = PyArray1::from_vec(py, remaining);
        let compressed = PyArray1::from_vec(py, compressed);
        Ok((compressed, remaining))
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

        let quantizer = DefaultLeakyQuantizer::new(min_supported_symbol..=max_supported_symbol);
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
    /// reverse order so as to simplify usage.
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

        let quantizer = DefaultLeakyQuantizer::new(min_supported_symbol..=max_supported_symbol);
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
    ///
    /// - all symbols are encoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian distribution.
    ///
    /// In detail, the categorical entropy model is constructed as follows:
    ///
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
        let model = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(
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
    ///
    /// - all symbols are decoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian model.
    ///
    /// See documentation of `encode_iid_categorical_symbols_reverse` for details of the
    /// categorical entropy model. See documentation of `decode_leaky_gaussian_symbols` for a
    /// discussion of the reverse order of decoding, and for a related usage
    /// example.
    pub fn decode_iid_categorical_symbols<'py>(
        &mut self,
        amt: usize,
        min_supported_symbol: i32,
        probabilities: PyReadonlyArray1<'_, f64>,
        py: Python<'py>,
    ) -> PyResult<&'py PyArray1<i32>> {
        let model = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(
            probabilities.as_slice()?,
        )
        .map_err(|()| {
            pyo3::exceptions::PyValueError::new_err(
                "Probability distribution is either degenerate or not normalizable.",
            )
        })?;

        let symbols = self
            .inner
            .decode_iid_symbols(amt, &model)
            .map(|symbol| symbol.map(|symbol| (symbol as i32).wrapping_add(min_supported_symbol)))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(PyArray1::from_vec(py, symbols))
    }

    #[text_signature = "(symbols, model)"]
    pub fn encode_iid_custom_model_reverse<'py>(
        &mut self,
        symbols: PyReadonlyArray1<'_, i32>,
        model: &CustomModel,
        py: Python<'py>,
    ) -> PyResult<()> {
        self.inner
            .encode_iid_symbols_reverse(symbols.as_slice()?, model.quantized(py))?;
        Ok(())
    }

    #[text_signature = "(amt, model)"]
    pub fn decode_iid_custom_model<'py>(
        &mut self,
        amt: usize,
        model: &CustomModel,
        py: Python<'py>,
    ) -> PyResult<&'py PyArray1<i32>> {
        let symbols = self
            .inner
            .decode_iid_symbols(amt, model.quantized(py))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(PyArray1::from_vec(py, symbols))
    }

    #[text_signature = "(symbols, model, model_parameters)"]
    pub fn encode_custom_model_reverse<'py>(
        &mut self,
        symbols: PyReadonlyArray1<'_, i32>,
        model: &CustomModel,
        model_parameters: PyReadonlyArray2<'_, f64>,
        py: Python<'py>,
    ) -> PyResult<()> {
        let dims = model_parameters.dims();
        let num_symbols = dims[0];
        let num_parameters = dims[1];
        if symbols.len() != num_symbols {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "`len(symbols)` must match first dimension of `model_parameters`.",
            ));
        }

        let model_parameters = model_parameters.as_slice()?.chunks_exact(num_parameters);
        let models = model_parameters.map(|params| {
            model.quantized_with_parameters(py, PyArray1::from_vec(py, params.to_vec()).readonly())
        });
        self.inner
            .encode_symbols_reverse(symbols.as_slice()?.iter().zip(models))?;
        Ok(())
    }

    #[text_signature = "(model, model_parameters)"]
    pub fn decode_custom_model<'py>(
        &mut self,
        model: &CustomModel,
        model_parameters: PyReadonlyArray2<'_, f64>,
        py: Python<'py>,
    ) -> PyResult<&'py PyArray1<i32>> {
        let num_parameters = model_parameters.dims()[1];
        let model_parameters = model_parameters.as_slice()?.chunks_exact(num_parameters);
        let models = model_parameters.map(|params| {
            model.quantized_with_parameters(py, PyArray1::from_vec(py, params.to_vec()).readonly())
        });

        let symbols = self
            .inner
            .decode_symbols(models)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PyArray1::from_vec(py, symbols))
    }
}

impl From<EncoderFrontendError> for PyErr {
    fn from(err: EncoderFrontendError) -> Self {
        match err {
            EncoderFrontendError::ImpossibleSymbol => {
                pyo3::exceptions::PyKeyError::new_err(err.to_string())
            }
            EncoderFrontendError::OutOfRemaining => {
                pyo3::exceptions::PyAssertionError::new_err(err.to_string())
            }
        }
    }
}

impl From<DecoderFrontendError> for PyErr {
    fn from(err: DecoderFrontendError) -> Self {
        match err {
            DecoderFrontendError::OutOfCompressedData => {
                pyo3::exceptions::PyAssertionError::new_err(err.to_string())
            }
        }
    }
}

impl<CompressedBackendError: Into<PyErr>, RemainingBackendError: Into<PyErr>>
    From<BackendError<CompressedBackendError, RemainingBackendError>> for PyErr
{
    fn from(err: BackendError<CompressedBackendError, RemainingBackendError>) -> Self {
        match err {
            BackendError::Compressed(err) => err.into(),
            BackendError::Remaining(err) => err.into(),
        }
    }
}
