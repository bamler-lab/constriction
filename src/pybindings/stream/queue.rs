use std::prelude::v1::*;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use probability::distribution::Gaussian;
use pyo3::prelude::*;

use crate::{
    stream::{
        model::{DefaultContiguousCategoricalEntropyModel, DefaultLeakyQuantizer},
        Decode, Encode,
    },
    UnwrapInfallible,
};

use super::model::CustomModel;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<RangeEncoder>()?;
    module.add_class::<RangeDecoder>()?;
    Ok(())
}

/// TODO: document
#[pyclass]
#[pyo3(text_signature = "()")]
#[derive(Debug, Default)]
pub struct RangeEncoder {
    inner: crate::stream::queue::DefaultRangeEncoder,
}

#[pymethods]
impl RangeEncoder {
    /// Constructs a new (empty) range encoder.
    #[new]
    pub fn new() -> Self {
        let inner = crate::stream::queue::DefaultRangeEncoder::new();
        Self { inner }
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

    /// Returns `True` iff the coder is in its default initial state.
    ///
    /// The default initial state is the state returned by the constructor when
    /// called without arguments, or the state to which the coder is set when
    /// calling `clear`.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a copy of the compressed data.
    pub fn get_compressed<'p>(&mut self, py: Python<'p>) -> &'p PyArray1<u32> {
        PyArray1::from_slice(py, &*self.inner.get_compressed())
    }

    #[pyo3(text_signature = "()")]
    pub fn get_decoder(&mut self) -> RangeDecoder {
        let compressed = self.inner.get_compressed().to_vec();
        RangeDecoder::from_vec(compressed)
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
    #[pyo3(text_signature = "(symbols, min_supported_symbol, max_supported_symbol, means, stds)")]
    pub fn encode_leaky_gaussian_symbols(
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
        self.inner
            .try_encode_symbols(symbols.iter().zip(means.iter()).zip(stds.iter()).map(
                |((&symbol, &mean), &std)| {
                    if std > 0.0 && std.is_finite() && mean.is_finite() {
                        Ok((symbol, quantizer.quantize(Gaussian::new(mean, std))))
                    } else {
                        Err(())
                    }
                },
            ))?;

        Ok(())
    }

    /// Encodes a sequence of symbols using a fixed categorical entropy model.
    ///
    /// This method is analogous to the method `encode_leaky_gaussian_symbols` except that
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
    pub fn encode_iid_categorical_symbols(
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

        self.inner.encode_iid_symbols(
            symbols
                .as_slice()?
                .iter()
                .map(|s| s.wrapping_sub(min_supported_symbol) as usize),
            &model,
        )?;

        Ok(())
    }

    /// Encodes a sequence of symbols with identical custom models.
    ///
    /// - For usage examples, see
    ///   [`CustomModel`](model.html#constriction.stream.model.CustomModel).
    /// - If the model parameters are different for each symbol then you'll want to use
    ///   [`encode_custom_model`](#constriction.stream.queue.RangeEncoder.encode_custom_model)
    ///   instead.
    #[pyo3(text_signature = "(symbols, model)")]
    pub fn encode_iid_custom_model<'py>(
        &mut self,
        symbols: PyReadonlyArray1<'_, i32>,
        model: &CustomModel,
        py: Python<'py>,
    ) -> PyResult<()> {
        self.inner
            .encode_iid_symbols(symbols.as_slice()?, model.quantized(py))?;
        Ok(())
    }

    /// Encodes a sequence of symbols with parameterized custom models.
    ///
    /// - For usage examples, see
    ///   [`CustomModel`](model.html#constriction.stream.model.CustomModel).
    /// - If all symbols use the same entropy model (with identical model parameters) then
    ///   you'll want to use
    ///   [`encode_iid_custom_model`](#constriction.stream.queue.RangeEncoder.encode_iid_custom_model)
    ///   instead.
    #[pyo3(text_signature = "(symbols, model, model_parameters)")]
    pub fn encode_custom_model<'py>(
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
            .encode_symbols(symbols.as_slice()?.iter().zip(models))?;
        Ok(())
    }
}

/// TODO: document
#[pyclass]
#[pyo3(text_signature = "(compressed)")]
#[derive(Debug)]
pub struct RangeDecoder {
    inner: crate::stream::queue::DefaultRangeDecoder,
}

#[pymethods]
impl RangeDecoder {
    /// Constructs a new (empty) range encoder.
    #[new]
    pub fn new(compressed: PyReadonlyArray1<'_, u32>) -> PyResult<Self> {
        Ok(Self::from_vec(compressed.to_vec()?))
    }

    pub fn maybe_exhausted(&self) -> bool {
        self.inner.maybe_exhausted()
    }

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
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("We use constant `PRECISION`.");

        Ok(PyArray1::from_vec(py, symbols))
    }

    /// Decodes a sequence of categorically distributed symbols
    ///
    /// This method is analogous to the method `decode_leaky_gaussian_symbols` except that
    ///
    /// - all symbols are decoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian model.
    ///
    /// See documentation of `encode_iid_categorical_symbols` for details of the
    /// categorical entropy model.
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

        Ok(PyArray1::from_iter(
            py,
            self.inner.decode_iid_symbols(amt, &model).map(|symbol| {
                (symbol.expect("We use constant `PRECISION`.") as i32)
                    .wrapping_add(min_supported_symbol)
            }),
        ))
    }

    /// Decodes a sequence of symbols with identical custom models.
    ///
    /// - For usage examples, see
    ///   [`CustomModel`](model.html#constriction.stream.model.CustomModel).
    /// - If the model parameters are different for each symbol then you'll want to use
    ///   [`decode_custom_model`](#constriction.stream.queue.RangeDecoder.decode_custom_model)
    ///   instead.
    #[pyo3(text_signature = "(amt, model)")]
    pub fn decode_iid_custom_model<'py>(
        &mut self,
        amt: usize,
        model: &CustomModel,
        py: Python<'py>,
    ) -> PyResult<&'py PyArray1<i32>> {
        Ok(PyArray1::from_iter(
            py,
            self.inner
                .decode_iid_symbols(amt, model.quantized(py))
                .map(|symbol| symbol.expect("We use constant `PRECISION`.") as i32),
        ))
    }

    /// Decodes a sequence of symbols with parameterized custom models.
    ///
    /// - For usage examples, see
    ///   [`CustomModel`](model.html#constriction.stream.model.CustomModel).
    /// - If all symbols use the same entropy model (with identical model parameters) then
    ///   you'll want to use
    ///   [`decode_iid_custom_model`](#constriction.stream.queue.RangeDecoder.decode_iid_custom_model)
    ///   instead.
    #[pyo3(text_signature = "(model, model_parameters)")]
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
            .map(|symbol| symbol.expect("We use constant `PRECISION`.") as i32)
            .collect::<Vec<_>>();

        Ok(PyArray1::from_vec(py, symbols))
    }
}

impl RangeDecoder {
    pub fn from_vec(compressed: Vec<u32>) -> Self {
        let inner = crate::stream::queue::DefaultRangeDecoder::from_compressed(compressed)
            .unwrap_infallible();
        Self { inner }
    }
}
