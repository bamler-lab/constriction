use std::prelude::v1::*;

use numpy::{PyArray1, PyReadonlyArray1};
use probability::distribution::Gaussian;
use pyo3::{prelude::*, types::PyTuple};

use crate::{
    pybindings::stream::model::Model,
    stream::{
        chain::{BackendError, DecoderFrontendError, EncoderFrontendError},
        model::{DefaultContiguousCategoricalEntropyModel, DefaultLeakyQuantizer},
        Decode, Encode,
    },
    UnwrapInfallible,
};

use super::model::internals::EncoderDecoderModel;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<ChainCoder>()?;
    Ok(())
}

/// See [above usage instructions](#usage-for-bits-back-coding) for explanation of
/// constructor arguments.
#[pyclass]
#[pyo3(text_signature = "(compressed, is_remainders=False, seal=False)")]
#[derive(Debug, Clone)]
pub struct ChainCoder {
    inner: crate::stream::chain::DefaultChainCoder,
}

#[pymethods]
impl ChainCoder {
    #[new]
    pub fn new(
        data: PyReadonlyArray1<'_, u32>,
        is_remainders: Option<bool>,
        seal: Option<bool>,
    ) -> PyResult<Self> {
        let data = data.to_vec()?;
        let inner = if is_remainders == Some(true) {
            if seal == Some(true) {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "Cannot seal remainders data.",
                ));
            } else {
                crate::stream::chain::ChainCoder::from_remainders(data).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Too little data provided, or provided data ends in zero word and `is_remainders==True`.",
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

    /// Returns a copy of the compressed data after re-encoding symbols, split into two
    /// arrays that you may want to concatenate.
    ///
    /// See [above usage instructions](#usage-for-bits-back-coding) for further explanation.
    #[pyo3(text_signature = "(unseal=False)")]
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
        let (remainders, compressed) = data.map_err(|_| {
            pyo3::exceptions::PyAssertionError::new_err(
                "Fractional number of words in compressed or remainders data.",
            )
        })?;

        let remainders = PyArray1::from_vec(py, remainders);
        let compressed = PyArray1::from_vec(py, compressed);
        Ok((remainders, compressed))
    }

    /// Returns a copy of the remainders after decoding some symbols, split into two arrays
    /// that you may want to concatenate.
    ///
    /// See [above usage instructions](#usage-for-bits-back-coding) for further explanation.
    #[pyo3(text_signature = "()")]
    pub fn get_remainders<'p>(
        &self,
        py: Python<'p>,
    ) -> PyResult<(&'p PyArray1<u32>, &'p PyArray1<u32>)> {
        let (compressed, remainders) = self.inner.clone().into_remainders().unwrap_infallible();
        let remainders = PyArray1::from_vec(py, remainders);
        let compressed = PyArray1::from_vec(py, compressed);
        Ok((compressed, remainders))
    }

    /// Encodes one or more symbols.
    ///
    /// Usage is analogous to [`AnsCoder.encode_reverse`](stack.html#constriction.stream.stack.AnsCoder.encode_reverse).
    ///
    /// Encoding appends the fixed amount of 24 bits per encoded symbol to the internal "compressed"
    /// buffer, regardless of the employed entropy model(s), and it consumes (24 bits - inf_content)
    /// per symbol from the internal "remainders" buffer (where "inf_content" is the information
    /// content of the encoded symbol under the employed entropy model).
    #[pyo3(text_signature = "(symbols, model, optional_model_params)")]
    #[args(symbols, model, params = "*")]
    pub fn encode_reverse(
        &mut self,
        py: Python<'_>,
        symbols: &PyAny,
        model: &Model,
        params: &PyTuple,
    ) -> PyResult<()> {
        if let Ok(symbol) = symbols.extract::<i32>() {
            if !params.is_empty() {
                return Err(pyo3::exceptions::PyAttributeError::new_err(
                    "To encode a single symbol, use a concrete model, i.e., pass the\n\
                    model parameters directly to the constructor of the model and not to the\n\
                    `encode` method of the entropy coder. Delaying the specification of model\n\
                    parameters until calling `encode_reverse` is only useful if you want to encode
                    several symbols in a row with individual model parameters for each symbol. If\n\
                    this is what you're trying to do then the `symbols` argument should be a numpy\n\
                    array, not a scalar.",
                ));
            }
            return model.0.as_parameterized(py, &mut |model| {
                self.inner
                    .encode_symbol(symbol, EncoderDecoderModel(model))?;
                Ok(())
            });
        }

        // Don't use an `else` branch here because, if the following `extract` fails, the returned
        // error message is actually pretty user friendly.
        let symbols = symbols.extract::<PyReadonlyArray1<'_, i32>>()?;
        let symbols = symbols.as_slice()?;

        if params.is_empty() {
            model.0.as_parameterized(py, &mut |model| {
                self.inner
                    .encode_iid_symbols_reverse(symbols, EncoderDecoderModel(model))?;
                Ok(())
            })?;
        } else {
            if symbols.len() != model.0.len(&params[0])? {
                return Err(pyo3::exceptions::PyAttributeError::new_err(
                    "`symbols` argument has wrong length.",
                ));
            }
            let mut symbol_iter = symbols.iter().rev();
            model.0.parameterize(py, params, true, &mut |model| {
                let symbol = symbol_iter.next().expect("TODO");
                self.inner
                    .encode_symbol(*symbol, EncoderDecoderModel(model))?;
                Ok(())
            })?;
        }

        Ok(())
    }

    /// .. deprecated:: 0.2.0
    ///    This method has been superseded by the new and more powerful generic
    ///    [`encode_reverse`](#constriction.stream.chain.ChainCoder.encode_reverse) method in conjunction with the
    ///    [`QuantizedGaussian`](model.html#constriction.stream.model.QuantizedGaussian) model.
    #[pyo3(text_signature = "(DEPRECATED)")]
    pub fn encode_leaky_gaussian_symbols_reverse(
        &mut self,
        py: Python<'_>,
        symbols: PyReadonlyArray1<'_, i32>,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        means: PyReadonlyArray1<'_, f64>,
        stds: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let _ = py.run(
            "print('WARNING: the method `encode_leaky_gaussian_symbols_reverse` is deprecated. Use method\\n\
            \x20        `encode_reverse` instead. For transition instructions with code examples, see:\\n\
            https://bamler-lab.github.io/constriction/apidoc/python/stream/model.html#examples')",
            None,
            None
        );

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

    /// .. deprecated:: 0.2.0
    ///    This method has been superseded by the new and more powerful generic
    ///    [`encode_reverse`](#constriction.stream.chain.ChainCoder.encode_reverse) method in conjunction with the
    ///    [`Categorical`](model.html#constriction.stream.model.Categorical) model.
    #[pyo3(text_signature = "(DEPRECATED)")]
    pub fn encode_iid_categorical_symbols_reverse(
        &mut self,
        py: Python<'_>,
        symbols: PyReadonlyArray1<'_, i32>,
        min_supported_symbol: i32,
        probabilities: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<()> {
        let _ = py.run(
            "print('WARNING: the method `encode_iid_categorical_symbols_reverse` is deprecated. Use method\\n\
            \x20        `encode_reverse` instead. For transition instructions with code examples, see:\\n\
            https://bamler-lab.github.io/constriction/apidoc/python/stream/model.html#constriction.stream.model.Categorical')",
            None,
            None
        );

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

    /// .. deprecated:: 0.2.0
    ///    This method has been superseded by the new and more powerful generic
    ///    [`encode_reverse`](#constriction.stream.chain.ChainCoder.encode_reverse) method in conjunction with the
    ///    [`CustomModel`](model.html#constriction.stream.model.CustomModel) or
    ///    [`ScipyModel`](model.html#constriction.stream.model.ScipyModel) model class.
    #[pyo3(text_signature = "(DEPRECATED)")]
    pub fn encode_iid_custom_model_reverse<'py>(
        &mut self,
        py: Python<'py>,
        symbols: PyReadonlyArray1<'_, i32>,
        model: &Model,
    ) -> PyResult<()> {
        let _ = py.run(
            "print('WARNING: the method `encode_iid_custom_model_reverse` is deprecated. Use method\\n\
            \x20        `encode_reverse` instead. For transition instructions with code examples, see:\\n\
            https://bamler-lab.github.io/constriction/apidoc/python/stream/model.html#constriction.stream.model.CustomModel')",
            None,
            None
        );

        self.encode_reverse(py, &symbols, model, PyTuple::empty(py))
    }

    /// Decodes one or more symbols.
    ///
    /// Usage is analogous to [`AnsCoder.decode`](stack.html#constriction.stream.stack.AnsCoder.decode).
    ///
    /// Decoding consumes the fixed amount of 24 bits per encoded symbol from the internal
    /// "compressed" buffer, regardless of the employed entropy model(s), and it appends (24
    /// bits - inf_content) per symbol to the internal "remainders" buffer (where
    /// "inf_content" is the information content of the decoded symbol under the employed
    /// entropy model).
    #[pyo3(text_signature = "(model, optional_amt_or_model_params)")]
    #[args(symbols, model, params = "*")]
    pub fn decode<'py>(
        &mut self,
        py: Python<'py>,
        model: &Model,
        params: &PyTuple,
    ) -> PyResult<PyObject> {
        match params.len() {
            0 => {
                let mut symbol = 0;
                model.0.as_parameterized(py, &mut |model| {
                    symbol = self
                        .inner
                        .decode_symbol(EncoderDecoderModel(model))
                        .expect("We use constant `PRECISION`.");
                    Ok(())
                })?;
                return Ok(symbol.to_object(py));
            }
            1 => {
                if let Ok(amt) = usize::extract(params.as_slice()[0]) {
                    let mut symbols = Vec::with_capacity(amt);
                    model.0.as_parameterized(py, &mut |model| {
                        for symbol in self
                            .inner
                            .decode_iid_symbols(amt, EncoderDecoderModel(model))
                        {
                            let symbol = symbol.expect("We use constant `PRECISION`.");
                            symbols.push(symbol);
                        }
                        Ok(())
                    })?;
                    return Ok(PyArray1::from_iter(py, symbols).to_object(py));
                }
            }
            _ => {} // Fall through to code below.
        };

        let mut symbols = Vec::with_capacity(model.0.len(&params[0])?);
        model.0.parameterize(py, params, false, &mut |model| {
            let symbol = self
                .inner
                .decode_symbol(EncoderDecoderModel(model))
                .expect("We use constant `PRECISION`.");
            symbols.push(symbol);
            Ok(())
        })?;

        Ok(PyArray1::from_vec(py, symbols).to_object(py))
    }

    /// .. deprecated:: 0.2.0
    ///    This method has been superseded by the new and more powerful generic
    ///    [`decode`](#constriction.stream.chain.ChainCoder.decode) method in conjunction with the
    ///    [`QuantizedGaussian`](model.html#constriction.stream.model.QuantizedGaussian) model.
    #[pyo3(text_signature = "(DEPRECATED)")]
    pub fn decode_leaky_gaussian_symbols<'p>(
        &mut self,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        means: PyReadonlyArray1<'_, f64>,
        stds: PyReadonlyArray1<'_, f64>,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray1<i32>> {
        let _ = py.run(
            "print('WARNING: the method `decode_leaky_gaussian_symbols` is deprecated. Use method\\n\
            \x20        `decode` instead. For transition instructions with code examples, see:\\n\
            https://bamler-lab.github.io/constriction/apidoc/python/stream/model.html#examples')",
            None,
            None
        );

        if means.len() != stds.len() {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "`means`, and `stds` must have the same length.",
            ));
        }

        let quantizer = DefaultLeakyQuantizer::new(min_supported_symbol..=max_supported_symbol);
        let symbols = self
            .inner
            .try_decode_symbols(means.as_array().iter().zip(stds.as_array()).map(|(&mean, &std)| {
                if std > 0.0 && std.is_finite() && mean.is_finite() {
                    Ok(quantizer.quantize(Gaussian::new(mean, std)))
                } else {
                    Err(())
                }
            }))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "Invalid model parameters (`std` must be strictly positive and both `std` and `mean` must be finite.).",
                )
            })?;

        Ok(PyArray1::from_vec(py, symbols))
    }

    /// .. deprecated:: 0.2.0
    ///    This method has been superseded by the new and more powerful generic
    ///    [`decode`](#constriction.stream.chain.ChainCoder.decode) method in conjunction with the
    ///    [`Categorical`](model.html#constriction.stream.model.Categorical) model.
    #[pyo3(text_signature = "(DEPRECATED)")]
    pub fn decode_iid_categorical_symbols<'py>(
        &mut self,
        amt: usize,
        min_supported_symbol: i32,
        probabilities: PyReadonlyArray1<'_, f64>,
        py: Python<'py>,
    ) -> PyResult<&'py PyArray1<i32>> {
        let _ = py.run(
            "print('WARNING: the method `decode_iid_categorical_symbols` is deprecated. Use method\\n\
            \x20        `decode` instead. For transition instructions with code examples, see:\\n\
            https://bamler-lab.github.io/constriction/apidoc/python/stream/model.html#constriction.stream.model.Categorical')",
            None,
            None
        );

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

    /// .. deprecated:: 0.2.0
    ///    This method has been superseded by the new and more powerful generic
    ///    [`decode`](#constriction.stream.chain.ChainCoder.decode) method in conjunction with the
    ///    [`CustomModel`](model.html#constriction.stream.model.CustomModel) or
    ///    [`ScipyModel`](model.html#constriction.stream.model.ScipyModel) model class.
    #[pyo3(text_signature = "(DEPRECATED)")]
    pub fn decode_iid_custom_model<'py>(
        &mut self,
        py: Python<'py>,
        amt: usize,
        model: &Model,
    ) -> PyResult<PyObject> {
        let _ = py.run(
            "print('WARNING: the method `decode_iid_custom_model` is deprecated. Use method\\n\
            \x20        `encode_reverse` instead. For transition instructions with code examples, see:\\n\
            https://bamler-lab.github.io/constriction/apidoc/python/stream/model.html#constriction.stream.model.CustomModel')",
            None,
            None
        );

        self.decode(py, model, PyTuple::new(py, [amt]))
    }

    /// Creates a deep copy of the coder and returns it.
    ///
    /// The returned copy will initially encapsulate the identical compressed data and
    /// remainders as the original coder, but the two coders can be used independently
    /// without influencing other.
    #[pyo3(text_signature = "()")]
    pub fn clone(&self) -> Self {
        Clone::clone(self)
    }
}

impl From<EncoderFrontendError> for PyErr {
    fn from(err: EncoderFrontendError) -> Self {
        match err {
            EncoderFrontendError::ImpossibleSymbol => {
                pyo3::exceptions::PyKeyError::new_err(err.to_string())
            }
            EncoderFrontendError::OutOfRemainders => {
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

impl<CompressedBackendError: Into<PyErr>, RemaindersBackendError: Into<PyErr>>
    From<BackendError<CompressedBackendError, RemaindersBackendError>> for PyErr
{
    fn from(err: BackendError<CompressedBackendError, RemaindersBackendError>) -> Self {
        match err {
            BackendError::Compressed(err) => err.into(),
            BackendError::Remainders(err) => err.into(),
        }
    }
}
