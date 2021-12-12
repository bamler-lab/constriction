use std::prelude::v1::*;

use numpy::PyReadonlyArray1;
use probability::distribution::{Distribution, Inverse};
use pyo3::prelude::*;

use crate::stream::model::{EntropyModel, LeakilyQuantizedDistribution, LeakyQuantizer};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<CustomModel>()?;
    Ok(())
}

/// Wrapper for a user defined (and possibly parameterized) entropy model.
///
/// The entropy coders in the sister modules provide optimized encoding and decoding methods
/// for a few common families of entropy models (currently: quantized Gaussian distributions
/// and categorical distributions). If you want to encode data with a different
/// distribution, you have to define it as a `CustomModel` and supply it as the `model`
/// argument to one of the the methods `{en,de}code_custom_model[_reverse]` or
/// `{en,de}code_iid_custom_model[_reverse]`, see examples below.
///
/// ## Arguments
///
/// - `cdf` --- the cumulative distribution function: a function (or closure) that maps a
///   scalar `x` to the probability to draw a symbol smaller than `x`; may (but does not
///   have to) accept a second argument that supplies model parameters as a 1-d numpy array
///   (see examples below).
/// - `approximate_inverse_cdf` --- the inverse of the cumulative distribution function, aka
///   quantile function or percent point function (PPF); may (but does not have to) accept a
///   second argument that supplies model parameters as a 1-d numpy array (see examples
///   below). This function is only used as a hint to speed the search for the correct
///   symbol during decoding. It therefore does not have to be a very precise inverse of
///   `cdf`. If `cdf` and `approximate_inverse_cdf` disagree then `cdf` will be considered
///   the source of truth.
/// - `min_symbol_inclusive` and `max_symbol_inclusive` --- two integers that specify the
///   range of supported symbols. The `CustomModel` will implicitly quantize the interval
///   `[min_symbol_inclusive - 0.5, max_symbol_inclusive + 0.5]` into bins of size one.
///
/// ## Examples
///
/// ### Fixed Model Parameters
///
/// The simplest use case of `CustomModel` is when you encode a sequence of symbols that all
/// use the exact same entropy model. The following example encodes four symbols modeled by
/// a quantized Cauchy distribution with scale 5.8 around mean 10.3.
///
/// ```python
/// import numpy as np
/// import constriction
/// import scipy.stats
///
/// model_scipy = scipy.stats.cauchy(loc=10.3, scale=5.8)
/// # Wrap the scipy-model in a `CustomModel`, which will implicitly
/// # quantize it to integers in the given range from -100 to 100 (both
/// # ends inclusively).
/// model = constriction.stream.model.CustomModel(
///     model_scipy.cdf, model_scipy.ppf, -100, 100)
///
/// symbols = np.array([5, 14, -1, 21], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # `RangeEncoder` also works.
/// coder.encode_iid_custom_model_reverse(symbols, model)
/// assert np.all(coder.decode_iid_custom_model(4, model) == symbols)
/// ```
///
/// ### Symbol-Dependent Model Parameters
///
/// It is very common to encode and decode a sequence of symbols where the entropy models of
/// all symbols are from the same family of probability distribution, but each symbol's
/// entropy model has different model parameters. The following example encodes a sequence
/// of four symbols where each symbol is modeled by a quantized Cauchy distribution, where
/// the location and scale of the Cauchy distribution is different for each symbol.
///
/// ```python
/// import numpy as np
/// import constriction
/// import scipy.stats
///
/// # The optional argument `params` will receive a 1-d python array when
/// # the model is used for encoding or decoding.
/// model = constriction.stream.model.CustomModel(
///     lambda x, params: scipy.stats.cauchy.cdf(
///         x, loc=params[0], scale=params[1]),
///     lambda x, params: scipy.stats.cauchy.ppf(
///         x, loc=params[0], scale=params[1]),
///     -100, 100)
///
/// model_parameters = np.array([
///     (7.3, 3.9),  # Location and scale of entropy model for 1st symbol.
///     (11.5, 5.2), # Location and scale of entropy model for 2nd symbol.
///     (-3.2, 4.9), # and so on ...
///     (25.9, 7.1),
/// ])
///
/// symbols = np.array([5, 14, -1, 21], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # `RangeEncoder` also works.
/// coder.encode_custom_model_reverse(symbols, model, model_parameters)
/// assert np.all(
///     coder.decode_custom_model(model, model_parameters) == symbols)
/// ```
///
/// ### Discrete Probability Distributions
///
/// You can naturally use `CustomModel` for discrete probability distributions. The
/// following example encodes a sequence of symbols from Binomial distributions (with
/// symbol-dependent success probability p).
///
/// ```python
/// import numpy as np
/// import constriction
/// import scipy.stats
///
/// model = constriction.stream.model.CustomModel(
///     lambda x, params: scipy.stats.binom.cdf(x, n=10, p=params[0]),
///     lambda x, params: scipy.stats.binom.ppf(x, n=10, p=params[0]),
///     0, 10)
///
/// success_probabilities = np.array([(0.3,), (0.7,), (0.2,), (0.6,)])
///
/// symbols = np.array([4, 8, 1, 5], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # `RangeEncoder` also works.
/// coder.encode_custom_model_reverse(symbols, model, success_probabilities)
/// assert np.all(
///     coder.decode_custom_model(model, success_probabilities) == symbols)
/// ```
#[pyclass]
#[pyo3(
    text_signature = "(cdf, approximate_inverse_cdf, min_symbol_inclusive, max_symbol_inclusive)"
)]
#[derive(Debug)]
pub struct CustomModel {
    cdf: PyObject,
    approximate_inverse_cdf: PyObject,
    quantizer: LeakyQuantizer<f64, i32, u32, 24>,
}

#[pymethods]
impl CustomModel {
    #[new]
    pub fn new(
        cdf: PyObject,
        approximate_inverse_cdf: PyObject,
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
    ) -> Self {
        Self {
            cdf,
            approximate_inverse_cdf,
            quantizer: LeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive),
        }
    }
}

impl CustomModel {
    pub fn quantized<'d, 'py>(
        &'d self,
        py: Python<'py>,
    ) -> LeakilyQuantizedDistribution<'d, f64, i32, u32, FixedCustomDistribution<'d, 'py>, 24> {
        let distribution = FixedCustomDistribution {
            cdf: &self.cdf,
            approximate_inverse_cdf: &self.approximate_inverse_cdf,
            py,
        };
        self.quantizer.quantize(distribution)
    }

    pub fn quantized_with_parameters<'d, 'py>(
        &'d self,
        py: Python<'py>,
        params: PyReadonlyArray1<'py, f64>,
    ) -> LeakilyQuantizedDistribution<'d, f64, i32, u32, ParameterizedCustomDistribution<'d, 'py>, 24>
    {
        let distribution = ParameterizedCustomDistribution {
            cdf: &self.cdf,
            approximate_inverse_cdf: &self.approximate_inverse_cdf,
            py,
            params,
        };
        self.quantizer.quantize(distribution)
    }
}

impl EntropyModel<24> for CustomModel {
    type Symbol = i32;
    type Probability = u32;
}

#[allow(missing_debug_implementations)]
#[derive(Clone, Copy)]
pub struct FixedCustomDistribution<'d, 'py> {
    py: Python<'py>,
    cdf: &'d PyObject,
    approximate_inverse_cdf: &'d PyObject,
}

impl<'d, 'py> Distribution for FixedCustomDistribution<'d, 'py> {
    type Value = f64;

    fn distribution(&self, x: f64) -> f64 {
        self.cdf
            .call1(self.py, (x,))
            .expect("Unable to call CDF.")
            .extract::<f64>(self.py)
            .expect("CDF did not return a float.")
    }
}

impl<'d, 'py> Inverse for FixedCustomDistribution<'d, 'py> {
    fn inverse(&self, xi: f64) -> Self::Value {
        self.approximate_inverse_cdf
            .call1(self.py, (xi,))
            .expect("Unable to call inverse CDF.")
            .extract::<f64>(self.py)
            .expect("Inverse CDF did not return a float.")
    }
}

#[allow(missing_debug_implementations)]
pub struct ParameterizedCustomDistribution<'d, 'py> {
    py: Python<'py>,
    cdf: &'d PyObject,
    approximate_inverse_cdf: &'d PyObject,
    params: PyReadonlyArray1<'py, f64>,
}

impl<'d, 'py> Distribution for ParameterizedCustomDistribution<'d, 'py> {
    type Value = f64;

    fn distribution(&self, x: f64) -> f64 {
        self.cdf
            .call1(self.py, (x, self.params.readonly()))
            .expect("Unable to call CDF.")
            .extract::<f64>(self.py)
            .expect("CDF did not return a float.")
    }
}

impl<'d, 'py> Inverse for ParameterizedCustomDistribution<'d, 'py> {
    fn inverse(&self, xi: f64) -> Self::Value {
        self.approximate_inverse_cdf
            .call1(self.py, (xi, self.params.readonly()))
            .expect("Unable to call inverse CDF.")
            .extract::<f64>(self.py)
            .expect("Inverse CDF did not return a float.")
    }
}
