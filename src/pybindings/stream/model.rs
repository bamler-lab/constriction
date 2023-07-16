pub mod internals;

use std::prelude::v1::*;

use alloc::sync::Arc;
use pyo3::prelude::*;

use crate::{
    pybindings::PyReadonlyFloatArray1,
    stream::model::{DefaultContiguousCategoricalEntropyModel, LeakyQuantizer, UniformModel},
};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<CustomModel>()?;
    module.add_class::<ScipyModel>()?;
    module.add_class::<Categorical>()?;
    module.add_class::<Uniform>()?;
    module.add_class::<QuantizedGaussian>()?;
    module.add_class::<QuantizedLaplace>()?;
    module.add_class::<QuantizedCauchy>()?;
    module.add_class::<Binomial>()?;
    module.add_class::<Bernoulli>()?;
    Ok(())
}

/// Abstract base class for all entropy models.
///
/// This class cannot be instantiated. Instantiate one of its concrete subclasses instead.
#[pyclass(subclass)]
#[allow(missing_debug_implementations)]
pub struct Model(pub Arc<dyn internals::Model>);

/// Wrapper for a model (or model family) defined via custom callback functions
///
/// A `CustomModel` provides maximum flexibility for defining entropy models. It
/// encapsulates a user-defined cumulative distribution function (CDF) and the corresponding
/// quantile function (inverse of the CDF, also called percent point function or PPF).
///
/// A `CustomModel` can define either a concrete model or a model family (see
/// [discussion above](#concrete-models-vs-model-families)). To define a model family, the
/// provided callbacks for the CDF and PPF should expect additional model parameters, see
/// second example below.
///
/// ## Before You Read on
///
/// If you use the `scipy` python package for defining your entropy model, then there's no
/// need to use `CustomModel`. The adapter
/// [`ScipyModel`](#constriction.stream.model.ScipyModel) will be more convenient.
///
/// ## Examples
///
/// Using a *concrete* (i.e., fully parameterized) custom model:
///
/// ```python
/// model = constriction.stream.model.CustomModel(
///     lambda x: ... TODO ...,  # define your CDF here
///     lambda xi: ... TODO ..., # provide an approximate inverse of the CDF
///    -100, 100) # (or whichever range your model has)
///
/// # Encode and decode an example message:
/// symbols = np.array([... TODO ...], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model)
/// print(coder.get_compressed())
///
/// reconstructed = coder.decode(model, 5) # (decodes 5 i.i.d. symbols)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
/// Using a model *family* so that we can provide individual model parameters for each
/// encoded or decoded symbol:
///
/// ```python
/// model_family = constriction.stream.model.CustomModel(
///     lambda x, model_param1, model_param2: ... TODO ...,  # CDF
///     lambda xi, model_param1, model_param2: ... TODO ..., # PPF
///    -100, 100) # (or whichever range your model has)
///
/// # Encode and decode an example message with per-symbol model parameters:
/// symbols       = np.array([... TODO ...], dtype=np.int32)
/// model_params1 = np.array([... TODO ...], dtype=np.float64)
/// model_params2 = np.array([... TODO ...], dtype=np.float64)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model_family, model_params1, model_params2)
/// print(coder.get_compressed())
///
/// reconstructed = coder.decode(model_family, model_params1, model_params2)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
/// ## Arguments
///
/// The following arguments always have to be provided directly to the constructor of the
/// model. They cannot be delayed until encoding or decoding. However, you may provide
/// callback functions `cdf` and `approximate_inverse_cdf` that expect additional model
/// parameters, which you then pass in as numpy arrays when calling the entropy coder's
/// encode or decode method, see second example above.
///
/// - **cdf** --- the cumulative distribution function; must be  a nondecreasing function
///   that returns a scalar between 0.0 and 1.0 (both inclusive) when evaluated at any
///   mid-point between two consecutive integers within the inclusive range from
///   `min_symbol_inclusive` to `max_symbol_inclusive`. The function signature must be
///   `cdf(x, [param1, [param2, [param3, ...]]])` where `x` is the value at which
///   `constriction` will evaluate the CDF, and `paramX` will be provided if the
///   `CustomModel` is used as a model *family*, as in the second example above.
/// - **approximate_inverse_cdf** --- the inverse of the CDF, also called quantile function
///   or percent point function (PPF). This function does not have to return very precise
///   results since `constriction` will use the provided `cdf` as the defining source of
///   truth and invert it exactly; the provided `approximate_inverse_cdf` is only used to
///   speed up the function inversion. The function signature must be analogous to above,
///   `approximate_inverse_cdf(xi, [param1, [param2, [param3, ...]]])`, where you may rely
///   on `0.0 <= xi <= 1.0`.
/// - **min_symbol_inclusive** and **max_symbol_inclusive** --- define the range of integer
///   symbols that you will be able to encode with this model, see "Guarantees And
///   Requirements" below.
///
/// ## Guarantees And Requirements
///
/// The `constriction` library takes care of ensuring that the resulting entropy model is
/// *exactly* invertible, which is crucial for correct encoding/decoding, and which is
/// nontrivial due to inevitable rounding errors. In addition, `constriction` ensures that
/// all symbols within the provided range {`min_symbol_inclusive`, ...,
/// `max_symbol_inclusive`} are assigned a nonzero probability (even if their actual
/// probability under the provided model is smaller than the smallest representable
/// probability), and that the probabilities of all symbols within this range add up to
/// *exactly* one, without rounding errors. This is important to ensure that all symbols
/// within the provided range can indeed be encoded, and that encoding with ANS is
/// surjective.
///
/// The above guarantees hold only as long as the provided CDF is nondecreasing, can be
/// evaluated on mid-points between integers, and returns a value >= 0.0 and <= 1.0
/// everywhere.
#[pyclass(extends=Model, subclass)]
#[derive(Debug)]
pub struct CustomModel;

#[pymethods]
impl CustomModel {
    #[new]
    #[pyo3(
        text_signature = "(self, cdf, approximate_inverse_cdf, min_symbol_inclusive, max_symbol_inclusive)"
    )]
    pub fn new(
        cdf: PyObject,
        approximate_inverse_cdf: PyObject,
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
    ) -> (Self, Model) {
        let model = internals::UnspecializedPythonModel::new(
            cdf,
            approximate_inverse_cdf,
            min_symbol_inclusive,
            max_symbol_inclusive,
        );
        (Self, Model(Arc::new(model)))
    }
}

/// Adapter for models and model families from the `scipy` python package.
///
/// This is similar to `CustomModel` but easier to use if your model's cumulative
/// distribution function and percent point function are already implemented in the popular
/// `scipy` python package. Just provide either a fully parameterized scipy-model or a scipy
/// model-class to the constructor. The adapter can be used both with both discrete models
/// (over a continuous integer domain) and continuous models. Continuous models will be
/// quantized to bins of width 1 centered at integers, analogous to the procedure described
/// in the documentation of
/// [`QuantizedGaussian`](#constriction.stream.model.QuantizedGaussian)
///
/// ## Compatibility Warning
///
/// The `scipy` package provides some of the same models for which `constriction` offers
/// builtin models too (e.g., Gaussian, Laplace, Binomial). While wrapping, e.g.,
/// `scipy.stats.norm` in a `ScipyModel` will result in an entropy model that is *similar*
/// to a `QuantizedGaussian` with the same parameters, the two models will differ slightly
/// due to different rounding operations. Even such tiny differences can have catastrophic
/// effects when the models are used for entropy coding. Thus, always make sure you use the
/// same implementation of entropy models on the encoder and decoder side. Generally prefer
/// `constriction`'s builtin models since they are considerably faster and also available in
/// `constriction`'s Rust API.
///
/// ## Examples
///
/// Using a *concrete* (i.e., fully parameterized) `scipy` model:
///
/// ```python
/// import scipy.stats
///
/// scipy_model = scipy.stats.cauchy(loc=6.7, scale=12.4)
/// model = constriction.stream.model.ScipyModel(scipy_model, -100, 100)
///
/// # Encode and decode an example message:
/// symbols = np.array([22, 14, 5, -3, 19, 7], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model)
/// print(coder.get_compressed()) # (prints: [3569876501    1944098])
///
/// reconstructed = coder.decode(model, 6) # (decodes 6 i.i.d. symbols)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
/// Using a model *family* so that we can provide individual model parameters for each
/// encoded or decoded symbol:
///
/// ```python
/// import scipy.stats
///
/// scipy_model_family = scipy.stats.cauchy
/// model_family = constriction.stream.model.ScipyModel(
///     scipy_model_family, -100, 100)
///
/// # Encode and decode an example message with per-symbol model parameters:
/// symbols = np.array([22,   14,   5,   -3,   19,   7  ], dtype=np.int32)
/// locs    = np.array([26.2, 10.9, 8.7, -6.3, 25.1, 8.9], dtype=np.float64)
/// scales  = np.array([ 4.3, 7.4,  2.9,  4.1,  9.7, 3.4], dtype=np.float64)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model_family, locs, scales)
/// print(coder.get_compressed()) # (prints: [3493721376, 17526])
///
/// reconstructed = coder.decode(model_family, locs, scales)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
/// ## Arguments
///
/// The following arguments always have to be provided directly to the constructor of the
/// model. They cannot be delayed until encoding or decoding. However, the encapsulated
/// scipy model may expect additional model parameters, which you can pass in at encoding or
/// decoding time as in the second example below.
///
/// - **model** --- a `scipy` model or model class as in the examples above.
/// - **min_symbol_inclusive** and **max_symbol_inclusive** --- define the range of integer
///   symbols that you will be able to encode with this model, see "Guarantees And
///   Requirements" below.
#[pyclass(extends=CustomModel)]
#[derive(Debug)]
pub struct ScipyModel;

#[pymethods]
impl ScipyModel {
    #[new]
    #[pyo3(text_signature = "(self, scipy_model, min_symbol_inclusive, max_symbol_inclusive)")]
    pub fn new(
        py: Python<'_>,
        model: PyObject,
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
    ) -> PyResult<PyClassInitializer<Self>> {
        let custom_model = CustomModel::new(
            model.getattr(py, "cdf")?,
            model.getattr(py, "ppf")?,
            min_symbol_inclusive,
            max_symbol_inclusive,
        );
        Ok(PyClassInitializer::from(custom_model).add_subclass(ScipyModel))
    }
}

/// A categorical distribution with explicitly provided probabilities.
///
/// Allows you to define any probability distribution over the alphabet `{0, 1, ... n-1}`
/// by explicitly providing the probability of each symbol in the alphabet.
///
/// ## Examples
///
/// Using a *concrete* (i.e., fully parameterized) `CategoricalModel`:
///
/// ```python
/// # Define a categorical distribution over the (implied) alphabet {0,1,2,3}
/// # with P(X=0) = 0.2, P(X=1) = 0.4, P(X=2) = 0.1, and P(X=3) = 0.3:
/// probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float64)
/// model = constriction.stream.model.Categorical(probabilities)
///
/// # Encode and decode an example message:
/// symbols = np.array([0, 3, 2, 3, 2, 0, 2, 1], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model)
/// print(coder.get_compressed()) # (prints: [488222996, 175])
///
/// reconstructed = coder.decode(model, 8) # (decodes 8 i.i.d. symbols)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
/// Using a model *family* so that we can provide individual probabilities for each
/// encoded or decoded symbol:
///
/// ```python
/// # Define 3 categorical distributions, each over the alphabet {0,1,2,3,4}:
/// model_family = constriction.stream.model.Categorical() # note empty `()`
/// probabilities = np.array(
///     [[0.3, 0.1, 0.1, 0.3, 0.2],  # (for symbols[0])
///      [0.1, 0.4, 0.2, 0.1, 0.2],  # (for symbols[1])
///      [0.4, 0.2, 0.1, 0.2, 0.1]], # (for symbols[2])
///     dtype=np.float64)
///
/// symbols = np.array([0, 4, 1], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model_family, probabilities)
/// print(coder.get_compressed()) # (prints: [152672664])
///
/// reconstructed = coder.decode(model_family, probabilities)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
///
///
/// ## Model Parameters
///
/// - **probabilities** --- the probability table, as a numpy array. You can specify the
///   probabilities either directly when constructing the model by passing a rank-1 numpy
///   array with `dtype=np.float64` and length `n` to the constructor; or you can call the
///   constructor with no arguments and instead provide a rank-2 tensor of shape `(m, n)`
///   when encoding or decoding an array of `m` symbols, as in the second example above.
///
/// The probability table for each symbol must be normalizable (i.e., all probabilities must
/// be nonnegative and finite), but the probabilities don't necessarily have to sum to one.
/// They will automatically be rescaled to an exactly normalized distribution. Further,
/// `constriction` guarantees to assign at least the smallest representable nonzero
/// probability to all symbols from the range `{0, 1, ..., n-1}` (where `n` is the number of
/// provided probabilities), even if the provided probability for some symbol is smaller
/// than the smallest representable probability (including if it is exactly `0.0`). This
/// ensures that all symbols from this range can in principle be encoded.
///
/// Note that, if you delay providing the probabilities until encoding or decoding as in
/// the second example above, you still have to *call* the constructor of the model, i.e.,
/// `model_family = constriction.stream.model.Categorical()` --- note the empty parentheses
/// `()` at the end.
#[pyclass(extends=Model)]
#[derive(Debug)]
struct Categorical;

#[pymethods]
impl Categorical {
    #[new]
    #[pyo3(text_signature = "(self, probabilities=None)")]
    pub fn new(probabilities: Option<PyReadonlyFloatArray1<'_>>) -> PyResult<(Self, Model)> {
        let model = match probabilities {
            None => Arc::new(internals::UnparameterizedCategoricalDistribution)
                as Arc<dyn internals::Model>,
            Some(probabilities) => {
                let model =
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(
                        probabilities.cast_f64()?.as_slice()?,
                    )
                    .map_err(|()| {
                        pyo3::exceptions::PyValueError::new_err(
                            "Probability distribution not normalizable (the array of probabilities\n\
                            might be empty, contain negative values or NaNs, or sum to infinity).",
                        )
                    })?;
                Arc::new(model) as Arc<dyn internals::Model>
            }
        };

        Ok((Self, Model(model)))
    }
}

/// A uniform distribution over the alphabet `{0, 1, ..., size-1}`, where `size` is an
/// integer model parameter.
///
/// Due to rounding effects, the symbol `size-1` typically has very slightly higher probability
/// than the other symbols.
///
/// ## Model Parameter
///
/// The model parameter can either be specified as a scalar when constructing the model, or
/// as a rank-1 numpy array with `dtype=np.int32` when calling the entropy coder's encode
/// or decode method. Note that, in the latter case, you still have to *call* the
/// constructor of the model, i.e.: `model_family = constriction.stream.model.Uniform()`
/// --- note the trailing `()`.
///
/// - **size** --- the size of the alphabet / domain of the model. Must be at least 2 since
///   `constriction` cannot model delta distributions. Must be smaller than
///   `2**24` ≈ 17 millions.
#[pyclass(extends=Model)]
#[derive(Debug)]
struct Uniform;

#[pymethods]
impl Uniform {
    #[new]
    #[pyo3(text_signature = "(self, size=None)")]
    pub fn new(size: Option<i32>) -> PyResult<(Self, Model)> {
        let model = match size {
            None => {
                let model = internals::ParameterizableModel::new(|(size,): (i32,)| {
                    UniformModel::new(size as u32)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            Some(size) => Arc::new(UniformModel::new(size as u32)) as Arc<dyn internals::Model>,
        };

        Ok((Self, Model(model)))
    }
}

/// A Gaussian distribution, quantized over bins of size 1 centered at integer values.
///
/// This kind of entropy model is often used in novel deep-learning based compression
/// methods. If you need a quantized continuous distribution that is not a Gaussian or a
/// Laplace, then maybe [`ScipyModel`](#constriction.stream.model.ScipyModel) or
/// [`CustomModel`](#constriction.stream.model.CustomModel) is for you.
///
/// A `QuantizedGaussian` distribution is a probability distribution over the alphabet
/// `{-min_symbol_inclusive, -min_symbol_inclusive + 1, ..., max_symbol_inclusive}`. It is
/// defined by taking a Gaussian (or "Normal") distribution with the specified mean and
/// standard deviation, clipping it to the interval
/// `[-min_symbol_inclusive - 0.5, max_symbol_inclusive + 0.5]`, renormalizing it to account
/// for the clipped off tails, and then integrating the probability density over the bins
/// `[symbol - 0.5, symbol + 0.5]` for each `symbol` in the above alphabet. We further
/// guarantee that all symbols within the above alphabet are assigned at least the smallest
/// representable nonzero probability (and thus can, in principle, be encoded), even if the
/// true probability mass on the interval `[symbol - 0.5, symbol + 0.5]` integrates to a
/// value that is smaller than the smallest representable nonzero probability.
///
/// ## Examples
///
/// See [module level examples](#examples).
///
/// ## Fixed Arguments
///
/// The following arguments always have to be provided directly to the constructor of the
/// model. They cannot be delayed until encoding or decoding.
///
/// - **min_symbol_inclusive** and **max_symbol_inclusive** --- specify the integer range on
///   which the model is defined.
///
/// ## Model Parameters
///
/// Each of the following model parameters can either be specified as a scalar when
/// constructing the model, or as a rank-1 numpy array (with `dtype=np.float64`) when
/// calling the entropy coder's encode or decode method.
///
/// - **mean** --- the mean of the Gaussian distribution before quantization.
/// - **std** --- the standard deviation of the Gaussian distribution before quantization.
#[pyclass(extends=Model)]
#[derive(Debug)]
struct QuantizedGaussian;

#[pymethods]
impl QuantizedGaussian {
    #[new]
    #[pyo3(
        text_signature = "(self, min_symbol_inclusive, max_symbol_inclusive, mean=None, std=None)"
    )]
    pub fn new(
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
        mean: Option<f64>,
        std: Option<f64>,
    ) -> PyResult<(Self, Model)> {
        let model = match (mean, std) {
            (None, None) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                let model = internals::ParameterizableModel::new(move |(mean, std): (f64, f64)| {
                    let distribution = probability::distribution::Gaussian::new(mean, std);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(mean), Some(std)) => {
                let distribution = probability::distribution::Gaussian::new(mean, std);
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                Arc::new(quantizer.quantize(distribution)) as Arc<dyn internals::Model>
            }
            (None, Some(std)) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                let model = internals::ParameterizableModel::new(move |(mean,): (f64,)| {
                    let distribution = probability::distribution::Gaussian::new(mean, std);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(mean), None) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                let model = internals::ParameterizableModel::new(move |(std,): (f64,)| {
                    let distribution = probability::distribution::Gaussian::new(mean, std);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
        };

        Ok((Self, Model(model)))
    }
}

/// A Laplace distribution, quantized over bins of size 1 centered at integer values.
///
/// Analogous to [`QuantizedGaussian`](#constriction.stream.model.QuantizedGaussian), just
/// starting from a Laplace distribution rather than a Gaussian.
///
/// ## Fixed Arguments
///
/// The following arguments always have to be provided directly to the constructor of the
/// model. They cannot be delayed until encoding or decoding.
///
/// - **min_symbol_inclusive** and **max_symbol_inclusive** --- specify the integer range on
///   which the model is defined.
///
/// ## Model Parameters
///
/// Each of the following model parameters can either be specified as a scalar when
/// constructing the model, or as a rank-1 numpy array (with `dtype=np.float64`) when
/// calling the entropy coder's encode or decode method.
///
/// - **mean** --- the mean of the Laplace distribution before quantization.
/// - **scale** --- the scale parameter `b` of the Laplace distribution before quantization
///   (resulting in a variance of `2 * scale**2`).
#[pyclass(extends=Model)]
#[derive(Debug)]
struct QuantizedLaplace;

#[pymethods]
impl QuantizedLaplace {
    #[new]
    #[pyo3(
        text_signature = "(self, min_symbol_inclusive, max_symbol_inclusive, mean=None, scale=None)"
    )]
    pub fn new(
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
        mean: Option<f64>,
        scale: Option<f64>,
    ) -> PyResult<(Self, Model)> {
        let model = match (mean, scale) {
            (None, None) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                let model =
                    internals::ParameterizableModel::new(move |(mean, scale): (f64, f64)| {
                        let distribution = probability::distribution::Laplace::new(mean, scale);
                        quantizer.quantize(distribution)
                    });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(mean), Some(scale)) => {
                let distribution = probability::distribution::Laplace::new(mean, scale);
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                Arc::new(quantizer.quantize(distribution)) as Arc<dyn internals::Model>
            }
            (Some(mean), None) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                let model = internals::ParameterizableModel::new(move |(scale,): (f64,)| {
                    let distribution = probability::distribution::Laplace::new(mean, scale);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (None, Some(scale)) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                let model = internals::ParameterizableModel::new(move |(mean,): (f64,)| {
                    let distribution = probability::distribution::Laplace::new(mean, scale);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
        };

        Ok((Self, Model(model)))
    }
}

/// A Cauchy distribution, quantized over bins of size 1 centered at integer values.
///
/// Analogous to [`QuantizedGaussian`](#constriction.stream.model.QuantizedGaussian), just
/// starting from a Cauchy distribution rather than a Gaussian.
///
/// Before quantization, the probability density function of a Cauchy distribution is:
///
/// `p(x) = 1 / (pi * gamma * (1 + ((x - loc) / gamma)^2))`
///
/// where the parameters `loc` and `scale` parameterize the location of the mode and the
/// width of the distribution.
///
/// ## Fixed Arguments
///
/// The following arguments always have to be provided directly to the constructor of the
/// model. They cannot be delayed until encoding or decoding.
///
/// - **min_symbol_inclusive** and **max_symbol_inclusive** --- specify the integer range on
///   which the model is defined.
///
/// ## Model Parameters
///
/// Each of the following model parameters can either be specified as a scalar when
/// constructing the model, or as a rank-1 numpy array (with `dtype=np.float64`) when
/// calling the entropy coder's encode or decode method.
///
/// - **loc** --- the location (mode) of the Cauchy distribution before quantization.
/// - **scale** --- the scale parameter `gamma` of the Cauchy distribution before
///   quantization (resulting in a full width at half maximum of `2 * scale`).
#[pyclass(extends=Model)]
#[derive(Debug)]
struct QuantizedCauchy;

#[pymethods]
impl QuantizedCauchy {
    #[new]
    #[pyo3(
        text_signature = "(self, min_symbol_inclusive, max_symbol_inclusive, loc=None, scale=None)"
    )]
    pub fn new(
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
        loc: Option<f64>,
        scale: Option<f64>,
    ) -> PyResult<(Self, Model)> {
        let model = match (loc, scale) {
            (None, None) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                let model =
                    internals::ParameterizableModel::new(move |(loc, scale): (f64, f64)| {
                        let distribution = probability::distribution::Cauchy::new(loc, scale);
                        quantizer.quantize(distribution)
                    });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(loc), Some(scale)) => {
                let distribution = probability::distribution::Cauchy::new(loc, scale);
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                Arc::new(quantizer.quantize(distribution)) as Arc<dyn internals::Model>
            }
            (Some(loc), None) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                let model = internals::ParameterizableModel::new(move |(scale,): (f64,)| {
                    let distribution = probability::distribution::Cauchy::new(loc, scale);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (None, Some(scale)) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(
                    min_symbol_inclusive..=max_symbol_inclusive,
                );
                let model = internals::ParameterizableModel::new(move |(loc,): (f64,)| {
                    let distribution = probability::distribution::Cauchy::new(loc, scale);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
        };

        Ok((Self, Model(model)))
    }
}

/// A Binomial distribution over the alphabet {0, 1, ..., n}.
///
/// Models the number of successful trials out of `n` trials where the trials are
/// independent from each other and each one succeeds with probability `p`.
///
/// ## Model Parameters
///
/// Each model parameter can either be specified as a scalar when constructing the model, or
/// as a rank-1 numpy array (with `dtype=np.int32` for `n` and `dtype=np.float64` for `p`)
/// when calling the entropy coder's encode or decode method (see [discussion
/// above](#concrete-models-vs-model-families)). Note that, even if you delay all model
/// parameters to the point of encoding or decoding, then  you still have to *call* the
/// constructor of the model, i.e.: `model_family = constriction.stream.model.Binomial()`
/// --- note the trailing `()`.
///
/// - **n** --- the number of trials;
/// - **p** --- the probability that any given trial succeeds; must be between 0.0 and 1.0
///   (both inclusive). For your convenience, `constriction` always assigns a (possibly
///   tiny but) nonzero probability to all symbols in the range {0, 1, ..., n}, even if you
///   set `p = 0.0` or `p = 1.0` so that all symbols in this range can in principle be
///   encoded, albeit possibly at a high bitrate.
#[pyclass(extends=Model)]
#[derive(Debug)]
struct Binomial;

#[pymethods]
impl Binomial {
    #[new]
    #[pyo3(text_signature = "(self, n=None, p=None)")]
    pub fn new(n: Option<i32>, p: Option<f64>) -> PyResult<(Self, Model)> {
        let model = match (n, p) {
            (None, None) => {
                let model = internals::ParameterizableModel::new(move |(n, p): (i32, f64)| {
                    let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(0..=n);
                    let distribution = probability::distribution::Binomial::new(n as usize, p);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(n), None) => {
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(0..=n);
                let model = internals::ParameterizableModel::new(move |(p,): (f64,)| {
                    let distribution = probability::distribution::Binomial::new(n as usize, p);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(n), Some(p)) => {
                let distribution = probability::distribution::Binomial::new(n as usize, p);
                let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(0..=n);
                Arc::new(quantizer.quantize(distribution)) as Arc<dyn internals::Model>
            }
            (None, Some(p)) => {
                let model = internals::ParameterizableModel::new(move |(n,): (i32,)| {
                    let quantizer = LeakyQuantizer::<f64, _, _, 24>::new(0..=n);
                    let distribution = probability::distribution::Binomial::new(n as usize, p);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
        };

        Ok((Self, Model(model)))
    }
}

/// A Bernoulli distribution over the alphabet {0, 1}.
///
/// ## Model Parameter
///
/// The model parameter can either be specified as a scalar when constructing the model, or
/// as a rank-1 numpy array with `dtype=np.float64` when calling the entropy coder's encode
/// or decode method (see [discussion above](#concrete-models-vs-model-families)). Note
/// that, in the latter case, you still have to *call* the constructor of the model, i.e.:
/// `model_family = constriction.stream.model.Bernoulli()` --- note the trailing `()`.
///
/// - **p** --- the probability for the symbol being `1` rather than `0`. Must be between
///   0.0 and 1.0 (both inclusive). Note that, even if you set `p = 0.0` or `p = 1.0`,
///   `constriction` still assigns a tiny probability to the disallowed outcome so that both
///   symbols `0` and `1` can always be encoded, albeit at a potentially large cost in
///   bitrate.
#[pyclass(extends=Model)]
#[derive(Debug)]
struct Bernoulli;

#[pymethods]
impl Bernoulli {
    #[new]
    #[pyo3(text_signature = "(self, p=None)")]
    pub fn new(p: Option<f64>) -> PyResult<(Self, Model)> {
        let model = match p {
            None => {
                let model = internals::ParameterizableModel::new(move |(p,): (f64,)| {
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(&[
                        1.0 - p,
                        p,
                    ])
                    .expect("`p` must be >= 0.0 and <= 1.0.")
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            Some(p) => {
                let model =
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(&[
                        1.0 - p,
                        p,
                    ])
                    .map_err(|()| {
                        pyo3::exceptions::PyValueError::new_err("`p` must be >= 0.0 and <= 1.0.")
                    })?;
                Arc::new(model) as Arc<dyn internals::Model>
            }
        };
        Ok((Self, Model(model)))
    }
}
