pub mod internals;

use core::{cell::Cell, iter::Sum};
use std::prelude::v1::*;
use std::sync::Mutex;

use alloc::sync::Arc;
use num_traits::{float::FloatCore, AsPrimitive};
use pyo3::prelude::*;

use crate::{
    pybindings::{PyReadonlyFloatArray, PyReadonlyFloatArray1},
    stream::model::{
        DefaultContiguousCategoricalEntropyModel, DefaultLazyContiguousCategoricalEntropyModel,
        DefaultLeakyQuantizer, UniformModel,
    },
};

use self::internals::DefaultEntropyModel;

/// Entropy models and model families for use with any of the stream codes from the sister
/// modules [`stack`](stack.html), [`queue`](queue.html), and [`chain`](chain.html).
///
/// This module provides tools to define probability distributions over symbols in fixed
/// point arithmetic, so that the models (more precisely, their cumulative distributions
/// functions) are *exactly* invertible without any rounding errors. Being exactly
/// invertible is crucial for for data compression since even tiny rounding errors can have
/// catastrophic consequences in an entropy coder (this issue is discussed in the
/// [motivating example of the `ChainCoder`](chain.html#motivation)). Further, the entropy
/// models in this module all have a well-defined domain, and they always assign a nonzero
/// probability to all symbols within this domain, even if the symbol is in the tail of some
/// distribution where its true probability would be lower than the smallest value that is
/// representable in the employed fixed point arithmetic. This ensures that symbols from the
/// well-defined domain of a model can, in principle, always be encoded without throwing an
/// error (symbols with the smallest representable probability will, however, have a very
/// high bitrate of 24 bits).
///
/// ## Concrete Models vs. Model Families
///
/// The entropy models in this module can be instantiated in two different ways:
///
/// - (a) as *concrete* models that are fully parameterized; simply provide all model
///   parameters to the constructor of the model (e.g., the mean and standard deviation of a
///   [`QuantizedGaussian`](#constriction.stream.model.QuantizedGaussian), or the domain of a
///   [`Uniform`](#constriction.stream.model.Uniform) model). You can use a concrete model
///   to either encode or decode single symbols, or to efficiently encode or decode a whole
///   array of *i.i.d.* symbols (i.e., using the same model for each symbol in the array,
///   see first example below).
/// - (b) as model *families*, i.e., models that still have some free parameters (again,
///   like the mean and standard deviation of a `QuantizedGaussian`, or the range of a
///   `Uniform` distribution); simply leave out any optional model parameters when calling
///   the model constructor. When you then use the resulting model family to encode or
///   decode an array of symbols, you can provide *arrays* of model parameters to the encode
///   and decode methods of the employed entropy coder. This will allow you to use
///   individual model parameters for each symbol, see second example below (this is more
///   efficient than constructing a new concrete model for each symbol and looping over the
///   symbols in Python).
///
/// ## Examples
///
/// Constructing and using a *concrete* [`QuantizedGaussian`](#constriction.stream.model.QuantizedGaussian)
/// model with mean 12.6 and standard deviation 7.3, and which is quantized to integers on the domain
/// {-100, -99, ..., 100}:
///
/// ```python
/// model = constriction.stream.model.QuantizedGaussian(-100, 100, 12.6, 7.3)
///
/// # Encode and decode an example message:
/// symbols = np.array([12, 15, 4, -2, 18, 5], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model)
/// print(coder.get_compressed()) # (prints: [745994372, 25704])
///
/// reconstructed = coder.decode(model, 6) # (decodes 6 i.i.d. symbols)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
/// We can generalize the above example and use model-specific means and standard deviations by
/// constructing and using a model *family* instead of a concrete model, and by providing arrays
/// of model parameters to the encode and decode methods:
///
/// ```python
/// model_family = constriction.stream.model.QuantizedGaussian(-100, 100)
/// # Note: we omitted the mean and standard deviation, but the quantization range
/// #       {-100, ..., 100} must always be specified when constructing the model.
///
/// # Define arrays of model parameters (means and standard deviations):
/// symbols = np.array([12,   15,   4,   -2,   18,   5  ], dtype=np.int32)
/// means   = np.array([13.2, 17.9, 7.3, -4.2, 25.1, 3.2], dtype=np.float32)
/// stds    = np.array([ 3.2,  4.7, 5.2,  3.1,  6.3, 2.9], dtype=np.float32)
///
/// # Encode and decode an example message:
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model_family, means, stds)
/// print(coder.get_compressed()) # (prints: [2051912079, 1549])
///
/// reconstructed = coder.decode(model_family, means, stds)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
///
#[pymodule]
#[pyo3(name = "model")]
pub fn init_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
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
/// model_params1 = np.array([... TODO ...], dtype=np.float32)
/// model_params2 = np.array([... TODO ...], dtype=np.float32)
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
        signature = (cdf, approximate_inverse_cdf, min_symbol_inclusive, max_symbol_inclusive)
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
/// model-class to the constructor. The adapter can be used with both discrete models
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
/// locs    = np.array([26.2, 10.9, 8.7, -6.3, 25.1, 8.9], dtype=np.float32)
/// scales  = np.array([ 4.3, 7.4,  2.9,  4.1,  9.7, 3.4], dtype=np.float32)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model_family, locs, scales)
/// print(coder.get_compressed()) # (prints: [3611353862, 17526])
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
    #[pyo3(signature = (scipy_model, min_symbol_inclusive, max_symbol_inclusive))]
    pub fn new(
        py: Python<'_>,
        scipy_model: PyObject,
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
    ) -> PyResult<PyClassInitializer<Self>> {
        let custom_model = CustomModel::new(
            scipy_model.getattr(py, "cdf")?,
            scipy_model.getattr(py, "ppf")?,
            min_symbol_inclusive,
            max_symbol_inclusive,
        );
        Ok(PyClassInitializer::from(custom_model).add_subclass(ScipyModel))
    }
}

/// A categorical distribution with explicitly provided probabilities.
///
/// Allows you to define any probability distribution over the alphabet `{0, 1, ... n-1}` by
/// explicitly providing the probability of each symbol in the alphabet.
///
/// ## Model Parameters
///
/// - **probabilities** --- the probability table, as a numpy array. You can specify the
///   probabilities either directly when constructing the model by passing a rank-1 numpy
///   array with a float `dtype` and length `n` to the constructor; or you can call the
///   constructor with no `probabilities` argument and instead provide a rank-2 tensor of
///   shape `(m, n)` when encoding or decoding an array of `m` symbols, as in the second
///   example below.
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
/// ## Fixed Arguments
///
/// The following arguments have to be provided directly to the constructor of the model.
/// They cannot be delayed until encoding or decoding.
///
/// - **lazy** --- set `lazy=True` if construction of the model should be delayed until the
///   model is used for encoding or decoding. This is faster if the model is used for only a
///   few symbols, but it is slower if you encode or decode lots of i.i.d. symbols. Note
///   that setting `lazy=True` implies `perfect=False`, see below. If you explicitly set
///   `perfect=False` anyway then the value of `lazy` only affects run time but has no
///   effect on the compression semantics. Thus, encoder and decoder may set `lazy`
///   differently as long as they both set `perfect=False`. Ignored if `perfect=False` and
///   `probabilities` is not given (in this case, lazy model construction is always used as
///   it is always faster and doesn't change semantics if `perfect=False`).
/// - **perfect** -- whether the constructor should accept a potentially long run time to
///   find the best possible approximation of the provided probability distribution (within
///   the limitations of fixed-point precision required to make the model exactly
///   invertible). If set to `False` (recommended in most cases and implied if `lazy=True`)
///   then the constructor will run faster but might find a *very slightly* worse
///   approximation of the provided probability distribution, thus leading to marginally
///   higher bit rates. Note that encoder and decoder have to use the same setting for
///   `perfect`. Most new code should set `perfect=False` as the differences in bit rate are
///   usually hardly noticeable. However, if neither `lazy` nor `perfect` are explicitly set
///   to any value, then `perfect` currently defaults to `True` for binary backward
///   compatibility with `constriction` version <= 0.3.5, which supported only
///   `perfect=True` (this default will change in the future, see discussion of defaults
///   below).
///
/// ## Default values of fixed arguments
///
/// - If neither `lazy` nor `perfect` are set, then `constriction` currently defaults to
///   `perfect=True` (and therefore `lazy=False`) to provide binary backward compatibility
///   with `constriction` version <= 0.3.5. If you don't need to exchange binary compressed
///   data with code that uses `constriction` version <= 0.3.5 then it is recommended to set
///   `perfect=False` to improve runtime performance.<br> **Warning:** this default will
///   change in `constriction` version 0.5, which will default to `perfect=False`.
/// - If one of `lazy` or `perfect` is specified but the other isn't, then the unspecified
///   argument defaults to `False` with the following exception:
/// - If `perfect=False` and `probabilities` is not specified (i.e., if you're constructing
///   a model *family*) then `lazy` is automatically always `True` since, in this case, lazy
///   model construction is always faster and doesn't change semantics if `perfect=False`.
///
/// ## Examples
///
/// Using a *concrete* (i.e., fully parameterized) `CategoricalModel`:
///
/// ```python
/// # Define a categorical distribution over the (implied) alphabet {0,1,2,3}
/// # with P(X=0) = 0.2, P(X=1) = 0.4, P(X=2) = 0.1, and P(X=3) = 0.3:
/// probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
/// model = constriction.stream.model.Categorical(probabilities, perfect=False)
///
/// # Encode and decode an example message:
/// symbols = np.array([0, 3, 2, 3, 2, 0, 2, 1], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model)
/// print(coder.get_compressed()) # (prints: [2484720979, 175])
///
/// reconstructed = coder.decode(model, 8) # (decodes 8 i.i.d. symbols)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
/// Using a model *family* so that we can provide individual probabilities for each encoded
/// or decoded symbol:
///
/// ```python
/// # Define 3 categorical distributions, each over the alphabet {0,1,2,3,4}:
/// model_family = constriction.stream.model.Categorical(perfect=False)
/// probabilities = np.array(
///     [[0.3, 0.1, 0.1, 0.3, 0.2],  # (for symbols[0])
///      [0.1, 0.4, 0.2, 0.1, 0.2],  # (for symbols[1])
///      [0.4, 0.2, 0.1, 0.2, 0.1]], # (for symbols[2])
///     dtype=np.float32)
///
/// symbols = np.array([0, 4, 1], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model_family, probabilities)
/// print(coder.get_compressed()) # (prints: [104018743])
///
/// reconstructed = coder.decode(model_family, probabilities)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
#[pyclass(extends=Model)]
#[derive(Debug)]
struct Categorical;

#[inline(always)]
fn parameterize_categorical<F>(
    probabilities: &[F],
    lazy: bool,
    perfect: bool,
) -> Result<Arc<dyn internals::Model>, ()>
where
    F: FloatCore + Sum<F> + AsPrimitive<u32> + Into<f64> + Send + Sync,
    u32: AsPrimitive<F>,
    usize: AsPrimitive<F>,
{
    if lazy {
        let model =
            DefaultLazyContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                probabilities.to_vec(),
                None,
            )?;
        Ok(Arc::new(model) as Arc<dyn internals::Model>)
    } else if perfect {
        let model =
            DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(
                probabilities,
            )?;
        Ok(Arc::new(model) as Arc<dyn internals::Model>)
    } else {
        let model =
            DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                probabilities,
                None,
            )?;
        Ok(Arc::new(model) as Arc<dyn internals::Model>)
    }
}

#[pymethods]
impl Categorical {
    #[new]
    #[pyo3(signature = (probabilities=None, lazy=None, perfect=None))]
    pub fn new(
        py: Python<'_>,
        probabilities: Option<PyReadonlyFloatArray1<'_>>,
        lazy: Option<bool>,
        perfect: Option<bool>,
    ) -> PyResult<(Self, Model)> {
        // It might be tempting to use an `AtomicBool` here, but rust considers "memory created by
        // [...] static items without interior mutability" to be read-only memory, and "all atomic
        // accesses on read-only memory are Undefined Behavior".
        // (source: https://doc.rust-lang.org/stable/std/sync/atomic/index.html).
        static WARNED: Mutex<Cell<bool>> = Mutex::new(Cell::new(false));

        let (lazy, perfect) = match (lazy, perfect) {
            (None, None) => {
                if !WARNED.lock().unwrap().replace(true) {
                    let _ = py.run_bound(
                        "print('WARNING: Neither argument `perfect` nor `lazy` were specified for `Categorical` entropy model.\\n\
                             \x20        In this case, `perfect` currently defaults to `True` for backward compatibility, but\\n\
                             \x20        this default will change to `perfect=False` in constriction version 0.5.\\n\
                             \x20        To suppress this warning, explicitly set:\\n\
                             \x20        - `perfect=False`: recommended for most new use cases; or\\n\
                             \x20        - `perfect=True`: if you need backward compatibility with constriction <= 0.3.5.')",
                        None,
                        None
                    );
                }
                (false, true)
            }
            (Some(true), Some(true)) => return Err(pyo3::exceptions::PyValueError::new_err(
                "Both arguments `lazy` and `perfect` cannot be set to `True` at the same time.\n\
                Lazy categorical entropy models cannot perfectly quantize probabilities.",
            )),
            (lazy, perfect) => (lazy.unwrap_or(false), perfect.unwrap_or(false)),
        };

        let model = match probabilities {
            None => {
                // We ignore the user's `lazy`-setting for unparameterized models because
                // these should always be lazy if possible (i.e., if not `perfect`).
                Arc::new(internals::UnparameterizedCategoricalDistribution::new(
                    perfect,
                )) as Arc<dyn internals::Model>
            }
            Some(probabilities) => {
                let model = match probabilities {
                    PyReadonlyFloatArray::F32(probabilities) => {
                        parameterize_categorical(probabilities.as_slice()?, lazy, perfect)
                    }
                    PyReadonlyFloatArray::F64(probabilities) => {
                        parameterize_categorical(probabilities.as_slice()?, lazy, perfect)
                    }
                };
                model.map_err(|()| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Probability distribution not normalizable (the array of probabilities\n\
                            might be empty, contain negative values or NaNs, or sum to infinity).",
                    )
                })?
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
    #[pyo3(signature = (size=None))]
    pub fn new(size: Option<i32>) -> PyResult<(Self, Model)> {
        let model = match size {
            None => {
                let model = internals::ParameterizableModel::new(|(size,): (i32,)| {
                    UniformModel::new(size as usize)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            Some(size) => Arc::new(UniformModel::new(size as usize)) as Arc<dyn internals::Model>,
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
/// constructing the model, or as a rank-1 numpy array (with a float `dtype`) when
/// calling the entropy coder's encode or decode method.
///
/// - **mean** --- the mean of the Gaussian distribution before quantization.
/// - **std** --- the standard deviation of the Gaussian distribution before quantization.
///   Must be strictly positive. If the standard deviation is calculated by a function
///   that might return zero, then add some small regularization (e.g., 1e-16) to it to
///   ensure the function argument is positive (note that, as with any parameters of the
///   entropy model, regularization has to be consistent between encoder and decoder side).
#[pyclass(extends=Model)]
#[derive(Debug)]
struct QuantizedGaussian;

fn quantized_gaussian(
    mean: f64,
    std: f64,
    quantizer: DefaultLeakyQuantizer<f64, i32>,
) -> impl DefaultEntropyModel + Send + Sync {
    assert!(
        std > 0.0,
        "Invalid model parameter: `std` must be positive."
    );
    let distribution = probability::distribution::Gaussian::new(mean, std);
    quantizer.quantize(distribution)
}

#[pymethods]
impl QuantizedGaussian {
    #[new]
    #[pyo3(
        signature = (min_symbol_inclusive, max_symbol_inclusive, mean=None, std=None)
    )]
    pub fn new(
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
        mean: Option<f64>,
        std: Option<f64>,
    ) -> PyResult<(Self, Model)> {
        let model = match (mean, std) {
            (None, None) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model = internals::ParameterizableModel::new(move |(mean, std): (f64, f64)| {
                    quantized_gaussian(mean, std, quantizer)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(mean), Some(std)) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                Arc::new(quantized_gaussian(mean, std, quantizer)) as Arc<dyn internals::Model>
            }
            (None, Some(std)) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model = internals::ParameterizableModel::new(move |(mean,): (f64,)| {
                    quantized_gaussian(mean, std, quantizer)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(mean), None) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model = internals::ParameterizableModel::new(move |(std,): (f64,)| {
                    quantized_gaussian(mean, std, quantizer)
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
/// constructing the model, or as a rank-1 numpy array (with a float `dtype`) when
/// calling the entropy coder's encode or decode method.
///
/// - **mean** --- the mean of the Laplace distribution before quantization.
/// - **scale** --- the scale parameter `b` of the Laplace distribution before quantization
///   (resulting in a variance of `2 * scale**2`). Must be strictly positive. If the scale
///   is calculated by a function that might return zero, then add some small regularization
///   (e.g., 1e-16) to it to ensure the function argument is positive (note that, as with
///   any parameters of the entropy model, regularization has to be consistent between
///   encoder and decoder side).
#[pyclass(extends=Model)]
#[derive(Debug)]
struct QuantizedLaplace;

fn quantized_laplace(
    mean: f64,
    scale: f64,
    quantizer: DefaultLeakyQuantizer<f64, i32>,
) -> impl DefaultEntropyModel + Send + Sync {
    assert!(
        scale > 0.0,
        "Invalid model parameter: `scale` must be positive."
    );
    let distribution = probability::distribution::Laplace::new(mean, scale);
    quantizer.quantize(distribution)
}

#[pymethods]
impl QuantizedLaplace {
    #[new]
    #[pyo3(
        signature = (min_symbol_inclusive, max_symbol_inclusive, mean=None, scale=None)
    )]
    pub fn new(
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
        mean: Option<f64>,
        scale: Option<f64>,
    ) -> PyResult<(Self, Model)> {
        let model = match (mean, scale) {
            (None, None) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model =
                    internals::ParameterizableModel::new(move |(mean, scale): (f64, f64)| {
                        quantized_laplace(mean, scale, quantizer)
                    });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(mean), Some(scale)) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model = quantized_laplace(mean, scale, quantizer);
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(mean), None) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model = internals::ParameterizableModel::new(move |(scale,): (f64,)| {
                    quantized_laplace(mean, scale, quantizer)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (None, Some(scale)) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model = internals::ParameterizableModel::new(move |(mean,): (f64,)| {
                    quantized_laplace(mean, scale, quantizer)
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
/// constructing the model, or as a rank-1 numpy array (with a float `dtype`) when
/// calling the entropy coder's encode or decode method.
///
/// - **loc** --- the location (mode) of the Cauchy distribution before quantization.
/// - **scale** --- the scale parameter `gamma` of the Cauchy distribution before
///   quantization (resulting in a full width at half maximum of `2 * scale`). Must be
///   strictly positive. If the scale is calculated by a function that might return zero,
///   then add some small regularization (e.g., 1e-16) to it to ensure the function argument
///   is positive (note that, as with any parameters of the entropy model, regularization
///   has to be consistent between encoder and decoder side).
#[pyclass(extends=Model)]
#[derive(Debug)]
struct QuantizedCauchy;

fn quantized_cauchy(
    mean: f64,
    scale: f64,
    quantizer: DefaultLeakyQuantizer<f64, i32>,
) -> impl DefaultEntropyModel + Send + Sync {
    assert!(
        scale > 0.0,
        "Invalid model parameter: `scale` must be positive."
    );
    let distribution = probability::distribution::Cauchy::new(mean, scale);
    quantizer.quantize(distribution)
}

#[pymethods]
impl QuantizedCauchy {
    #[new]
    #[pyo3(
        signature = (min_symbol_inclusive, max_symbol_inclusive, loc=None, scale=None)
    )]
    pub fn new(
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
        loc: Option<f64>,
        scale: Option<f64>,
    ) -> PyResult<(Self, Model)> {
        let model = match (loc, scale) {
            (None, None) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model =
                    internals::ParameterizableModel::new(move |(loc, scale): (f64, f64)| {
                        quantized_cauchy(loc, scale, quantizer)
                    });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(loc), Some(scale)) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                Arc::new(quantized_cauchy(loc, scale, quantizer)) as Arc<dyn internals::Model>
            }
            (Some(loc), None) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model = internals::ParameterizableModel::new(move |(scale,): (f64,)| {
                    quantized_cauchy(loc, scale, quantizer)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (None, Some(scale)) => {
                let quantizer =
                    DefaultLeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive);
                let model = internals::ParameterizableModel::new(move |(loc,): (f64,)| {
                    quantized_cauchy(loc, scale, quantizer)
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
/// as a rank-1 numpy array (with `dtype=np.int32` for `n` and a float `dtype` for `p`)
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
    #[pyo3(signature = (n=None, p=None))]
    pub fn new(n: Option<i32>, p: Option<f64>) -> PyResult<(Self, Model)> {
        let model = match (n, p) {
            (None, None) => {
                let model = internals::ParameterizableModel::new(move |(n, p): (i32, f64)| {
                    let quantizer = DefaultLeakyQuantizer::new(0..=n);
                    let distribution = probability::distribution::Binomial::new(n as usize, p);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(n), None) => {
                let quantizer = DefaultLeakyQuantizer::new(0..=n);
                let model = internals::ParameterizableModel::new(move |(p,): (f64,)| {
                    let distribution = probability::distribution::Binomial::new(n as usize, p);
                    quantizer.quantize(distribution)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(n), Some(p)) => {
                let distribution = probability::distribution::Binomial::new(n as usize, p);
                let quantizer = DefaultLeakyQuantizer::new(0..=n);
                Arc::new(quantizer.quantize(distribution)) as Arc<dyn internals::Model>
            }
            (None, Some(p)) => {
                let model = internals::ParameterizableModel::new(move |(n,): (i32,)| {
                    let quantizer = DefaultLeakyQuantizer::new(0..=n);
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
/// as a rank-1 numpy array with a float `dtype` when calling the entropy coder's encode
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
    #[pyo3(signature = (p=None, perfect=None))]
    pub fn new(py: Python<'_>, p: Option<f64>, perfect: Option<bool>) -> PyResult<(Self, Model)> {
        // See comment in `Categorical::new` for why we don't use an `AtomicBool` here.
        static WARNED: Mutex<Cell<bool>> = Mutex::new(Cell::new(false));

        if perfect.is_none() && !WARNED.lock().unwrap().replace(true) {
            let _ = py.run_bound(
                "print('WARNING: Argument `perfect` was not specified for `Bernoulli` distribution.\\n\
                     \x20        It currently defaults to `perfect=True` for backward compatibility, but this default\\n\
                     \x20        will change to `perfect=False` in constriction version 0.5. To suppress this warning,\\n\
                     \x20        explicitly set `perfect=False` (recommended for most new use cases) or explicitly set\\n\
                     \x20        `perfect=True` (if you need backward compatibility with constriction <= 0.3.5).')",
                None,
                None
            );
        }

        let model = match (p, perfect) {
            (None, Some(false)) => {
                let model = internals::ParameterizableModel::new(move |(p,): (f64,)| {
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                        &[1.0 - p, p],
                        None
                    )
                    .expect("`p` must be >= 0.0 and <= 1.0.")
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (None, _) => {
                let model = internals::ParameterizableModel::new(move |(p,): (f64,)| {
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(
                        &[1.0 - p, p]
                    )
                    .expect("`p` must be >= 0.0 and <= 1.0.")
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(p), Some(false)) => {
                let model =
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                        &[1.0 - p, p],
                        None
                    )
                    .map_err(|()| {
                        pyo3::exceptions::PyValueError::new_err("`p` must be >= 0.0 and <= 1.0.")
                    })?;
                Arc::new(model) as Arc<dyn internals::Model>
            }
            (Some(p), _) => {
                let model =
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(
                        &[1.0 - p, p]
                    )
                    .map_err(|()| {
                        pyo3::exceptions::PyValueError::new_err("`p` must be >= 0.0 and <= 1.0.")
                    })?;
                Arc::new(model) as Arc<dyn internals::Model>
            }
        };
        Ok((Self, Model(model)))
    }
}
