pub mod internals;

use std::prelude::v1::*;

use alloc::sync::Arc;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::stream::model::{
    DefaultContiguousCategoricalEntropyModel, LeakyQuantizer, UniformModel,
};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<CustomModel>()?;
    module.add_class::<ScipyModel>()?;
    module.add_class::<Categorical>()?;
    module.add_class::<Uniform>()?;
    module.add_class::<QuantizedGaussian>()?;
    module.add_class::<QuantizedLaplace>()?;
    module.add_class::<Binomial>()?;
    module.add_class::<Bernoulli>()?;
    Ok(())
}

#[pyclass(subclass)]
#[allow(missing_debug_implementations)]
pub struct Model(pub Arc<dyn internals::Model>);

/// Wrapper for a model (or model family) defined via custom callback functions
///
/// A `CustomModel` provides maximum flexibility for defining entropy models. It
/// encapsulates a user-defined cumulative distribution function (CDF) and the corresponding
/// quantile function (inverse of the CDF, also called percent point function or PPF).
/// 
/// **Note:** If you use the `scipy` python package for defining CDFs and PDFs, then
/// [`ScipyModel`](#constriction.stream.model.ScipyModel) will be a more convenient wrapper
/// type for you.
///
/// A `CustomModel` can define either a concrete model or a model family (see
/// [discussion above](#concrete-models-vs-model-families)). To define a model family, the
/// provided callbacks for the CDF and PPF should expect additional model parameters, see
/// below.
///
/// ## Example
///
/// See [`ScipyModel`](#constriction.stream.model.ScipyModel).
///
/// ## Arguments
///
/// - **cdf** --- the cumulative distribution function; a nondecreasing function that returns a scalar
///   between 0.0 and 1.0 (both inclusive), and which will be evaluated by constriction on
///   mid-points between integers in order to integrate the probability distribution over
///   bins centered at each integer. The function signature must be 
///   `cdf(x, [param1, [param2, [param3, ...]]])` where `x` is the value at which
///   `constriction` will evaluate the CDF and `paramX` will be provided if the
///   `CustomModel` is used as a model *family*.
/// - **approximate_inverse_cdf** --- the inverse of the CDF, also called quantile function
///   or percent point function (PPF). This function does not have to return very precise
///   results since `constriction` will use the provided `cdf` as the defining source of
///   truth and invert it exactly; the provided `approximate_inverse_cdf` is only used to
///   speed up this function inversion. The function signature must be analogous to above,
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
/// nontrivial for probability distributions that are evaluated with a limited floating
/// point precision. In addition, `constriction` ensures that all symbols within the
/// provided range {`min_symbol_inclusive`, ..., `max_symbol_inclusive`} are assigned a
/// nonzero probability (even if their actual probability under the provided model is
/// smaller than the smallest representable probability), and that the probabilities of all
/// symbols within this range add up to *exactly* one, without rounding errors. This is
/// important to ensure that all symbols within the provided range can indeed be encoded,
/// and that encoding with ANS is surjective.
/// 
/// All guarantees only hold as long as the provided CDF is nondecreasing, that it can be
/// evaluated on mid-points between integers, its value is >= 0.0 and <= 1.0 everywhere.
#[pyclass(extends=Model, subclass)]
#[pyo3(
    text_signature = "(cdf, approximate_inverse_cdf, min_symbol_inclusive, max_symbol_inclusive)"
)]
#[derive(Debug)]
pub struct CustomModel;

#[pymethods]
impl CustomModel {
    #[new]
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

#[pyclass(extends=CustomModel)]
#[pyo3(text_signature = "(scipy_model, min_symbol_inclusive, max_symbol_inclusive)")]
#[derive(Debug)]
pub struct ScipyModel;

#[pymethods]
impl ScipyModel {
    #[new]
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

#[pyclass(extends=Model)]
#[pyo3(text_signature = "([probabilities])")]
#[derive(Debug)]
struct Categorical;

#[pymethods]
impl Categorical {
    #[new]
    pub fn new(probabilities: Option<PyReadonlyArray1<'_, f64>>) -> PyResult<(Self, Model)> {
        let model = match probabilities {
            None => Arc::new(internals::UnparameterizedCategoricalDistribution)
                as Arc<dyn internals::Model>,
            Some(probabilities) => {
                let model =
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(
                        probabilities.as_slice()?,
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

#[pyclass(extends=Model)]
#[pyo3(text_signature = "([range])")]
#[derive(Debug)]
struct Uniform;

#[pymethods]
impl Uniform {
    #[new]
    pub fn new(range: Option<i32>) -> PyResult<(Self, Model)> {
        let model = match range {
            None => {
                let model = internals::ParameterizableModel::new(|(range,): (i32,)| {
                    UniformModel::new(range as u32)
                });
                Arc::new(model) as Arc<dyn internals::Model>
            }
            Some(range) => Arc::new(UniformModel::new(range as u32)) as Arc<dyn internals::Model>,
        };

        Ok((Self, Model(model)))
    }
}

#[pyclass(extends=Model)]
#[pyo3(text_signature = "(min_symbol_inclusive, max_symbol_inclusive, [mean, std])")]
#[derive(Debug)]
struct QuantizedGaussian;

#[pymethods]
impl QuantizedGaussian {
    #[new]
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
            _ => {
                return Err(pyo3::exceptions::PyAttributeError::new_err(
                    "Either none or both of `mean` and `std` must be specified.",
                ));
            }
        };

        Ok((Self, Model(model)))
    }
}

#[pyclass(extends=Model)]
#[pyo3(text_signature = "(min_symbol_inclusive, max_symbol_inclusive, [mean, scale])")]
#[derive(Debug)]
struct QuantizedLaplace;

#[pymethods]
impl QuantizedLaplace {
    #[new]
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
            _ => {
                return Err(pyo3::exceptions::PyAttributeError::new_err(
                    "Either none or both of `mean` and `scale` must be specified.",
                ));
            }
        };

        Ok((Self, Model(model)))
    }
}

#[pyclass(extends=Model)]
#[pyo3(text_signature = "([n, [p]])")]
#[derive(Debug)]
struct Binomial;

#[pymethods]
impl Binomial {
    #[new]
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
            _ => {
                panic!("Should be unreachable.")
            }
        };

        Ok((Self, Model(model)))
    }
}

#[pyclass(extends=Model)]
#[pyo3(text_signature = "([p])")]
#[derive(Debug)]
struct Bernoulli;

#[pymethods]
impl Bernoulli {
    #[new]
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
