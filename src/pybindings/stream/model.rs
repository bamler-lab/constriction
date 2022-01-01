pub mod internals;

use std::prelude::v1::*;

use alloc::sync::Arc;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::stream::model::{DefaultContiguousCategoricalEntropyModel, LeakyQuantizer};

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

#[pyclass(extends=Model, subclass)]
#[pyo3(
    text_signature = "(min_symbol_inclusive, max_symbol_inclusive, cdf, approximate_inverse_cdf)"
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
#[pyo3(text_signature = "(min_symbol_inclusive, max_symbol_inclusive, scipy_model)")]
#[derive(Debug)]
pub struct ScipyModel;

#[pymethods]
impl ScipyModel {
    #[new]
    pub fn new(
        py: Python<'_>,
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
        model: PyObject,
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
        let make_model = |(range,): (i32,)| {
            assert!(range >= 2 && range <= (1 << 24));
            let total_probability = 1 << 24;
            let prob_per_symbol = (total_probability / range) as u32;
            let remainder = (total_probability % range) as usize;
            let probabilities = core::iter::repeat(prob_per_symbol + 1)
                .take(remainder)
                .chain(core::iter::repeat(prob_per_symbol).take(range as usize - remainder));
            DefaultContiguousCategoricalEntropyModel::from_nonzero_fixed_point_probabilities(
                probabilities,
                false,
            )
            .expect("Trivially correct model.")
        };

        let model = match range {
            None => {
                let model = internals::ParameterizableModel::new(make_model);
                Arc::new(model) as Arc<dyn internals::Model>
            }
            Some(range) => Arc::new((make_model)((range,))) as Arc<dyn internals::Model>,
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
