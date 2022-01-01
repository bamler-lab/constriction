pub mod internals;

use std::prelude::v1::*;

use alloc::sync::Arc;
use pyo3::prelude::*;

use crate::stream::model::LeakyQuantizer;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<CustomModel>()?;
    module.add_class::<ScipyModel>()?;
    module.add_class::<Gaussian>()?;
    Ok(())
}

#[pyclass(subclass)]
#[allow(missing_debug_implementations)]
pub struct Model(pub Arc<dyn internals::Model>);

#[pyclass(extends=Model)]
#[pyo3(text_signature = "(min_symbol_inclusive, max_symbol_inclusive, [mean, std])")]
#[derive(Debug)]
struct Gaussian;

#[pymethods]
impl Gaussian {
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

                Arc::new(model)
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
