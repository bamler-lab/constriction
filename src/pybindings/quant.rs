use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;

use crate::{
    quant::{
        vbq::vbq_quadratic_distortion, DynamicEmpiricalDistribution, EmpiricalDistribution as ED,
    },
    NonNanFloat,
};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<EmpiricalDistribution>()?;
    module.add_function(wrap_pyfunction!(vbq_, module)?)?;
    Ok(())
}

#[pyclass]
#[derive(Debug)]
pub struct EmpiricalDistribution(DynamicEmpiricalDistribution);

#[pymethods]
impl EmpiricalDistribution {
    #[new]
    #[pyo3(text_signature = "(points=None)")]
    pub fn new(py: Python<'_>, points: Option<PyReadonlyArrayDyn<'_, f32>>) -> PyResult<Py<Self>> {
        let mut distribution = DynamicEmpiricalDistribution::new();
        if let Some(points) = points {
            distribution.try_add_points(points.as_array())?;
        }

        Py::new(py, Self(distribution))
    }

    pub fn add_points(&mut self, points: PyReadonlyArrayDyn<'_, f32>) -> PyResult<()> {
        self.0.try_add_points(points.as_array())?;
        Ok(())
    }

    #[pyo3(text_signature = "(self, old, new)")]
    pub fn update(
        &mut self,
        old: PyReadonlyArrayDyn<'_, f32>,
        new: PyReadonlyArrayDyn<'_, f32>,
    ) -> PyResult<()> {
        if old.dims() != new.dims() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "`old` and `new` must have the same shape.",
            ));
        }

        for (dst, &src) in old.as_array().iter().zip(&new.as_array()) {
            self.0.remove(NonNanFloat::new(*dst)?).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(
                    "One of the entries in `old` does not exist in the distribution.",
                )
            })?;
            self.0.insert(NonNanFloat::new(src)?);
        }

        Ok(())
    }

    #[pyo3(signature = ())]
    pub fn total(&self) -> u32 {
        self.0.total()
    }

    #[pyo3(signature = ())]
    pub fn entropy_base2(&self) -> f32 {
        self.0.entropy_base2()
    }
}

#[pyfunction]
fn vbq_(
    py: Python<'_>,
    mut unquantized: PyReadwriteArrayDyn<'_, f32>,
    prior: Py<EmpiricalDistribution>,
    posterior_variance: f32,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
) -> PyResult<()> {
    let posterior_variance = NonNanFloat::new(posterior_variance)?;
    let coarseness = NonNanFloat::new(coarseness)?;
    let mut unquantized = unquantized.as_array_mut();

    if let Some(mut reference) = reference {
        if update_prior == Some(false) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "providing a `reference` implies `update_prior=True`",
            ));
        }
        if reference.dims() != unquantized.dim() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "`reference` must have the same shape as `unquantized`.",
            ));
        }

        let prior = &mut *prior.borrow_mut(py);
        let mut reference = reference.as_array_mut();

        for (x, reference) in unquantized.iter_mut().zip(&mut reference) {
            let old_value = NonNanFloat::new(*x)?;
            let reference_val = NonNanFloat::new(*reference)?;
            let new_value = vbq_quadratic_distortion::<f32, _, _>(
                &prior.0,
                old_value,
                posterior_variance,
                coarseness,
            );
            prior.0.remove(reference_val).ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err(
                    "An uncompressed value does not exist in the distribution. \
                    You might have to provide a `reference` argument.",
                )
            })?;
            prior.0.insert(new_value);
            *x = new_value.get();
            *reference = new_value.get();
        }
    } else {
        let prior = &mut *prior.borrow_mut(py);
        for x in unquantized.iter_mut() {
            let old_value = NonNanFloat::new(*x)?;
            let new_value = vbq_quadratic_distortion::<f32, _, _>(
                &prior.0,
                old_value,
                posterior_variance,
                coarseness,
            );
            *x = new_value.get();
            if update_prior == Some(true) {
                prior.0.remove(old_value).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(
                        "An uncompressed value does not exist in the distribution. \
                        You might have to provide a `reference` argument.",
                    )
                })?;
                prior.0.insert(new_value);
            }
        }
    }

    Ok(())
}
