use std::{prelude::v1::*, vec};

use numpy::PyReadonlyArray1;
use probability::distribution::{Distribution, Inverse};
use pyo3::prelude::*;

use crate::stream::model::{EntropyModel, LeakilyQuantizedDistribution, LeakyQuantizer};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<CustomModel>()?;
    Ok(())
}

#[pyclass]
#[text_signature = "(cdf, approximate_inverse_cdf, min_symbol_inclusive, max_symbol_inclusive)"]
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
            params: params,
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
