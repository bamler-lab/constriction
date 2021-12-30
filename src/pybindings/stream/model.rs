pub mod internals;

use core::num::NonZeroU32;
use core::{fmt::Debug, marker::PhantomData};
use std::prelude::v1::*;

use alloc::sync::Arc;
use numpy::PyReadonlyArray1;
use probability::distribution::{Distribution, Inverse};
use pyo3::prelude::*;

use crate::stream::model::{EntropyModel, LeakilyQuantizedDistribution, LeakyQuantizer};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<CustomModel>()?;
    module.add_class::<ScipyModel>()?;
    module.add_class::<Gaussian>()?;
    Ok(())
}

#[pyclass(subclass)]
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

#[pyclass(extends=Model, subclass)]
#[pyo3(text_signature = "(min_symbol_inclusive, max_symbol_inclusive, [mean, std])")]
#[derive(Debug)]
pub struct Gaussian;

#[pymethods]
impl Gaussian {
    #[new]
    pub fn new(
        _py: Python<'_>,
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
        mean: Option<f64>,
        std: Option<f64>,
    ) -> PyResult<(Self, Model)> {
        let model = match (mean, std) {
            (None, None) => {
                let model = internals::UnspecializedRustModel::new(
                    |(mean, std)| probability::distribution::Gaussian::new(mean, std),
                    min_symbol_inclusive,
                    max_symbol_inclusive,
                );
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

// macro_rules! declare_rust_model_methods {
//     {$cdf:ident, $ppf:ident, $num_params:literal $(,$param:ident)* $(,)?} => {
//         fn $cdf(&self, py: Python<'_>, _x: f64, $($param: f64),*) -> PyResult<(u32, NonZeroU32)> {
//             Err(pyo3::exceptions::PyAttributeError::new_err(
//                 concat!("Wrong number of parameters supplied to model: ", $num_params, "."),
//             ))
//         }

//         fn $ppf(&self, py: Python<'_>, _xi: f64, $($param: f64),*) -> PyResult<(i32, u32, NonZeroU32)> {
//             Err(pyo3::exceptions::PyAttributeError::new_err(
//                 concat!("Wrong number of parameters supplied to model: ", $num_params, "."),
//             ))
//         }
//     };
// }

// trait RustModel: Debug + Send {
//     declare_rust_model_methods! {cdf0, ppf0, 0}
//     declare_rust_model_methods! {cdf1, ppf1, 1, _p0}
//     declare_rust_model_methods! {cdf2, ppf2, 2, _p0, _p1}
//     declare_rust_model_methods! {cdf3, ppf3, 3, _p0, _p1, _p2}
//     declare_rust_model_methods! {cdf4, ppf4, 4, _p0, _p1, _p2, _p3}
//     declare_rust_model_methods! {cdf5, ppf5, 5, _p0, _p1, _p2, _p3, _p4}
//     declare_rust_model_methods! {cdf6, ppf6, 6, _p0, _p1, _p2, _p3, _p4, _p5}
//     declare_rust_model_methods! {cdf7, ppf7, 7, _p0, _p1, _p2, _p3, _p4, _p5, _p6}
//     declare_rust_model_methods! {cdf8, ppf8, 8, _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7}
// }

// macro_rules! declare_unquantized_rust_model_methods {
//     {$cdf:ident, $ppf:ident, $num_params:literal $(,$param:ident)* $(,)?} => {
//         fn $cdf(&self, py: Python<'_>, _x: f64, $($param: f64),*) -> PyResult<f64> {
//             Err(pyo3::exceptions::PyAttributeError::new_err(
//                 concat!("Wrong number of parameters supplied to model: ", $num_params, "."),
//             ))
//         }

//         fn $ppf(&self, py: Python<'_>, _xi: f64, $($param: f64),*) -> PyResult<f64> {
//             Err(pyo3::exceptions::PyAttributeError::new_err(
//                 concat!("Wrong number of parameters supplied to model: ", $num_params, "."),
//             ))
//         }
//     };
// }

// trait UnquantizedRustModel: Debug + Send {
//     declare_unquantized_rust_model_methods! {cdf0, ppf0, 0}
//     declare_unquantized_rust_model_methods! {cdf1, ppf1, 1, _p0}
//     declare_unquantized_rust_model_methods! {cdf2, ppf2, 2, _p0, _p1}
//     declare_unquantized_rust_model_methods! {cdf3, ppf3, 3, _p0, _p1, _p2}
//     declare_unquantized_rust_model_methods! {cdf4, ppf4, 4, _p0, _p1, _p2, _p3}
//     declare_unquantized_rust_model_methods! {cdf5, ppf5, 5, _p0, _p1, _p2, _p3, _p4}
//     declare_unquantized_rust_model_methods! {cdf6, ppf6, 6, _p0, _p1, _p2, _p3, _p4, _p5}
//     declare_unquantized_rust_model_methods! {cdf7, ppf7, 7, _p0, _p1, _p2, _p3, _p4, _p5, _p6}
//     declare_unquantized_rust_model_methods! {cdf8, ppf8, 8, _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7}
// }

// macro_rules! impl_rust_model_methods {
//     {$cdf:ident, $ppf:ident, $num_params:literal $(,$param:ident)* $(,)?} => {
//         fn $cdf(&self, py: Python<'_>, symbol: i32, $($param: f64),*) -> PyResult<(u32, NonZeroU32)> {
//             let distribution=GenericDistribution{
//                 cdf: |x| <self as UnquantizedRustModel>::$cdf(x, ($($param,)*)).expect("TODO"),
//                 ppf: |xi| <self as UnquantizedRustModel>::$ppf(xi, ($($param,)*)).expect("TODO")
//             };
//             let model = self.quantizer.quantize(distribution);
//             model.left_cumulative_and_probability()
//             self.cdf.call1(py, ($($param,)*))?.extract::<f64>(py)
//             left_cumulative_and_probability
//         }

//         fn $ppf(&self, py: Python<'_>, _xi: f64, $($param: f64),*) -> PyResult<(i32, u32, NonZeroU32)> {
//             self.approximate_inverse_cdf.call1(py, ($($param,)*))?.extract::<f64>(py)
//         }
//     };
// }

// impl<M: UnquantizedRustModel> RustModel for M {
//     impl_rust_model_methods! {cdf0, ppf0, 0}
//     impl_rust_model_methods! {cdf1, ppf1, 1, p0}
//     impl_rust_model_methods! {cdf2, ppf2, 2, p0, p1}
//     impl_rust_model_methods! {cdf3, ppf3, 3, p0, p1, p2}
//     impl_rust_model_methods! {cdf4, ppf4, 4, p0, p1, p2, p3}
//     impl_rust_model_methods! {cdf5, ppf5, 5, p0, p1, p2, p3, p4}
//     impl_rust_model_methods! {cdf6, ppf6, 6, p0, p1, p2, p3, p4, p5}
//     impl_rust_model_methods! {cdf7, ppf7, 7, p0, p1, p2, p3, p4, p5, p6}
//     impl_rust_model_methods! {cdf8, ppf8, 8, p0, p1, p2, p3, p4, p5, p6, p7}
// }

// #[derive(Debug)]
// pub struct RustCustomModel {
//     cdf: PyObject,
//     approximate_inverse_cdf: PyObject,
//     quantizer: LeakyQuantizer<f64, i32, u32, 24>,
// }

// macro_rules! implement_rust_custom_model_methods {
//     {$cdf:ident, $ppf:ident, $num_params:literal $(,$param:ident)* $(,)?} => {
//         fn $cdf(&self, py: Python<'_>, x: f64, $($param: f64),*) ->  PyResult<f64> {
//             left_cumulative_and_probability
//             self.cdf.call1(py, ($($param,)*))?.extract::<f64>(py)
//         }

//         fn $ppf(&self, py: Python<'_>, _x: f64, $($param: f64),*) -> PyResult<f64> {
//             self.approximate_inverse_cdf.call1(py, ($($param,)*))?.extract::<f64>(py)
//         }
//     };
// }

// impl UnquantizedRustModel for RustCustomModel {
//     implement_rust_custom_model_methods! {cdf0, ppf0, 0}
//     implement_rust_custom_model_methods! {cdf1, ppf1, 1, p0}
//     implement_rust_custom_model_methods! {cdf2, ppf2, 2, p0, p1}
//     implement_rust_custom_model_methods! {cdf3, ppf3, 3, p0, p1, p2}
//     implement_rust_custom_model_methods! {cdf4, ppf4, 4, p0, p1, p2, p3}
//     implement_rust_custom_model_methods! {cdf5, ppf5, 5, p0, p1, p2, p3, p4}
//     implement_rust_custom_model_methods! {cdf6, ppf6, 6, p0, p1, p2, p3, p4, p5}
//     implement_rust_custom_model_methods! {cdf7, ppf7, 7, p0, p1, p2, p3, p4, p5, p6}
//     implement_rust_custom_model_methods! {cdf8, ppf8, 8, p0, p1, p2, p3, p4, p5, p6, p7}
// }

// struct GenericDistribution<Cdf: Fn(f64) -> f64, Ppf: Fn(f64) -> f64> {
//     cdf: Cdf,
//     ppf: Ppf,
// }

// impl<Cdf, Ppf> Distribution for GenericDistribution<Cdf, Ppf>
// where
//     Cdf: Fn(f64) -> f64,
//     Ppf: Fn(f64) -> f64,
// {
//     type Value = f64;

//     #[inline]
//     fn distribution(&self, x: f64) -> f64 {
//         self.cdf(x)
//     }
// }

// impl<Cdf, Ppf> Inverse for GenericDistribution<Cdf, Ppf>
// where
//     Cdf: Fn(f64) -> f64,
//     Ppf: Fn(f64) -> f64,
// {
//     fn inverse(&self, xi: f64) -> Self::Value {
//         self.ppf(xi)
//     }
// }

// impl CustomModel {
//     pub fn quantized<'d, 'py>(
//         &'d self,
//         py: Python<'py>,
//     ) -> LeakilyQuantizedDistribution<'d, f64, i32, u32, FixedCustomDistribution<'d, 'py>, 24> {
//         let distribution = FixedCustomDistribution {
//             cdf: &self.cdf,
//             approximate_inverse_cdf: &self.approximate_inverse_cdf,
//             py,
//         };
//         self.quantizer.quantize(distribution)
//     }

//     pub fn quantized_with_parameters<'d, 'py>(
//         &'d self,
//         py: Python<'py>,
//         params: PyReadonlyArray1<'py, f64>,
//     ) -> LeakilyQuantizedDistribution<'d, f64, i32, u32, ParameterizedCustomDistribution<'d, 'py>, 24>
//     {
//         let distribution = ParameterizedCustomDistribution {
//             cdf: &self.cdf,
//             approximate_inverse_cdf: &self.approximate_inverse_cdf,
//             py,
//             params,
//         };
//         self.quantizer.quantize(distribution)
//     }
// }

// impl EntropyModel<24> for CustomModel {
//     type Symbol = i32;
//     type Probability = u32;
// }

// // #[pyclass]
// // #[pyo3(text_signature = "(model, min_symbol_inclusive, max_symbol_inclusive)")]
// // #[derive(Debug)]
// // pub struct ScipyModel(CustomModel);

// // #[pymethods]
// // impl ScipyModel {
// //     #[new]
// //     pub fn new(
// //         py: Python<'_>,
// //         model: PyObject,
// //         min_symbol_inclusive: i32,
// //         max_symbol_inclusive: i32,
// //     ) -> PyResult<Self> {
// //         Ok(Self(CustomModel::new(
// //             model.getattr(py, "cdf")?,
// //             model.getattr(py, "pdf")?,
// //             min_symbol_inclusive,
// //             max_symbol_inclusive,
// //         )))
// //     }
// // }

// // impl core::ops::Deref for ScipyModel {
// //     type Target = CustomModel;

// //     fn deref(&self) -> &Self::Target {
// //         &self.0
// //     }
// // }

// #[allow(missing_debug_implementations)]
// #[derive(Clone, Copy)]
// pub struct FixedCustomDistribution<'d, 'py> {
//     py: Python<'py>,
//     cdf: &'d PyObject,
//     approximate_inverse_cdf: &'d PyObject,
// }

// impl<'d, 'py> Distribution for FixedCustomDistribution<'d, 'py> {
//     type Value = f64;

//     fn distribution(&self, x: f64) -> f64 {
//         self.cdf
//             .call1(self.py, (x,))
//             .expect("Unable to call CDF.")
//             .extract::<f64>(self.py)
//             .expect("CDF did not return a float.")
//     }
// }

// impl<'d, 'py> Inverse for FixedCustomDistribution<'d, 'py> {
//     fn inverse(&self, xi: f64) -> Self::Value {
//         self.approximate_inverse_cdf
//             .call1(self.py, (xi,))
//             .expect("Unable to call inverse CDF.")
//             .extract::<f64>(self.py)
//             .expect("Inverse CDF did not return a float.")
//     }
// }

// #[allow(missing_debug_implementations)]
// pub struct ParameterizedCustomDistribution<'d, 'py> {
//     py: Python<'py>,
//     cdf: &'d PyObject,
//     approximate_inverse_cdf: &'d PyObject,
//     params: PyReadonlyArray1<'py, f64>,
// }

// impl<'d, 'py> Distribution for ParameterizedCustomDistribution<'d, 'py> {
//     type Value = f64;

//     fn distribution(&self, x: f64) -> f64 {
//         self.cdf
//             .call1(self.py, (x, self.params.readonly()))
//             .expect("Unable to call CDF.")
//             .extract::<f64>(self.py)
//             .expect("CDF did not return a float.")
//     }
// }

// impl<'d, 'py> Inverse for ParameterizedCustomDistribution<'d, 'py> {
//     fn inverse(&self, xi: f64) -> Self::Value {
//         self.approximate_inverse_cdf
//             .call1(self.py, (xi, self.params.readonly()))
//             .expect("Unable to call inverse CDF.")
//             .extract::<f64>(self.py)
//             .expect("Inverse CDF did not return a float.")
//     }
// }
