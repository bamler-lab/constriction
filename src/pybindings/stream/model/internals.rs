use core::{cell::RefCell, marker::PhantomData, num::NonZeroU32};
use std::prelude::v1::*;

use alloc::vec;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use probability::distribution::{Distribution, Inverse};
use pyo3::{prelude::*, types::PyTuple};

use crate::stream::model::{
    DecoderModel, DefaultContiguousCategoricalEntropyModel, EncoderModel, EntropyModel,
    LeakyQuantizer, UniformModel,
};

/// Workaround for the fact that rust for some reason cannot create
/// joint vtables for `dyn Trait1 + Trait2`.
pub trait DefaultEntropyModel {
    fn left_cumulative_and_probability(&self, symbol: i32) -> Option<(u32, NonZeroU32)>;
    fn quantile_function(&self, quantile: u32) -> (i32, u32, NonZeroU32);
}

impl<M> DefaultEntropyModel for M
where
    M: EncoderModel<24, Symbol = i32, Probability = u32> + DecoderModel<24>,
{
    #[inline]
    fn left_cumulative_and_probability(&self, symbol: i32) -> Option<(u32, NonZeroU32)> {
        M::left_cumulative_and_probability(self, symbol)
    }

    #[inline]
    fn quantile_function(&self, quantile: u32) -> (i32, u32, NonZeroU32) {
        M::quantile_function(self, quantile)
    }
}

pub struct EncoderDecoderModel<M>(pub M);

impl<'m, M> Clone for EncoderDecoderModel<&'m M>
where
    M: DefaultEntropyModel + ?Sized,
{
    #[inline]
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<'m, M> Copy for EncoderDecoderModel<&'m M> where M: DefaultEntropyModel + ?Sized {}

impl<'m, M> EntropyModel<24> for EncoderDecoderModel<&'m M>
where
    M: DefaultEntropyModel + ?Sized,
{
    type Symbol = i32;
    type Probability = u32;
}

impl<'m, M> EncoderModel<24> for EncoderDecoderModel<&'m M>
where
    M: DefaultEntropyModel + ?Sized,
{
    #[inline]
    fn left_cumulative_and_probability(
        &self,
        symbol: impl core::borrow::Borrow<Self::Symbol>,
    ) -> Option<(Self::Probability, NonZeroU32)> {
        self.0.left_cumulative_and_probability(*symbol.borrow())
    }
}

impl<'m, M> DecoderModel<24> for EncoderDecoderModel<&'m M>
where
    M: DefaultEntropyModel + ?Sized,
{
    #[inline]
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Self::Symbol, Self::Probability, NonZeroU32) {
        self.0.quantile_function(quantile)
    }
}

pub trait Model: Send + Sync {
    fn as_parameterized(
        &self,
        _py: Python<'_>,
        _callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        Err(pyo3::exceptions::PyAttributeError::new_err(
            "No model parameters specified.",
        ))
    }

    fn parameterize(
        &self,
        _py: Python<'_>,
        _params: &PyTuple,
        _reverse: bool,
        _callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        Err(pyo3::exceptions::PyAttributeError::new_err(
            "Model parameters were specified but the model is already fully parameterized.",
        ))
    }

    fn len(&self, _param0: &PyAny) -> PyResult<usize> {
        Err(pyo3::exceptions::PyAttributeError::new_err(
            "Model parameters were specified but the model is already fully parameterized.",
        ))
    }
}

pub struct ParameterizableModel<P, M, F>
where
    M: DefaultEntropyModel,
    F: Fn(P) -> M,
{
    build_model: F,
    phantom: PhantomData<P>,
}

impl<P, M, F> ParameterizableModel<P, M, F>
where
    M: DefaultEntropyModel,
    F: Fn(P) -> M,
{
    pub fn new(build_model: F) -> Self {
        Self {
            build_model,
            phantom: PhantomData,
        }
    }
}

impl<M> Model for M
where
    M: DefaultEntropyModel + Send + Sync,
{
    fn as_parameterized(
        &self,
        _py: Python<'_>,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        (callback)(self)
    }
}

macro_rules! impl_model_for_parameterizable_model {
    {$expected_len: literal, $p0:ident: $ty0:tt $(, $ps:ident: $tys:tt)* $(,)?} => {
        impl<$ty0, $($tys,)* M, F> Model for ParameterizableModel<($ty0, $($tys,)*), M, F>
        where
            $ty0: numpy::Element + Copy + Send + Sync,
            $($tys: numpy::Element + Copy + Send + Sync,)*
            M: DefaultEntropyModel,
            F: Fn(($ty0, $($tys,)*)) -> M + Send + Sync,
        {
            fn parameterize(
                &self,
                _py: Python<'_>,
                params: &PyTuple,
                reverse: bool,
                callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
            ) -> PyResult<()> {
                if params.len() != $expected_len {
                    return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                        "Wrong number of model parameters: expected {}, got {}.",
                        $expected_len,
                        params.len()
                    )));
                }

                let $p0 = params[0].extract::<PyReadonlyArray1<'_, $ty0>>()?;

                #[allow(unused_variables)] // (`len` remains unused when macro is invoked with only one parameter.)
                let len = $p0.len();
                $(
                    let $ps = params[1].extract::<PyReadonlyArray1<'_, $tys>>()?;
                    if $ps.len() != len {
                        return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                            "Model parameters have unequal shape",
                        )));
                    }
                )*

                if reverse{
                    $(
                        let mut $ps = $ps.as_slice()?.iter().rev();
                    )*
                    for &$p0 in $p0.as_slice()?.iter().rev() {
                        $(
                            let $ps = *$ps.next().expect("We checked that all params have same length.");
                        )*
                        callback(&(self.build_model)(($p0, $($ps,)*)))?;
                    }
                } else {
                    $(
                        let mut $ps = $ps.iter()?;
                    )*
                    for &$p0 in $p0.iter()? {
                        $(
                            let $ps = *$ps.next().expect("We checked that all params have same length.");
                        )*
                        callback(&(self.build_model)(($p0, $($ps,)*)))?;
                    }
                }

                Ok(())
            }

            fn len(&self, $p0: &PyAny) -> PyResult<usize> {
                Ok($p0.extract::<PyReadonlyArray1<'_, $ty0>>()?.len())
            }
        }
    }
}

impl_model_for_parameterizable_model! {1, p0: P0}
impl_model_for_parameterizable_model! {2, p0: P0, p1: P1}

#[derive(Debug)]
pub struct UnspecializedPythonModel {
    cdf: PyObject,
    approximate_inverse_cdf: PyObject,
    quantizer: LeakyQuantizer<f64, i32, u32, 24>,
}

impl UnspecializedPythonModel {
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

impl Model for UnspecializedPythonModel {
    fn as_parameterized(
        &self,
        py: Python<'_>,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        let mut value_and_params = [0.0f64];
        let distribution = SpecializedPythonDistribution {
            cdf: &self.cdf,
            approximate_inverse_cdf: &self.approximate_inverse_cdf,
            value_and_params: RefCell::new(&mut value_and_params),
            py,
        };
        (callback)(&self.quantizer.quantize(distribution))
    }

    fn parameterize<'py>(
        &self,
        py: Python<'py>,
        params: &PyTuple,
        reverse: bool,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        let params = params.as_slice();
        let p0 = params[0].extract::<PyReadonlyArray1<'_, f64>>()?;
        let len = p0.len();

        let mut value_and_params = vec![0.0f64; params.len() + 1];
        if reverse {
            let mut remaining_params = params[1..]
                .iter()
                .map(|&param| {
                    let param = param.extract::<&PyArray1<f64>>()?;
                    if param.len() != len {
                        return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                            "Model parameters have unequal lengths.",
                        )));
                    };
                    Ok(param
                        .as_cell_slice()
                        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
                        .iter()
                        .rev())
                })
                .collect::<PyResult<Vec<_>>>()?;

            for &p0 in p0.as_slice()?.iter().rev() {
                value_and_params[1] = p0;
                for (src, dst) in remaining_params.iter_mut().zip(&mut value_and_params[2..]) {
                    *dst = src
                        .next()
                        .expect("We checked that all arrays have the same size.")
                        .get();
                }

                let distribution = SpecializedPythonDistribution {
                    cdf: &self.cdf,
                    approximate_inverse_cdf: &self.approximate_inverse_cdf,
                    value_and_params: RefCell::new(&mut value_and_params),
                    py,
                };

                (callback)(&self.quantizer.quantize(distribution))?;
            }
        } else {
            let mut remaining_params = params[1..]
                .iter()
                .map(|&param| {
                    let param = param.extract::<PyReadonlyArray1<'_, f64>>()?;
                    if param.len() != len {
                        return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                            "Model parameters have unequal lengths.",
                        )));
                    };
                    param.iter()
                })
                .collect::<PyResult<Vec<_>>>()?;

            for &p0 in p0.iter()? {
                value_and_params[1] = p0;
                for (src, dst) in remaining_params.iter_mut().zip(&mut value_and_params[2..]) {
                    *dst = *src
                        .next()
                        .expect("We checked that all arrays have the same size.");
                }

                let distribution = SpecializedPythonDistribution {
                    cdf: &self.cdf,
                    approximate_inverse_cdf: &self.approximate_inverse_cdf,
                    value_and_params: RefCell::new(&mut value_and_params),
                    py,
                };

                (callback)(&self.quantizer.quantize(distribution))?;
            }
        }

        Ok(())
    }

    fn len(&self, param0: &PyAny) -> PyResult<usize> {
        Ok(param0.extract::<PyReadonlyArray1<'_, f64>>()?.len())
    }
}

struct SpecializedPythonDistribution<'py, 'p> {
    cdf: &'py PyObject,
    approximate_inverse_cdf: &'py PyObject,
    value_and_params: RefCell<&'p mut [f64]>,
    py: Python<'py>,
}

impl<'py, 'p> Distribution for SpecializedPythonDistribution<'py, 'p> {
    type Value = f64;

    fn distribution(&self, x: f64) -> f64 {
        self.value_and_params.borrow_mut()[0] = x;
        self.cdf
            .call1(
                self.py,
                PyTuple::new(self.py, &**self.value_and_params.borrow()),
            )
            .expect("TODO")
            .extract::<f64>(self.py)
            .expect("TODO")
    }
}

impl<'py, 'p> Inverse for SpecializedPythonDistribution<'py, 'p> {
    fn inverse(&self, xi: f64) -> f64 {
        self.value_and_params.borrow_mut()[0] = xi;
        self.approximate_inverse_cdf
            .call1(
                self.py,
                PyTuple::new(self.py, &**self.value_and_params.borrow()),
            )
            .expect("TODO")
            .extract::<f64>(self.py)
            .expect("TODO")
    }
}

pub struct UnparameterizedCategoricalDistribution;

impl Model for UnparameterizedCategoricalDistribution {
    fn parameterize(
        &self,
        _py: Python<'_>,
        params: &PyTuple,
        reverse: bool,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        if params.len() != 1 {
            return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                "Wrong number of model parameters: expected 1, got {}. To use a\n\
                categorical distribution, either provide a rank-1 numpy array of probabilities\n\
                to the constructor of the model and no model parameters to the entropy coder's
                `encode` or `decode` method; or, if you want to encode several symbols in a row\n\
                with an individual categorical probability distribution for each symbol, provide
                no model parameters to the constructor and then provide a single rank-2 numpy\n\
                array to the entropy coder's `encode` or `decode` method.",
                params.len()
            )));
        }

        let probabilities = params[0].extract::<PyReadonlyArray2<'_, f64>>()?;
        let range = probabilities.shape()[1];
        let probabilities = probabilities.as_slice()?;

        if reverse {
            for probabilities in probabilities.chunks_exact(range).rev() {
                let model =
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(
                        probabilities,
                    )
                    .map_err(|()| {
                        pyo3::exceptions::PyValueError::new_err(
                        "Probability distribution not normalizable (the array of probabilities\n\
                        might be empty, contain negative values or NaNs, or sum to infinity).",
                    )
                    })?;
                callback(&model)?;
            }
        } else {
            for probabilities in probabilities.chunks_exact(range) {
                let model =
                    DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(
                        probabilities,
                    )
                    .map_err(|()| {
                        pyo3::exceptions::PyValueError::new_err(
                        "Probability distribution not normalizable (the array of probabilities\n\
                        might be empty, contain negative values or NaNs, or sum to infinity).",
                    )
                    })?;
                callback(&model)?;
            }
        }

        Ok(())
    }

    fn len(&self, param0: &PyAny) -> PyResult<usize> {
        Ok(param0.extract::<PyReadonlyArray2<'_, f64>>()?.shape()[0])
    }
}

impl DefaultEntropyModel for DefaultContiguousCategoricalEntropyModel {
    #[inline]
    fn left_cumulative_and_probability(&self, symbol: i32) -> Option<(u32, NonZeroU32)> {
        EncoderModel::left_cumulative_and_probability(self, symbol as usize)
    }

    #[inline]
    fn quantile_function(&self, quantile: u32) -> (i32, u32, NonZeroU32) {
        let (symbol, left_cumulative, probability) =
            DecoderModel::quantile_function(self, quantile);
        (symbol as i32, left_cumulative, probability)
    }
}

impl DefaultEntropyModel for UniformModel<u32, 24> {
    #[inline]
    fn left_cumulative_and_probability(&self, symbol: i32) -> Option<(u32, NonZeroU32)> {
        EncoderModel::left_cumulative_and_probability(self, symbol as u32)
    }

    #[inline]
    fn quantile_function(&self, quantile: u32) -> (i32, u32, NonZeroU32) {
        let (symbol, left_cumulative, probability) =
            DecoderModel::quantile_function(self, quantile);
        (symbol as i32, left_cumulative, probability)
    }
}
