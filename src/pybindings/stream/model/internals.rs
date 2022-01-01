use core::{cell::RefCell, marker::PhantomData, num::NonZeroU32};
use std::prelude::v1::*;

use alloc::vec;
use numpy::PyArray1;
use probability::distribution::{Distribution, Inverse};
use pyo3::{prelude::*, types::PyTuple};

use crate::stream::model::{DecoderModel, EncoderModel, EntropyModel, LeakyQuantizer};

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
        _callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        Err(pyo3::exceptions::PyAttributeError::new_err(
            "Model parameters were specified when no parameters were expected.",
        ))
    }

    fn len(&self, _param0: &PyAny) -> PyResult<usize> {
        Err(pyo3::exceptions::PyAttributeError::new_err(
            "Model parameters were specified when no parameters were expected.",
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

impl<M, F> Model for ParameterizableModel<(f64,), M, F>
where
    M: DefaultEntropyModel + Send + Sync,
    F: Fn((f64,)) -> M + Send + Sync,
{
    fn parameterize(
        &self,
        _py: Python<'_>,
        params: &PyTuple,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        if params.len() != 1 {
            return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                "Wrong number of model parameters: expected 1, got {}.",
                params.len()
            )));
        }
        let p0 = params[0].downcast::<PyArray1<f64>>()?.readonly();
        // TODO: check that arrays are rank 1.
        let p0 = p0.as_slice()?.iter();

        for &p0 in p0 {
            callback(&(self.build_model)((p0,)))?
        }

        Ok(())
    }

    fn len(&self, param0: &PyAny) -> PyResult<usize> {
        Ok(param0.downcast::<PyArray1<f64>>()?.len())
    }
}

impl<M, F> Model for ParameterizableModel<(f64, f64), M, F>
where
    M: DefaultEntropyModel,
    F: Fn((f64, f64)) -> M + Send + Sync,
{
    fn parameterize(
        &self,
        _py: Python<'_>,
        params: &PyTuple,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        if params.len() != 2 {
            return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                "Wrong number of model parameters: expected 2, got {}.",
                params.len()
            )));
        }

        let p0 = params[0].downcast::<PyArray1<f64>>()?.readonly();
        let p0 = p0.as_slice()?;
        let len = p0.len();
        let p0 = p0.iter();

        let p1 = params[1].downcast::<PyArray1<f64>>()?.readonly();
        let p1 = p1.as_slice()?;
        if p1.len() != len {
            return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                "Model parameters have unequal size",
            )));
        }
        let p1 = p1.iter();

        for (&p0, &p1) in p0.zip(p1) {
            callback(&(self.build_model)((p0, p1)))?;
        }

        Ok(())
    }

    fn len(&self, param0: &PyAny) -> PyResult<usize> {
        Ok(param0.downcast::<PyArray1<f64>>()?.len())
    }
}

impl<M, F> Model for ParameterizableModel<(i32, f64), M, F>
where
    M: DefaultEntropyModel,
    F: Fn((i32, f64)) -> M + Send + Sync,
{
    fn parameterize(
        &self,
        _py: Python<'_>,
        params: &PyTuple,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        if params.len() != 2 {
            return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                "Wrong number of model parameters: expected 2, got {}.",
                params.len()
            )));
        }

        let p0 = params[0].downcast::<PyArray1<i32>>()?.readonly();
        let p0 = p0.as_slice()?;
        let len = p0.len();
        let p0 = p0.iter();

        let p1 = params[1].downcast::<PyArray1<f64>>()?.readonly();
        let p1 = p1.as_slice()?;
        if p1.len() != len {
            return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                "Model parameters have unequal size",
            )));
        }
        let p1 = p1.iter();

        for (&p0, &p1) in p0.zip(p1) {
            callback(&(self.build_model)((p0, p1)))?;
        }

        Ok(())
    }

    fn len(&self, param0: &PyAny) -> PyResult<usize> {
        Ok(param0.downcast::<PyArray1<f64>>()?.len())
    }
}

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

    fn parameterize(
        &self,
        py: Python<'_>,
        params: &PyTuple,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        let params = params.as_slice();

        let p0 = params[0].downcast::<PyArray1<f64>>()?.readonly();
        let p0 = p0.as_slice()?;
        let len = p0.len();
        let p0iter = p0.iter();

        let mut param_iters = params[1..]
            .iter()
            .map(|&param| {
                let param = param.downcast::<PyArray1<f64>>()?.readonly();
                if param.len() != len {
                    return Err(pyo3::exceptions::PyAttributeError::new_err(alloc::format!(
                        "Model parameters have unequal lengths.",
                    )));
                };
                param.iter()
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut value_and_params = vec![0.0f64; params.len() + 1];
        for &p0 in p0iter {
            value_and_params[1] = p0;
            for (src, dst) in param_iters.iter_mut().zip(&mut value_and_params[2..]) {
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

        Ok(())
    }

    fn len(&self, param0: &PyAny) -> PyResult<usize> {
        Ok(param0.downcast::<PyArray1<f64>>()?.len())
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
