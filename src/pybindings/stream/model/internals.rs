use core::{cell::RefCell, fmt::Debug, marker::PhantomData, num::NonZeroU32};
use std::prelude::v1::*;

use alloc::sync::Arc;
use num::cast::AsPrimitive;
use probability::distribution::{Distribution, Inverse};
use pyo3::{prelude::*, types::PyTuple};

use crate::stream::model::{DecoderModel, EncoderModel, EntropyModel, LeakyQuantizer};

pub trait DefaultEntropyModel {
    fn left_cumulative_and_probability(&self, symbol: i32) -> Option<(u32, NonZeroU32)>;

    fn quantile_function(&self, quantile: u32) -> (i32, u32, NonZeroU32);
}

impl<M> DefaultEntropyModel for M
where
    M: EncoderModel<24, Symbol = i32, Probability = u32> + DecoderModel<24>,
{
    fn left_cumulative_and_probability(&self, symbol: i32) -> Option<(u32, NonZeroU32)> {
        M::left_cumulative_and_probability(self, symbol)
    }

    fn quantile_function(&self, quantile: u32) -> (i32, u32, NonZeroU32) {
        M::quantile_function(self, quantile)
    }
}

pub struct EncoderDecoderModel<M>(pub M);

impl<M> Clone for EncoderDecoderModel<Arc<M>>
where
    M: DefaultEntropyModel + ?Sized,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<M> EntropyModel<24> for EncoderDecoderModel<Arc<M>>
where
    M: DefaultEntropyModel + ?Sized,
{
    type Symbol = i32;
    type Probability = u32;
}

impl<M> EncoderModel<24> for EncoderDecoderModel<Arc<M>>
where
    M: DefaultEntropyModel + ?Sized,
{
    fn left_cumulative_and_probability(
        &self,
        symbol: impl core::borrow::Borrow<Self::Symbol>,
    ) -> Option<(Self::Probability, NonZeroU32)> {
        self.0.left_cumulative_and_probability(*symbol.borrow())
    }
}

impl<M> DecoderModel<24> for EncoderDecoderModel<Arc<M>>
where
    M: DefaultEntropyModel + ?Sized,
{
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Self::Symbol, Self::Probability, NonZeroU32) {
        self.0.quantile_function(quantile)
    }
}

impl<'m, M> Clone for EncoderDecoderModel<&'m M>
where
    M: DefaultEntropyModel + ?Sized,
{
    #[inline(always)]
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
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Self::Symbol, Self::Probability, NonZeroU32) {
        self.0.quantile_function(quantile)
    }
}

trait UnquantizedRustModel {}

pub trait Model: Send + Sync {
    fn specialize0<'py>(
        self: Arc<Self>,
        _py: Python<'py>,
    ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
        Err(pyo3::exceptions::PyAttributeError::new_err(concat!(
            "Wrong number of model parameters supplied: 0"
        )))
    }

    fn specialize2<'py>(
        self: Arc<Self>,
        _py: Python<'py>,
        _p1: f64,
        _p2: f64,
    ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
        Err(pyo3::exceptions::PyAttributeError::new_err(concat!(
            "Wrong number of model parameters supplied: 2"
        )))
    }
}

impl<M> Model for M
where
    M: EncoderModel<24, Symbol = i32, Probability = u32> + DecoderModel<24> + Send + Sync + 'static,
{
    #[inline]
    fn specialize0<'py>(
        self: Arc<Self>,
        _py: Python<'py>,
    ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
        Ok(self as Arc<dyn DefaultEntropyModel>)
    }
}

#[derive(Debug)]
pub struct UnspecializedRustModel<P, D: Distribution + 'static, F: Fn(P) -> D> {
    build_distribution: F,
    quantizer: LeakyQuantizer<f64, i32, u32, 24>,
    phantom: PhantomData<(P, D)>,
}

impl<P, D: Distribution + 'static, F: Fn(P) -> D> UnspecializedRustModel<P, D, F> {
    pub fn new(
        build_distribution: F,
        min_symbol_inclusive: i32,
        max_symbol_inclusive: i32,
    ) -> Self {
        Self {
            build_distribution,
            quantizer: LeakyQuantizer::new(min_symbol_inclusive..=max_symbol_inclusive),
            phantom: PhantomData,
        }
    }
}

impl<D, F> Model for UnspecializedRustModel<(f64, f64), D, F>
where
    D: Distribution + Inverse + Send + Sync + 'static,
    D::Value: AsPrimitive<i32>,
    F: Fn((f64, f64)) -> D + Send + Sync,
{
    fn specialize2<'py>(
        self: Arc<Self>,
        _py: Python<'py>,
        p1: f64,
        p2: f64,
    ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
        let distribution = (self.build_distribution)((p1, p2));
        Ok(Arc::new(self.quantizer.quantize(distribution)))
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
    fn specialize0<'py>(
        self: Arc<Self>,
        py: Python<'py>,
    ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
        let distribution = SpecializedPythonDistribution {
            cdf: self.cdf.clone(),
            approximate_inverse_cdf: self.approximate_inverse_cdf.clone(),
            value_and_params: RefCell::new([0.0f64]),
            py,
        };
        Ok(Arc::new(self.quantizer.quantize(distribution)))
    }

    fn specialize2<'py>(
        self: Arc<Self>,
        py: Python<'py>,
        p1: f64,
        p2: f64,
    ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
        let distribution = SpecializedPythonDistribution {
            cdf: self.cdf.clone(),
            approximate_inverse_cdf: self.approximate_inverse_cdf.clone(),
            value_and_params: RefCell::new([0.0, p1, p2]),
            py,
        };
        Ok(Arc::new(self.quantizer.quantize(distribution)))
    }
}

struct SpecializedPythonDistribution<'py, P> {
    cdf: PyObject,
    approximate_inverse_cdf: PyObject,
    value_and_params: RefCell<P>,
    py: Python<'py>,
}

impl<'py, const N: usize> Distribution for SpecializedPythonDistribution<'py, [f64; N]> {
    type Value = f64;

    fn distribution(&self, x: f64) -> f64 {
        self.value_and_params.borrow_mut()[0] = x;
        self.cdf
            .call1(
                self.py,
                PyTuple::new(self.py, &*self.value_and_params.borrow()),
            )
            .expect("TODO")
            .extract::<f64>(self.py)
            .expect("TODO")
    }
}

impl<'py, const N: usize> Inverse for SpecializedPythonDistribution<'py, [f64; N]> {
    fn inverse(&self, xi: f64) -> f64 {
        self.value_and_params.borrow_mut()[0] = xi;
        self.approximate_inverse_cdf
            .call1(
                self.py,
                PyTuple::new(self.py, &*self.value_and_params.borrow()),
            )
            .expect("TODO")
            .extract::<f64>(self.py)
            .expect("TODO")
    }
}
