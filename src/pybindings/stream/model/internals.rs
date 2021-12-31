use core::{cell::RefCell, fmt::Debug, marker::PhantomData, num::NonZeroU32};
use std::prelude::v1::*;

use alloc::sync::Arc;
use num::cast::AsPrimitive;
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

impl<M> Clone for EncoderDecoderModel<Arc<M>>
where
    M: DefaultEntropyModel + ?Sized,
{
    #[inline]
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
    #[inline]
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
    #[inline]
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

macro_rules! declare_model_method {
    {$name:ident, $num_params:literal} => {
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            self: Arc<Self>,
            _py: Python<'py>,
            _params: [f64; $num_params],
        ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
            Err(pyo3::exceptions::PyAttributeError::new_err(
                concat!("Wrong number of model parameters supplied: ", $num_params, "."),
            ))
        }
    };
}

pub trait Model: Send + Sync {
    declare_model_method! {specialize0, 0}
    declare_model_method! {specialize1, 1}
    declare_model_method! {specialize2, 2}
    declare_model_method! {specialize3, 3}
    declare_model_method! {specialize4, 4}
    declare_model_method! {specialize5, 5}
    declare_model_method! {specialize6, 6}
    declare_model_method! {specialize7, 7}
    declare_model_method! {specialize8, 8}
}

impl<M> Model for M
where
    M: EncoderModel<24, Symbol = i32, Probability = u32> + DecoderModel<24> + Send + Sync + 'static,
{
    /// `EncoderModel`s can be trivially specialized by providing no arguments.
    #[inline]
    fn specialize0<'py>(
        self: Arc<Self>,
        _py: Python<'py>,
        _params: [f64; 0],
    ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
        Ok(self as Arc<dyn DefaultEntropyModel>)
    }
}

#[derive(Debug)]
pub struct UnspecializedRustModel<D: Distribution + 'static, F: Fn([f64; N]) -> D, const N: usize> {
    build_distribution: F,
    quantizer: LeakyQuantizer<f64, i32, u32, 24>,
    phantom: PhantomData<([f64; N], D)>,
}

impl<D: Distribution + 'static, F: Fn([f64; N]) -> D, const N: usize>
    UnspecializedRustModel<D, F, N>
{
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

macro_rules! impl_unspecialized_rust_model {
    {$name:ident, $num_params:literal} => {
        impl<D, F> Model for UnspecializedRustModel<D, F, $num_params>
        where
            D: Distribution + Inverse + Send + Sync + 'static,
            D::Value: AsPrimitive<i32>,
            F: Fn([f64; $num_params]) -> D + Send + Sync,
        {
            #[allow(clippy::too_many_arguments)]
            fn $name<'py>(
                self: Arc<Self>,
                _py: Python<'py>,
                params: [f64; $num_params],
            ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
                let distribution = (self.build_distribution)(params);
                Ok(Arc::new(self.quantizer.quantize(distribution)))
            }
        }
    };
}

impl_unspecialized_rust_model! {specialize0, 0}
impl_unspecialized_rust_model! {specialize1, 1}
impl_unspecialized_rust_model! {specialize2, 2}
impl_unspecialized_rust_model! {specialize3, 3}
impl_unspecialized_rust_model! {specialize4, 4}
impl_unspecialized_rust_model! {specialize5, 5}
impl_unspecialized_rust_model! {specialize6, 6}
impl_unspecialized_rust_model! {specialize7, 7}
impl_unspecialized_rust_model! {specialize8, 8}

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

macro_rules! impl_unspecialized_python_model_method {
    {$name:ident, $num_params:literal, ($($param:ident),*)} => {
        #[allow(clippy::too_many_arguments)]
        fn $name<'py>(
            self: Arc<Self>,
            py: Python<'py>,
            [$($param),*]: [f64; $num_params],
        ) -> PyResult<Arc<dyn DefaultEntropyModel + 'py>> {
            let distribution = SpecializedPythonDistribution {
                cdf: self.cdf.clone(),
                approximate_inverse_cdf: self.approximate_inverse_cdf.clone(),
                value_and_params: RefCell::new([0.0f64 $(, $param)*]),
                py,
            };
            Ok(Arc::new(self.quantizer.quantize(distribution)))
        }
    };
}

impl Model for UnspecializedPythonModel {
    impl_unspecialized_python_model_method! {specialize0, 0, ()}
    impl_unspecialized_python_model_method! {specialize1, 1, (p0)}
    impl_unspecialized_python_model_method! {specialize2, 2, (p0, p1)}
    impl_unspecialized_python_model_method! {specialize3, 3, (p0, p1, p2)}
    impl_unspecialized_python_model_method! {specialize4, 4, (p0, p1, p2, p3)}
    impl_unspecialized_python_model_method! {specialize5, 5, (p0, p1, p2, p3, p4)}
    impl_unspecialized_python_model_method! {specialize6, 6, (p0, p1, p2, p3, p4, p5)}
    impl_unspecialized_python_model_method! {specialize7, 7, (p0, p1, p2, p3, p4, p5, p6)}
    impl_unspecialized_python_model_method! {specialize8, 8, (p0, p1, p2, p3, p4, p5, p6, p7)}
}

struct SpecializedPythonDistribution<'py, const N_PLUS1: usize> {
    cdf: PyObject,
    approximate_inverse_cdf: PyObject,
    value_and_params: RefCell<[f64; N_PLUS1]>,
    py: Python<'py>,
}

impl<'py, const N_PLUS1: usize> Distribution for SpecializedPythonDistribution<'py, N_PLUS1> {
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

impl<'py, const N_PLUS1: usize> Inverse for SpecializedPythonDistribution<'py, N_PLUS1> {
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
