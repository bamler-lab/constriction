use core::{cell::RefCell, iter::Sum, marker::PhantomData, num::NonZeroU32};
use std::prelude::v1::*;

use alloc::{borrow::Cow, vec};
use num_traits::{float::FloatCore, AsPrimitive};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use probability::distribution::{Distribution, Inverse};
use pyo3::{prelude::*, types::PyTuple};

use crate::{
    pybindings::{PyReadonlyFloatArray, PyReadonlyFloatArray1, PyReadonlyFloatArray2},
    stream::model::{
        DecoderModel, DefaultContiguousCategoricalEntropyModel,
        DefaultLazyContiguousCategoricalEntropyModel, EncoderModel, EntropyModel, LeakyQuantizer,
        UniformModel,
    },
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
        *self
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
        Err(pyo3::exceptions::PyValueError::new_err(
            "No model parameters specified.",
        ))
    }

    fn parameterize(
        &self,
        _py: Python<'_>,
        _params: &Bound<'_, PyTuple>,
        _reverse: bool,
        _callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        Err(pyo3::exceptions::PyValueError::new_err(
            "Model parameters were specified but the model is already fully parameterized.",
        ))
    }

    fn len(&self, _param0: Borrowed<'_, '_, PyAny>) -> PyResult<usize> {
        Err(pyo3::exceptions::PyValueError::new_err(
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

trait ParameterExtract<'source, Target: numpy::Element + 'source> {
    type Extracted: pyo3::FromPyObject<'source> + 'source;
    fn cast(param: &Self::Extracted) -> PyResult<Cow<'_, PyReadonlyArray1<'_, Target>>>;
}

struct ParameterExtractor<Target>(PhantomData<Target>);

impl<'source> ParameterExtract<'source, i32> for ParameterExtractor<i32> {
    type Extracted = PyReadonlyArray1<'source, i32>;

    fn cast(param: &Self::Extracted) -> PyResult<Cow<'_, PyReadonlyArray1<'_, i32>>> {
        Ok(Cow::Borrowed(param))
    }
}

impl<'source> ParameterExtract<'source, f64> for ParameterExtractor<f64> {
    type Extracted = PyReadonlyFloatArray1<'source>;

    fn cast(param: &Self::Extracted) -> PyResult<Cow<'_, PyReadonlyArray1<'_, f64>>> {
        param.cast_f64()
    }
}

macro_rules! impl_model_for_parameterizable_model {
    {$expected_len: literal, $p0:ident: $ty0:tt $(, $ps:ident: $tys:tt)* $(,)?} => {
        impl<$ty0, $($tys,)* M, F> Model for ParameterizableModel<($ty0, $($tys,)*), M, F>
        where
            $ty0: numpy::Element + Copy + Send + Sync,
            $($tys: numpy::Element + Copy + Send + Sync,)*
            for<'py> ParameterExtractor<$ty0>: ParameterExtract<'py, $ty0>,
            $(for<'py> ParameterExtractor<$tys>: ParameterExtract<'py, $tys>,)*
            M: DefaultEntropyModel,
            F: Fn(($ty0, $($tys,)*)) -> M + Send + Sync,
        {
            fn parameterize(
                &self,
                _py: Python<'_>,
                params: &Bound<'_, PyTuple>,
                reverse: bool,
                callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
            ) -> PyResult<()> {
                if params.len() != $expected_len {
                    return Err(pyo3::exceptions::PyValueError::new_err(alloc::format!(
                        "Wrong number of model parameters: expected {}, got {}.",
                        $expected_len,
                        params.len()
                    )));
                }

                let mut params = params.iter_borrowed();
                let $p0 = params.next().expect("len checked above").extract::<<ParameterExtractor<$ty0> as ParameterExtract<'_, $ty0>>::Extracted>()?;
                let $p0 = <ParameterExtractor<$ty0> as ParameterExtract<'_, $ty0>>::cast(&$p0)?;
                let $p0 = $p0.as_array();

                #[allow(unused_variables)] // (`len` remains unused when macro is invoked with only one parameter.)
                let len = $p0.len();
                // `.len()` returns the *total* number of entries, but that's OK here since this macro only
                // implements parameterization with rank-1 arrays.

                #[allow(unused_variables)] // (`i` remains unused when macro is invoked with only one parameter.)
                $(
                    let $ps = params.next().expect("len checked above").extract::<<ParameterExtractor<$tys> as ParameterExtract<'_, $tys>>::Extracted>()?;
                    let $ps = <ParameterExtractor<$tys> as ParameterExtract<'_, $tys>>::cast(&$ps)?;
                    let $ps = $ps.as_array();

                    if $ps.len() != len {
                        return Err(pyo3::exceptions::PyValueError::new_err(alloc::format!(
                            "Model parameters have unequal shape",
                        )));
                    }
                )*

                if reverse{
                    $(
                        let mut $ps = $ps.iter().rev();
                    )*
                    for &$p0 in $p0.iter().rev() {
                        $(
                            let $ps = *$ps.next().expect("We checked that all params have same length.");
                        )*
                        callback(&(self.build_model)(($p0, $($ps,)*)))?;
                    }
                } else {
                    $(
                        let mut $ps = $ps.iter();
                    )*
                    for &$p0 in $p0.iter() {
                        $(
                            let $ps = *$ps.next().expect("We checked that all params have same length.");
                        )*
                        callback(&(self.build_model)(($p0, $($ps,)*)))?;
                    }
                }

                Ok(())
            }

            fn len(&self, $p0: Borrowed<'_, '_, PyAny>) -> PyResult<usize> {
                $p0.len()
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

    fn parameterize(
        &self,
        py: Python<'_>,
        params: &Bound<'_, PyTuple>,
        reverse: bool,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        let params = params
            .iter_borrowed()
            .map(|param| param.extract::<PyReadonlyFloatArray1<'_>>())
            .collect::<Result<Vec<_>, _>>()?;
        let len = params[0].len();

        for param in &params[1..] {
            if param.len() != len {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Model parameters have unequal lengths.",
                ));
            }
        }

        let mut value_and_params = vec![0.0f64; params.len() + 1];

        let mut iteration_step = |i: usize| {
            for (src, dst) in params.iter().zip(&mut value_and_params[1..]) {
                *dst = src
                    .get_f64(i)
                    .expect("We checked that all arrays have the same size.");
            }

            let distribution = SpecializedPythonDistribution {
                cdf: &self.cdf,
                approximate_inverse_cdf: &self.approximate_inverse_cdf,
                value_and_params: RefCell::new(&mut value_and_params),
                py,
            };

            (callback)(&self.quantizer.quantize(distribution))
        };

        if reverse {
            for i in (0..len).rev() {
                iteration_step(i)?;
            }
        } else {
            for i in 0..len {
                iteration_step(i)?;
            }
        }

        Ok(())
    }

    fn len(&self, param0: Borrowed<'_, '_, PyAny>) -> PyResult<usize> {
        Ok(param0.extract::<PyReadonlyFloatArray1<'_>>()?.len())
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
        let mut value_and_params = self.value_and_params.borrow_mut();
        value_and_params[0] = x;
        self.cdf
            .call1(self.py, PyTuple::new_bound(self.py, &**value_and_params))
            .expect("Calling the provided cdf raised an exception.")
            .extract::<f64>(self.py)
            .expect("The provided cdf did not return a number.")
    }
}

impl<'py, 'p> Inverse for SpecializedPythonDistribution<'py, 'p> {
    fn inverse(&self, xi: f64) -> f64 {
        let mut value_and_params = self.value_and_params.borrow_mut();
        value_and_params[0] = xi;
        self.approximate_inverse_cdf
            .call1(self.py, PyTuple::new_bound(self.py, &**value_and_params))
            .expect("Calling the provided ppf raised an exception.")
            .extract::<f64>(self.py)
            .expect("The provided ppf did not return a number.")
    }
}

pub struct UnparameterizedCategoricalDistribution {
    perfect: bool,
}

impl UnparameterizedCategoricalDistribution {
    pub fn new(perfect: bool) -> Self {
        Self { perfect }
    }
}

#[inline(always)]
fn parameterize_categorical_with_model_builder<'a, F, M>(
    probabilities: impl Iterator<Item = &'a [F]>,
    build_model: impl Fn(&'a [F]) -> Result<M, ()>,
    callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
) -> PyResult<()>
where
    F: FloatCore + Sum<F> + AsPrimitive<u32>,
    u32: AsPrimitive<F>,
    M: DefaultEntropyModel + 'a,
{
    for probabilities in probabilities {
        let model = build_model(probabilities).map_err(|()| {
            pyo3::exceptions::PyValueError::new_err(
                "Probability distribution not normalizable (the array of probabilities\n\
                might be empty, contain negative values or NaNs, or sum to infinity).",
            )
        })?;
        callback(&model)?;
    }
    Ok(())
}

#[inline(always)]
fn parameterize_categorical_with_float_type<'a, F>(
    probabilities: impl Iterator<Item = &'a [F]>,
    perfect: bool,
    callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
) -> PyResult<()>
where
    F: FloatCore + Sum<F> + AsPrimitive<u32> + Into<f64>,
    u32: AsPrimitive<F>,
    usize: AsPrimitive<F>,
{
    if perfect {
        parameterize_categorical_with_model_builder(
            probabilities,
            DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect,
            callback,
        )
    } else {
        parameterize_categorical_with_model_builder(
            probabilities,
            |probabilities| {
                DefaultLazyContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
                    probabilities,
                    None,
                )
            },
            callback,
        )
    }
}

#[inline(always)]
fn parameterize_categorical<F>(
    probabilities: PyReadonlyArray2<'_, F>,
    reverse: bool,
    perfect: bool,
    callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
) -> PyResult<()>
where
    F: FloatCore + Sum<F> + AsPrimitive<u32> + Into<f64> + numpy::Element,
    u32: AsPrimitive<F>,
    usize: AsPrimitive<F>,
{
    let range = probabilities.shape()[1];
    let probabilities = probabilities.as_slice()?.chunks_exact(range);
    if reverse {
        parameterize_categorical_with_float_type(probabilities.rev(), perfect, callback)
    } else {
        parameterize_categorical_with_float_type(probabilities, perfect, callback)
    }
}

impl Model for UnparameterizedCategoricalDistribution {
    fn parameterize(
        &self,
        _py: Python<'_>,
        params: &Bound<'_, PyTuple>,
        reverse: bool,
        callback: &mut dyn FnMut(&dyn DefaultEntropyModel) -> PyResult<()>,
    ) -> PyResult<()> {
        if params.len() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(alloc::format!(
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

        let probabilities = params
            .get_borrowed_item(0)
            .expect("len checked above")
            .extract::<PyReadonlyFloatArray2<'_>>()?;

        match probabilities {
            PyReadonlyFloatArray::F32(probabilities) => {
                parameterize_categorical(probabilities, reverse, self.perfect, callback)
            }
            PyReadonlyFloatArray::F64(probabilities) => {
                parameterize_categorical(probabilities, reverse, self.perfect, callback)
            }
        }
    }

    fn len(&self, param0: Borrowed<'_, '_, PyAny>) -> PyResult<usize> {
        Ok(param0.extract::<PyReadonlyFloatArray2<'_>>()?.shape()[0])
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

impl<F, Cdf> DefaultEntropyModel for DefaultLazyContiguousCategoricalEntropyModel<F, Cdf>
where
    F: FloatCore + Sum<F> + AsPrimitive<u32>,
    u32: AsPrimitive<F>,
    Cdf: AsRef<[F]>,
{
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
        EncoderModel::left_cumulative_and_probability(self, symbol as usize)
    }

    #[inline]
    fn quantile_function(&self, quantile: u32) -> (i32, u32, NonZeroU32) {
        let (symbol, left_cumulative, probability) =
            DecoderModel::quantile_function(self, quantile);
        (symbol as i32, left_cumulative, probability)
    }
}
