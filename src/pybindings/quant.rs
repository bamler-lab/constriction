use alloc::vec::Vec;
use core::{borrow::Borrow, convert::Infallible};

use ndarray::parallel::prelude::*;
use ndarray::{ArrayBase, Axis, IxDyn};
use numpy::{PyArray, PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::{prelude::*, types::PyTuple};

use crate::quant::UnnormalizedDistribution;
use crate::F32;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<EmpiricalDistribution>()?;
    module.add_function(wrap_pyfunction!(vbq, module)?)?;
    module.add_function(wrap_pyfunction!(vbq_, module)?)?;
    Ok(())
}

#[derive(Debug, Clone, FromPyObject)]
pub enum PyReadonlyF32ArrayOrScalar<'py> {
    Array(PyReadonlyArrayDyn<'py, f32>),
    Scalar(f32),
}
use PyReadonlyF32ArrayOrScalar::*;

#[pyclass]
#[derive(Debug)]
pub struct EmpiricalDistribution(EmpiricalDistributionImpl);

#[derive(Debug)]
enum EmpiricalDistributionImpl {
    Single(crate::quant::EmpiricalDistribution),
    Multiple {
        distributions: Vec<crate::quant::EmpiricalDistribution>,
        axis: usize,
    },
}

#[pymethods]
impl EmpiricalDistribution {
    #[new]
    pub fn new(
        py: Python<'_>,
        points: PyReadonlyArrayDyn<'_, f32>,
        specialize_along_axis: Option<usize>,
    ) -> PyResult<Py<Self>> {
        let points = points.as_array();
        let distribution = if let Some(axis) = specialize_along_axis {
            let distributions = points
                .axis_iter(Axis(axis))
                .into_par_iter()
                .map(|points| {
                    crate::quant::EmpiricalDistribution::try_from_points(points.iter().copied())
                })
                .collect::<Result<Vec<_>, _>>()?;
            EmpiricalDistributionImpl::Multiple {
                distributions,
                axis,
            }
        } else {
            let distribution =
                crate::quant::EmpiricalDistribution::try_from_points(points.iter().copied())?;
            EmpiricalDistributionImpl::Single(distribution)
        };

        Py::new(py, Self(distribution))
    }

    #[pyo3(signature = (*args))]
    pub fn add_points(&mut self, args: &PyTuple) -> PyResult<()> {
        match &mut self.0 {
            EmpiricalDistributionImpl::Single(distribution) => {
                for points in args.iter() {
                    let points = points.extract::<PyReadonlyArrayDyn<'_, f32>>()?;
                    let points = points.as_array();
                    distribution.try_add_points(points.iter().copied())?;
                }
            }
            EmpiricalDistributionImpl::Multiple {
                distributions,
                axis,
            } => {
                let len = distributions.len();
                for points in args.iter() {
                    let points = points.extract::<PyReadonlyArrayDyn<'_, f32>>()?;
                    let points = points.as_array();
                    let points = points.axis_iter(Axis(*axis));
                    if points.len() != len {
                        return Err(pyo3::exceptions::PyIndexError::new_err(alloc::format!(
                            "Axis {} has wrong dimension: expected {} but found {}.",
                            axis,
                            len,
                            points.len()
                        )));
                    }
                    points
                        .into_par_iter()
                        .zip(distributions.into_par_iter())
                        .try_for_each(|(points, distribution)| {
                            distribution.try_add_points(points.iter().copied())
                        })?;
                }
            }
        }

        Ok(())
    }

    pub fn update(
        &mut self,
        old: PyReadonlyF32ArrayOrScalar<'_>,
        new: PyReadonlyF32ArrayOrScalar<'_>,
    ) -> PyResult<()> {
        match (old, new) {
            (Scalar(old), Scalar(new)) => match &mut self.0 {
                EmpiricalDistributionImpl::Single(distribution) => {
                    distribution.remove(F32::new(old)?).ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(
                            "The `old` value does not exist in the distribution.",
                        )
                    })?;
                    distribution.insert(F32::new(new)?);
                }
                EmpiricalDistributionImpl::Multiple { .. } => {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "Scalar updates not possible with an empirical distribution that is specialized along an axis."
                    ));
                }
            },
            (Array(old), Array(new)) if old.dims() == new.dims() => {
                let old = old.as_array();
                let new = new.as_array();

                match &mut self.0 {
                    EmpiricalDistributionImpl::Single(distribution) => {
                        for (old, &new) in old.iter().zip(&new) {
                            distribution.remove(F32::new(*old)?).ok_or_else(|| {
                                pyo3::exceptions::PyKeyError::new_err(
                                    "One of the entries in `old` does not exist in the distribution.",
                                )
                            })?;
                            distribution.insert(F32::new(new)?);
                        }
                    }
                    EmpiricalDistributionImpl::Multiple {
                        distributions,
                        axis,
                    } => {
                        let old = old.axis_iter(Axis(*axis));
                        if old.len() != distributions.len() {
                            return Err(pyo3::exceptions::PyIndexError::new_err(alloc::format!(
                                "Axis {} has wrong dimension: expected {} but found {}.",
                                axis,
                                distributions.len(),
                                old.len()
                            )));
                        }

                        old
                            .into_par_iter()
                            .zip(new.axis_iter(Axis(*axis)))
                            .zip(distributions)
                            .try_for_each(|((old, new), distribution)| {
                                for (old, &new) in old.iter().zip(&new) {
                                     distribution.remove(F32::new(*old)?).ok_or_else(|| {
                                        pyo3::exceptions::PyKeyError::new_err(
                                            "One of the entries in `old` does not exist in the distribution.",
                                        )
                                    })?;
                                    distribution.insert(F32::new(new)?);
                                }
                                Ok::<(),PyErr>(())
                            })?;
                    }
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "`old` and `new` must have the same shape.",
                ))
            }
        }

        Ok(())
    }

    #[pyo3(signature = ())]
    pub fn total(&self, py: Python<'_>) -> PyObject {
        match &self.0 {
            EmpiricalDistributionImpl::Single(distribution) => distribution.total().to_object(py),
            EmpiricalDistributionImpl::Multiple {
                distributions,
                axis: _,
            } => {
                let totals = distributions.iter().map(|d| d.total()).collect::<Vec<_>>();
                PyArray::from_vec(py, totals).to_object(py)
            }
        }
    }

    #[pyo3(signature = ())]
    pub fn entropy_base2(&self, py: Python<'_>) -> PyObject {
        match &self.0 {
            EmpiricalDistributionImpl::Single(distribution) => {
                distribution.entropy_base2::<f32>().to_object(py)
            }
            EmpiricalDistributionImpl::Multiple {
                distributions,
                axis: _,
            } => {
                let totals = distributions
                    .iter()
                    .map(|d| d.entropy_base2::<f32>())
                    .collect::<Vec<_>>();
                PyArray::from_vec(py, totals).to_object(py)
            }
        }
    }

    #[pyo3(signature = ())]
    pub fn points_and_counts<'p>(&self, py: Python<'p>) -> (PyObject, PyObject) {
        match &self.0 {
            EmpiricalDistributionImpl::Single(distribution) => {
                let (points, counts): (Vec<_>, Vec<_>) = distribution
                    .iter()
                    .map(|(value, count)| (value.get(), count))
                    .unzip();
                let points = PyArray::from_vec(py, points).to_object(py);
                let counts = PyArray::from_vec(py, counts).to_object(py);
                (points, counts)
            }
            EmpiricalDistributionImpl::Multiple {
                distributions,
                axis: _,
            } => {
                let vecs = distributions
                    .par_iter()
                    .map(|distribution| {
                        distribution
                            .iter()
                            .map(|(value, count)| (value.get(), count))
                            .unzip()
                    })
                    .collect::<Vec<(Vec<_>, Vec<_>)>>();
                let (points, counts): (Vec<_>, Vec<_>) = vecs
                    .into_iter()
                    .map(|(points, counts)| {
                        let points = PyArray::from_vec(py, points).to_object(py);
                        let counts = PyArray::from_vec(py, counts).to_object(py);
                        (points, counts)
                    })
                    .unzip();
                let points = PyArray::from_vec(py, points).to_object(py);
                let counts = PyArray::from_vec(py, counts).to_object(py);
                (points, counts)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn vbq<'a>(
    py: Python<'a>,
    unquantized: PyReadonlyArrayDyn<'a, f32>,
    prior: Py<EmpiricalDistribution>,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<&'a PyArrayDyn<f32>> {
    match &mut prior.borrow_mut(py).0 {
        EmpiricalDistributionImpl::Single(distribution) => vbq_single(
            py,
            unquantized,
            distribution,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
        ),
        EmpiricalDistributionImpl::Multiple {
            distributions,
            axis,
        } => vbq_multiplexed(
            py,
            unquantized,
            distributions,
            *axis,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
        ),
    }
}

fn vbq_single<'a>(
    py: Python<'a>,
    unquantized: PyReadonlyArrayDyn<'a, f32>,
    prior: &mut crate::quant::EmpiricalDistribution,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<&'a PyArrayDyn<f32>> {
    let len = unquantized.len();
    let mut quantized: Vec<f32> = Vec::with_capacity(len);
    let unquantized = unquantized.as_array();
    let dim = unquantized.raw_dim();

    {
        let mut quantized = ArrayBase::<ndarray::ViewRepr<&mut _>, _>::from_shape(
            dim.clone(),
            quantized.spare_capacity_mut(),
        )
        .expect("len was chosen to match shape");

        match (update_prior, reference.is_some()) {
            (Some(false), true) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "providing a `reference` implies `update_prior=True`",
                ))
            }
            (None | Some(false), false) => {
                // Prior doesn't get mutated, so we can run VBQ in parallel.
                let src_and_dst = par_azip!(&unquantized, &mut quantized);
                match posterior_variance {
                    Array(posterior_variance) => {
                        if posterior_variance.dims() != dim {
                            return Err(pyo3::exceptions::PyAssertionError::new_err(
                                "`posterior_variance` must have the same shape as `unquantized`.",
                            ));
                        }
                        let posterior_variance = posterior_variance.as_array();
                        let two_coarseness = 2.0 * coarseness;
                        let src_dst_penalty = src_and_dst
                            .and(&posterior_variance)
                            .into_par_iter()
                            .map(|(src, dst, var)| ((src, dst), F32::new(two_coarseness * *var)));
                        vbq_parallel(
                            src_dst_penalty,
                            prior,
                            |(src, _dst)| **src,
                            |(_src, dst), value| {
                                dst.write(value);
                            },
                        )
                    }
                    Scalar(posterior_variance) => {
                        let bit_penalty = Result::<_, Infallible>::Ok(F32::new(
                            2.0 * coarseness * posterior_variance,
                        )?);
                        let src_dst_penalty =
                            src_and_dst.into_par_iter().map(|sd| (sd, bit_penalty));
                        vbq_parallel(
                            src_dst_penalty,
                            prior,
                            |(src, _dst)| **src,
                            |(_src, dst), value| {
                                dst.write(value);
                            },
                        )
                    }
                }?;
            }
            _ => {
                // We mutate the prior after each update, and therefore the result depends on the
                // order in which we iterate over items. Fall back to a sequential implementation
                // since running VBQ in parallel would make it nondeterministic here.
                vbq_sequential(
                    unquantized.iter().zip(quantized.iter_mut()),
                    dim,
                    prior,
                    posterior_variance,
                    coarseness,
                    reference,
                    |_src, dst, value| {
                        dst.write(value);
                    },
                )?
            }
        }
    }
    unsafe {
        // SAFETY: We created `quantized` with the required capacity and initialized all of its
        // entries above.
        quantized.set_len(len);
    }
    let quantized = ArrayBase::from_shape_vec(unquantized.dim(), quantized)
        .expect("Vec should have correct len");
    Ok(PyArray::from_owned_array(py, quantized))
}

#[allow(clippy::too_many_arguments)]
fn vbq_multiplexed<'a>(
    py: Python<'a>,
    unquantized: PyReadonlyArrayDyn<'a, f32>,
    priors: &mut [crate::quant::EmpiricalDistribution],
    axis: usize,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<&'a PyArrayDyn<f32>> {
    let len = unquantized.len();

    let shape = unquantized.shape();
    let unquantized = unquantized.as_array();
    let unquantized = unquantized.axis_iter(Axis(axis));
    if unquantized.len() != priors.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err(alloc::format!(
            "Axis {} has wrong dimension: expected {} but found {}.",
            axis,
            priors.len(),
            unquantized.len()
        )));
    }

    let mut quantized: Vec<f32> = Vec::with_capacity(len);

    {
        let mut quantized = ArrayBase::<ndarray::ViewRepr<&mut _>, _>::from_shape(
            shape,
            quantized.spare_capacity_mut(),
        )
        .expect("len was chosen to match shape");

        let unquantized = unquantized.into_par_iter();
        let quantized = quantized.axis_iter_mut(Axis(axis));
        let quantized = quantized.into_par_iter();

        vbq_multiplexed_generic(
            shape,
            unquantized,
            quantized,
            priors,
            axis,
            posterior_variance,
            coarseness,
            |_src, dst, new| {
                dst.write(new);
            },
            update_prior,
            reference,
        )?;
    }
    unsafe {
        // SAFETY: We created `quantized` with the required capacity and initialized all of its
        // entries above.
        quantized.set_len(len);
    }
    let quantized =
        ArrayBase::from_shape_vec(shape, quantized).expect("Vec should have correct len");
    Ok(PyArray::from_owned_array(py, quantized))
}

#[allow(clippy::too_many_arguments)]
fn vbq_multiplexed_inplace<'a>(
    mut unquantized: PyReadwriteArrayDyn<'a, f32>,
    priors: &mut [crate::quant::EmpiricalDistribution],
    axis: usize,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<()> {
    let shape = unquantized.shape().to_vec();
    let mut unquantized = unquantized.as_array_mut();
    let unquantized = unquantized.axis_iter_mut(Axis(axis));
    if unquantized.len() != priors.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err(alloc::format!(
            "Axis {} has wrong dimension: expected {} but found {}.",
            axis,
            priors.len(),
            unquantized.len()
        )));
    }

    let unquantized = unquantized.into_par_iter();

    vbq_multiplexed_generic(
        &shape,
        unquantized,
        rayon::iter::repeat(core::iter::repeat(())).take(priors.len()),
        priors,
        axis,
        posterior_variance,
        coarseness,
        |src, _dst, new| {
            *src = new;
        },
        update_prior,
        reference,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn vbq_multiplexed_generic<'a, I1, I2, U, Src, Dst>(
    shape: &[usize],
    unquantized: I1,
    quantized: I2,
    priors: &mut [crate::quant::EmpiricalDistribution],
    axis: usize,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update: U,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<()>
where
    I1: IndexedParallelIterator,
    <I1 as ParallelIterator>::Item: IntoIterator<Item = Src>,
    I2: IndexedParallelIterator,
    <I2 as ParallelIterator>::Item: IntoIterator<Item = Dst>,
    Src: Borrow<f32>,
    U: Fn(Src, Dst, f32) + Send + Sync,
{
    fn inner<I1, I2, I3, R, E, U1, U2, Src, Dst>(
        shape: &[usize],
        unquantized: I1,
        quantized: I2,
        priors: &mut [crate::quant::EmpiricalDistribution],
        axis: usize,
        posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
        coarseness: f32,
        update: U1,
        update_prior: U2,
        reference: I3,
    ) -> PyResult<()>
    where
        I1: IndexedParallelIterator,
        <I1 as ParallelIterator>::Item: IntoIterator<Item = Src>,
        I2: IndexedParallelIterator,
        <I2 as ParallelIterator>::Item: IntoIterator<Item = Dst>,
        I3: IndexedParallelIterator,
        <I3 as ParallelIterator>::Item: IntoIterator<Item = R>,
        Src: Borrow<f32>,
        U1: Fn(Src, Dst, f32) + Send + Sync,
        U2: Fn(&mut crate::quant::EmpiricalDistribution, F32, F32, R) -> Result<(), E>
            + Send
            + Sync,
        E: Send + Sync,
        PyErr: From<E>,
    {
        match posterior_variance {
            Array(posterior_variance) => {
                if posterior_variance.shape() != shape {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "`posterior_variance` must have the same shape as `unquantized`.",
                    ));
                }
                let posterior_variance = posterior_variance.as_array();
                let two_coarseness = 2.0 * coarseness;

                unquantized
                    .zip(quantized)
                    .zip(posterior_variance.axis_iter(Axis(axis)))
                    .zip(priors)
                    .zip(reference)
                    .try_for_each(|((((src, dst), var), prior), reference)| {
                        for (((src, dst), &var), reference) in
                            src.into_iter().zip(dst).zip(&var).zip(reference)
                        {
                            let old = F32::new(*src.borrow())?;
                            let new = crate::quant::vbq(
                                old,
                                prior,
                                |x| x * x,
                                F32::new(two_coarseness * var)?,
                            );
                            update(src, dst, new.get());
                            update_prior(prior, old, new, reference)?;
                        }
                        Ok::<(), PyErr>(())
                    })?;
            }
            Scalar(posterior_variance) => {
                let bit_penalty = F32::new(2.0 * coarseness * posterior_variance)?;
                unquantized
                    .zip(quantized)
                    .zip(priors)
                    .zip(reference)
                    .try_for_each(|(((src, dst), prior), reference)| {
                        for ((src, dst), reference) in src.into_iter().zip(dst).zip(reference) {
                            let old = F32::new(*src.borrow())?;
                            let new = crate::quant::vbq(old, prior, |x| x * x, bit_penalty);
                            update(src, dst, new.get());
                            update_prior(prior, old, new, reference)?
                        }
                        Ok::<(), PyErr>(())
                    })?;
            }
        }

        Ok(())
    }

    match (update_prior, reference) {
        (Some(false), Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "providing a `reference` implies `update_prior=True`",
            ));
        }
        (None | Some(false), None) => {
            // The caller set `update_prior=False` (either implicitly or explicitly).
            inner(
                shape,
                unquantized,
                quantized,
                priors,
                axis,
                posterior_variance,
                coarseness,
                update,
                |_prior, _old, _new, _reference| Ok::<(), Infallible>(()),
                rayon::iter::repeat(core::iter::repeat(())).take(priors.len()),
            )?;
        }
        (Some(true), None) => {
            // The caller set `update_prior=False` without providing a reference
            inner(
                shape,
                unquantized,
                quantized,
                priors,
                axis,
                posterior_variance,
                coarseness,
                update,
                |prior, old, new, _reference| {
                    prior.remove(old).ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(
                            "An uncompressed value does not exist in the distribution. \
                                You might have to provide a `reference` argument.",
                        )
                    })?;
                    prior.insert(new);
                    Ok::<(), PyErr>(())
                },
                rayon::iter::repeat(core::iter::repeat(())).take(priors.len()),
            )?;
        }
        (None | Some(true), Some(mut reference)) => {
            // The caller provided a reference for prior updates.
            if reference.shape() != shape {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "`reference` must have the same shape as `unquantized`.",
                ));
            }
            let mut reference = reference.as_array_mut();

            inner(
                shape,
                unquantized,
                quantized,
                priors,
                axis,
                posterior_variance,
                coarseness,
                update,
                |prior, _old, new, reference| {
                    prior.remove(F32::new(*reference)?).ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(
                            "A reference value does not exist in the distribution.",
                        )
                    })?;
                    prior.insert(new);
                    *reference = new.get();
                    Ok::<(), PyErr>(())
                },
                reference.axis_iter_mut(Axis(axis)).into_par_iter(),
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn vbq_(
    py: Python<'_>,
    unquantized: PyReadwriteArrayDyn<'_, f32>,
    prior: Py<EmpiricalDistribution>,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
) -> PyResult<()> {
    match &mut prior.borrow_mut(py).0 {
        EmpiricalDistributionImpl::Single(distribution) => vbq_single_inplace(
            unquantized,
            distribution,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
        ),
        EmpiricalDistributionImpl::Multiple {
            distributions,
            axis,
        } => vbq_multiplexed_inplace(
            unquantized,
            distributions,
            *axis,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
        ),
    }
}

fn vbq_single_inplace(
    mut unquantized: PyReadwriteArrayDyn<'_, f32>,
    prior: &mut crate::quant::EmpiricalDistribution,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
) -> PyResult<()> {
    let mut unquantized = unquantized.as_array_mut();
    let dim: ndarray::prelude::Dim<ndarray::IxDynImpl> = unquantized.raw_dim();

    match (update_prior, reference.is_some()) {
        (Some(false), true) => Err(pyo3::exceptions::PyValueError::new_err(
            "providing a `reference` implies `update_prior=True`",
        )),
        (None | Some(false), false) => {
            // Prior doesn't get mutated, so we can run VBQ in parallel.
            let src_and_dst = par_azip!(&mut unquantized);
            match posterior_variance {
                Array(posterior_variance) => {
                    if posterior_variance.dims() != dim {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(
                            "`posterior_variance` must have the same shape as `unquantized`.",
                        ));
                    }
                    let posterior_variance = posterior_variance.as_array();
                    let two_coarseness = 2.0 * coarseness;
                    let src_dst_penalty = src_and_dst
                        .and(&posterior_variance)
                        .into_par_iter()
                        .map(|(sd, var)| (sd, F32::new(two_coarseness * *var)));
                    vbq_parallel(src_dst_penalty, prior, |sd| **sd, |sd, value| *sd = value)
                }
                Scalar(posterior_variance) => {
                    let bit_penalty = Result::<_, Infallible>::Ok(F32::new(
                        2.0 * coarseness * posterior_variance,
                    )?);
                    let src_dst_penalty = src_and_dst.into_par_iter().map(|sd| (sd, bit_penalty));
                    vbq_parallel(
                        src_dst_penalty,
                        prior,
                        |(sd,)| **sd,
                        |(sd,), value| *sd = value,
                    )
                }
            }
        }
        _ => {
            // We mutate the prior after each update, and therefore the result depends on the
            // order in which we iterate over items. Fall back to a sequential implementation
            // since running VBQ in parallel would make it nondeterministic here.
            vbq_sequential(
                unquantized.iter_mut().map(|src| (src, ())),
                dim,
                prior,
                posterior_variance,
                coarseness,
                reference,
                |src, _dst, value| *src = value,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn vbq_sequential<Src, Dst>(
    src_and_dst: impl ExactSizeIterator<Item = (Src, Dst)>,
    dim: IxDyn,
    prior: &mut crate::quant::EmpiricalDistribution,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    update: impl FnMut(Src, Dst, f32),
) -> PyResult<()>
where
    Src: Borrow<f32>,
{
    return match posterior_variance {
        Array(posterior_variance) => {
            if posterior_variance.dims() != dim {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "`posterior_variance` must have the same shape as `unquantized`.",
                ));
            }
            let two_coarseness = 2.0 * coarseness;
            let bit_penalty = posterior_variance.as_array();
            let bit_penalty = bit_penalty.iter().map(|&x| F32::new(two_coarseness * x));
            internal(src_and_dst.zip(bit_penalty), dim, prior, reference, update)
        }
        Scalar(posterior_variance) => {
            let bit_penalty =
                Result::<_, Infallible>::Ok(F32::new(2.0 * coarseness * posterior_variance)?);
            internal(
                src_and_dst.map(|(src, dst)| ((src, dst), bit_penalty)),
                dim,
                prior,
                reference,
                update,
            )
        }
    };

    fn internal<Src, Dst, E>(
        src_dst_penalty: impl ExactSizeIterator<Item = ((Src, Dst), Result<F32, E>)>,
        dim: IxDyn,
        prior: &mut crate::quant::EmpiricalDistribution,
        reference: Option<PyReadwriteArrayDyn<'_, f32>>,
        mut update: impl FnMut(Src, Dst, f32),
    ) -> PyResult<()>
    where
        Src: Borrow<f32>,
        PyErr: From<E>,
    {
        if let Some(mut reference) = reference {
            if reference.dims() != dim {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "`reference` must have the same shape as `unquantized`.",
                ));
            }

            let mut reference = reference.as_array_mut();

            for (((src, dst), bit_penalty), reference) in src_dst_penalty.zip(&mut reference) {
                let unquantized = F32::new(*src.borrow())?;
                let reference_val = F32::new(*reference)?;
                let quantized = crate::quant::vbq(unquantized, prior, |x| x * x, bit_penalty?);
                prior.remove(reference_val).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(
                        "An uncompressed value does not exist in the distribution. \
                    You might have to provide a `reference` argument.",
                    )
                })?;
                prior.insert(quantized);
                update(src, dst, quantized.get());
                *reference = quantized.get();
            }
        } else {
            for ((src, dst), bit_penalty) in src_dst_penalty {
                let unquantized = F32::new(*src.borrow())?;
                let quantized = crate::quant::vbq(unquantized, prior, |x| x * x, bit_penalty?);
                update(src, dst, quantized.get());
                prior.remove(unquantized).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(
                        "An uncompressed value does not exist in the distribution. \
                        You might have to provide a `reference` argument.",
                    )
                })?;
                prior.insert(quantized);
            }
        }

        Ok(())
    }
}

fn vbq_parallel<SrcAndDst, E>(
    src_dst_penalty: impl ParallelIterator<Item = (SrcAndDst, Result<F32, E>)>,
    prior: &mut crate::quant::EmpiricalDistribution,
    extract: impl Fn(&SrcAndDst) -> f32 + Sync + Send,
    update: impl Fn(SrcAndDst, f32) + Sync + Send,
) -> PyResult<()>
where
    SrcAndDst: Send + Sync,
    PyErr: From<E>,
{
    src_dst_penalty.try_for_each(move |(sd, bit_penalty)| {
        let unquantized = F32::new(extract(&sd))?;
        let quantized = crate::quant::vbq(unquantized, prior, |x| x * x, bit_penalty?);
        update(sd, quantized.get());
        PyResult::Ok(())
    })
}
