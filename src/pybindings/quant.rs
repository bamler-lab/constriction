use core::{borrow::Borrow, convert::Infallible};

use crate::{quant::UnnormalizedDistribution, F32};
use alloc::vec::Vec;
use ndarray::parallel::prelude::*;
use ndarray::{ArrayBase, IxDyn};
use numpy::{PyArray, PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::{prelude::*, types::PyTuple};

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
pub struct EmpiricalDistribution(crate::quant::EmpiricalDistribution);

#[pymethods]
impl EmpiricalDistribution {
    #[new]
    #[pyo3(signature = (*args))]
    pub fn new(py: Python<'_>, args: &PyTuple) -> PyResult<Py<Self>> {
        let mut distribution = Self(crate::quant::EmpiricalDistribution::new());
        distribution.add_points(args)?;
        Py::new(py, distribution)
    }

    #[pyo3(signature = (*args))]
    pub fn add_points(&mut self, args: &PyTuple) -> PyResult<()> {
        for points in args.iter() {
            let points = points.extract::<PyReadonlyArrayDyn<'_, f32>>()?;
            self.0.try_add_points(points.as_array().iter().copied())?;
        }
        Ok(())
    }

    pub fn update(
        &mut self,
        old: PyReadonlyF32ArrayOrScalar<'_>,
        new: PyReadonlyF32ArrayOrScalar<'_>,
    ) -> PyResult<()> {
        match (old, new) {
            (Scalar(old), Scalar(new)) => {
                self.0.remove(F32::new(old)?).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(
                        "The `old` value does not exist in the distribution.",
                    )
                })?;
                self.0.insert(F32::new(new)?);
                Ok(())
            }
            (Array(old), Array(new)) if old.dims() == new.dims() => {
                for (old, &new) in old.as_array().iter().zip(&new.as_array()) {
                    self.0.remove(F32::new(*old)?).ok_or_else(|| {
                        pyo3::exceptions::PyKeyError::new_err(
                            "One of the entries in `old` does not exist in the distribution.",
                        )
                    })?;
                    self.0.insert(F32::new(new)?);
                }
                Ok(())
            }
            _ => Err(pyo3::exceptions::PyAssertionError::new_err(
                "`old` and `new` must have the same shape.",
            )),
        }
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
                            py,
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
                            py,
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
                    py,
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
#[pyfunction]
fn vbq_(
    py: Python<'_>,
    mut unquantized: PyReadwriteArrayDyn<'_, f32>,
    prior: Py<EmpiricalDistribution>,
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
                    vbq_parallel(
                        py,
                        src_dst_penalty,
                        prior,
                        |sd| **sd,
                        |sd, value| *sd = value,
                    )
                }
                Scalar(posterior_variance) => {
                    let bit_penalty = Result::<_, Infallible>::Ok(F32::new(
                        2.0 * coarseness * posterior_variance,
                    )?);
                    let src_dst_penalty = src_and_dst.into_par_iter().map(|sd| (sd, bit_penalty));
                    vbq_parallel(
                        py,
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
                py,
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
    py: Python<'_>,
    src_and_dst: impl ExactSizeIterator<Item = (Src, Dst)>,
    dim: IxDyn,
    prior: Py<EmpiricalDistribution>,
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
            internal(
                py,
                src_and_dst.zip(bit_penalty),
                dim,
                prior,
                reference,
                update,
            )
        }
        Scalar(posterior_variance) => {
            let bit_penalty =
                Result::<_, Infallible>::Ok(F32::new(2.0 * coarseness * posterior_variance)?);
            internal(
                py,
                src_and_dst.map(|(src, dst)| ((src, dst), bit_penalty)),
                dim,
                prior,
                reference,
                update,
            )
        }
    };

    fn internal<Src, Dst, E>(
        py: Python<'_>,
        src_dst_penalty: impl ExactSizeIterator<Item = ((Src, Dst), Result<F32, E>)>,
        dim: IxDyn,
        prior: Py<EmpiricalDistribution>,
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

            let prior = &mut *prior.borrow_mut(py);
            let mut reference = reference.as_array_mut();

            for (((src, dst), bit_penalty), reference) in src_dst_penalty.zip(&mut reference) {
                let unquantized = F32::new(*src.borrow())?;
                let reference_val = F32::new(*reference)?;
                let quantized = crate::quant::vbq(unquantized, &prior.0, |x| x * x, bit_penalty?);
                prior.0.remove(reference_val).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(
                        "An uncompressed value does not exist in the distribution. \
                    You might have to provide a `reference` argument.",
                    )
                })?;
                prior.0.insert(quantized);
                update(src, dst, quantized.get());
                *reference = quantized.get();
            }
        } else {
            let prior = &mut *prior.borrow_mut(py);
            for ((src, dst), bit_penalty) in src_dst_penalty {
                let unquantized = F32::new(*src.borrow())?;
                let quantized = crate::quant::vbq(unquantized, &prior.0, |x| x * x, bit_penalty?);
                update(src, dst, quantized.get());
                prior.0.remove(unquantized).ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err(
                        "An uncompressed value does not exist in the distribution. \
                        You might have to provide a `reference` argument.",
                    )
                })?;
                prior.0.insert(quantized);
            }
        }

        Ok(())
    }
}

fn vbq_parallel<SrcAndDst, E>(
    py: Python<'_>,
    src_dst_penalty: impl ParallelIterator<Item = (SrcAndDst, Result<F32, E>)>,
    prior: Py<EmpiricalDistribution>,
    extract: impl Fn(&SrcAndDst) -> f32 + Sync + Send,
    update: impl Fn(SrcAndDst, f32) + Sync + Send,
) -> PyResult<()>
where
    SrcAndDst: Send + Sync,
    PyErr: From<E>,
{
    let prior = &mut *prior.borrow_mut(py);
    src_dst_penalty.try_for_each(move |(sd, bit_penalty)| {
        let unquantized = F32::new(extract(&sd))?;
        let quantized = crate::quant::vbq(unquantized, &prior.0, |x| x * x, bit_penalty?);
        update(sd, quantized.get());
        PyResult::Ok(())
    })
}
