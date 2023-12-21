//! TODO: much simpler:
//! - outsource all indexing stuff to python/numpy
//! - implement a wrapper for `quant::EmpiricalDistribution`
//!   - has a method `update` that takes two slices of tensors,
//! - in `quant`, implement a function `vbq` that takes
//!   - an `&mut EmpiricalDistribution`
//!   - a mutable slice of entries to be quantized
//!   - a posterior variance (for now just a scalar; later it will broadcast)
//!   - a boolean switch `dynamic`, and possibly an optional parameter `reference`

use std::sync::Mutex;

use alloc::vec::Vec;
use alloc::{sync::Arc, vec};
use ndarray::{IxDyn, SliceInfo, SliceInfoElem};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::{
    prelude::*,
    types::{PySlice, PySliceIndices, PyTuple},
};

use crate::{
    quant::{vbq::vbq_quadratic_distortion, DynamicEmpiricalDistribution, EmpiricalDistribution},
    NonNanFloat,
};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<Vbq>()?;
    Ok(())
}

#[pyclass]
#[derive(Debug, Clone)]
struct Vbq(Arc<Mutex<VbqInternal>>);

#[derive(Debug, Clone)]
struct VbqInternal {
    prior: DynamicEmpiricalDistribution,
    data: Py<PyArrayDyn<f32>>,
}

type Slice = SliceInfo<Vec<ndarray::SliceInfoElem>, IxDyn, IxDyn>;

#[pyclass]
#[derive(Debug, Clone)]
struct VbqView {
    vbq: Arc<Mutex<VbqInternal>>,
    slice: Slice,
}

#[pymethods]
impl Vbq {
    #[new]
    #[pyo3(signature = (unquantized))]
    pub fn new(py: Python<'_>, unquantized: Py<PyArrayDyn<f32>>) -> PyResult<Self> {
        let prior = {
            let unquantized = unquantized.as_ref(py).readonly();
            let unquantized = unquantized.as_array();
            DynamicEmpiricalDistribution::try_from_points(unquantized.iter())
                .expect("NaN encountered")
        };
        Ok(Self(Arc::new(Mutex::new(VbqInternal {
            prior,
            data: unquantized,
        }))))
    }

    #[getter]
    pub fn data(&self, py: Python<'_>) -> Py<PyArrayDyn<f32>> {
        let this = self.0.lock().unwrap();
        this.data.as_ref(py).to_owned()
    }

    pub fn __getitem__(&mut self, py: Python<'_>, index: &PyAny) -> VbqView {
        let this = self.0.lock().unwrap();
        let data = this.data.as_ref(py);
        let slice = parse_slice(index, data.shape());
        VbqView {
            vbq: Arc::clone(&self.0),
            slice,
        }
    }

    #[pyo3(signature = ())]
    pub fn prior_entropy_base2(&self) -> f32 {
        let this = self.0.lock().unwrap();
        this.prior.entropy_base2()
    }
}

#[pymethods]
impl VbqView {
    #[pyo3(
        text_signature = "(self, index, posterior_variance, coarseness, update_prior=False, reference=None)"
    )]
    pub fn quantize_with_quadratic_distortion(
        &mut self,
        py: Python<'_>,
        posterior_variance: f32,
        coarseness: f32,
        update_prior: Option<bool>,
        reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    ) -> PyResult<()> {
        let vbq = &mut *self.vbq.lock().unwrap();
        let mut data = vbq.data.as_ref(py).readwrite();
        let mut data = data.as_array_mut();
        let mut data = data.slice_mut(&self.slice);
        let posterior_variance = NonNanFloat::new(posterior_variance).unwrap();
        let coarseness = NonNanFloat::new(coarseness).unwrap();
        let prior = &mut vbq.prior;

        if let Some(mut reference) = reference {
            if update_prior == Some(false) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Providing a `referenc` implies `update_prior=True`",
                ));
            }
            let mut reference = reference.as_array_mut();

            for (x, reference) in data.iter_mut().zip(reference.iter_mut()) {
                let old_value = NonNanFloat::new(*x).unwrap();
                let reference_val = NonNanFloat::new(*reference).unwrap();
                let new_value = vbq_quadratic_distortion::<f32, _, _>(
                    prior,
                    old_value,
                    posterior_variance,
                    coarseness,
                );
                prior.remove(reference_val).expect("prior out of sync");
                prior.insert(new_value);
                *x = new_value.get();
                *reference = new_value.get();
            }
        } else {
            for x in data.iter_mut() {
                let old_value = NonNanFloat::new(*x).unwrap();
                let new_value = vbq_quadratic_distortion::<f32, _, _>(
                    prior,
                    old_value,
                    posterior_variance,
                    coarseness,
                );
                *x = new_value.get();
                if update_prior == Some(true) {
                    prior.remove(old_value).expect("prior out of sync");
                    prior.insert(new_value);
                }
            }
        }

        Ok(())
    }

    #[pyo3(text_signature = "(self, new_values, update_prior=False, reference=None)")]
    pub fn update(
        &mut self,
        py: Python<'_>,
        new_values: PyReadonlyArrayDyn<'_, f32>,
        update_prior: Option<bool>,
        reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    ) -> PyResult<()> {
        let vbq = &mut *self.vbq.lock().unwrap();
        let mut data = vbq.data.as_ref(py).readwrite();
        let mut data = data.as_array_mut();
        let mut data = data.slice_mut(&self.slice);

        let new_values = new_values.as_array();
        let prior = &mut vbq.prior;

        if let Some(mut reference) = reference {
            if update_prior == Some(false) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Providing a `referenc` implies `update_prior=True`",
                ));
            }
            let mut reference = reference.as_array_mut();
            for ((dst, &src), reference) in data
                .iter_mut()
                .zip(new_values.iter())
                .zip(reference.iter_mut())
            {
                prior
                    .remove(NonNanFloat::new(*reference).unwrap())
                    .expect("prior out of sync");
                prior.insert(NonNanFloat::new(src).unwrap());
                *dst = src;
                *reference = src;
            }
        } else if update_prior == Some(true) {
            for (dst, &src) in data.iter_mut().zip(new_values.iter()) {
                prior
                    .remove(NonNanFloat::new(*dst).unwrap())
                    .expect("prior out of sync");
                prior.insert(NonNanFloat::new(src).unwrap());
                *dst = src;
            }
        } else {
            for (dst, &src) in data.iter_mut().zip(new_values.iter()) {
                *dst = src;
            }
        }

        Ok(())
    }
}

fn parse_slice(index: &PyAny, target_dims: &[usize]) -> Slice {
    fn convert_slice(i: &PyAny, dims: &mut impl Iterator<Item = usize>) -> SliceInfoElem {
        match i
            .extract::<Option<SimpleSlicePart<'_>>>()
            .expect("can't parse")
        {
            Some(SimpleSlicePart::Slice(slice)) => {
                let PySliceIndices {
                    start,
                    stop,
                    step,
                    slicelength: _,
                } = slice
                    .indices(dims.next().expect("too long") as core::ffi::c_long)
                    .expect("doesn't fit");
                SliceInfoElem::Slice {
                    start,
                    end: Some(stop),
                    step,
                }
            }
            Some(SimpleSlicePart::Index(index)) => {
                dims.next().expect("too long2");
                SliceInfoElem::Index(index)
            }
            None => SliceInfoElem::NewAxis,
        }
    }

    let mut dims = target_dims.iter().cloned();
    let mut index = if let Ok(index) = index.extract::<&PyTuple>() {
        index
            .as_slice()
            .iter()
            .map(|&i| convert_slice(i, &mut dims))
            .collect::<Vec<_>>()
    } else {
        vec![convert_slice(index, &mut dims)]
    };
    for _ in dims {
        index.push(SliceInfoElem::Slice {
            start: 0,
            end: None,
            step: 1,
        });
    }
    unsafe { SliceInfo::<_, IxDyn, IxDyn>::new(index).unwrap() }
}

#[derive(FromPyObject, Debug)]
enum SimpleSlicePart<'a> {
    Slice(&'a PySlice),
    Index(isize),
}
