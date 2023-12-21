//! TODO: much simpler:
//! - outsource all indexing stuff to python/numpy
//! - implement a wrapper for `quant::EmpiricalDistribution`
//!   - has a method `update` that takes two slices of tensors,
//! - in `quant`, implement a function `vbq` that takes
//!   - an `&mut EmpiricalDistribution`
//!   - a mutable slice of entries to be quantized
//!   - a posterior variance (for now just a scalar; later it will broadcast)
//!   - a boolean switch `dynamic`, and possibly an optional parameter `reference`

use alloc::vec;
use alloc::vec::Vec;
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
struct Vbq {
    prior: DynamicEmpiricalDistribution,

    #[pyo3(get)]
    data: Py<PyArrayDyn<f32>>,
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
        Ok(Self {
            prior,
            data: unquantized,
        })
    }

    #[pyo3(
        text_signature = "(self, index, posterior_variance, coarseness, [update_prior], [reference])"
    )]
    pub fn quantize_with_quadratic_distortion(
        &mut self,
        py: Python<'_>,
        index: &PyAny,
        posterior_variance: f32,
        coarseness: f32,
        update_prior: Option<bool>,
        reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    ) -> PyResult<()> {
        let index = parse_slice_indices(index, self.data.as_ref(py).shape());
        let mut data = self.data.as_ref(py).readwrite();
        let mut data = data.as_array_mut();
        let mut data = data.slice_mut(&index);
        let posterior_variance = NonNanFloat::new(posterior_variance).unwrap();
        let coarseness = NonNanFloat::new(coarseness).unwrap();

        if let Some(mut reference) = reference {
            if update_prior == Some(false) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Providing a `referenc` implies `update_prior=True`",
                ));
            }
            let mut reference = reference.as_array_mut();
            let mut reference = reference.slice_mut(index);

            for (x, reference) in data.iter_mut().zip(reference.iter_mut()) {
                let old_value = NonNanFloat::new(*x).unwrap();
                let reference_val = NonNanFloat::new(*reference).unwrap();
                let new_value = vbq_quadratic_distortion::<f32, _, _>(
                    &self.prior,
                    old_value,
                    posterior_variance,
                    coarseness,
                );
                *x = new_value.get();
                *reference = new_value.get();
                self.prior.remove(reference_val).expect("prior out of sync");
                self.prior.insert(new_value);
            }
        } else {
            for x in data.iter_mut() {
                let old_value = NonNanFloat::new(*x).unwrap();
                let new_value = vbq_quadratic_distortion::<f32, _, _>(
                    &self.prior,
                    old_value,
                    posterior_variance,
                    coarseness,
                );
                *x = new_value.get();
                if update_prior == Some(true) {
                    self.prior.remove(old_value).expect("prior out of sync");
                    self.prior.insert(new_value);
                }
            }
        }

        Ok(())
    }

    #[pyo3(signature = (index, new_values, update_prior))]
    #[pyo3(text_signature = "(self, index, new_values, [update_prior])")]
    pub fn update(
        &mut self,
        py: Python<'_>,
        index: &PyAny,
        new_values: PyReadonlyArrayDyn<'_, f32>,
        update_prior: Option<bool>,
    ) {
        let index = parse_slice_indices(index, self.data.as_ref(py).shape());
        let mut data = self.data.as_ref(py).readwrite();
        let mut data = data.as_array_mut();
        let mut data = data.slice_mut(&index);

        let new_values = new_values.as_array();
        let new_values = new_values.slice(index);

        if update_prior == Some(true) {
            for (dst, &src) in data.iter_mut().zip(new_values.iter()) {
                *dst = src;
                self.prior
                    .remove(NonNanFloat::new(*dst).unwrap())
                    .expect("prior out of sync");
                self.prior.insert(NonNanFloat::new(src).unwrap());
            }
        } else {
            for (dst, &src) in data.iter_mut().zip(new_values.iter()) {
                *dst = src;
            }
        }
    }

    #[pyo3(signature = ())]
    pub fn prior_entropy_base2(&self) -> f32 {
        self.prior.entropy_base2()
    }
}

fn parse_slice_indices(
    index: &PyAny,
    target_dims: &[usize],
) -> SliceInfo<Vec<ndarray::SliceInfoElem>, IxDyn, IxDyn> {
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
                    .indices(dims.next().expect("too long") as i64)
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
