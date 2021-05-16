pub mod chain;
pub mod model;
pub mod queue;
pub mod stack;

use pyo3::{prelude::*, wrap_pymodule};

use std::prelude::v1::*;

use crate::{stream::TryCodingError, CoderError, DefaultEncoderFrontendError};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(model))?;
    module.add_wrapped(wrap_pymodule!(queue))?;
    module.add_wrapped(wrap_pymodule!(stack))?;
    module.add_wrapped(wrap_pymodule!(chain))?;
    Ok(())
}

/// Entropy models for individual symbols
/// 
/// This module provides tools to define probability distributions over symbols in fixed
/// point arithmetic, so that the models (more precisely, the cumulative distribution
/// functions) are *exactly* invertible without any rounding errors. Such exactly invertible
/// models are used by all entropy coders in the sister modules `stack`, `queue`, and
/// `chain`.
#[pymodule]
fn model(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    model::init_module(py, module)
}

/// Entropy coding with queue semantics (first in first out) using a Range Coder [1].
/// 
/// See last example in the top-level API documentation of `constriction`.
/// 
/// ## References
///
/// [1] Pasco, Richard Clark. Source coding algorithms for fast data compression. Diss.
/// Stanford University, 1976.
#[pymodule]
fn queue(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    queue::init_module(py, module)
}

/// Entropy coding with stack semantics (last in first out) using Asymmetric Numeral Systems
/// (ANS) [1].
/// 
/// See first two examples in the top-level API documentation of `constriction`.
/// 
/// ## References
///
/// [1] Duda, Jarek, et al. "The use of asymmetric numeral systems as an accurate
/// replacement for Huffman coding." 2015 Picture Coding Symposium (PCS). IEEE, 2015.
#[pymodule]
fn stack(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    stack::init_module(py, module)
}

/// Experimental new entropy coder for joint inference, quantization, and bits-back coding.
#[pymodule]
fn chain(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    chain::init_module(py, module)
}

impl<CodingError: Into<PyErr>, ModelError> From<TryCodingError<CodingError, ModelError>> for PyErr {
    fn from(err: TryCodingError<CodingError, ModelError>) -> Self {
        match err {
            crate::stream::TryCodingError::CodingError(err) => err.into(),
            crate::stream::TryCodingError::InvalidEntropyModel(_) => {
                pyo3::exceptions::PyValueError::new_err("Invalid parameters for entropy model")
            }
        }
    }
}

impl<FrontendError: Into<PyErr>, BackendError: Into<PyErr>>
    From<CoderError<FrontendError, BackendError>> for PyErr
{
    fn from(err: CoderError<FrontendError, BackendError>) -> Self {
        match err {
            CoderError::Frontend(err) => err.into(),
            CoderError::Backend(err) => err.into(),
        }
    }
}

impl From<DefaultEncoderFrontendError> for PyErr {
    fn from(err: DefaultEncoderFrontendError) -> Self {
        match err {
            DefaultEncoderFrontendError::ImpossibleSymbol => {
                pyo3::exceptions::PyKeyError::new_err(err.to_string())
            }
        }
    }
}
