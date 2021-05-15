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

/// Docstring of `model` module
#[pymodule]
fn model(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    model::init_module(py, module)
}

/// Docstring of `queue` module
#[pymodule]
fn queue(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    queue::init_module(py, module)
}

/// Docstring of `stack` module
#[pymodule]
fn stack(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    stack::init_module(py, module)
}

/// Docstring of `chain` module
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
