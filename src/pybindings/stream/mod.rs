pub mod model;
pub mod stack;

use pyo3::{prelude::*, wrap_pymodule};

use std::prelude::v1::*;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(ans))?;
    module.add_wrapped(wrap_pymodule!(model))?;
    Ok(())
}

/// Docstring of `ans` module
#[pymodule]
fn ans(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    stack::init_module(py, module)
}

/// Docstring of `model` module
#[pymodule]
fn model(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    model::init_module(py, module)
}
