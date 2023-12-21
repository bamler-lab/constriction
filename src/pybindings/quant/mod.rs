pub mod vbq;
use pyo3::{prelude::*, wrap_pymodule};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(init_vbq))?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "vbq")]
fn init_vbq(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    vbq::init_module(py, module)
}
