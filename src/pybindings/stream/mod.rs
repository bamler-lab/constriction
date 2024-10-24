mod chain;
mod model;
mod queue;
mod stack;

use pyo3::{prelude::*, wrap_pymodule};

use std::prelude::v1::*;

use crate::{stream::TryCodingError, CoderError, DefaultEncoderFrontendError};

/// Stream codes, i.e., entropy codes that amortize compressed bits over several symbols.
///
/// We provide two main stream codes:
///
/// - **Range Coding** [1, 2] (in submodule [`queue`](stream/queue.html)), which is a computationally
///   more efficient variant of Arithmetic Coding [3], and which has "queue" semantics ("first in
///   first out", i.e., symbols get decoded in the same order in which they were encoded); and
/// - **Asymmetric Numeral Systems (ANS)** [4] (in submodule [`stack`](stream/stack.html)), which has
///   "stack" semantics ("last in first out", i.e., symbols get decoded in *reverse* order compared
///   to the the order  in which they got encoded).
///
/// In addition, the submodule [`model`](stream/model.html) provides common entropy models and
/// wrappers for defining your own entropy models (or for using models from the popular `scipy`
/// package in `constriction`). These entropy models can be used with both of the above stream
/// codes.
///
/// We further provide an experimental new "Chain Coder"  (in submodule [`chain`](stream/chain.html)),
/// which is intended for special new compression methods.
///
/// ## Examples
///
/// See top of the documentations of both submodules [`queue`](stream/queue.html) and
/// [`stack`](stream/stack.html).
///
/// ## References
///
/// [1] Pasco, Richard Clark. Source coding algorithms for fast data compression. Diss.
/// Stanford University, 1976.
///
/// [2] Martin, G. Nigel N. "Range encoding: an algorithm for removing redundancy from a
/// digitised message." Proc. Institution of Electronic and Radio Engineers International
/// Conference on Video and Data Recording. 1979.
///
/// [3] Rissanen, Jorma, and Glen G. Langdon. "Arithmetic coding." IBM Journal of research
/// and development 23.2 (1979): 149-162.
///
/// [4] Duda, Jarek, et al. "The use of asymmetric numeral systems as an accurate
/// replacement for Huffman coding." 2015 Picture Coding Symposium (PCS). IEEE, 2015.
#[pymodule]
#[pyo3(name = "stream")]
pub fn init_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(model::init_module))?;
    module.add_wrapped(wrap_pymodule!(queue::init_module))?;
    module.add_wrapped(wrap_pymodule!(stack::init_module))?;
    module.add_wrapped(wrap_pymodule!(chain::init_module))?;
    Ok(())
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
