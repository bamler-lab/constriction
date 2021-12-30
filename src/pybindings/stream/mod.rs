// mod chain;
mod model;
mod queue;
// mod stack;

use pyo3::{prelude::*, wrap_pymodule};

use std::prelude::v1::*;

use crate::{stream::TryCodingError, CoderError, DefaultEncoderFrontendError};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(model))?;
    module.add_wrapped(wrap_pymodule!(queue))?;
    // module.add_wrapped(wrap_pymodule!(stack))?;
    // module.add_wrapped(wrap_pymodule!(chain))?;
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
// #[pymodule]
// fn stack(py: Python<'_>, module: &PyModule) -> PyResult<()> {
//     stack::init_module(py, module)
// }

/// Experimental entropy coding algorithm for advanced variants of bitsback coding.
///
/// This module provides the `ChainCoder`, an experimental entropy coder that is similar
/// to an `AnsCoder` in that it operates as a stack (i.e., a last-in-first-out data
/// structure). However, different to an `AnsCoder`, a `ChainCoder` treats each symbol
/// independently. Thus, when decoding some bit string into a sequence of symbols, any
/// modification to the entropy model for one symbol does not affect decoding for any other
/// symbol (by contrast, when decoding with an `AnsCoder` then changing the entropy model
/// for one symbol can affect *all* subsequently decoded symbols too, see
/// [Motivation](#motivation) below).
///
/// This property of treating symbols independently upon decoding can be useful for advanced
/// compression methods that combine inference, quantization, and bits-back coding.
///
/// # Motivation
///
/// The following example illustrates how decoding differs between an `AnsCoder` and a
/// `ChainCoder`. We decode the same bitstring `data` twice with each coder: once with a
/// sequence of toy entropy models, and then a second time with slightly different sequence
/// of entropy models. Importantly, only the entropy model for the first decoded symbol
/// differs between the two applications of each coder. We then observe that
///
/// - with the `AnsCoder`, changing the first entropy model affects not only the first
///   decoded symbol but also has a ripple effect that can affect subsequently decoded
///   symbols; while
/// - with the `ChainCoder`, changing the first entropy model affects only the first decoded
///   symbol; all subsequently decoded symbols remain unchanged.
///
/// ```python
/// # Some sample binary data and sample probabilities for our entropy models
/// data = np.array(
///     [0x80d1_4131, 0xdda9_7c6c, 0x5017_a640, 0x0117_0a3d], np.uint32)
/// probs = np.array([
///     [0.1, 0.7, 0.1, 0.1],
///     [0.2, 0.2, 0.1, 0.5],
///     [0.2, 0.1, 0.4, 0.3],
/// ])
///
/// # Decoding `data` with an `AnsCoder` results in the symbols `[0, 0, 1]`.
/// ansCoder = constriction.stream.stack.AnsCoder(data, True)
/// assert ansCoder.decode_iid_categorical_symbols(1, 0, probs[0, :]) == [0]
/// assert ansCoder.decode_iid_categorical_symbols(1, 0, probs[1, :]) == [0]
/// assert ansCoder.decode_iid_categorical_symbols(1, 0, probs[2, :]) == [1]
///
/// # Even if we change only the first entropy model (slightly), *all* decoded
/// # symbols can change:
/// probs[0, :] = np.array([0.09, 0.71, 0.1, 0.1])
/// ansCoder = constriction.stream.stack.AnsCoder(data, True)
/// assert ansCoder.decode_iid_categorical_symbols(1, 0, probs[0, :]) == [1]
/// assert ansCoder.decode_iid_categorical_symbols(1, 0, probs[1, :]) == [0]
/// assert ansCoder.decode_iid_categorical_symbols(1, 0, probs[2, :]) == [3]
/// # It's no surprise that the first symbol changed since we changed its entropy
/// # model. But note that the third symbol changed too even though we hadn't
/// # changed its entropy model.
/// # --> Changes to entropy models have a *global* effect.
///
/// # Let's try the same with a `ChainCoder`:
/// probs[0, :] = np.array([0.1, 0.7, 0.1, 0.1]) # Restore original model.
/// chainCoder = constriction.stream.chain.ChainCoder(data, False, True)
/// assert chainCoder.decode_iid_categorical_symbols(1, 0, probs[0, :]) == [0]
/// assert chainCoder.decode_iid_categorical_symbols(1, 0, probs[1, :]) == [3]
/// assert chainCoder.decode_iid_categorical_symbols(1, 0, probs[2, :]) == [3]

/// # We obtain different symbols than for the `AnsCoder`, of course, but that's
/// # not the point here.
///
/// probs[0, :] = np.array([0.09, 0.71, 0.1, 0.1]) # Change the first model again.
/// chainCoder = constriction.stream.chain.ChainCoder(data, False, True)
/// assert chainCoder.decode_iid_categorical_symbols(1, 0, probs[0, :]) == [1]
/// assert chainCoder.decode_iid_categorical_symbols(1, 0, probs[1, :]) == [3]
/// assert chainCoder.decode_iid_categorical_symbols(1, 0, probs[2, :]) == [3]
/// # The only symbol that changed was the one whose entropy model we had changed.
/// # --> In a `ChainCoder`, changes to entropy models (and also to compressed
/// #     bits) only have a *local* effect on the decompressed symbols.
/// ```
///
/// # How does this work?
///
/// TODO
// #[pymodule]
// fn chain(py: Python<'_>, module: &PyModule) -> PyResult<()> {
//     chain::init_module(py, module)
// }

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
