// mod chain;
mod model;
mod queue;
mod stack;

use pyo3::{prelude::*, wrap_pymodule};

use std::prelude::v1::*;

use crate::{stream::TryCodingError, CoderError, DefaultEncoderFrontendError};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(model))?;
    module.add_wrapped(wrap_pymodule!(queue))?;
    module.add_wrapped(wrap_pymodule!(stack))?;
    // module.add_wrapped(wrap_pymodule!(chain))?;
    Ok(())
}

/// Entropy models and model families for use with any of the provided stream codes.
///
/// This module provides tools to define probability distributions over symbols in fixed
/// point arithmetic, so that the models (more precisely, the cumulative distribution
/// functions) are *exactly* invertible without any rounding errors. Such exactly invertible
/// models are used by all entropy coders in the sister modules [`stack`](stack.html),
/// [`queue`](queue.html), and [`chain`](chain.html).
///
/// ## Concrete Models vs. Model Families
///
/// The entropy models in this module can be instantiated in two different ways:
///
/// - (a) As *concrete* models that are fully parameterized models. Simply provide all model
///   parameters to the constructor of the model (e.g., the mean and standard deviation of a
///   [`QuantizedGaussian`](#constriction.stream.model.QuantizedGaussian), or the domain of a
///   [`Uniform`](#constriction.stream.model.Uniform) model. You can then use the resulting
///   model to either encode or decode single symbols, or to efficiently encode or decode a whole
///   array of *i.i.d.* symbols (i.e., using the same model for each symbol in the array).
/// - (b) As model *families*, i.e., models that still have some free parameters (again, like the
///   mean and standard deviation, or the range of a uniform distribution). Simply leave out any
///   optional model parameters when calling the model constructor. When you then use the resulting
///   model family to encode or decode an array of symbols, then you can provide arrays of model
///   parameters to the encode and decode methods of the employed entropy coder. This will allow you
///   to use individual model parameters for each symbol in a sequence (and it is more efficient
///   constructing a new concrete model for each symbol)
///
/// ## Examples
///
/// Constructing and using a *concrete* [`QuantizedGaussian`](#constriction.stream.model.QuantizedGaussian)
/// model with mean 12.6 and standard deviation 7.3, and which is quantized to integers on the domain
/// {-100, ..., 100}:
///
/// ```python
/// model = constriction.stream.model.QuantizedGaussian(-100, 100, 12.6, 7.3)
///
/// # Encode and decode an example message:
/// symbols = np.array([12, 15, 4, -2, 18, 5], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # (`RangeEncoder` also works)
/// coder.encode_reverse(symbols, model)
/// print(coder.get_compressed()) # (prints: [745994372, 25704])
///
/// reconstructed = coder.decode(model, 6) # (decodes 6 i.i.d. symbols)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
/// We can generalize the above example and use model-specific means and standard deviations by 
/// constructing and using a model *family* instead of a concrete model, and by providing arrays:
/// of model parameters to the encode and decode methods:
///
/// ```python
/// model_family = constriction.stream.model.QuantizedGaussian(-100, 100)
/// # Note: we omitted the mean and standard deviation, but the quantization range
/// #       {-100, ..., 100} must always be specified when constructing the model.
///
/// # Define arrays of model parameters (means and standard deviations):
/// symbols = np.array([12,   15,   4,   -2,   18,   5  ], dtype=np.int32)
/// means   = np.array([13.2, 17.9, 7.3, -4.2, 25.1, 3.2], dtype=np.float64)
/// stds    = np.array([ 3.2,  4.7, 5.2,  3.1,  6.3, 2.9], dtype=np.float64)
///
/// # Encode and decode an example message:
/// coder = constriction.stream.stack.AnsCoder() # (`RangeEncoder` also works)
/// coder.encode_reverse(symbols, model_family, means, stds)
/// print(coder.get_compressed()) # (prints: [2051958011, 1549])
///
/// reconstructed = coder.decode(model_family, means, stds)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
///
#[pymodule]
fn model(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    model::init_module(py, module)
}

/// Range Coding: a stream code with queue semantics (i.e., "first in first out") [1, 2].
///
/// The Range Coding algorithm is a variation on Arithmetic Coding [1, 3] that runs more efficiently
/// on standard computing hardware.
///
/// ## Example
///
/// The following example shows a full round trip that encodes some message, prints the compressed
/// representation, and then decodes the message again. The message is a sequence of 11 integers
/// (symbols) and comprised of two parts: the first 7 symbols are encoded with an i.i.d. entropy
/// model, i.e., using the same categorical distribution for each symbol; and the remaining 4
/// symbols are each encoded with a different entropy model, but all of these 4 models are from the
/// same family of [`QuantizedGaussian`](model.html#constriction.stream.model.QuantizedGaussian)s,
/// just with different model parameters for each of the 4 symbols.
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Define the two parts of the message and their respective entropy models.
/// message_part1       = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
/// probabilities_part1 = np.array([0.2, 0.4, 0.1, 0.2], dtype=np.float64)
/// model_part1       = constriction.stream.model.Categorical(probabilities_part1)
///
/// message_part2       = np.array([6,   10,   -4,    2  ], dtype=np.int32)
/// means_part2         = np.array([2.5, 13.1, -1.1, -3.0], dtype=np.float64)
/// stds_part2          = np.array([4.1,  8.7,  6.2,  5.4], dtype=np.float64)
/// model_family_part2  = constriction.stream.model.QuantizedGaussian(-100, 100)
///
/// print(f"Original message: {np.concatenate([message_part1, message_part2])}")
///
/// # Encode both parts of the message in sequence:
/// encoder = constriction.stream.queue.RangeEncoder()
/// encoder.encode(message_part1, model_part1)
/// encoder.encode(message_part2, model_family_part2, means_part2, stds_part2)
///
/// # Get and print the compressed representation:
/// compressed = encoder.get_compressed()
/// print(f"compressed representation: {compressed}")
/// print(f"(in binary: {[bin(word) for word in compressed]})")
///
/// # Decode the message:
/// decoder = constriction.stream.queue.RangeDecoder(compressed)
/// decoded_part1 = decoder.decode(model_part1, 7) # (decodes 7 symbols)
/// decoded_part2 = decoder.decode(model_family_part2, means_part2, stds_part2)
/// print(f"Decoded message: {np.concatenate([decoded_part1, decoded_part2])}")
/// assert np.all(decoded_part1 == message_part1)
/// assert np.all(decoded_part2 == message_part2)
/// ```
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
#[pymodule]
fn queue(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    queue::init_module(py, module)
}

/// Asymmetric Numeral Systems (ANS): a stream code with stack semantics
/// (i.e., "last in first out") [1].
///
/// The ANS entropy coding algorithm is a popular choice for bits-back coding with latent variable
/// models. It uses only a single data structure, `AnsCoder`, which operates as both encoder and
/// decoder. This allows you to easily switch back and forth between encoding and decoding
/// operations. Further, `constrictions` ANS implementation (unlike its Range Coding implementation
/// in the sister module `queue`) is *surjective*. This means that you can decode symbols from any
/// bitstring, regardless of its origin, and then re-encode the symbols to exactly reconstruct the
/// original bitstring.
///
/// ## Stack Semantics
///
/// ANS operates as a *stack*: encoding *pushes* (i.e., appends) data onto the top of the stack and
/// decoding *pops* (i.e., consumes) data from the top of the stack. This means that encoding and
/// decoding operate in opposite directions to each other. When using an `AnsCoder`, we recommend to
/// encode sequences of symbols in reverse order so that you can later decode them in their original
/// order. The method `encode_reverse` does this automatically when given an array of symbols. If
/// you call `encode_reverse` several times to encode several slices of a message, then start with
/// the last slice of your message, as in the example below.
///
/// ## Example
///
/// The following example shows a full round trip that encodes some message, prints the compressed
/// representation, and then decodes the message again. The message is a sequence of 11 integers
/// (symbols) and comprised of two parts: the first 7 symbols are encoded with an i.i.d. entropy
/// model, i.e., using the same categorical distribution for each symbol; and the remaining 4
/// symbols are each encoded with a different entropy model, but all of these 4 models are from the
/// same family of [`QuantizedGaussian`](model.html#constriction.stream.model.QuantizedGaussian)s,
/// just with different model parameters for each of the 4 symbols.
///
/// Notice that we encode part 2 before part 1, but when we decode, we first obtain part 1 and then
/// part 2.
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Define the two parts of the message and their respective entropy models.
/// message_part1       = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
/// probabilities_part1 = np.array([0.2, 0.4, 0.1, 0.2], dtype=np.float64)
/// model_part1       = constriction.stream.model.Categorical(probabilities_part1)
///
/// message_part2       = np.array([6,   10,   -4,    2  ], dtype=np.int32)
/// means_part2         = np.array([2.5, 13.1, -1.1, -3.0], dtype=np.float64)
/// stds_part2          = np.array([4.1,  8.7,  6.2,  5.4], dtype=np.float64)
/// model_family_part2  = constriction.stream.model.QuantizedGaussian(-100, 100)
///
/// print(f"Original message: {np.concatenate([message_part1, message_part2])}")
///
/// # Encode both parts of the message in sequence (in reverse order):
/// coder = constriction.stream.stack.AnsCoder()
/// coder.encode_reverse(
///     message_part2, model_family_part2, means_part2, stds_part2)
/// coder.encode_reverse(message_part1, model_part1)
///
/// # Get and print the compressed representation:
/// compressed = coder.get_compressed()
/// print(f"compressed representation: {compressed}")
/// print(f"(in binary: {[bin(word) for word in compressed]})")
///
/// # Below, we'll decode the message directly from `coder` for simplicity.
/// # If decoding happens at a later time than encoding, then you can safe
/// # `compressed` to a file (using `compressed.tofile("filename")`), read the
/// # file back later (using `compressed = np.fromfile("filename")) and re-create
/// # the coder using `coder = constriction.stream.stack.AnsCoder(compressed)`.
///
/// # Decode the message:
/// decoded_part1 = coder.decode(model_part1, 7) # (decodes 7 symbols)
/// decoded_part2 = coder.decode(model_family_part2, means_part2, stds_part2)
/// print(f"Decoded message: {np.concatenate([decoded_part1, decoded_part2])}")
/// assert np.all(decoded_part1 == message_part1)
/// assert np.all(decoded_part2 == message_part2)
/// ```
///
/// ## References
///
/// [1] Duda, Jarek, et al. "The use of asymmetric numeral systems as an accurate
/// replacement for Huffman coding." 2015 Picture Coding Symposium (PCS). IEEE, 2015.
#[pymodule]
fn stack(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    stack::init_module(py, module)
}

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
