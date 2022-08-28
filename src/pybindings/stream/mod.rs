mod chain;
mod model;
mod queue;
mod stack;

use pyo3::{prelude::*, wrap_pymodule};

use std::prelude::v1::*;

use crate::{stream::TryCodingError, CoderError, DefaultEncoderFrontendError};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(init_model))?;
    module.add_wrapped(wrap_pymodule!(init_queue))?;
    module.add_wrapped(wrap_pymodule!(init_stack))?;
    module.add_wrapped(wrap_pymodule!(init_chain))?;
    Ok(())
}

/// Entropy models and model families for use with any of the stream codes from the sister
/// modules [`stack`](stack.html), [`queue`](queue.html), and [`chain`](chain.html).
///
/// This module provides tools to define probability distributions over symbols in fixed
/// point arithmetic, so that the models (more precisely, their cumulative distributions
/// functions) are *exactly* invertible without any rounding errors. Being exactly
/// invertible is crucial for for data compression since even tiny rounding errors can have
/// catastrophic consequences in an entropy coder (this issue is discussed in the
/// [motivating example of the `ChainCoder`](chain.html#motivation)). Further, the entropy
/// models in this module all have a well-defined domain, and they always assign a nonzero
/// probability to all symbols within this domain, even if the symbol is in the tail of some
/// distribution where its true probability would be lower than the smallest value that is
/// representable in the employed fixed point arithmetic. This ensures that symbols from the
/// well-defined domain of a model can, in principle, always be encoded without throwing an
/// error (symbols with the smallest representable probability will, however, have a very
/// high bitrate of 24 bits).
///
/// ## Concrete Models vs. Model Families
///
/// The entropy models in this module can be instantiated in two different ways:
///
/// - (a) as *concrete* models that are fully parameterized; simply provide all model
///   parameters to the constructor of the model (e.g., the mean and standard deviation of a
///   [`QuantizedGaussian`](#constriction.stream.model.QuantizedGaussian), or the domain of a
///   [`Uniform`](#constriction.stream.model.Uniform) model). You can use a concrete model
///   to either encode or decode single symbols, or to efficiently encode or decode a whole
///   array of *i.i.d.* symbols (i.e., using the same model for each symbol in the array,
///   see first example below).
/// - (b) as model *families*, i.e., models that still have some free parameters (again,
///   like the mean and standard deviation of a `QuantizedGaussian`, or the range of a
///   `Uniform` distribution); simply leave out any optional model parameters when calling
///   the model constructor. When you then use the resulting model family to encode or
///   decode an array of symbols, you can provide *arrays* of model parameters to the encode
///   and decode methods of the employed entropy coder. This will allow you to use
///   individual model parameters for each symbol, see second example below (this is more
///   efficient than constructing a new concrete model for each symbol and looping over the
///   symbols in Python).
///
/// ## Examples
///
/// Constructing and using a *concrete* [`QuantizedGaussian`](#constriction.stream.model.QuantizedGaussian)
/// model with mean 12.6 and standard deviation 7.3, and which is quantized to integers on the domain
/// {-100, -99, ..., 100}:
///
/// ```python
/// model = constriction.stream.model.QuantizedGaussian(-100, 100, 12.6, 7.3)
///
/// # Encode and decode an example message:
/// symbols = np.array([12, 15, 4, -2, 18, 5], dtype=np.int32)
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model)
/// print(coder.get_compressed()) # (prints: [745994372, 25704])
///
/// reconstructed = coder.decode(model, 6) # (decodes 6 i.i.d. symbols)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
/// We can generalize the above example and use model-specific means and standard deviations by
/// constructing and using a model *family* instead of a concrete model, and by providing arrays
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
/// coder = constriction.stream.stack.AnsCoder() # (RangeEncoder also works)
/// coder.encode_reverse(symbols, model_family, means, stds)
/// print(coder.get_compressed()) # (prints: [2051958011, 1549])
///
/// reconstructed = coder.decode(model_family, means, stds)
/// assert np.all(reconstructed == symbols) # (verify correctness)
/// ```
///
///
#[pymodule]
#[pyo3(name = "model")]
fn init_model(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    model::init_module(py, module)
}

/// Range Coding: a stream code with queue semantics (i.e., "first in first out") [1, 2].
///
/// The Range Coding algorithm is a variation on Arithmetic Coding [1, 3] that runs more efficiently
/// on standard computing hardware.
///
/// ## Example
///
/// The following example shows a full round trip that encodes a message, prints its compressed
/// representation, and then decodes the message again. The message is a sequence of 11 integers
/// (referred to as "symbols") and comprised of two parts: the first 7 symbols are encoded with an
/// i.i.d. entropy model, i.e., using the same distribution for each symbol, which happens to be a
/// [`Categorical`](model.html#constriction.stream.model.Categorical) distribution; and the remaining
/// 4 symbols are each encoded with a different entropy model, but all of these 4 models are from
/// the same family of [`QuantizedGaussian`](model.html#constriction.stream.model.QuantizedGaussian)s,
/// just with different model parameters (means and standard deviations) for each of the 4 symbols.
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Define the two parts of the message and their respective entropy models:
/// message_part1       = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
/// probabilities_part1 = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float64)
/// model_part1       = constriction.stream.model.Categorical(probabilities_part1)
/// # `model_part1` is a categorical distribution over the (implied) alphabet
/// # {0,1,2,3} with P(X=0) = 0.2, P(X=1) = 0.4, P(X=2) = 0.1, and P(X=3) = 0.3;
/// # we will use it below to encode each of the 7 symbols in `message_part1`.
///
/// message_part2       = np.array([6,   10,   -4,    2  ], dtype=np.int32)
/// means_part2         = np.array([2.5, 13.1, -1.1, -3.0], dtype=np.float64)
/// stds_part2          = np.array([4.1,  8.7,  6.2,  5.4], dtype=np.float64)
/// model_family_part2  = constriction.stream.model.QuantizedGaussian(-100, 100)
/// # `model_part2` is a *family* of Gaussian distributions that are quantized
/// # to bins of with 1 centered at the integers -100, -99, ..., 100. We could
/// # have provided a fixed mean and standard deviation to the constructor of
/// # `QuantizedGaussian` but we'll instead provide individual means and standard
/// # deviations for each symbol when we encode and decode `message_part2` below.
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
/// # You could save `compressed` to a file using `compressed.tofile("filename")`
/// # and read it back in: `compressed = np.fromfile("filename", dtype=np.uint32).
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
#[pyo3(name = "queue")]
fn init_queue(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    queue::init_module(py, module)
}

/// Asymmetric Numeral Systems (ANS): a stream code with stack semantics
/// (i.e., "last in first out") [1].
///
/// The ANS entropy coding algorithm is a popular choice for bits-back coding with latent variable
/// models. It uses only a single data structure, `AnsCoder`, which operates as both encoder and
/// decoder. This allows you to easily switch back and forth between encoding and decoding
/// operations. A further, more subtle property that distinguishes `constrictions` ANS
/// implementation from its Range Coding implementation in the sister module `queue`) is that
/// encoding with an `AnsCoder` is *surjective* and therefore decoding is injective. This means that
/// you can decode some symbols from any bitstring, regardless of its origin, and then re-encode the
/// symbols to exactly reconstruct the original bitstring (e.g., for bits-back coding).
///
/// ## Stack Semantics
///
/// ANS operates as a *stack*: encoding *pushes* (i.e., appends) data onto the top of the stack and
/// decoding *pops*  data from the top of the stack (i.e., it consumes data from the *same* end).
/// This means that encoding and  decoding operate in opposite directions to each other. When using
/// an `AnsCoder`, we recommend to encode sequences of symbols in reverse order so that you can
/// later decode them in their original order. The method `encode_reverse` does this automatically
/// when given an array of symbols. If you call `encode_reverse` several times to encode several
/// parts of a message, then start with the last part of your message and work your way back, as in
/// the example below.
///
/// ## Example
///
/// The following example shows a full round trip that encodes a message, prints its compressed
/// representation, and then decodes the message again. The message is a sequence of 11 integers
/// (referred to as "symbols") and comprised of two parts: the first 7 symbols are encoded with an
/// i.i.d. entropy model, i.e., using the same distribution for each symbol, which happens to be a
/// [`Categorical`](model.html#constriction.stream.model.Categorical) distribution; and the remaining
/// 4 symbols are each encoded with a different entropy model, but all of these 4 models are from
/// the same family of [`QuantizedGaussian`](model.html#constriction.stream.model.QuantizedGaussian)s,
/// just with different model parameters (means and standard deviations) for each of the 4 symbols.
///
/// Notice that we encode part 2 before part 1, but when we decode, we first obtain part 1 and then
/// part 2. This is because of the `AnsCoder`'s [stack semantics](#stack-semantics).
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Define the two parts of the message and their respective entropy models:
/// message_part1       = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
/// probabilities_part1 = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float64)
/// model_part1       = constriction.stream.model.Categorical(probabilities_part1)
/// # `model_part1` is a categorical distribution over the (implied) alphabet
/// # {0,1,2,3} with P(X=0) = 0.2, P(X=1) = 0.4, P(X=2) = 0.1, and P(X=3) = 0.3;
/// # we will use it below to encode each of the 7 symbols in `message_part1`.
///
/// message_part2       = np.array([6,   10,   -4,    2  ], dtype=np.int32)
/// means_part2         = np.array([2.5, 13.1, -1.1, -3.0], dtype=np.float64)
/// stds_part2          = np.array([4.1,  8.7,  6.2,  5.4], dtype=np.float64)
/// model_family_part2  = constriction.stream.model.QuantizedGaussian(-100, 100)
/// # `model_part2` is a *family* of Gaussian distributions that are quantized
/// # to bins of with 1 centered at the integers -100, -99, ..., 100. We could
/// # have provided a fixed mean and standard deviation to the constructor of
/// # `QuantizedGaussian` but we'll instead provide individual means and standard
/// # deviations for each symbol when we encode and decode `message_part2` below.
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
/// # You could save `compressed` to a file using `compressed.tofile("filename")`,
/// # read it back in: `compressed = np.fromfile("filename", dtype=np.uint32) and
/// # then re-create `coder = constriction.stream.stack.AnsCoder(compressed)`.
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
#[pyo3(name = "stack")]
fn init_stack(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    stack::init_module(py, module)
}

/// Experimental entropy coding algorithm for advanced variants of bits-back coding.
///
/// This module provides the `ChainCoder`, an experimental entropy coder that is similar
/// to an [`AnsCoder`](stack.html#constriction.stream.stack.AnsCoder) in that it operates as
/// a stack (i.e., a last-in-first-out data structure). However, different to an `AnsCoder`,
/// a `ChainCoder` treats each symbol independently. Thus, when decoding some bit string
/// into a sequence of symbols, any modification to the entropy model for one symbol does
/// not affect decoding for any other symbol (by contrast, when decoding with an `AnsCoder`
/// or `RangeDecoder`, a change to the entropy model for one symbol can affect *all*
/// subsequently decoded symbols too, see [Motivation](#motivation) below).
///
/// This property of treating symbols independently upon decoding can be useful for advanced
/// compression methods that combine inference, quantization, and bits-back coding.
///
/// ## Motivation
///
/// The following two examples illustrate how decoding differs between an `AnsCoder` and a
/// `ChainCoder`. In the first example, we decode from the same bitstring `data` twice with
/// an `AnsCoder`: a first time with an initial sequence of toy entropy models, and then a
/// second time with a slightly different sequence of entropy models. Importantly, we change
/// only the entropy model for the first decoded symbol (and the change is small). The
/// entropy models for all other symbols remain exactly unmodified. We observe, however,
/// that changing the first entropy model affects not only the first decoded symbol. It also
/// has a ripple effect on subsequently decoded symbols.
///
/// ```python
/// # Some sample binary data and sample probabilities for our entropy models
/// data = np.array([0x80d14131, 0xdda97c6c, 0x5017a640, 0x01170a3d], np.uint32)
/// probabilities = np.array(
///     [[0.1, 0.7, 0.1, 0.1],  # (<-- probabilities for first decoded symbol)
///      [0.2, 0.2, 0.1, 0.5],  # (<-- probabilities for second decoded symbol)
///      [0.2, 0.1, 0.4, 0.3]]) # (<-- probabilities for third decoded symbol)
/// model_family = constriction.stream.model.Categorical()
///
/// # Decoding `data` with an `AnsCoder` results in the symbols `[0, 0, 1]`:
/// ansCoder = constriction.stream.stack.AnsCoder(data, seal=True)
/// print(ansCoder.decode(model_family, probabilities)) # (prints: [0, 0, 1])
///
/// # Even if we change only the first entropy model (slightly), *all* decoded
/// # symbols can change:
/// probabilities[0, :] = np.array([0.09, 0.71, 0.1, 0.1])
/// ansCoder = constriction.stream.stack.AnsCoder(data, seal=True)
/// print(ansCoder.decode(model_family, probabilities)) # (prints: [1, 0, 3])
/// ```
///
/// It's no surprise that the first symbol changed since we changed its entropy model. But
/// note that the third symbol changed too even though we hadn't changed its entropy model.
/// Thus, in ANS (as in most codes), changes to entropy models have a  *global* effect.
///
/// Now let's try the same with a `ChainCoder`:
///
/// ```python
/// # Same compressed data and original entropy models as in our first example
/// data = np.array([0x80d14131, 0xdda97c6c, 0x5017a640, 0x01170a3d], np.uint32)
/// probabilities = np.array(
///     [[0.1, 0.7, 0.1, 0.1],
///      [0.2, 0.2, 0.1, 0.5],
///      [0.2, 0.1, 0.4, 0.3]])
/// model_family = constriction.stream.model.Categorical()
///
/// # Decode with the original entropy models, this time using a `ChainCoder`:
/// chainCoder = constriction.stream.chain.ChainCoder(data, seal=True)
/// print(chainCoder.decode(model_family, probabilities)) # (prints: [0, 3, 3])
///
/// # We obtain different symbols than for the `AnsCoder`, of course, but that's
/// # not the point here. Now let's change the first model again:
/// probabilities[0, :] = np.array([0.09, 0.71, 0.1, 0.1])
/// chainCoder = constriction.stream.chain.ChainCoder(data, seal=True)
/// print(chainCoder.decode(model_family, probabilities)) # (prints: [1, 3, 3])
/// ```
///
/// Notice that the only symbol that changed was the one whose entropy model we had changed.
/// Thus, in a `ChainCoder`, changes to entropy models (and also to compressed bits) only
/// have a *local* effect on the decompressed symbols.
///
/// ## How does this work?
///
/// A `ChainCoder` is a variant on ANS. To understand how a `ChainCoder` works, one first
/// has to understand how an `AnsCoder` works. When an `AnsCoder` decodes a symbol, it
/// performs the following three steps:
///
/// 1. The `AnsCoder` consumes a fixed amount of bits (24 bits in `constriction`'s default
///    configuration) from the end of its encapsulated compressed data; then
/// 2. the `AnsCoder` interprets these 24 bits as a number `xi` between 0.0 and 1.0 in fixed
///    point representation, and it maps `xi` to a symbol using the entropy model's quantile
///    function (aka percent point function); finally
/// 3. the `AnsCoder` uses the bits-back trick to write (24 - inf_content) bits worth of
///    (amortized) bits back onto the encapsulated compressed data; these "returned" bits
///    reflect the information contained in the precise position of the number `xi` within
///    the whole range of numbers that the quantile function would map to the same symbol.
///
/// A `ChainCoder` breaks these three steps apart. It encapsulates *two* rather than just
/// one bitstring buffers, referred to in the following as "compressed" and "remainders".
/// When you call the constructor of `ChainCoder` with argument `is_remainders=False` (which
/// is the default) then the provided argument `data` is used to initialize the "compressed"
/// buffer. When the `ChainCoder` decodes a symbol, it executes steps 1-3 from above, with
/// the modification that step 1 reads from the "compressed" buffer while step 3 writes to
/// the "remainders" buffer. Since step 1 is independent of the employed entropy model,
/// changes to the entropy model have no effect on subsequently decoded symbols.
///
/// Once you're done decoding a sequence of symbols, you can concatenate the two buffers to
/// a single one.
///
/// ## Usage for Bits-Back Coding
///
/// A typical usage cycle of a `ChainCoder` goes along the following steps:
///
/// ### When compressing data using the bits-back trick
///
/// 1. Start with some array of (typically already compressed) binary data, which you want
///    to piggy-back into the choice of latent variables in a latent variable entropy model.
/// 2. Create a `ChainCoder`, passing into the constructor the binary data and the arguments
///    `is_remainders=False` and `seal=True` (you can set `seal=False` if you know that the
///    data cannot end in a zero word, e.g., if it comes from
///    `AnsCoder.get_compressed(unseal=False)`).
/// 3. Decode the latent variables from the `ChainCoder` as usual in bits-back coding.
/// 4. Export the remaining data on the `ChainCoder` by calling its method
///    `.get_remainders()`. The method returns two numpy arrays, which you may concatenate in
///    order.
/// 5. You can now use this remaining binary data in an `AnsCoder` (it is guaranteed not to
///    have a zero word at the end, so you can set `seal=False` in the constructor of
///    `AnsCoder`), and you can encode a message on top of it using entropy models that are
///    conditioned on the latent variables you just decoded.
///
/// ### When decompressing the data
///
/// 1. Recreate the binary data that you had after encoder step 4 above, and the latent
///    variables that you decoded in encoder step 3 above, as usual in bits-back coding.
/// 2. Pass it to the constructor of `ChainCoder`, with additional arguments
///    `is_remainders=True` (`seal` always has to be `False` when `is_remainders=True`
///    because remainders data is guaranteed not to end in a zero word).
/// 3. Encode the latent variables back onto the new `ChainCoder` (in reverse order), using
///    the same entropy models as during decoding.
/// 4. Recover the original binary data from encoder step 1 above by calling
///    `.get_data(unseal=True)` (if you set `seal=False` in encoder step 2 above, then set
///    `unseal=False` now). The method returns two arrays, which you may concatenate in
///    order.
#[pymodule]
#[pyo3(name = "chain")]
fn init_chain(py: Python<'_>, module: &PyModule) -> PyResult<()> {
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
