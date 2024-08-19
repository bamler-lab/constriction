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
/// # `model_family_part2` is a *family* of Gaussian distributions, quantized to
/// # bins of width 1 centered at the integers -100, -99, ..., 100. We could
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
/// # `model_family_part2` is a *family* of Gaussian distributions, quantized to
/// # bins of width 1 centered at the integers -100, -99, ..., 100. We could
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
/// Treating symbols independently upon encoding and decoding can be useful for advanced
/// compression methods that combine inference, quantization, and bits-back coding. In
/// treating symbols independently, a `ChainCoder` bears some resemblance with symbol codes.
/// However, in contrast to symbol codes, a `ChainCoder` still amortizes compressed bits
/// over symbols and therefore has better compression effectiveness than a symbol code; in a
/// sense, it just delays amortization to the very end of its life cycle (see "[How Does
/// `ChainCoder` Work?](#how-does-chaincoder-work)" below).
///
/// ## Usage Example
///
/// A `ChainCoder` is meant to be used as a building block for advanced applications of the
/// bits-back trick [1, 2]. One therefore typically starts on the encoder side by *decoding*
/// some "side information" into a sequence of symbols. On the decoder side, one then
/// recovers the side information by *(re-)encoding* these symbols. It is important to be
/// aware that both the initialization of a `ChainCoder` and the way how we obtain its
/// encapsulated data differ from the encoder and decoder side, as illustrated in the
/// following example of a full round trip:
///
/// ```python
/// # Parameters for a few example Gaussian entropy models:
/// leaky_gaussian = constriction.stream.model.QuantizedGaussian(-100, 100)
/// means = np.array([3.2, -14.3, 5.7])
/// stds = np.array([6.4, 4.2, 3.9])
///
/// def run_encoder_part(side_information):
///     # Construct a `ChainCoder` for *decoding*:
///     coder = constriction.stream.chain.ChainCoder(
///         side_information,    # Provided bit string.
///         is_remainders=False, # Bit string is *not* remaining data after decoding.
///         seal=True            # Bit string comes from an external source here.
///     )
///     # Decode side information into a sequence of symbols as usual in bits-back coding:
///     symbols = coder.decode(leaky_gaussian, means, stds)
///     # Obtain what's *remaining* on the coder after decoding the symbols:
///     remaining1, remaining2 = coder.get_remainders()
///     return symbols, np.concatenate([remaining1, remaining2])
///
/// def run_decoder_part(symbols, remaining):
///     # Construct a `ChainCoder` for *encoding*:
///     coder = constriction.stream.chain.ChainCoder(
///         remaining,           # Provided bit string.
///         is_remainders=True,  # Bit string *is* remaining data after decoding.
///         seal=False           # Bit string comes from a `ChainCoder`, no need to seal it.
///     )
///     # Re-encode the symbols to recover the side information:
///     coder.encode_reverse(symbols, leaky_gaussian, means, stds)
///     # Obtain the reconstructed data
///     data1, data2 = coder.get_data(unseal=True)
///     return np.concatenate([data1, data2])
///
/// np.random.seed(123)
/// sample_side_information = np.random.randint(2**32, size=10, dtype=np.uint32)
/// symbols, remaining = run_encoder_part(sample_side_information)
/// recovered = run_decoder_part(symbols, remaining)
/// assert np.all(recovered == sample_side_information)
/// ```
///
/// Notice that:
///
/// - we construct the `ChainCoder` with argument `is_remainders=False` on the encoder side
///   (which *decodes* symbols from the side information), and with argument
///   `is_remainders=True` on the decoder side (which *re-encodes* the symbols to recover the
///   side information); and
/// - we export the remaining bit string on the `ChainCoder` after decoding some symbols
///   from it by calling `get_remainders`, and we export the recovered side information after
///   re-encoding the symbols by calling `get_data`. Both methods return a pair of bit
///   strings (more precisely, `uint32` arrays) where only the second item of the pair is
///   strictly required to invert the process we just performed, but we may concatenate the
///   two bit strings without loss of information.
///
/// To understand why this asymmetry between encoder and decoder side is necessary, see
/// section "[How Does `ChainCoder` Work?](#how-does-chaincoder-work)" below.
///
/// [A side remark concerning the `seal` and `unseal` arguments: when constructing a
/// `ChainCoder` from some bit string that we obtained from either
/// `ChainCoder.get_remaining()` or from `AnsCoder.get_compressed()` then we may set
/// `seal=False` in the constructor since these methods guarantee that the last word of the
/// bit string is nonzero. By contrast, when we construct a `ChainCoder` from some bit
/// string that is outside of our control (and that may therefore end in a zero word), then
/// we have to set `seal=True` in the constructor (and we then have to set `unseal=True`
/// when we want to get that bit string back via `get_data`).]
///
/// ## Motivation
///
/// The following two examples illustrate how the `ChainCoder` provided in this module
/// differs from an `AnsCoder` from the sister module `constriction.stream.stack`.
///
/// ### The Problem With `AnsCoder`
///
/// We start with an `AnsCoder`, and we decode a fixed bitstring `data` two times, using
/// slightly different sequences of entropy models each time. As expected, decoding the same
/// data with different entropy models leads to different sequences of decoded symbols.
/// Somewhat surprisingly, however, changes to entropy models can have a ripple effect: in
/// the example below, the line marked with "<-- change first entropy model" changes *only*
/// the entropy model for the first symbol. Nevertheless, this change of a single entropy
/// model causes the coder to decode different symbols in more than just that one place.
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
/// In the above example, it's no surprise that changing the first entropy model made the
/// first decoded symbol change (from "0" to "1"). But notice that the third symbol also
/// changes (from "1" to "3") even though we didn't change its entropy model. This ripple
/// effect is a result of the fact that the internal state of `ansCoder` after decoding the
/// first symbol depends on `model1`. This is usually what we'd want from a good entropy
/// coder that packs as much information into as few bits as possible. However, it can
/// become a problem in certain advanced compression methods that combine bits-back coding
/// with quantization and inference. For those applications, a `ChainCoder`, illustrated in
/// the next example, may be more suitable.
///
/// ### The Solution With a `ChainCoder`
///
/// In the following example, we replace the `AnsCoder` with a `ChainCoder`. This means,
/// firstly and somewhat trivially, that we will decode the same bit string `data` into a
/// different sequence of symbols than in the last example because `ChainCoder` uses a
/// different entropy coding algorithm than `AnsCoder`. The more profound difference to the
/// example with the `AnsCoder` above is, however, that changes to a single entropy model no
/// longer have a ripple effect: when we now change the entropy model for one of the
/// symbols, this change affects only the corresponding symbol. All subsequently decoded
/// symbols remain unchanged.
///
/// ```python
/// # Same compressed data and original entropy models as in our first example
/// data = np.array([0x80d14131, 0xdda97c6c, 0x5017a640, 0x01170a3d], np.uint32)
/// probabilities = np.array(
///     [[0.1, 0.7, 0.1, 0.1],  # (<-- probabilities for first decoded symbol)
///      [0.2, 0.2, 0.1, 0.5],  # (<-- probabilities for second decoded symbol)
///      [0.2, 0.1, 0.4, 0.3]]) # (<-- probabilities for third decoded symbol)
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
/// Notice that the only symbol that changes is the one whose entropy model we changed.
/// Thus, in a `ChainCoder`, changes to entropy models (and also to compressed bits) only
/// have a *local* effect on the decompressed symbols.
///
/// ## How Does `ChainCoder` Work?
///
/// The class `ChainCoder` uses an experimental new entropy coding algorithm that is
/// inspired by but different to Asymmetric Numeral Systems (ANS) [3] as used by the
/// `AnsCoder` in the sister module `constriction.stream.stack`. This experimental new
/// entropy coding algorithm was proposed in Ref. [4].
///
/// **In ANS**, decoding a single symbol can conceptually be divided into three steps:
///
/// 1. We chop off a *fixed integer* number of bits (the `AnsCoder` uses 24 bits by default)
///    from the end of the compressed bit string.
/// 2. We interpret the 24 bits obtained from Step 1 above as the fixed-point binary
///    representation of a fractional number between zero and one and we map this number to
///    a symbol via the quantile function of the entropy model (the quantile function is the
///    inverse of the cumulative distribution function; it is sometimes also called the
///    percent-point function).
/// 3. We put back some (typically non-integer amount of) information content to the
///    compressed bit string by using the bits-back trick [1, 2]. These "returned" bits
///    account for the difference between the 24 bits that we consumed in Step 1 above and
///    the true information content of the symbol that we decoded in Step 2 above.
///
/// The ripple effect from a single changed entropy model that we observed in the [ANS
/// example above](#the-problem-with-anscoder) comes from Step 3 of the ANS algorithm since
/// the information content that we put back (and therefore also the internal state of the
/// coder after putting back the information) depends on the employed entropy model. By
/// contrast, Step 1 is independent of the entropy model and Step 2 does not mutate the
/// coder's internal state. To avoid a ripple effect when changing entropy models, a
/// `ChainCoder` therefore effectively postpones Step 3 to a later point.
///
/// In detail, a `ChainCoder` keeps track of not just one but two bit strings, which we
/// call `compressed` and `remaining`. When we use a `ChainCoder` to *decode* a symbol then
/// we chop off 24 bits from `compressed` (analogous to Step 1 of the above ANS algorithm)
/// and we identify the decoded symbol (Step 2); different to Step 3 of the ANS algorithm,
/// however, we then put back the variable-length superfluous  information to the *separate*
/// bit string `remaining`. Thus, if we were to change the entropy model, this change can
/// only affect the decoded symbol and the contents of `remaining` after decoding the
/// symbol, but it will not affect the contents of `compressed` after decoding the symbol.
/// Thus, any subsequent symbols that we decode from the bit string `compressed` remain
/// unaffected.
///
/// If we want to use a `ChainCoder` to *encode* data then we have to initialize `remaining`
/// with some sufficiently long bit string (by setting `is_remaining=True` in the
/// constructor, see [usage example](#usage-example). The methods `get_remaining` and
/// `get_data` return both bit strings `compressed` and `remaining` in the appropriate order
/// and with an appropriate finishing that allows the caller to concatenate them without a
/// delimiter (see again [usage example](#usage-example)).
///
/// ## Etymology
///
/// The name `ChainCoder` refers to the fact that the coder consumes and generates
/// compressed bits in fixed-sized quanta (24 bits per symbol by default), reminiscent of
/// how a chain (as opposed to, say, a rope) can only be shortened or extended by
/// fixed-sized quanta (the chain links).
///
/// ## References
///
/// - [1] Townsend, James, Tom Bird, and David Barber. "Practical lossless compression with
///   latent variables using bits back coding." arXiv preprint arXiv:1901.04866 (2019).
/// - [2] Duda, Jarek, et al. "The use of asymmetric numeral systems as an accurate
///   replacement for Huffman coding." 2015 Picture Coding Symposium (PCS). IEEE, 2015.
/// - [3] Wallace, Chris S. "Classification by minimum-message-length inference."
///   International Conference on Computing and Information. Springer, Berlin, Heidelberg,
///   1990.
/// - [4] Bamler, Robert. "Understanding Entropy Coding With Asymmetric Numeral Systems
///   (ANS): a Statistician's Perspective." arXiv preprint arXiv:2201.01741 (2022).
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
