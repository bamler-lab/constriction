use std::prelude::v1::*;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, types::PyTuple};

use crate::{
    pybindings::array1_to_vec,
    stream::{Decode, Encode},
    Pos, Seek, UnwrapInfallible,
};

use super::model::{internals::EncoderDecoderModel, Model};

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
/// probabilities_part1 = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
/// model_part1       = constriction.stream.model.Categorical(probabilities_part1, perfect=False)
/// # `model_part1` is a categorical distribution over the (implied) alphabet
/// # {0,1,2,3} with P(X=0) = 0.2, P(X=1) = 0.4, P(X=2) = 0.1, and P(X=3) = 0.3;
/// # we will use it below to encode each of the 7 symbols in `message_part1`.
///
/// message_part2       = np.array([6,   10,   -4,    2  ], dtype=np.int32)
/// means_part2         = np.array([2.5, 13.1, -1.1, -3.0], dtype=np.float32)
/// stds_part2          = np.array([4.1,  8.7,  6.2,  5.4], dtype=np.float32)
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
pub fn init_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<AnsCoder>()?;
    Ok(())
}

/// An entropy coder based on [Asymmetric Numeral Systems (ANS)] [1].
///
/// This is a wrapper around the Rust type [`constriction::stream::stack::DefaultAnsCoder`]
/// with python bindings.
///
/// Note that this entropy coder is a stack (a "last in first out" data
/// structure). You can push symbols on the stack using the method`encode_reverse`,
/// and then pop them off *in reverse order* using the method `decode`.
///
/// To copy out the compressed data that is currently on the stack, call
/// `get_compressed`. You would typically want write this to a binary file in some
/// well-documented byte order. After reading it back in at a later time, you can
/// decompress it by constructing an `AnsCoder` where you pass in the compressed
/// data as an argument to the constructor.
///
/// If you're only interested in the compressed file size, calling `num_bits` will
/// be cheaper as it won't actually copy out the compressed data.
///
/// ## Examples
///
/// ### Compression:
///
/// ```python
/// import sys
/// import constriction
/// import numpy as np
///
/// ans = constriction.stream.stack.AnsCoder()  # No args => empty ANS coder
///
/// symbols = np.array([2, -1, 0, 2, 3], dtype=np.int32)
/// min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
/// model = constriction.stream.model.QuantizedGaussian(
///     min_supported_symbol, max_supported_symbol)
/// means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype=np.float32)
/// stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype=np.float32)
///
/// ans.encode_reverse(symbols, model, means, stds)
///
/// print(f"Compressed size: {ans.num_valid_bits()} bits")
///
/// compressed = ans.get_compressed()
/// if sys.byteorder == "big":
///     # Convert native byte order to a consistent one (here: little endian).
///     compressed.byteswap(inplace=True)
/// compressed.tofile("compressed.bin")
/// ```
///
/// ### Decompression:
///
/// ```python
/// import sys
/// import constriction
/// import numpy as np
///
/// compressed = np.fromfile("compressed.bin", dtype=np.uint32)
/// if sys.byteorder == "big":
///     # Convert little endian byte order to native byte order.
///     compressed.byteswap(inplace=True)
///
/// ans = constriction.stream.stack.AnsCoder( compressed )
/// min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
/// model = constriction.stream.model.QuantizedGaussian(
///     min_supported_symbol, max_supported_symbol)
/// means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype=np.float32)
/// stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype=np.float32)
///
/// reconstructed = ans.decode(model, means, stds)
/// assert ans.is_empty()
/// print(reconstructed)  # Should print [2, -1, 0, 2, 3]
/// ```
///
/// ## Constructor
///
/// AnsCoder(compressed)
///
/// Arguments:
/// compressed (optional) -- initial compressed data, as a numpy array with
///     dtype `uint32`.
///
/// [Asymmetric Numeral Systems (ANS)]: https://en.wikipedia.org/wiki/Asymmetric_numeral_systems
/// [`constriction::stream::ans::DefaultAnsCoder`]: crate::stream::stack::DefaultAnsCoder
///
/// ## References
///
/// [1] Duda, Jarek, et al. "The use of asymmetric numeral systems as an accurate
/// replacement for Huffman coding." 2015 Picture Coding Symposium (PCS). IEEE, 2015.
#[pyclass]
#[derive(Debug, Clone)]
pub struct AnsCoder {
    inner: crate::stream::stack::DefaultAnsCoder,
}

#[pymethods]
impl AnsCoder {
    /// The constructor has the call signature `AnsCoder([compressed, [seal=False]])`.
    ///
    /// - If you want to encode a message, call the constructor with no arguments.
    /// - If you want to decode a message that was previously encoded with an `AnsCoder`, call the
    ///   constructor with a single argument `compressed`, which must be a rank-1 numpy array with
    ///   `dtype=np.uint32` (as returned by the method
    ///   [`get_compressed`](#constriction.stream.stack.AnsCoder.get_compressed) when invoked with
    ///   no arguments).
    /// - For bits-back related compression techniques, it can sometimes be useful to decode symbols
    ///   from some arbitrary bit string that was *not* generated by ANS. To do so, call the
    ///   constructor with the additional argument `seal=True` (if you don't set `seal` to `True`
    ///   then the `AnsCoder` will truncate any trailing zero words from `compressed`). Once you've
    ///   decoded and re-encoded some symbols, you can get back the original `compressed` data by
    ///   calling `.get_compressed(unseal=True)`.
    #[new]
    #[pyo3(signature = (compressed=None, seal=false))]
    pub fn new(compressed: Option<PyReadonlyArray1<'_, u32>>, seal: bool) -> PyResult<Self> {
        if compressed.is_none() && seal {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Need compressed data to seal.",
            ));
        }
        let inner = if let Some(compressed) = compressed {
            let compressed = array1_to_vec(compressed);
            if seal {
                crate::stream::stack::AnsCoder::from_binary(compressed).unwrap_infallible()
            } else {
                crate::stream::stack::AnsCoder::from_compressed(compressed).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Invalid compressed data: ANS compressed data never ends in a zero word.",
                    )
                })?
            }
        } else {
            crate::stream::stack::AnsCoder::new()
        };

        Ok(Self { inner })
    }

    /// Records a checkpoint to which you can jump during decoding using
    /// [`seek`](#constriction.stream.stack.AnsCoder.seek).
    ///
    /// Returns a tuple `(position, state)` where `position` is an integer that specifies how many
    /// 32-bit words of compressed data have been produced so far, and `state` is an integer that
    /// defines the `RangeEncoder`'s internal state (so that it can be restored upon
    /// [`seek`ing](#constriction.stream.stack.AnsCoder.seek).
    ///
    /// **Note:** Don't call `pos` if you just want to find out how much compressed data has been
    /// produced so far. Call [`num_words`](#constriction.stream.stack.AnsCoder.num_words)
    /// instead.
    ///
    /// ## Example
    ///
    /// See [`seek`](#constriction.stream.stack.AnsCoder.seek).
    #[pyo3(signature = ())]
    pub fn pos(&mut self) -> (usize, u64) {
        self.inner.pos()
    }

    /// Jumps to a checkpoint recorded with method
    /// [`pos`](#constriction.stream.stack.AnsCoder.pos) during encoding.
    ///
    /// This allows random-access decoding. The arguments `position` and `state` are the two values
    /// returned by the method [`pos`](#constriction.stream.stack
    ///
    /// **Note:** in an ANS coder, both decoding and seeking *consume* compressed data. The Python
    /// API of `constriction`'s ANS coder currently supports only seeking forward but not backward
    /// (seeking backward is supported for Range Coding, and for both ANS and Range Coding in
    /// `constriction`'s Rust API).
    ///
    /// ## Example
    ///
    /// ```python
    /// probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
    /// model         = constriction.stream.model.Categorical(probabilities, perfect=False)
    /// message_part1 = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
    /// message_part2 = np.array([2, 2, 0, 1, 3], dtype=np.int32)
    ///
    /// # Encode both parts of the message (in reverse order, because ANS
    /// # operates as a stack) and record a checkpoint in-between:
    /// coder = constriction.stream.stack.AnsCoder()
    /// coder.encode_reverse(message_part2, model)
    /// (position, state) = coder.pos() # Records a checkpoint.
    /// coder.encode_reverse(message_part1, model)
    ///
    /// # We could now call `coder.get_compressed()` but we'll just decode
    /// # directly from the original `coder` for simplicity.
    ///
    /// # Decode first symbol:
    /// print(coder.decode(model)) # (prints: 1)
    ///
    /// # Jump to part 2 and decode it:
    /// coder.seek(position, state)
    /// decoded_part2 = coder.decode(model, 5)
    /// assert np.all(decoded_part2 == message_part2)
    /// ```
    #[pyo3(signature = (position, state))]
    pub fn seek(&mut self, position: usize, state: u64) -> PyResult<()> {
        self.inner.seek((position, state)).map_err(|()| {
            pyo3::exceptions::PyValueError::new_err(
                "Tried to seek past end of stream. Note: in an ANS coder,\n\
                both decoding and seeking *consume* compressed data. The Python API of\n\
                `constriction`'s ANS coder currently does not support seeking backward.",
            )
        })
    }

    /// Resets the encoder to an empty state.
    ///
    /// This removes any existing compressed data on the encoder. It is equivalent to replacing the
    /// encoder with a new one but slightly more efficient.
    #[pyo3(signature = ())]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Returns the current size of the encapsulated compressed data, in `np.uint32` words.
    ///
    /// Thus, the number returned by this method is the length of the array that you would get if
    /// you called [`get_compressed`](#constriction.stream.queue.RangeEncoder.get_compressed)
    /// without arguments.
    #[pyo3(signature = ())]
    pub fn num_words(&self) -> usize {
        self.inner.num_words()
    }

    /// Returns the current size of the compressed data, in bits, rounded up to full words.
    ///
    /// This is 32 times the result of what [`num_words`](#constriction.stream.queue.RangeEncoder.num_words)
    /// would return.
    #[pyo3(signature = ())]
    pub fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    /// The current size of the compressed data, in bits, not rounded up to full words.
    ///
    /// This can be at most 32 smaller than `.num_bits()`.
    #[pyo3(signature = ())]
    pub fn num_valid_bits(&self) -> usize {
        self.inner.num_valid_bits()
    }

    /// Returns `True` iff the coder is in its default initial state.
    ///
    /// The default initial state is the state returned by the constructor when
    /// called without arguments, or the state to which the coder is set when
    /// calling `clear`.
    #[pyo3(signature = ())]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a copy of the compressed data.
    ///
    /// You'll almost always want to call this method without arguments (which will default to
    /// `unseal=False`). See below for an explanation of the advanced use case with argument
    /// `unseal=True`.
    ///
    /// You will typically only want to call this method at the very end of your encoding task,
    /// i.e., once you've encoded the *entire* message. There is usually no need to call this method
    /// after encoding each symbol or other portion of your message. The encoders in `constriction`
    /// *accumulate* compressed data in an internal buffer, and encoding (semantically) *appends* to
    /// this buffer.
    ///
    /// That said, calling `get_compressed` has no side effects, so you *can* call `get_compressed`,
    /// then continue to encode more symbols, and then call `get_compressed` again. The first call
    /// of `get_compressed` will have no effect on the return value of the second call of
    /// `get_compressed`.
    ///
    /// The return value is a rank-1 numpy array of `dtype=np.uint32`. You can write it to a file by
    /// calling `to_file` on it, but we recommend to convert it into an architecture-independent
    /// byte order first:
    ///
    /// ```python
    /// import sys
    ///
    /// encoder = constriction.stream.stack.AnsCoder()
    /// # ... encode some message (skipped here) ...
    /// compressed = encoder.get_compressed() # returns a numpy array.
    /// if sys.byteorder != 'little':
    ///     # Let's save data in little-endian byte order by convention.
    ///     compressed.byteswap(inplace=True)
    /// compressed.tofile('compressed-file.bin')
    ///
    /// # At a later point, you might want to read and decode the file:
    /// compressed = np.fromfile('compressed-file.bin', dtype=np.uint32)
    /// if sys.byteorder != 'little':
    ///     # Restore native byte order before passing it to `constriction`.
    ///     compressed.byteswap(inplace=True)
    /// decoder = constriction.stream.stack.AnsCoder(compressed)
    /// # ... decode the message (skipped here) ...
    /// ```    
    ///
    /// ## Explanation of the optional argument `unseal`
    ///
    /// The optional argument `unseal` of this method is the counterpart to the optional argument
    /// `seal` of the constructor. Calling `.get_compressed(unseal=True)` tells the ANS coder that
    /// you expect it to be in a "sealed" state and instructs it to reverse the "sealing" operation.
    /// An ANS coder is in a sealed state if its encapsulated compressed data ends in a single "1"
    /// word. Calling the constructor of `AnsCoder` with argument `seal=True` constructs a coder
    /// that is guaranteed to be in a sealed state because the constructor will append a single "1"
    /// word to the provided `compressed` data. This sealing/unsealing operation makes sure that any
    /// trailing zero words are conserved since an `AnsCoder` would otherwise truncate them.
    ///
    /// Note that calling `.get_compressed(unseal=True)` fails if the coder is not in a "sealed"
    /// state.
    #[pyo3(signature = (unseal=false))]
    pub fn get_compressed<'p>(
        &mut self,
        py: Python<'p>,
        unseal: bool,
    ) -> PyResult<Bound<'p, PyArray1<u32>>> {
        if unseal {
            let binary = self.inner.get_binary().map_err(|_|
                pyo3::exceptions::PyAssertionError::new_err(
                    "Cannot unseal compressed data because it doesn't fit into integer number of words. Did you create the encoder with `seal=True` and restore its original state?",
                ))?;
            Ok(PyArray1::from_slice_bound(py, &binary))
        } else {
            Ok(PyArray1::from_slice_bound(
                py,
                &self.inner.get_compressed().unwrap_infallible(),
            ))
        }
    }

    /// Encodes one or more symbols, appending them to the encapsulated compressed data.
    ///
    /// This method can be called in 3 different ways:
    ///
    /// ## Option 1: encode_reverse(symbol, model)
    ///
    /// Encodes a *single* symbol with a concrete (i.e., fully parameterized) entropy model; the
    /// suffix "_reverse" of the method name has no significance when called this way.
    ///
    /// For optimal computational efficiency, don't use this option in a loop if you can instead
    /// use one of the two alternative options below.
    ///
    /// For example:
    ///
    /// ```python
    /// # Define a concrete categorical entropy model over the (implied)
    /// # alphabet {0, 1, 2}:
    /// probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    /// model = constriction.stream.model.Categorical(probabilities, perfect=False)
    ///
    /// # Encode a single symbol with this entropy model:
    /// coder = constriction.stream.stack.AnsCoder()
    /// coder.encode_reverse(2, model) # Encodes the symbol `2`.
    /// # ... then encode some more symbols ...
    /// ```
    ///
    /// ## Option 2: encode_reverse(symbols, model)
    ///
    /// Encodes multiple i.i.d. symbols, i.e., all symbols in the rank-1 array `symbols` will be
    /// encoded with the same concrete (i.e., fully parameterized) entropy model. The symbols are
    /// encoded in *reverse* order so that subsequent decoding will retrieve them in forward order
    /// (see [module-level example](#example)).
    ///
    /// For example:
    ///
    /// ```python
    /// # Use the same concrete entropy model as in the previous example:
    /// probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    /// model = constriction.stream.model.Categorical(probabilities, perfect=False)
    ///
    /// # Encode an example message using the above `model` for all symbols:
    /// symbols = np.array([0, 2, 1, 2, 0, 2, 0, 2, 1], dtype=np.int32)
    /// coder = constriction.stream.stack.AnsCoder()
    /// coder.encode_reverse(symbols, model)
    /// print(coder.get_compressed()) # (prints: [1276732052, 172])
    /// ```
    ///
    /// ## Option 3: encode_reverse(symbols, model_family, params1, params2, ...)
    ///
    /// Encodes multiple symbols, using the same *family* of entropy models (e.g., categorical or
    /// quantized Gaussian) for all symbols, but with different model parameters for each symbol;
    /// here, each `paramsX` argument is an array of the same length as `symbols`. The number of
    /// required `paramsX` arguments and their shapes and `dtype`s depend on the model family. The
    /// symbols are encoded in *reverse* order so that subsequent decoding will retrieve them in
    /// forward order (see [module-level example](#example)). But the mapping between symbols and
    /// model parameters is as you'd expect it to be (i.e., `symbols[i]` gets encoded with model
    /// parameters `params1[i]`, `params2[i]`, and so on, where `i` counts backwards).
    ///
    /// For example, the
    /// [`QuantizedGaussian`](model.html#constriction.stream.model.QuantizedGaussian) model family
    /// expects two rank-1 model parameters with float `dtype`, which specify the mean and
    /// standard deviation for each entropy model:
    ///
    /// ```python
    /// # Define a generic quantized Gaussian distribution for all integers
    /// # in the range from -100 to 100 (both ends inclusive):
    /// model_family = constriction.stream.model.QuantizedGaussian(-100, 100)
    ///    
    /// # Specify the model parameters for each symbol:
    /// means = np.array([10.3, -4.7, 20.5], dtype=np.float32)
    /// stds  = np.array([ 5.2, 24.2,  3.1], dtype=np.float32)
    ///    
    /// # Encode an example message:
    /// # (needs `len(symbols) == len(means) == len(stds)`)
    /// symbols = np.array([12, -13, 25], dtype=np.int32)
    /// coder = constriction.stream.stack.AnsCoder()
    /// coder.encode_reverse(symbols, model_family, means, stds)
    /// print(coder.get_compressed()) # (prints: [597775281, 3])
    /// ```
    ///
    /// By contrast, the [`Categorical`](model.html#constriction.stream.model.Categorical) model
    /// family expects a single rank-2 model parameter where the i'th row lists the
    /// probabilities for each possible value of the i'th symbol:
    ///
    /// ```python
    /// # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    /// probabilities = np.array(
    ///     [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for symbols[0])
    ///      [0.3, 0.2, 0.2, 0.2, 0.1]], # (for symbols[1])
    ///     dtype=np.float32)
    /// model_family = constriction.stream.model.Categorical(perfect=False)
    ///
    /// # Encode 2 symbols (needs `len(symbols) == probabilities.shape[0]`):
    /// symbols = np.array([3, 1], dtype=np.int32)
    /// coder = constriction.stream.stack.AnsCoder()
    /// coder.encode_reverse(symbols, model_family, probabilities)
    /// print(coder.get_compressed()) # (prints: [45298482])
    /// ```
    #[pyo3(signature = (symbols, model, *optional_model_params))]
    pub fn encode_reverse(
        &mut self,
        py: Python<'_>,
        symbols: &Bound<'_, PyAny>,
        model: &Model,
        optional_model_params: &Bound<'_, PyTuple>,
    ) -> PyResult<()> {
        if let Ok(symbol) = symbols.extract::<i32>() {
            if !optional_model_params.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "To encode a single symbol, use a concrete model, i.e., pass the\n\
                    model parameters directly to the constructor of the model and not to the\n\
                    `encode` method of the entropy coder. Delaying the specification of model\n\
                    parameters until calling `encode_reverse` is only useful if you want to encode
                    several symbols in a row with individual model parameters for each symbol. If\n\
                    this is what you're trying to do then the `symbols` argument should be a numpy\n\
                    array, not a scalar.",
                ));
            }
            return model.0.as_parameterized(py, &mut |model| {
                self.inner
                    .encode_symbol(symbol, EncoderDecoderModel(model))?;
                Ok(())
            });
        }

        // Don't use an `else` branch here because, if the following `extract` fails, the returned
        // error message is actually pretty user friendly.
        let symbols = symbols.extract::<PyReadonlyArray1<'_, i32>>()?;
        let symbols = symbols.as_array();

        if optional_model_params.is_empty() {
            model.0.as_parameterized(py, &mut |model| {
                self.inner
                    .encode_iid_symbols_reverse(symbols, EncoderDecoderModel(model))?;
                Ok(())
            })?;
        } else {
            if symbols.len()
                != model.0.len(
                    optional_model_params
                        .get_borrowed_item(0)
                        .expect("len checked above"),
                )?
            {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "`symbols` argument has wrong length.",
                ));
            }
            let mut symbol_iter = symbols.iter().rev();
            model
                .0
                .parameterize(py, optional_model_params, true, &mut |model| {
                    let symbol = symbol_iter.next().expect("TODO");
                    self.inner
                        .encode_symbol(*symbol, EncoderDecoderModel(model))?;
                    Ok(())
                })?;
        }

        Ok(())
    }

    /// Decodes one or more symbols, consuming them from the encapsulated compressed data.
    ///
    /// This method can be called in 3 different ways:
    ///
    /// ## Option 1: decode(model)
    ///
    /// Decodes a *single* symbol with a concrete (i.e., fully parameterized) entropy model and
    /// returns the decoded symbol; (for optimal computational efficiency, don't use this option in
    /// a loop if you can instead use one of the two alternative options below.)
    ///
    /// For example:
    ///
    /// ```python
    /// # Define a concrete categorical entropy model over the (implied)
    /// # alphabet {0, 1, 2}:
    /// probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    /// model = constriction.stream.model.Categorical(probabilities, perfect=False)
    ///
    /// # Decode a single symbol from some example compressed data:
    /// compressed = np.array([2514924296, 114], dtype=np.uint32)
    /// coder = constriction.stream.stack.AnsCoder(compressed)
    /// symbol = coder.decode(model)
    /// print(symbol) # (prints: 2)
    /// # ... then decode some more symbols ...
    /// ```
    ///
    /// ## Option 2: decode(model, amt) [where `amt` is an integer]
    ///
    /// Decodes `amt` i.i.d. symbols using the same concrete (i.e., fully parametrized) entropy
    /// model for each symbol, and returns the decoded symbols as a rank-1 numpy array with
    /// `dtype=np.int32` and length `amt`;
    ///
    /// For example:
    ///
    /// ```python
    /// # Use the same concrete entropy model as in the previous example:
    /// probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    /// model = constriction.stream.model.Categorical(probabilities, perfect=False)
    ///
    /// # Decode 9 symbols from some example compressed data, using the
    /// # same (fixed) entropy model defined above for all symbols:
    /// compressed = np.array([2514924296, 114], dtype=np.uint32)
    /// coder = constriction.stream.stack.AnsCoder(compressed)
    /// symbols = coder.decode(model, 9)
    /// print(symbols) # (prints: [2, 0, 0, 1, 2, 2, 1, 2, 2])
    /// ```
    ///
    /// ## Option 3: decode(model_family, params1, params2, ...)
    ///
    /// Decodes multiple symbols, using the same *family* of entropy models (e.g., categorical or
    /// quantized Gaussian) for all symbols, but with different model parameters for each symbol,
    /// and returns the decoded symbols as a rank-1 numpy array with `dtype=np.int32`; here, all
    /// `paramsX` arguments are arrays of equal length (the number of symbols to be decoded). The
    /// number of required `paramsX` arguments and their shapes and `dtype`s depend on the model
    /// family.
    ///
    /// For example, the
    /// [`QuantizedGaussian`](model.html#constriction.stream.model.QuantizedGaussian) model family
    /// expects two rank-1 model parameters with float `dtype`, which specify the mean and
    /// standard deviation for each entropy model:
    ///
    /// ```python
    /// # Define a generic quantized Gaussian distribution for all integers
    /// # in the range from -100 to 100 (both ends inclusive):
    /// model_family = constriction.stream.model.QuantizedGaussian(-100, 100)
    ///
    /// # Specify the model parameters for each symbol:
    /// means = np.array([10.3, -4.7, 20.5], dtype=np.float32)
    /// stds  = np.array([ 5.2, 24.2,  3.1], dtype=np.float32)
    ///
    /// # Decode a message from some example compressed data:
    /// compressed = np.array([597775281, 3], dtype=np.uint32)
    /// coder = constriction.stream.stack.AnsCoder(compressed)
    /// symbols = coder.decode(model_family, means, stds)
    /// print(symbols) # (prints: [12, -13, 25])
    /// ```
    ///
    /// By contrast, the [`Categorical`](model.html#constriction.stream.model.Categorical) model
    /// family expects a single rank-2 model parameter where the i'th row lists the
    /// probabilities for each possible value of the i'th symbol:
    ///
    /// ```python
    /// # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    /// probabilities = np.array(
    ///     [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for first decoded symbol)
    ///      [0.3, 0.2, 0.2, 0.2, 0.1]], # (for second decoded symbol)
    ///     dtype=np.float32)
    /// model_family = constriction.stream.model.Categorical(perfect=False)
    ///
    /// # Decode 2 symbols:
    /// compressed = np.array([2142112014, 31], dtype=np.uint32)
    /// coder = constriction.stream.stack.AnsCoder(compressed)
    /// symbols = coder.decode(model_family, probabilities)
    /// print(symbols) # (prints: [3, 1])
    /// ```
    #[pyo3(signature = (model, *optional_amt_or_model_params))]
    pub fn decode(
        &mut self,
        py: Python<'_>,
        model: &Model,
        optional_amt_or_model_params: &Bound<'_, PyTuple>,
    ) -> PyResult<PyObject> {
        match optional_amt_or_model_params.len() {
            0 => {
                let mut symbol = 0;
                model.0.as_parameterized(py, &mut |model| {
                    symbol = self
                        .inner
                        .decode_symbol(EncoderDecoderModel(model))
                        .unwrap_infallible();
                    Ok(())
                })?;
                return Ok(symbol.to_object(py));
            }
            1 => {
                if let Ok(amt) = optional_amt_or_model_params
                    .get_borrowed_item(0)
                    .expect("len checked above")
                    .extract::<usize>()
                {
                    let mut symbols = Vec::with_capacity(amt);
                    model.0.as_parameterized(py, &mut |model| {
                        for symbol in self
                            .inner
                            .decode_iid_symbols(amt, EncoderDecoderModel(model))
                        {
                            symbols.push(symbol.unwrap_infallible());
                        }
                        Ok(())
                    })?;
                    return Ok(PyArray1::from_iter_bound(py, symbols).to_object(py));
                }
            }
            _ => {} // Fall through to code below.
        };

        let mut symbols = Vec::with_capacity(
            model.0.len(
                optional_amt_or_model_params
                    .get_borrowed_item(0)
                    .expect("len checked above"),
            )?,
        );
        model
            .0
            .parameterize(py, optional_amt_or_model_params, false, &mut |model| {
                let symbol = self
                    .inner
                    .decode_symbol(EncoderDecoderModel(model))
                    .unwrap_infallible();
                symbols.push(symbol);
                Ok(())
            })?;

        Ok(PyArray1::from_vec_bound(py, symbols).to_object(py))
    }

    /// Creates a deep copy of the coder and returns it.
    ///
    /// The returned copy will initially encapsulate the identical compressed data as the
    /// original coder, but the two coders can be used independently without influencing
    /// other.
    #[pyo3(signature = ())]
    pub fn clone(&self) -> Self {
        Clone::clone(self)
    }
}
