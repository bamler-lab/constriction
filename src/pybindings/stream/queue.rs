use std::prelude::v1::*;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, types::PyTuple};

use crate::{
    stream::{
        queue::{DecoderFrontendError, RangeCoderState},
        Decode, Encode,
    },
    Pos, Seek, UnwrapInfallible,
};

use super::model::{internals::EncoderDecoderModel, Model};

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<RangeEncoder>()?;
    module.add_class::<RangeDecoder>()?;
    Ok(())
}

/// An encoder that uses the range coding algorithm.
///
/// To encode data with a `RangeEncoder`, call its method
/// [`encode`](#constriction.stream.queue.RangeEncoder.encode) one or more times. A `RangeEncoder`
/// has an internal buffer of compressed data, and each `encode` operation appends to this internal
/// buffer. You can copy out the contents of the internal buffer by calling the method
/// [`get_compressed`](#constriction.stream.queue.RangeEncoder.get_compressed). This will return a
/// rank-1 numpy array with `dtype=np.uint32` that you can pass to the constructor of a
/// `RangeDecoder` or write to a file for decoding at some later time (see example in the
/// documentation of the method
/// [`get_compressed`](#constriction.stream.queue.RangeEncoder.get_compressed)).
///
/// ## Example
///
/// See [module level example](#example).
#[pyclass]
#[pyo3(text_signature = "(self)")]
#[derive(Debug, Default, Clone)]
pub struct RangeEncoder {
    inner: crate::stream::queue::DefaultRangeEncoder,
}

#[pymethods]
impl RangeEncoder {
    /// Constructs a new (empty) range encoder.
    #[new]
    pub fn new() -> Self {
        let inner = crate::stream::queue::DefaultRangeEncoder::new();
        Self { inner }
    }

    /// Resets the encoder to an empty state.
    ///
    /// This removes any existing compressed data on the coder. It is equivalent to replacing the
    /// coder with a new one but slightly more efficient.
    #[pyo3(text_signature = "(self)")]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Records a checkpoint to which you can jump during decoding using
    /// [`seek`](#constriction.stream.queue.RangeDecoder.seek).
    ///
    /// Returns a tuple `(position, state)` where `position` is an integer that specifies how many
    /// 32-bit words of compressed data have been produced so far, and `state` is a tuple of two
    /// integers that define the `RangeEncoder`'s internal state (so that it can be restored upon
    /// [`seek`ing](#constriction.stream.queue.RangeDecoder.seek).
    ///
    /// **Note:** Don't call `pos` if you just want to find out how much compressed data has been
    /// produced so far. Call [`num_words`](#constriction.stream.queue.RangeEncoder.num_words)
    /// instead.
    ///
    /// ## Example
    ///
    /// See [`seek`](#constriction.stream.queue.RangeDecoder.seek).
    #[pyo3(text_signature = "(self)")]
    pub fn pos(&mut self) -> (usize, (u64, u64)) {
        let (pos, state) = self.inner.pos();
        (pos, (state.lower(), state.range().get()))
    }

    /// Returns the current size of the encapsulated compressed data, in `np.uint32` words.
    ///
    /// Thus, the number returned by this method is the length of the array that you would get if
    /// you called [`get_compressed`](#constriction.stream.queue.RangeEncoder.get_compressed).
    #[pyo3(text_signature = "(self)")]
    pub fn num_words(&self) -> usize {
        self.inner.num_words()
    }

    /// Returns the current size of the compressed data, in bits, rounded up to full words.
    ///
    /// This is 32 times the result of what [`num_words`](#constriction.stream.queue.RangeEncoder.num_words)
    /// would return.
    #[pyo3(text_signature = "(self)")]
    pub fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    /// Returns `True` iff the coder is in its default initial state.
    ///
    /// The default initial state is the state returned by the constructor when
    /// called without arguments, or the state to which the coder is set when
    /// calling `clear`.
    #[pyo3(text_signature = "(self)")]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns a copy of the compressed data accumulated so far, as a rank-1 numpy array of
    /// `dtype=np.uint32`.
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
    /// encoder = constriction.stream.queue.RangeEncoder()
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
    /// decoder = constriction.stream.queue.RangeDecoder(compressed)
    /// # ... decode the message (skipped here) ...
    /// ```
    #[pyo3(text_signature = "(self)")]
    pub fn get_compressed<'p>(&mut self, py: Python<'p>) -> &'p PyArray1<u32> {
        PyArray1::from_slice(py, &self.inner.get_compressed())
    }

    /// Returns a `RangeDecoder` that is initialized with a copy of the compressed data currently on
    /// this `RangeEncoder`.
    ///
    /// If `encoder` is a `RangeEncoder`, then
    ///
    /// ```python
    /// decoder = encoder.get_decoder()
    /// ```
    ///
    /// is equivalent to:
    ///
    /// ```python
    /// compressed = encoder.get_compressed()
    /// decoder = constriction.stream.stack.RangeDecoder(compressed)
    /// ```
    ///
    /// Calling `get_decoder` is more efficient since it copies the compressed data only once
    /// whereas the longhand version copies the data twice.
    #[pyo3(text_signature = "(self)")]
    pub fn get_decoder(&mut self) -> RangeDecoder {
        let compressed = self.inner.get_compressed().to_vec();
        RangeDecoder::from_vec(compressed)
    }

    /// Encodes one or more symbols, appending them to the encapsulated compressed data.
    ///
    /// This method can be called in 3 different ways:
    ///
    /// ## Option 1: encode(symbol, model)
    ///
    /// Encodes a *single* symbol with a concrete (i.e., fully parameterized) entropy model; (for
    /// optimal computational efficiency, don't use this option in a loop if you can instead use one
    /// of the two alternative options below.)
    ///
    /// For example:
    ///
    /// ```python
    /// # Define a concrete categorical entropy model over the (implied)
    /// # alphabet {0, 1, 2}:
    /// probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float64)
    /// model = constriction.stream.model.Categorical(probabilities)
    ///
    /// # Encode a single symbol with this entropy model:
    /// encoder = constriction.stream.queue.RangeEncoder()
    /// encoder.encode(2, model) # Encodes the symbol `2`.
    /// # ... then encode some more symbols ...
    /// ```
    ///
    /// ## Option 2: encode(symbols, model)
    ///
    /// Encodes multiple i.i.d. symbols, i.e., all symbols in the rank-1 array `symbols` will be
    /// encoded with the same concrete (i.e., fully parameterized) entropy model.
    ///
    /// For example:
    ///
    /// ```python
    /// # Use the same concrete entropy model as in the previous example:
    /// probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float64)
    /// model = constriction.stream.model.Categorical(probabilities)
    ///
    /// # Encode an example message using the above `model` for all symbols:
    /// symbols = np.array([0, 2, 1, 2, 0, 2, 0, 2, 1], dtype=np.int32)
    /// encoder = constriction.stream.queue.RangeEncoder()
    /// encoder.encode(symbols, model)
    /// print(encoder.get_compressed()) # (prints: [369323576])
    /// ```
    ///
    /// ## Option 3: encode(symbols, model_family, params1, params2, ...)
    ///
    /// Encodes multiple symbols, using the same *family* of entropy models (e.g., categorical or
    /// quantized Gaussian) for all symbols, but with different model parameters for each symbol;
    /// here, each `paramsX` argument is an array of the same length as `symbols`. The number of
    /// required `paramsX` arguments and their shapes and `dtype`s depend on the model family.
    ///
    /// For example, the
    /// [`QuantizedGaussian`](model.html#constriction.stream.model.QuantizedGaussian) model family
    /// expects two rank-1 model parameters of dtype `np.float64`, which specify the mean and
    /// standard deviation for each entropy model:
    ///
    /// ```python
    /// # Define a generic quantized Gaussian distribution for all integers
    /// # in the range from -100 to 100 (both ends inclusive):
    /// model_family = constriction.stream.model.QuantizedGaussian(-100, 100)
    ///    
    /// # Specify the model parameters for each symbol:
    /// means = np.array([10.3, -4.7, 20.5], dtype=np.float64)
    /// stds  = np.array([ 5.2, 24.2,  3.1], dtype=np.float64)
    ///    
    /// # Encode an example message:
    /// # (needs `len(symbols) == len(means) == len(stds)`)
    /// symbols = np.array([12, -13, 25], dtype=np.int32)
    /// encoder = constriction.stream.queue.RangeEncoder()
    /// encoder.encode(symbols, model_family, means, stds)
    /// print(encoder.get_compressed()) # (prints: [2655472005])
    /// ```
    ///
    /// By contrast, the [`Categorical`](model.html#constriction.stream.model.Categorical) model
    /// family expects a single rank-2 model parameter where the i'th row lists the
    /// probabilities for each possible value of the i'th symbol:
    ///
    /// ```python
    /// # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    /// probabilities = np.array(
    ///     [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for first encoded symbol)
    ///      [0.3, 0.2, 0.2, 0.2, 0.1]], # (for second encoded symbol)
    ///     dtype=np.float64)
    /// model_family = constriction.stream.model.Categorical()
    ///
    /// # Encode 2 symbols (needs `len(symbols) == probabilities.shape[0]`):
    /// symbols = np.array([3, 1], dtype=np.int32)
    /// encoder = constriction.stream.queue.RangeEncoder()
    /// encoder.encode(symbols, model_family, probabilities)
    /// print(encoder.get_compressed()) # (prints: [2705829535])
    /// ```
    #[pyo3(signature = (symbols, model, *params), text_signature = "(self, symbols, model, *optional_model_params)")]
    pub fn encode(
        &mut self,
        py: Python<'_>,
        symbols: &PyAny,
        model: &Model,
        params: &PyTuple,
    ) -> PyResult<()> {
        // TODO: also allow encoding and decoding with model type instead of instance for
        // models that take no range.
        if let Ok(symbol) = symbols.extract::<i32>() {
            if !params.is_empty() {
                return Err(pyo3::exceptions::PyAttributeError::new_err(
                    "To encode a single symbol, use a concrete model, i.e., pass the\n\
                    model parameters directly to the constructor of the model and not to the\n\
                    `encode` method of the entropy coder. Delaying the specification of model\n\
                    parameters until calling `encode` is only useful if you want to encode several\n\
                    symbols in a row with individual model parameters for each symbol. If this is\n\
                    what you're trying to do then the `symbols` argument should be a numpy array,\n\
                    not a scalar.",
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

        if params.is_empty() {
            model.0.as_parameterized(py, &mut |model| {
                self.inner
                    .encode_iid_symbols(symbols, EncoderDecoderModel(model))?;
                Ok(())
            })?;
        } else {
            if symbols.len() != model.0.len(&params[0])? {
                return Err(pyo3::exceptions::PyAttributeError::new_err(
                    "`symbols` argument has wrong length.",
                ));
            }
            let mut symbol_iter = symbols.iter();
            model.0.parameterize(py, params, false, &mut |model| {
                let symbol = symbol_iter.next().expect("TODO");
                self.inner
                    .encode_symbol(*symbol, EncoderDecoderModel(model))?;
                Ok(())
            })?;
        }

        Ok(())
    }

    /// Creates a deep copy of the coder and returns it.
    ///
    /// The returned copy will initially encapsulate the identical compressed data as the
    /// original coder, but the two coders can be used independently without influencing
    /// other.
    #[pyo3(text_signature = "(self)")]
    pub fn clone(&self) -> Self {
        Clone::clone(self)
    }
}

/// A decoder of data that was previously encoded with a `RangeEncoder`.
///
/// The constructor expects a single argument `compressed`, which has to be a rank-1 numpy array
/// with `dtype=np.uint32` that contains the compressed data (as returned by the method
/// [`get_compressed`](#constriction.stream.queue.RangeEncoder.get_compressed) of a `RangeEncoder`).
/// The provided compressed data gets *copied* in to an internal buffer of the `RangeDecoder`.
///
/// To decode data with a `RangeDecoder`, call its method
/// [`decode`](#constriction.stream.queue.RangeDecoder.decode) one or more times. Each decoding
/// operation consumes some portion of the compressed data from the `RangeDecoder`'s internal
/// buffer.
///
/// ## Example
///
/// See [module level example](#example).
#[pyclass]
#[pyo3(text_signature = "(self, compressed)")]
#[derive(Debug, Clone)]
pub struct RangeDecoder {
    inner: crate::stream::queue::DefaultRangeDecoder,
}

#[pymethods]
impl RangeDecoder {
    #[new]
    pub fn new(compressed: PyReadonlyArray1<'_, u32>) -> PyResult<Self> {
        Ok(Self::from_vec(compressed.to_vec()?))
    }

    /// Jumps to a checkpoint recorded with method
    /// [`pos`](#constriction.stream.queue.RangeEncoder.pos) during encoding.
    ///
    /// This allows random-access decoding. The arguments `position` and `state` are the two values
    /// returned by the `RangeEncoder`'s method [`pos`](#constriction.stream.queue.RangeEncoder.pos).
    ///
    /// ## Example
    ///
    /// ```python
    /// probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float64)
    /// model         = constriction.stream.model.Categorical(probabilities)
    /// message_part1 = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
    /// message_part2 = np.array([2, 2, 0, 1, 3], dtype=np.int32)
    ///
    /// # Encode both parts of the message and record a checkpoint in-between:
    /// encoder = constriction.stream.queue.RangeEncoder()
    /// encoder.encode(message_part1, model)
    /// (position, state) = encoder.pos() # Records a checkpoint.
    /// encoder.encode(message_part2, model)
    ///
    /// compressed = encoder.get_compressed()
    /// decoder = constriction.stream.queue.RangeDecoder(compressed)
    ///
    /// # Decode first symbol:
    /// print(decoder.decode(model)) # (prints: 1)
    ///
    /// # Jump to part 2 and decode it:
    /// decoder.seek(position, state)
    /// decoded_part2 = decoder.decode(model, 5)
    /// assert np.all(decoded_part2 == message_part2)
    /// ```
    #[pyo3(text_signature = "(self, position, state)")]
    pub fn seek(&mut self, position: usize, state: (u64, u64)) -> PyResult<()> {
        let (lower, range) = state;
        let state = RangeCoderState::new(lower, range)
            .map_err(|()| pyo3::exceptions::PyAttributeError::new_err("Invalid coder state."))?;
        self.inner.seek((position, state)).map_err(|()| {
            pyo3::exceptions::PyAttributeError::new_err("Tried to seek past end of stream.")
        })
    }

    /// Returns `True` if all compressed data *may* have already been decoded and `False` if there
    /// is definitely still some more data available to decode.
    ///
    /// A return value of `True` does not necessarily mean that there is no data left on the
    /// decoder because `constriction`'s range coding implementation--by design--cannot detect end-
    /// of-stream in all cases. If you need ot be able to decode variable-length messages then you
    /// can introduce an "end of stream" sentinel symbol, which you append to all messages before
    /// encoding them.
    #[pyo3(text_signature = "(self)")]
    pub fn maybe_exhausted(&self) -> bool {
        self.inner.maybe_exhausted()
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
    /// probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float64)
    /// model = constriction.stream.model.Categorical(probabilities)
    ///
    /// # Decode a single symbol from some example compressed data:
    /// compressed = np.array([3089773345, 1894195597], dtype=np.uint32)
    /// decoder = constriction.stream.queue.RangeDecoder(compressed)
    /// symbol = decoder.decode(model)
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
    /// probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float64)
    /// model = constriction.stream.model.Categorical(probabilities)
    ///
    /// # Decode 9 symbols from some example compressed data, using the
    /// # same (fixed) entropy model defined above for all symbols:
    /// compressed = np.array([369323576], dtype=np.uint32)
    /// decoder = constriction.stream.queue.RangeDecoder(compressed)
    /// symbols = decoder.decode(model, 9)
    /// print(symbols) # (prints: [0, 2, 1, 2, 0, 2, 0, 2, 1])
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
    /// expects two rank-1 model parameters of dtype `np.float64`, which specify the mean and
    /// standard deviation for each entropy model:
    ///
    /// ```python
    /// # Define a generic quantized Gaussian distribution for all integers
    /// # in the range from -100 to 100 (both ends inclusive):
    /// model_family = constriction.stream.model.QuantizedGaussian(-100, 100)
    ///
    /// # Specify the model parameters for each symbol:
    /// means = np.array([10.3, -4.7, 20.5], dtype=np.float64)
    /// stds  = np.array([ 5.2, 24.2,  3.1], dtype=np.float64)
    ///
    /// # Decode a message from some example compressed data:
    /// compressed = np.array([2655472005], dtype=np.uint32)
    /// decoder = constriction.stream.queue.RangeDecoder(compressed)
    /// symbols = decoder.decode(model_family, means, stds)
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
    ///     dtype=np.float64)
    /// model_family = constriction.stream.model.Categorical()
    ///
    /// # Decode 2 symbols:
    /// compressed = np.array([2705829535], dtype=np.uint32)
    /// decoder = constriction.stream.queue.RangeDecoder(compressed)
    /// symbols = decoder.decode(model_family, probabilities)
    /// print(symbols) # (prints: [3, 1])
    /// ```
    #[pyo3(signature = (model, *params), text_signature = "(self, model, *optional_amt_or_model_params)")]
    pub fn decode(
        &mut self,
        py: Python<'_>,
        model: &Model,
        params: &PyTuple,
    ) -> PyResult<PyObject> {
        match params.len() {
            0 => {
                let mut symbol = 0;
                model.0.as_parameterized(py, &mut |model| {
                    symbol = self.inner.decode_symbol(EncoderDecoderModel(model))?;
                    Ok(())
                })?;
                return Ok(symbol.to_object(py));
            }
            1 => {
                if let Ok(amt) = usize::extract(params.as_slice()[0]) {
                    let mut symbols = Vec::with_capacity(amt);
                    model.0.as_parameterized(py, &mut |model| {
                        for symbol in self
                            .inner
                            .decode_iid_symbols(amt, EncoderDecoderModel(model))
                        {
                            symbols.push(symbol?);
                        }
                        Ok(())
                    })?;
                    return Ok(PyArray1::from_iter(py, symbols).to_object(py));
                }
            }
            _ => {} // Fall through to code below.
        };

        let mut symbols = Vec::with_capacity(model.0.len(&params[0])?);
        model.0.parameterize(py, params, false, &mut |model| {
            let symbol = self.inner.decode_symbol(EncoderDecoderModel(model))?;
            symbols.push(symbol);
            Ok(())
        })?;

        Ok(PyArray1::from_vec(py, symbols).to_object(py))
    }

    /// Creates a deep copy of the coder and returns it.
    ///
    /// The returned copy will initially encapsulate the identical compressed data as the
    /// original coder, but the two coders can be used independently without influencing
    /// other.
    #[pyo3(text_signature = "(self)")]
    pub fn clone(&self) -> Self {
        Clone::clone(self)
    }
}

impl RangeDecoder {
    pub fn from_vec(compressed: Vec<u32>) -> Self {
        let inner = crate::stream::queue::DefaultRangeDecoder::from_compressed(compressed)
            .unwrap_infallible();
        Self { inner }
    }
}

impl From<DecoderFrontendError> for pyo3::PyErr {
    fn from(err: DecoderFrontendError) -> Self {
        match err {
            DecoderFrontendError::InvalidData => {
                pyo3::exceptions::PyAssertionError::new_err(err.to_string())
            }
        }
    }
}
