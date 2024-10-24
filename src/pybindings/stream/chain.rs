use std::prelude::v1::*;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{prelude::*, types::PyTuple};

use crate::{
    pybindings::{array1_to_vec, stream::model::Model},
    stream::{
        chain::{BackendError, DecoderFrontendError, EncoderFrontendError},
        Decode, Encode,
    },
    UnwrapInfallible,
};

use super::model::internals::EncoderDecoderModel;

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
/// data = np.array([0x80d14131, 0xdda97c6c, 0x5017a640, 0x01170a3e], np.uint32)
/// probabilities = np.array(
///     [[0.1, 0.7, 0.1, 0.1],  # (<-- probabilities for first decoded symbol)
///      [0.2, 0.2, 0.1, 0.5],  # (<-- probabilities for second decoded symbol)
///      [0.2, 0.1, 0.4, 0.3]]) # (<-- probabilities for third decoded symbol)
/// model_family = constriction.stream.model.Categorical(perfect=False)
///
/// # Decoding `data` with an `AnsCoder` results in the symbols `[0, 0, 2]`:
/// ansCoder = constriction.stream.stack.AnsCoder(data, seal=True)
/// print(ansCoder.decode(model_family, probabilities)) # (prints: [0, 0, 2])
///
/// # Even if we change only the first entropy model (slightly), *all* decoded
/// # symbols can change:
/// probabilities[0, :] = np.array([0.09, 0.71, 0.1, 0.1])
/// ansCoder = constriction.stream.stack.AnsCoder(data, seal=True)
/// print(ansCoder.decode(model_family, probabilities)) # (prints: [1, 0, 0])
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
/// data = np.array([0x80d14131, 0xdda97c6c, 0x5017a640, 0x01170a3e], np.uint32)
/// probabilities = np.array(
///     [[0.1, 0.7, 0.1, 0.1],  # (<-- probabilities for first decoded symbol)
///      [0.2, 0.2, 0.1, 0.5],  # (<-- probabilities for second decoded symbol)
///      [0.2, 0.1, 0.4, 0.3]]) # (<-- probabilities for third decoded symbol)
/// model_family = constriction.stream.model.Categorical(perfect=False)
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
pub fn init_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ChainCoder>()?;
    Ok(())
}

/// See module level documentation.
///
/// Constructor arguments:
///
/// - `data` is a one-dimensional numpy array of dtype `np.uint32`.
/// - `is_remaining` should be `False` if you intend to *decode* symbols from `data` and
///   `True` if you intend to *encode* symbols.
/// - `seal` must be `False` (the default) if `is_remainders==True`, and it should be `True`
///   if `is_remainders==False` unless you can guarantee that the last word of `data` is
///   nonzero (e.g., if you obtained `data` from either
///   `np.concatenate(ChainCoder.get_remaining())` or from `AnsCoder.get_compressed()` then
///   the last word, if existent, is guaranteed to be nonzero and you may set `seal=False`).
///
/// See [above usage example](#usage-example) for a more elaborate explanation of the
/// constructor arguments.
#[pyclass]
#[derive(Debug, Clone)]
pub struct ChainCoder {
    inner: crate::stream::chain::DefaultChainCoder,
}

#[pymethods]
impl ChainCoder {
    #[new]
    #[pyo3(signature = (data, is_remainders=false, seal=false))]
    pub fn new(data: PyReadonlyArray1<'_, u32>, is_remainders: bool, seal: bool) -> PyResult<Self> {
        let data = array1_to_vec(data);
        let inner = if is_remainders {
            if seal {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "Cannot seal remainders data.",
                ));
            } else {
                crate::stream::chain::ChainCoder::from_remainders(data).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Too little data provided, or provided data ends in zero word and `is_remainders==True`.",
                    )
                })?
            }
        } else if seal {
            crate::stream::chain::ChainCoder::from_binary(data)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Too little data provided."))?
        } else {
            crate::stream::chain::ChainCoder::from_compressed(data).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "Too little data provided, or provided data ends in zero word and `seal==False`.",
                    )
                })?
        };

        Ok(Self { inner })
    }

    /// Returns a copy of the compressed data after re-encoding symbols, split into two
    /// arrays that you may want to concatenate.
    ///
    /// See [above usage instructions](#usage-for-bits-back-coding) for further explanation.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (unseal=false))]
    pub fn get_data<'p>(
        &self,
        unseal: bool,
        py: Python<'p>,
    ) -> PyResult<(Bound<'p, PyArray1<u32>>, Bound<'p, PyArray1<u32>>)> {
        let cloned = self.inner.clone();
        let data = if unseal {
            cloned.into_binary()
        } else {
            cloned.into_compressed()
        };
        let (remainders, compressed) = data.map_err(|_| {
            pyo3::exceptions::PyAssertionError::new_err(
                "Fractional number of words in compressed or remainders data.",
            )
        })?;

        let remainders = PyArray1::from_vec_bound(py, remainders);
        let compressed = PyArray1::from_vec_bound(py, compressed);
        Ok((remainders, compressed))
    }

    /// Returns a copy of the remainders after decoding some symbols, split into two arrays
    /// that you may want to concatenate.
    ///
    /// See [above usage instructions](#usage-for-bits-back-coding) for further explanation.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = ())]
    pub fn get_remainders<'p>(
        &self,
        py: Python<'p>,
    ) -> PyResult<(Bound<'p, PyArray1<u32>>, Bound<'p, PyArray1<u32>>)> {
        let (compressed, remainders) = self.inner.clone().into_remainders().unwrap_infallible();
        let remainders = PyArray1::from_vec_bound(py, remainders);
        let compressed = PyArray1::from_vec_bound(py, compressed);
        Ok((compressed, remainders))
    }

    /// Encodes one or more symbols.
    ///
    /// Usage is analogous to [`AnsCoder.encode_reverse`](stack.html#constriction.stream.stack.AnsCoder.encode_reverse).
    ///
    /// Encoding appends the fixed amount of 24 bits per encoded symbol to the internal "compressed"
    /// buffer, regardless of the employed entropy model(s), and it consumes (24 bits - inf_content)
    /// per symbol from the internal "remainders" buffer (where "inf_content" is the information
    /// content of the encoded symbol under the employed entropy model).
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

    /// Decodes one or more symbols.
    ///
    /// Usage is analogous to [`AnsCoder.decode`](stack.html#constriction.stream.stack.AnsCoder.decode).
    ///
    /// Decoding consumes the fixed amount of 24 bits per encoded symbol from the internal
    /// "compressed" buffer, regardless of the employed entropy model(s), and it appends (24
    /// bits - inf_content) per symbol to the internal "remainders" buffer (where
    /// "inf_content" is the information content of the decoded symbol under the employed
    /// entropy model).
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
                        .expect("We use constant `PRECISION`.");
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
                            let symbol = symbol.expect("We use constant `PRECISION`.");
                            symbols.push(symbol);
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
                    .expect("We use constant `PRECISION`.");
                symbols.push(symbol);
                Ok(())
            })?;

        Ok(PyArray1::from_vec_bound(py, symbols).to_object(py))
    }

    /// Creates a deep copy of the coder and returns it.
    ///
    /// The returned copy will initially encapsulate the identical compressed data and
    /// remainders as the original coder, but the two coders can be used independently
    /// without influencing other.
    #[pyo3(signature = ())]
    pub fn clone(&self) -> Self {
        Clone::clone(self)
    }
}

impl From<EncoderFrontendError> for PyErr {
    fn from(err: EncoderFrontendError) -> Self {
        match err {
            EncoderFrontendError::ImpossibleSymbol => {
                pyo3::exceptions::PyKeyError::new_err(err.to_string())
            }
            EncoderFrontendError::OutOfRemainders => {
                pyo3::exceptions::PyAssertionError::new_err(err.to_string())
            }
        }
    }
}

impl From<DecoderFrontendError> for PyErr {
    fn from(err: DecoderFrontendError) -> Self {
        match err {
            DecoderFrontendError::OutOfCompressedData => {
                pyo3::exceptions::PyAssertionError::new_err(err.to_string())
            }
        }
    }
}

impl<CompressedBackendError: Into<PyErr>, RemaindersBackendError: Into<PyErr>>
    From<BackendError<CompressedBackendError, RemaindersBackendError>> for PyErr
{
    fn from(err: BackendError<CompressedBackendError, RemaindersBackendError>) -> Self {
        match err {
            BackendError::Compressed(err) => err.into(),
            BackendError::Remainders(err) => err.into(),
        }
    }
}
