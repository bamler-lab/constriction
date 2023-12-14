pub mod stream;
pub mod symbol;

use std::prelude::v1::*;

use alloc::borrow::Cow;
use numpy::PyReadonlyArray;
use pyo3::{prelude::*, wrap_pymodule};

/// ## Entropy Coders for Research and Production
///
/// The `constriction` library provides a set of composable entropy coding algorithms with a
/// focus on correctness, versatility, ease of use, compression performance, and
/// computational efficiency. The goals of `constriction` are three-fold:
///
/// 1. **to facilitate research on novel lossless and lossy compression methods** by
///    providing a *composable* set of primitives (e.g., you can can easily switch out a
///    Range Coder for an ANS coder without having to find a new library or change how you
///    represent exactly invertible entropy models);
/// 2. **to simplify the transition from research code to deployed software** by providing
///    similar APIs and binary compatible entropy coders for both Python (for rapid
///    prototyping on research code) and Rust (for turning successful prototypes into
///    standalone binaries, libraries, or WebAssembly modules); and
/// 3. **to serve as a teaching resource** by providing a variety of entropy coding
///    primitives within a single consistent framework. Check out our [additional teaching
///    material](https://robamler.github.io/teaching/compress21/) from a university course
///    on data compression, which contains some problem sets where you use `constriction`
///    (with solutions).
///
/// **More Information:** [project website](https://bamler-lab.github.io/constriction)
///
/// **Live demo:** [here's a web app](https://robamler.github.io/linguistic-flux-capacitor)
/// that started out as a machine-learning research project in Python and was later turned
/// into a web app by using `constriction` in a WebAssembly module).
///
/// ## Quick Start
///
/// ### Installing `constriction` for Python
///
/// ```bash
/// pip install constriction~=0.3.5
/// ```
///
/// ### Hello, World
///
/// You'll mostly use the `stream` submodule, which provides stream codes (like Range
/// Coding or ANS). The following example shows a simple encoding-decoding round trip. More
/// complex entropy models and other entropy coders are also supported, see
/// [more examples](#more-examples) below.
///
/// ```python
/// import constriction
/// import numpy as np
///
/// message = np.array([6, 10, -4, 2, 5, 2, 1, 0, 2], dtype=np.int32)
///
/// # Define an i.i.d. entropy model (see below for more complex models):
/// entropy_model = constriction.stream.model.QuantizedGaussian(-50, 50, 3.2, 9.6)
///
/// # Let's use an ANS coder in this example. See below for a Range Coder example.
/// encoder = constriction.stream.stack.AnsCoder()
/// encoder.encode_reverse(message, entropy_model)
///
/// compressed = encoder.get_compressed()
/// print(f"compressed representation: {compressed}")
/// print(f"(in binary: {[bin(word) for word in compressed]})")
///
/// decoder = constriction.stream.stack.AnsCoder(compressed)
/// decoded = decoder.decode(entropy_model, 9) # (decodes 9 symbols)
/// assert np.all(decoded == message)
/// ```
///
/// ## More Examples
///
/// ### Switching Out the Entropy Coding Algorithm
///
/// Let's take our ["Hello, World"](#hello-world) example from above and assume we want to
/// switch the entropy coding algorithm from ANS to Range Coding. But we don't want to
/// look for a new library or change how we represent entropy *models* and compressed data.
/// Luckily, we only have to modify a few lines of code:
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Same representation of message and entropy model as in the previous example:
/// message = np.array([6, 10, -4, 2, 5, 2, 1, 0, 2], dtype=np.int32)
/// entropy_model = constriction.stream.model.QuantizedGaussian(-50, 50, 3.2, 9.6)
///
/// # Let's use a Range coder now:
/// encoder = constriction.stream.queue.RangeEncoder()         # <-- CHANGED LINE
/// encoder.encode(message, entropy_model)          # <-- (slightly) CHANGED LINE
///
/// compressed = encoder.get_compressed()
/// print(f"compressed representation: {compressed}")
/// print(f"(in binary: {[bin(word) for word in compressed]})")
///
/// decoder = constriction.stream.queue.RangeDecoder(compressed) #<--CHANGED LINE
/// decoded = decoder.decode(entropy_model, 9) # (decodes 9 symbols)
/// assert np.all(decoded == message)
/// ```
///
/// ### Complex Entropy Models
///
/// This time, let's keep the entropy coding algorithm as it is but make the entropy *model*
/// more complex. We'll encode the first 5 symbols of the message again with a
/// `QuantizedGaussian` distribution, but this time we'll use individual model parameters
/// (means and standard deviations) for each of the 5 symbols. For the remaining 4 symbols,
/// we'll use a fixed categorical distribution, just to make it more interesting:
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Same message as above, but a complex entropy model consisting of two parts:
/// message = np.array([6,   10,   -4,   2,   5,    2, 1, 0, 2], dtype=np.int32)
/// means   = np.array([2.3,  6.1, -8.5, 4.1, 1.3], dtype=np.float64)
/// stds    = np.array([6.2,  5.3,  3.8, 3.2, 4.7], dtype=np.float64)
/// entropy_model1 = constriction.stream.model.QuantizedGaussian(-50, 50)
/// entropy_model2 = constriction.stream.model.Categorical(np.array(
///     [0.2, 0.5, 0.3], dtype=np.float64))  # Probabilities of the symbols 0,1,2.
///
/// # Simply encode both parts in sequence with their respective models:
/// encoder = constriction.stream.queue.RangeEncoder()
/// encoder.encode(message[0:5], entropy_model1, means, stds) # per-symbol params.
/// encoder.encode(message[5:9], entropy_model2)
///
/// compressed = encoder.get_compressed()
/// print(f"compressed representation: {compressed}")
/// print(f"(in binary: {[bin(word) for word in compressed]})")
///
/// decoder = constriction.stream.queue.RangeDecoder(compressed)
/// decoded_part1 = decoder.decode(entropy_model1, means, stds)
/// decoded_part2 = decoder.decode(entropy_model2, 4)
/// assert np.all(np.concatenate((decoded_part1, decoded_part2)) == message)
/// ```
///
/// You can define even more complex entropy models by providing an arbitrary Python
/// function for the cumulative distribution function (see
/// [`CustomModel`](stream/model.html#constriction.stream.model.CustomModel) and
/// [`ScipyModel`](stream/model.html#constriction.stream.model.CustomModel)). The
/// `constriction` library provides wrappers that turn your models into *exactly*
/// invertible fixed-point arithmetic since even tiny rounding errors could otherwise
/// completely break an entropy coding algorithm.
///
/// ### Exercise
///
/// We've shown examples of [ANS coding with a simple entropy model](#hello-world), of
/// [Range Coding with the same simple entropy
/// model](#switching-out-the-entropy-coding-algorithm) and of [Range coding with a complex
/// entropy model](#complex-entropy-models). One combination is still missing: ANS coding
/// with the complex entropy model from the last example above. This should be no problem
/// now, so try it out yourself:
///
/// - In the last example above, change both "queue.RangeEncoder" and "queue.RangeDecoder"
///   to "stack.AnsCoder" (ANS uses the same data structure for both encoding and decoding).
/// - Then change both occurrences of `.encode(...)` to `.encode_reverse(...)` (since ANS
///   operates as a stack, i.e., last-in-first-out, we encode the symbols in reverse order
///   so that we can decode them in their normal order).
/// - Finally, there's one slightly subtle change: when encoding the message, switch the
///   order of the two lines that encode `message[0:5]` and `message[5:9]`, respectively.
///   Do *not* change the order of decoding though. This is again necessary because ANS
///   operates as a stack.
///
/// Congratulations, you've successfully implemented your first own compression scheme with
/// `constriction`.
///
/// ## Further Reading
///
/// You can find links to more examples and tutorials on the [project
/// website](https://bamler-lab.github.io/constriction). Or just dive right into the
/// documentation of [range coding](stream/queue.html), [ANS](stream/stack.html), and
/// [entropy models](stream/model.html).
#[pymodule]
#[pyo3(name = "constriction")]
fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(init_stream))?;
    module.add_wrapped(wrap_pymodule!(init_symbol))?;
    Ok(())
}

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
fn init_stream(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    stream::init_module(py, module)
}

/// Symbol codes. Mainly provided for teaching purpose. You'll probably want to use a
/// [stream code](stream.html) instead.
///
/// Symbol codes encode messages (= sequences of symbols) by compressing each symbol
/// individually and then concatenating the compressed representations of the symbols
/// (called code words). This has the advantage of being conceptually simple, but the
/// disadvantage of incurring an overhead of up to almost 1 bit *per symbol*. The overhead
/// is most significant in the regime of low entropy per symbol (often the regime of novel
/// machine-learning based compression methods), since each (nontrivial) symbol in a symbol
/// code contributes at least 1 bit to the total bitrate of the message, even if its
/// information content is only marginally above zero.
///
/// This module defines the types `QueueEncoder`, `QueueDecoder`, and `StackCoder`, which
/// are essentially just efficient containers for of growable bit strings. The interesting
/// part happens in the so-called code book, which defines how symbols map to code words. We
/// provide the Huffman Coding algorithm in the submodule [`huffman`](symbol/huffman.html),
/// which calculates an optimal code book for a given probability distribution.
///
/// ## Examples
///
/// The following two examples both encode and then decode a sequence of symbols using the
/// same entropy model for each symbol. We could as well use a different entropy model for
/// each symbol, but it would make the examples more lengthy.
///
/// - Example 1: Encoding and decoding with "queue" semantics ("first in first out", i.e.,
///   we'll decode symbols in the same order in which we encode them):
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Define an entropy model over the (implied) alphabet {0, 1, 2, 3}:
/// probabils = np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float64)
///
/// # Encode some example message, using the same model for each symbol here:
/// message = [1, 3, 2, 3, 0, 1, 3, 0, 2, 1, 1, 3, 3, 1, 2, 0, 1, 3, 1]
/// encoder = constriction.symbol.QueueEncoder()
/// encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(probabils)
/// for symbol in message:
///     encoder.encode_symbol(symbol, encoder_codebook)
///
/// # Obtain the compressed representation and the bitrate:
/// compressed, bitrate = encoder.get_compressed()
/// print(compressed, bitrate) # (prints: [3756389791, 61358], 48)
/// print(f"(in binary: {[bin(word) for word in compressed]}")
///
/// # Decode the message
/// decoder = constriction.symbol.QueueDecoder(compressed)
/// decoded = []
/// decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(probabils)
/// for symbol in range(19):
///     decoded.append(decoder.decode_symbol(decoder_codebook))
///
/// assert decoded == message # (verifies correctness)
/// ```
///
/// - Example 2: Encoding and decoding with "stack" semantics ("last in first out", i.e.,
///   we'll encode symbols from back to front and then decode them from front to back):
///
/// ```python
/// import constriction
/// import numpy as np
///
/// # Define an entropy model over the (implied) alphabet {0, 1, 2, 3}:
/// probabils = np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float64)
///
/// # Encode some example message, using the same model for each symbol here:
/// message = [1, 3, 2, 3, 0, 1, 3, 0, 2, 1, 1, 3, 3, 1, 2, 0, 1, 3, 1]
/// coder = constriction.symbol.StackCoder()
/// encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(probabils)
/// for symbol in reversed(message): # Note: reversed
///     coder.encode_symbol(symbol, encoder_codebook)
///
/// # Obtain the compressed representation and the bitrate:
/// compressed, bitrate = coder.get_compressed()
/// print(compressed, bitrate) # (prints: [[2818274807, 129455] 48)
/// print(f"(in binary: {[bin(word) for word in compressed]}")
///
/// # Decode the message (we could explicitly construct a decoder:
/// # `decoder = constritcion.symbol.StackCoder(compressed)`
/// # but we can also also reuse our existing `coder` for decoding):
/// decoded = []
/// decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(probabils)
/// for symbol in range(19):
///     decoded.append(coder.decode_symbol(decoder_codebook))
///
/// assert decoded == message # (verifies correctness)
/// ```
#[pymodule]
#[pyo3(name = "symbol")]
fn init_symbol(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    symbol::init_module(py, module)
}

#[derive(Debug, Clone)]
pub enum PyReadonlyFloatArray<'py, D: ndarray::Dimension> {
    F32(PyReadonlyArray<'py, f32, D>),
    F64(PyReadonlyArray<'py, f64, D>),
}

pub type PyReadonlyFloatArray1<'py> = PyReadonlyFloatArray<'py, numpy::Ix1>;
pub type PyReadonlyFloatArray2<'py> = PyReadonlyFloatArray<'py, numpy::Ix2>;

impl<'py, D: ndarray::Dimension> FromPyObject<'py> for PyReadonlyFloatArray<'py, D> {
    fn extract(ob: &'py PyAny) -> PyResult<Self> {
        if let Ok(x) = ob.extract::<PyReadonlyArray<'_, f32, D>>() {
            Ok(PyReadonlyFloatArray::F32(x))
        } else {
            // This should also return a well crafted error in case it fails.
            ob.extract::<PyReadonlyArray<'_, f64, D>>()
                .map(PyReadonlyFloatArray::F64)
        }
    }
}

impl<'py, D: ndarray::Dimension> PyReadonlyFloatArray<'py, D> {
    fn cast_f64(&'py self) -> PyResult<Cow<'py, PyReadonlyArray<'py, f64, D>>> {
        match self {
            PyReadonlyFloatArray::F32(x) => Ok(Cow::Owned(x.cast::<f64>(false)?.readonly())),
            PyReadonlyFloatArray::F64(x) => Ok(Cow::Borrowed(x)),
        }
    }

    fn len(&self) -> usize {
        match self {
            PyReadonlyFloatArray::F32(x) => x.len(),
            PyReadonlyFloatArray::F64(x) => x.len(),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            PyReadonlyFloatArray::F32(x) => x.shape(),
            PyReadonlyFloatArray::F64(x) => x.shape(),
        }
    }

    fn get_f64<I: numpy::NpyIndex<Dim = D>>(&self, index: I) -> Option<f64> {
        match self {
            PyReadonlyFloatArray::F32(x) => x.get(index).map(|&x| x as f64),
            PyReadonlyFloatArray::F64(x) => x.get(index).copied(),
        }
    }
}
