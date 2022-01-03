pub mod stream;
pub mod symbol;

use std::prelude::v1::*;

use pyo3::{prelude::*, wrap_pymodule};

/// ## Entropy Coding Primitives for Research and Production
///
/// The `constriction` library provides a set of composable entropy coding algorithms with a
/// focus on correctness, versatility, ease of use, compression performance, and
/// computational efficiency. The goals of `constriction` are to three-fold:
///
/// 1. **to facilitate research on novel lossless and lossy compression methods** by
///    providing a *composable* set of entropy coding primitives rather than a rigid
///    implementation of a single preconfigured method;
/// 2. **to simplify the transition from research code to production software** by exposing
///    the exact same functionality via both a Python API (for rapid prototyping on research
///    code) and a Rust API (for turning successful prototypes into production); and
/// 3. **to serve as a teaching resource** by providing a wide range of entropy coding
///    algorithms within a single consistent framework, thus making the various algorithms
///    easily discoverable and comparable on example models and data. [Additional teaching
///    material](https://robamler.github.io/teaching/compress21/) is being made publicly
///    available as a by-product of an ongoing university course on data compression with
///    deep probabilistic models.
///
/// For an example of a compression codec that started as research code in Python and was
/// then deployed as a fast and dependency-free WebAssembly module using `constriction`'s
/// Rust API, have a look at [The Linguistic Flux
/// Capacitor](https://robamler.github.io/linguistic-flux-capacitor).
///
/// ## Project Status
///
/// We currently provide implementations of the following entropy coding algorithms:
///
/// - **Asymmetric Numeral Systems (ANS):** a fast modern entropy coder with near-optimal
///   compression effectiveness that supports advanced use cases like bits-back coding.
/// - **Range Coding:** a computationally efficient variant of Arithmetic Coding, that has
///   essentially the same compression effectiveness as ANS Coding but operates as a queue
///   ("first in first out"), which makes it preferable for autoregressive models.
/// - **Chain Coding:** an experimental new entropy coder that combines the (net)
///   effectiveness of stream codes with the locality of symbol codes; it is meant for
///   experimental new compression approaches that perform joint inference, quantization,
///   and bits-back coding in an end-to-end optimization. This experimental coder is mainly
///   provided to prove to ourselves that the API for encoding and decoding, which is shared
///   across all stream coders, is flexible enough to express complex novel tasks.
/// - **Huffman Coding:** a well-known symbol code, mainly provided here for teaching
///   purpose; you'll usually want to use a stream code like ANS or Range Coding instead
///   since symbol codes can have a considerable overhead on the bitrate, especially in the
///   regime of low entropy per symbol, which is common in machine-learning based
///   compression methods.
///
/// Further, `constriction` provides implementations of common probability distributions in
/// fixed-point arithmetic, which can be used as entropy models in either of the above
/// stream codes. The library also provides adapters for turning custom probability
/// distributions into exactly invertible fixed-point arithmetic.
///
/// The provided implementations of entropy coding algorithms and probability distributions
/// are extensively tested and should be considered reliable (except for the still
/// experimental Chain Coder). However, their APIs may change in future versions of
/// `constriction` if more user experience reveals any shortcomings of the current APIs in
/// terms of ergonomics. Please [file an
/// issue](https://github.com/bamler-lab/constriction/issues) if you run into a scenario
/// where the current APIs are suboptimal.
///
/// ## Quick Start With the Python API
///
/// You are currently reading the documentation of `constriction`'s Python API. If Python is
/// not your language of choice then head over to the [Rust API
/// Documentation](https://docs.rs/constriction). The Python API focuses on ease of use and
/// rapid iteration and is targeted mainly towards data scientists and machine learning
/// researchers. The Rust API provides binary identical implementations of everything that's
/// available through the Python API. Additionally, the Rust API, provides optional finer
/// grained control over technical details (such as word size or numerical precision) using
/// Rust's generic type system.
///
/// ### Installation
///
/// The easiest and recommended way to install `constriction` for Python is via `pip`:
///
/// ```bash
/// pip install constriction numpy
/// ```
///
/// ### Example
///
/// Let's encode a sequence of symbols and write the compressed data to a binary file. We'll
/// use a quantized Gaussian distribution as entropy model, with a different mean and
/// standard deviation of the Gaussian for each symbol so that the example is not too
/// simplistic. Further, we'll use an Asymmetric Numeral Systems (ANS) Coder here for its
/// speed and compression performance. We'll discuss how you could replace the ANS Coder
/// with a Range Coder or a symbol code like Huffman Coding [below](#exercise).
///
/// ```python
/// import constriction
/// import numpy as np
/// import sys
///
/// # Create an empty Asymmetric Numeral Systems (ANS) Coder:
/// coder = constriction.stream.stack.AnsCoder()
///
/// # Some made up data and entropy models for demonstration purpose:
/// min_supported_symbol, max_supported_symbol = -100, 100  # both inclusively
/// symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
/// means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
/// stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)
///
/// # Encode the data (in reverse order, since ANS is a stack):
/// coder.encode_leaky_gaussian_symbols_reverse(
///     symbols, min_supported_symbol, max_supported_symbol, means, stds)
///
/// print(f"Compressed size: {coder.num_bits()} bits")
/// print(f"(without unnecessary trailing zeros: {coder.num_valid_bits()} bits)")
///
/// # Get the compressed bit string, convert it into an architecture-independent
/// # byte order, and write it to a binary file:
/// compressed = coder.get_compressed()
/// if sys.byteorder == "big":
///     compressed.byteswap(inplace=True)
/// compressed.tofile("compressed.bin")
/// ```
///
/// Now let's read the compressed bit string back in and decode it.
///
/// ```python
/// import constriction
/// import numpy as np
/// import sys
///
/// # Read the compressed bit string from the file we created above and convert
/// # it into the right byte order for the current computer architecture:
/// compressed = np.fromfile("compressed.bin", dtype=np.uint32)
/// if sys.byteorder == "big":
///     compressed.byteswap(inplace=True)
///
/// # Initialize an ANS coder from the compressed bit string:
/// coder = constriction.stream.stack.AnsCoder(compressed)
///
/// # Use the same entropy models that we used for encoding:
/// min_supported_symbol, max_supported_symbol = -100, 100  # both inclusively
/// means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
/// stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)
///
/// # Decode and print the data:
/// reconstructed = coder.decode_leaky_gaussian_symbols(
///     min_supported_symbol, max_supported_symbol, means, stds)
/// assert coder.is_empty()
/// print(reconstructed)  # Should print [23, -15, 78, 43, -69].
/// ```
///
/// ### Exercise
///
/// Try out the above example and verify that decoding reconstructs the original data. Then
/// see how easy `constriction` makes it to replace the ANS coder with a range coder by
/// making the following substitutions:
///
/// **In the encoder,**
///
/// - replace `constriction.stream.stack.AnsCoder` with
///   `constriction.stream.queue.RangeEncoder`; and
/// - replace `coder.encode_leaky_gaussian_symbols_reverse` with
///   `coder.encode_leaky_gaussian_symbols` (we no longer need to encode symbols in reverse
///   order since Range Coding is a queue, i.e., first-in-first-out; we only had to reverse
///   the order for the ANS coder since ANS is a stack, i.e., last-in-first-out).
///
/// **In the decoder,**
///
/// - replace `constriction.stream.stack.AnsCoder` with
///   `constriction.stream.queue.RangeDecoder` (note that Range Coding distinguishes between
///   an encoder and a decoder type since the encoder writes to the back while the decoder
///   reads from the front; by contrast, ANS Coding is a stack, i.e., it reads and writes at
///   the same position and allows interleaving reads and writes).
///
/// You could also use a symbol code like Huffman Coding (see submodule `symbol`) but that
/// would have considerably worse compression performance, especially on large files, since
/// symbol codes always emit an integer number of bits per compressed symbol, even if the
/// information content of the symbol is a fractional number (stream codes like ANS and
/// Range Coding *effectively* emit a fractional number of bits per symbol since they
/// amortize over several symbols).
///
/// The above replacements should lead you to something like the following:
///
/// ```python
/// import constriction
/// import numpy as np
/// import sys
///
/// # Create an empty Range Encoder:
/// encoder = constriction.stream.queue.RangeEncoder()
///
/// # Same made up data and entropy models as in the ANS Coding example above:
/// min_supported_symbol, max_supported_symbol = -100, 100  # both inclusively
/// symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
/// means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
/// stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)
///
/// # Encode the data (this time in normal order, since Range Coding is a queue):
/// encoder.encode_leaky_gaussian_symbols(
///     symbols, min_supported_symbol, max_supported_symbol, means, stds)
///
/// print(f"Compressed size: {encoder.num_bits()} bits")
///
/// # Get the compressed bit string (sealed up to full words):
/// compressed = encoder.get_compressed()
///
/// # ... writing and reading from file same as above (skipped here) ...
///
/// # Initialize a Range Decoder from the compressed bit string:
/// decoder = constriction.stream.queue.RangeDecoder(compressed)
///
/// # Decode the data and verify it's correct:
/// reconstructed = decoder.decode_leaky_gaussian_symbols(
///     min_supported_symbol, max_supported_symbol, means, stds)
/// assert decoder.maybe_exhausted()
/// assert np.all(reconstructed == symbols)
/// ```
#[pymodule]
#[pyo3(name = "constriction")]
fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(stream))?;
    module.add_wrapped(wrap_pymodule!(symbol))?;
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
fn stream(py: Python<'_>, module: &PyModule) -> PyResult<()> {
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
fn symbol(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    symbol::init_module(py, module)
}
