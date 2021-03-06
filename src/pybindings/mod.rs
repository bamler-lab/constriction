pub mod stream;
pub mod symbol;

use std::prelude::v1::*;

use pyo3::{prelude::*, wrap_pymodule};

/// ## Entropy Coding Primitives for Research and Production
///
/// The `constriction` library provides a set of composable entropy coding algorithms with a
/// focus on ease of use, flexibility, compression performance, and computational
/// efficiency. The goals of `constriction` are to three-fold:
///
/// 1. **to facilitate research on novel lossless and lossy compression methods** by
///    bridging the gap between declarative tools for data modeling from the machine
///    learning community and imperative tools for algorithm design from the source coding
///    literature;
/// 2. **to simplify the transition from research code to production software** by providing
///    both an easy-to-use Python API (rapid iteration on research code) and a highly
///    generic Rust API (for turning research code into optimized standalone binary programs
///    or libraries with minimal dependencies); and
/// 3. **to serve as a teaching resource** by providing a wide range of entropy coding
///    algorithms within a single consistent framework, thus making the various algorithms
///    easily discoverable and comparable on example models and data. [Additional teaching
///    material](https://robamler.github.io/teaching/compress21/) will be made publicly
///    available as a by-product of an upcoming university course on data compression with
///    deep probabilistic models.
///
/// ## Currently Supported Entropy Coding Algorithms
///
/// The `constriction` library currently supports the following algorithms:
///
/// - **Asymmetric Numeral Systems (ANS):** a highly efficient modern entropy coder with
///   near-optimal compression performance that supports advanced use cases like bits-back
///   coding;
///   - A "split" variant of ANS coding is provided for advanced use cases in hierarchical
///     models with cyclic dependencies.
/// - **Range Coding:** a variant of Arithmetic Coding that is optimized for realistic
///   computing hardware; it has similar compression performance and almost the same
///   computational performance as ANS. The main practical difference is that Range Coding
///   is a queue (FIFO) while ANS is a stack (LIFO).
/// - **Huffman Coding:** a well-known symbol code, mainly provided here for teaching
///   purpose; you'll usually want to use a stream code like ANS or Range Coding instead.
///
/// ## Quick Start With the Python API
///
/// You are currently reading the documentation of `constriction`'s Python API. If Python is
/// not your language of choice then head over to the [Rust API Documentation](TODO). The
/// Python API focuses on ease of use and rapid iteration for data scientists and machine
/// learning researchers. The Rust API provides binary identical implementations of
/// everything that's available through the Python API. Additionally, the Rust API, provides
/// optional finer grained control over technical details (such as word size or numerical
/// precision) using Rust's generic typesystem.
///
/// ### Installation
///
/// The easiest and recommended way to install `constriction` for Python is via `pip`:
///
/// ```bash
/// pip install constriction
/// ```
///
/// ### Example
///
/// Let's encode a sequence of symbols and write the compressed data to a binary file. We'll
/// use a quantized Gaussian distribution as entropy model, with a different mean and
/// standard deviation of the Gaussian for each symbol so that the example is not too
/// simplistic. Further, we'll use an Asymmetric Numeral Systems (ANS) coder here for its
/// speed and compression performance. We'll discuss how you could replace the ANS coder
/// with a range coder or a symbol code like Huffman coding [below](#exercise).
///
/// ```python
/// import constriction
/// import numpy as np
/// import sys
///
/// # Create an empty Asymmetric Numeral Systems (ANS) Coder:
/// coder = constriction.stream.ans.AnsCoder()
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
/// coder = constriction.stream.ans.AnsCoder(compressed)
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
/// making the following substitutions (TODO: verify that this is actually correct)
///
/// **In the encoder,**
///
/// - replace `constriction.stream.ans.AnsCoder` with
///   `constriction.stream.range.RangeEncoder`; and
/// - replace `coder.encode_leaky_gaussian_symbols_reverse` with
///   `coder.encode_leaky_gaussian_symbols` (we no longer need to encode symbols in reverse
///   order since Range Coding is a queue, i.e., first-in-first-out; we only had to reverse
///   the order for the ANS coder since ANS is a stack, i.e., last-in-first-out).
///
/// **In the decoder,**
///
/// - replace `constriction.stream.ans.AnsCoder` with
///   `constriction.stream.range.RangeDecoder` (note that Range Coding distinguishes between
///   an encoder and a decoder type since the encoder writes to the back while the decoder
///   reads from the front; by contrast, ANS Coding reads and writes at the same position
///   and allows interleaving reads and writes).
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
/// encoder = constriction.stream.range.RangeEncoder()
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
/// print(f"(without sealing up to full words: {encoder.num_valid_bits()} bits)")
///
/// # Get the compressed bit string (sealed up to full words):
/// compressed = encoder.get_compressed()
///
/// # ... writing and reading from file same as above (skipped here) ...
///
/// # Initialize a Range Decoder from the compressed bit string:
/// decoder = constriction.stream.range.RangeDecoder(compressed)
///
/// # Decode the data and verify it's correct:
/// reconstructed = decoder.decode_leaky_gaussian_symbols(
///     min_supported_symbol, max_supported_symbol, means, stds)
/// assert decoder.is_empty()
/// assert np.all(reconstructed == symbols)
/// ```
#[pymodule(constriction)]
fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_wrapped(wrap_pymodule!(stream))?;
    module.add_wrapped(wrap_pymodule!(symbol))?;
    Ok(())
}

/// Stream codes, i.e., entropy codes that amortize compressed bits over several symbols.
///
/// TODO
#[pymodule]
fn stream(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    stream::init_module(py, module)
}

/// Symbol codes. Mainly provided for teaching purpose. You probably want to use a [stream
/// code](stream.html) instead.
///
/// TODO
#[pymodule]
fn symbol(py: Python<'_>, module: &PyModule) -> PyResult<()> {
    symbol::init_module(py, module)
}
