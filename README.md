# Composable Entropy Coding Primitives for Research and Production

The `constriction` library provides a set of composable implementations of entropy coding
algorithms. It has APIs for both the **Python and Rust** languages and it focuses on
correctness, versatility, ease of use, compression performance, and computational
efficiency. The goals of `constriction` are to three-fold:

1. **to facilitate research on novel lossless and lossy compression methods** by providing a
   *composable* set of entropy coding primitives rather than a rigid implementation of a
   single preconfigured method; in compression research, different applications put
   different requirements on the entropy coding method. For example, you may prefer a Range
   Coder for an autoregressive entropy model (because it preserves the order of encoded
   symbols), but you may prefer an ANS Coder for a hierarchical entropy model (because it
   supports bits-back coding). With many other libraries, swapping out a Range Coder for an
   ANS Coder would mean that you not only have to find and learn how to use new library, but
   you would also have to port the part of your code that represents probabilistic entropy
   models so that it adheres to the rules of the new library. By contrast, the composable
   architecture of `constriction` lets you seamlessly swap out individual components of your
   compression pipeline (such as the core entropy coding algorithm) independently from other
   components (such as the fixed-point representation of entropy models or the strategy for
   dealing with zero probability symbols).
2. **to simplify the transition from research code to reliable software products** by
   exposing the exact same functionality via both a Python API (for rapid prototyping on
   research code) and a Rust API (for turning successful prototypes into production); This
   approach bridges the gap between two communities that have vastly different requirements
   on their software development tools: while *data scientists and machine learning
   researchers* need the quick iteration cycles that scripting languages like Python
   provide, *real-world compression codecs* that are to be used outside of laboratory
   conditions have to be implemented in a compiled language that runs fast and that doesn't
   require setting up a complex runtime environment with lots of dependencies. With
   `constriction`, you can seamlessly turn your Python research code into a high-performance
   standalone binary, library, or WebAssembly module. By default, the Python and Rust API
   are binary compatible, so you can gradually port one component at a time without breaking
   things. On top of this, the Rust API provides optional fine-grained control over issues
   relevant to real-world deployments such as the trade-off between compression
   effectiveness, memory usage, and run-time efficiency, as well as hooks into the backing
   data sources and sinks, while preventing accidental misuse through Rust's powerful type
   system.
3. **to serve as a teaching resource** by providing a collection of several complementary
   entropy coding algorithms within a single consistent framework, thus making the various
   algorithms easily discoverable and comparable on practical examples; [additional teaching
   material](https://robamler.github.io/teaching/compress21) is being made publicly
   available as a by-product of an ongoing university course on data compression with deep
   probabilistic models.

For an example of a compression codec that started as research code in Python and was then
deployed as a fast and dependency-free WebAssembly module using `constriction`'s Rust API,
have a look at [The Linguistic Flux
Capacitor](https://robamler.github.io/linguistic-flux-capacitor).

## Project Status

We currently provide implementations of the following entropy coding algorithms:

- **Asymmetric Numeral Systems (ANS):** a fast modern entropy coder with near-optimal
  compression effectiveness that supports advanced use cases like bits-back coding;
- **Range Coding:** a computationally efficient variant of Arithmetic Coding, that has
  essentially the same compression effectiveness as ANS Coding but operates as a queue
  ("first in first out"), which makes it preferable for autoregressive models.
- **Chain Coding:** an experimental new entropy coder that combines the (net) effectiveness
  of stream codes with the locality of symbol codes; it is meant for experimental new
  compression approaches that perform joint inference, quantization, and bits-back coding in
  an end-to-end optimization. This experimental coder is mainly provided to prove to
  ourselves that the API for encoding and decoding, which is shared across all stream
  coders, is flexible enough to express complex novel tasks.
- **Huffman Coding:** a well-known symbol code, mainly provided here for teaching purpose;
  you'll usually want to use a stream code like ANS or Range Coding instead since symbol
  codes can have a considerable overhead on the bitrate, especially in the regime of low
  entropy per symbol, which is common in machine-learning based compression methods.

Further, `constriction` provides implementations of common probability distributions in
fixed-point arithmetic, which can be used as entropy models in either of the above stream
codes. The library also provides adapters for turning custom probability distributions into
exactly invertible fixed-point arithmetic.

The provided implementations of entropy coding algorithms and probability distributions are
extensively tested and should be considered reliable (except for the still experimental
Chain Coder). However, their APIs may change in future versions of `constriction` if more
user experience reveals any shortcomings of the current APIs in terms of ergonomics. Please
[file an issue](https://github.com/bamler-lab/constriction/issues) if you run into a
scenario where the current APIs are suboptimal.

## Quick Start Guides And Examples in Python and Rust

### Python

The easiest way to install `constriction` for Python is via `pip` (the following command
also installs `scipy`, which is not required but useful if you want to use `constriction`
with custom probability distributions):

```bash
pip install constriction numpy scipy
```

Then go ahead and use it:

```python
import constriction
import numpy as np

# Let's use a Range Coder in this example. Constriction also provides an ANS 
# Coder, a Huffman Coder, and an experimental new "Chain Coder".
encoder = constriction.stream.queue.RangeEncoder()

# Define some data and a sequence of entropy models. We use quantized Gaussians
# here, but you could also use other models or even provide your own.
min_supported_symbol, max_supported_symbol = -100, 100
symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)

# Encode the symbols and get the compressed data.
encoder.encode_leaky_gaussian_symbols(
    symbols, min_supported_symbol, max_supported_symbol, means, stds)
compressed = encoder.get_compressed()
print(compressed)

# Create a decoder and recover the original symbols.
decoder = constriction.stream.queue.RangeDecoder(compressed)
reconstructed = decoder1.decode_leaky_gaussian_symbols(
    min_supported_symbol, max_supported_symbol, means, stds)
assert np.all(reconstructed == symbols)
```

There's a lot more you can do with `constriction`'s Python API. Please check out the [Python
API Documentation](https://bamler-lab.github.io/constriction/apidoc/python/).

### Rust

Add this line to your `Cargo.toml`:

```toml
constriction = "0.1"
probability = "0.17" # Not strictly required but useful for defining quantized entropy models.
```

Then go ahead and use it:

```rust
use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode};

// Let's use an ANS Coder in this example. Constriction also provides a Range
// Coder, a Huffman Coder, and an experimental new "Chain Coder".
let mut coder = DefaultAnsCoder::new();
 
// Define some data and a sequence of entropy models. We use quantized Gaussians here,
// but `constriction` also provides other models and allows you to implement your own.
let symbols = vec![23i32, -15, 78, 43, -69];
let quantizer = DefaultLeakyQuantizer::new(-100..=100);
let means = vec![35.2f64, -1.7, 30.1, 71.2, -75.1];
let stds = vec![10.1f64, 25.3, 23.8, 35.4, 3.9];
let models = means.iter().zip(&stds).map(
    |(&mean, &std)| quantizer.quantize(probability::distribution::Gaussian::new(mean, std))
);

// Encode symbols (in *reverse* order, because ANS Coding operates as a stack).
coder.encode_symbols_reverse(symbols.iter().zip(models.clone())).unwrap();

// Obtain temporary shared access to the compressed bit string. If you want ownership of the
// compressed bit string, call `.into_compressed()` instead of `.get_compressed()`.
println!("Encoded into {} bits: {:?}", coder.num_bits(), &*coder.get_compressed().unwrap());

// Decode the symbols and verify correctness.
let reconstructed = coder.decode_symbols(models).collect::<Result<Vec<_>, _>>().unwrap();
assert_eq!(reconstructed, symbols);
```

There's a lot more you can do with `constriction`'s Rust API. Please check out the [Rust API
Documentation](https://docs.rs/constriction).

## Compiling From Source

Users of `constriction` typically don't need to manually compile the library from source.
Just install `constriction` via `pip` or `cargo` as described in the above [quick start guides](#quick-start-guides-and-examples-in-python-and-rust).

Contributors can compile `constriction` manually as follows:

1. Prepare your system:
   - If you don't have a Rust toolchain, install one as described on <https://rustup.rs>
   - If you already have a Rust toolchain, make sure it's on version 1.51 or later. Run
     `rustc --version` to find out and `rustup update stable` if you need to update.
2. `git clone` the repository and `cd` into it.
3. To compile the Rust library:
   - compile in development mode and execute all tests: `cargo test`
   - compile in release mode (i.e., with optimizations) and run the benchmarks: `cargo
     bench`
4. If you want to compile the Python module:
   - install [poetry](https://python-poetry.org/).
   - install Python dependencies: `cd` into the repository and run `poetry install`
   - build the Python module: `poetry run maturin develop '--cargo-extra-args=--features
     pybindings'`
   - run Python unit tests: `poetry run pytest tests/python`
   - start a Python REPL that sees the compiled Python module: `poetry run ipython`

## Contributing

Pull requests and issue reports are welcome. Unless contributors explicitly state otherwise
at the time of contributing, all contributions will be assumed to be licensed under either
one of MIT license, Apache License Version 2.0, or Boost Software License Version 1.0, at
the choice of each licensee.

There's no official guide for contributions since nobody reads those anyway. Just be nice to
other people and act like a grown-up (i.e., it's OK to make mistakes as long as you strive
for improvement and are open to respectfully phrased opinions of other people).

## License

This work is licensed under the terms of the MIT license, Apache License Version 2.0, or
Boost Software License Version 1.0. You can choose between one of them if you use this work.
See the files whose name start with `LICENSE` in this directory. The compiled python
extension module is linked with a number of third party libraries. Binary distributions of
the `constriction` python extension module contain a file `LICENSE.html` that includes all
licenses of all dependencies (the file is also available
[online](https://bamler-lab.github.io/constriction/license.html)).

## What's With the Name?

Constriction is a library of compression primitives with bindings for Rust and
[Python](https://en.wikipedia.org/wiki/Python_(programming_language)).
[Pythons](https://en.wikipedia.org/wiki/Pythonidae) are a family of nonvenomous snakes that
subdue their prey by "compressing" it, a method known as
[constriction](https://en.wikipedia.org/wiki/Constriction).
