# Entropy Coders for Research and Production

![test status](https://github.com/bamler-lab/constriction/workflows/test/badge.svg?branch=main)

The `constriction` library provides a set of composable entropy coding algorithms with a
focus on correctness, versatility, ease of use, compression performance, and
computational efficiency. The goals of `constriction` are three-fold:

1. **to facilitate research on novel lossless and lossy compression methods** by
   providing a *composable* set of primitives (e.g., you can can easily switch out a
   Range Coder for an ANS coder without having to find a new library or change how you
   represent exactly invertible entropy models);
2. **to simplify the transition from research code to deployed software** by providing
   similar APIs and binary compatible entropy coders for both Python (for rapid
   prototyping on research code) and Rust (for turning successful prototypes into
   standalone binaries, libraries, or WebAssembly modules); and
3. **to serve as a teaching resource** by providing a variety of entropy coding
   primitives within a single consistent framework. Check out our [additional teaching
   material](https://robamler.github.io/teaching/compress21/) from a university course
   on data compression, which contains some problem sets where you use `constriction`
   (with solutions).

**More Information:** [project website](https://bamler-lab.github.io/constriction)

**Live demo:** [here's a web app](https://robamler.github.io/linguistic-flux-capacitor)
that started out as a machine-learning research project in Python and was later turned
into a web app by using `constriction` in a WebAssembly module).

## Project Status

We currently provide implementations of the following entropy coding algorithms:

- **Asymmetric Numeral Systems (ANS):** a fast modern entropy coder with near-optimal
  compression effectiveness that supports advanced use cases like bits-back coding.
- **Range Coding:** a computationally efficient variant of Arithmetic Coding, that has
  essentially the same compression effectiveness as ANS Coding but operates as a queue
  ("first in first out"), which makes it preferable for autoregressive models.
- **Chain Coding:** an experimental new entropy coder that combines the (net) effectiveness
  of stream codes with the locality of symbol codes (for details, see Section&nbsp;4.3 in
  [this paper](https://arxiv.org/pdf/2201.01741)]); it admits experimental new compression
  techniques that perform joint inference, quantization, and bits-back coding in an
  end-to-end optimization. This experimental coder is mainly provided to prove to ourselves
  that the API for encoding and decoding, which is shared across all stream coders, is
  flexible enough to express complex novel tasks.
- **Huffman Coding:** a well-known symbol code, mainly provided here for teaching purpose;
  you'll usually want to use a stream code like ANS or Range Coding instead since symbol
  codes can have a considerable overhead on the bitrate, especially in the regime of low
  entropy per symbol, which is common in machine-learning based compression methods.

Further, `constriction` provides implementations of common probability distributions in
fixed-point arithmetic, which can be used as entropy models in either of the above stream
codes. The library also provides adapters for turning custom probability distributions into
exactly invertible fixed-point arithmetic.

The provided implementations of entropy coding algorithms and probability distributions are
continuously and extensively tested. We consider updates that can affect the encoder or
decoder output in existing code as breaking changes that necessitate a bump in the leading
nonzero number of the version string (this is a stronger guarantee than SemVer in that we
apply it even to 0.y.z versions). Please [file an
issue](https://github.com/bamler-lab/constriction/issues) if you find a bug, are missing a
particular feature, or run into a scenario where the current APIs are confusing or
unnecessarily limit what you can achieve with `constriction`.

## Quick Start Guides And Examples in Python and Rust

### Python

Install `constriction` for Python:

```bash
pip install constriction~=0.2.2
```

Then go ahead and encode and decode some data:

```python
import constriction
import numpy as np

message = np.array([6, 10, -4, 2, 5, 2, 1, 0, 2], dtype=np.int32)

# Define an i.i.d. entropy model (see links below for more complex models):
entropy_model = constriction.stream.model.QuantizedGaussian(-50, 50, 3.2, 9.6)

# Let's use an ANS coder in this example (see links below for Range Coding examples).
encoder = constriction.stream.stack.AnsCoder()
encoder.encode_reverse(message, entropy_model)

compressed = encoder.get_compressed()
print(f"compressed representation: {compressed}")
print(f"(in binary: {[bin(word) for word in compressed]})")

decoder = constriction.stream.stack.AnsCoder(compressed)
decoded = decoder.decode(entropy_model, 9) # (decodes 9 symbols)
assert np.all(decoded == message) # (verifies correctness)
```

There's a lot more you can do with `constriction`'s Python API. Please check out the [Python
API Documentation](https://bamler-lab.github.io/constriction/apidoc/python/) or our [example
jupyter notebooks](https://github.com/bamler-lab/constriction/tree/main/examples/python).

### Rust

Add this line to your `Cargo.toml`:

```toml
[dependencies]
constriction = "0.2.2"
probability = "0.17" # Not strictly required but used in many code examples.
```

If you compile in `no_std` mode then you have to deactivate `constriction`'s default
features (and you can't use the `probability` crate):

```toml
[dependencies]
constriction = {version = "0.1.2", default-features = false} # for `no_std` mode
```

Then go ahead and encode and decode some data:

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

There's a lot more you can do with `constriction`'s Rust API. d check out the [Rust API
Documentation](https://docs.rs/constriction).

## Citing

I'd appreciate attribution if you use constriction in your scientific work. You can cite the
following paper, which announces `constriction` (Section&nbsp;5.1) and analyzes its
compression performance and runtime efficiency (Section&nbsp;5.2):

- R. Bamler, Understanding Entropy Coding With Asymmetric Numeral Systems (ANS): a
  Statistician's Perspective, arXiv preprint
  [arXiv:2201.01741](https://arxiv.org/pdf/2201.01741).

**BibTex:**

```bibtex
@article{bamler2022constriction,
  title   = {Understanding Entropy Coding With Asymmetric Numeral Systems (ANS): a Statistician's Perspective},
  author  = {Bamler, Robert},
  journal = {arXiv preprint arXiv:2201.01741},
  year    = {2022}
}
```

## Compiling From Source

Users of `constriction` typically don't need to manually compile the library from source.
Just install `constriction` via `pip` or `cargo` as described in the above [quick start
guides](#quick-start-guides-and-examples-in-python-and-rust).

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
for improvement and are open to consider respectfully phrased opinions of other people).

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
