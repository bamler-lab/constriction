# Entropy Coding Primitives for Research and Production

The `constriction` library provides a set of composable implementations of entropy coding
algorithms with APIs for both the Python and Rust languages and with a focus on ease of use,
flexibility, compression performance, and computational efficiency. The goals of
`constriction` are to three-fold:

1. **to facilitate research on novel lossless and lossy compression methods** by bridging
   the gap between popular machine-learning software tools for the (declarative) modeling of
   data sources, and source-coding algorithms for the (imperative) task of entropy coding
   with a given model;
2. **to simplify the transition from research code to production software** by exposing two
   interoperable APIs: an easy-to-use and opinionated Python API (for rapid iteration on
   research code) and a highly flexible Rust API (for turning research code into optimized
   standalone binary programs or libraries with minimal dependencies); and
3. **to serve as a teaching resource** by providing a collection of entropy coding
   algorithms within a single consistent framework, thus making the various algorithms
   easily discoverable and comparable on practical examples. [Additional teaching
   material](https://robamler.github.io/teaching/compress21/) will be made publicly
   available as a by-product of an upcoming university course on data compression with deep
   probabilistic models.

## Project Status

The `constriction` library currently provides solid implementations of the following entropy
coding algorithms:

- **Asymmetric Numeral Systems (ANS):** a highly efficient modern entropy coder with
  near-optimal compression performance that supports advanced use cases like bits-back
  coding;
  - A "split" variant of ANS Coding is provided for advanced use cases in hierarchical
    models with cyclic dependencies.
- **Range Coding:** a variant of Arithmetic Coding that is optimized for realistic computing
  hardware; it has similar compression performance and almost the same computational
  performance as ANS Coding. The main practical difference is that Range Coding operates as
  a queue (first-in-first-out, which is useful for autoregressive entropy models) while ANS
  Coding operates as a stack (last-in-first-out, which is useful for hierarchical models).
- **Huffman Coding:** a well-known symbol code, mainly provided here for teaching purpose;
  you'll usually want to use a stream code like ANS or Range Coding instead.

Further, `constriction` provides implementations of common probability distributions in
fixed-point arithmetic, which can be used as entropy models for all of the above entropy
coding algorithms.

The provided implementations of entropy coding algorithms and probability distributions are
extensively tested and should be considered reliable. However, their APIs may change in
future versions of `constriction` if more user experience reveals any unnecessary
restrictions or repetitiveness of the current APIs. Please [file an
issue](https://github.com/bamler-lab/constriction/issues) if you run into a scenario where
the current APIs are suboptimal.

## Quick Start Guides And Examples in Python and Rust

See the [Python API documentation](https://bamler-lab.github.io/constriction/apidoc/python/)
and the [Rust API documentation](https://bamler-lab.github.io/constriction/apidoc/rust/),
respectively.

## Python API vs Rust API

The `constriction` library provides both a [Python
API](https://bamler-lab.github.io/constriction/apidoc/python/) and a [Rust
API](https://bamler-lab.github.io/constriction/apidoc/rust/). Use `constriction` in
whichever of the two languages you feel more comfortable with. Exposing the same
functionality to two such different languages has the following advantages for users:

- **The Python API** allows data scientists to quickly iterate and experiment with new
  entropy models using their favorite Python packages for machine learning. The
  `constriction` Python package is implemented as a native extension module that is
  basically a thin wrapper around the Rust implementation, thus making it much faster than
  any pure Python implementation of the same algorithms could be.
- **The Rust API** makes it easy to turn research code into efficient standalone software
  products with minimal dependencies (binaries, system libraries, or WebAssembly modules).
  To make the transition from research to production seamless, the default settings for all
  entropy coding algorithms in the Rust API are exactly identical to those exposed by the
  Python API. Thus, your (Python) research code and your (Rust) production code remain
  compatible with each other by default, i.e., each one can decode compressed data encoded
  by the respective other.
- Advanced users may need more **fine grained control** over details of the entropy coding
  algorithms (such as word size, numeric precision, or rounding strategies). The Rust API
  makes it possible to adjust these details, thus deviating from the default settings in
  flexible ways while still producing optimized binaries and preventing erroneous usage by
  exploiting Rust's powerful type system. You can use these advanced features to optimize
  production code for a specific target platform, or to expose specialized functionality to
  Python by copying parts of the provided Python bindings and modifying just the generic
  type parameters and const generics.

## Compiling From Source

Users of `constriction` typically don't need to manually compile the library from source.
Just install `constriction` via `pip` or `cargo` as described in the quick start guides of
the [Python API documentation](https://bamler-lab.github.io/constriction/apidoc/python/) or
the [Rust API documentation](https://bamler-lab.github.io/constriction/apidoc/rust/),
respectively.

Contributors can compile `constriction` manually as follows:

1. Prepare your system:
   - If you don't have a Rust toolchain, install one as described on <https://rustup.rs>
   - If you already have a Rust toolchain, make sure it's on version 1.51 or later. Run
     `rustc --version` to find out and `rustup update stable` if you need to update.
2. `git clone` the repository and `cd` into it.
3. Compile the library
   - To compile the Rust version in development mode and run tests, type: `cargo test`
   - To compile the Python module, TODO

## Contributing

Pull requests and issue reports are welcome. Unless contributors explicitly state otherwise
at the time of contributing, all contributions will be assumed to be licensed under either
one of MIT license, Apache License Version 2.0, or Boost Software License Version 1.0, at
the choice of each licensee.

There's no official guide for contributions since nobody reads those anyway. Just be nice to
other people and act like a grown-up (i.e., it's OK to make mistakes as long as you strive
for improvement and are open to respectfully phrased opinions of other people).

## License

TODO (use `cargo-deny`)

- This repository's source code: MIT OR Apache-2.0 OR BSL
- Dependencies may be more specific.

## What's With the Name?

Constriction is

- a Rust and [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) library
  of compression primitives; and
- a method by which [pythons](https://en.wikipedia.org/wiki/Pythonidae) subdue their prey by
  "compressing" it.
