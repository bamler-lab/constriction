# Entropy Coding Primitives for Research and Production

The `constriction` library provides a set of composable implementations of entropy coding
algorithms. It has APIs for both the **Python and Rust** languages and it focuses on
versatility, ease of use, compression performance, and computational efficiency. The goals
of `constriction` are to three-fold:

1. **to facilitate research on novel lossless and lossy compression methods** by bridging
   the gap between vastly different software stacks commonly used for machine learning
   research vs real-world compression codecs; machine learning researchers will likely want
   to start using `constriction` through its Python/numpy API. It exposes both highly
   optimized entropy coders that are easy to use in common cases as well as a composable set
   of more low-level primitives that allow researchers to come up with more specialized
   variants of existing source coding algorithms (e.g., adapters for making custom defined
   probability distributions *exactly* invertable in fixed-point arithmetic; TODO).
2. **to simplify the transition from research code to production software** by exposing a
   superset of the exact same algorithms that the Python API provides also as a Rust crate;
   if your research lead to a successful prototype of a new compression method then you can
   use `constriction`'s Rust API to turn your Python research code into a small and highly
   optimized standalone (statically linked) program, library, or WebAssembly module that
   runs efficiently and that can be used by customers who don't want to deal with Python's
   dependency hell. By default, the Rust and Python APIs are binary compatible, so you can,
   e.g., continue to compress data with your Python research code while decompressing it
   with your optimized Rust deployment or vice-versa. For deployments with tighter resource
   constraints, the Rust API provides optional fine-grained control over the trade-off
   between compression effectiveness, memory usage, and run-time efficency, as well as hooks
   into the backing data sources and sinks, while preventing accidental misuse through
   Rust's powerful type system.
3. **to serve as a teaching resource** by providing a collection of several different
   entropy coding algorithms within a single consistent framework, thus making the various
   algorithms easily discoverable and comparable on practical examples; [additional teaching
   material](https://robamler.github.io/teaching/compress21) will be made publicly available
   as a by-product of an upcoming university course on data compression with deep
   probabilistic models.

For an example of a compression codec that started as research code in Python and was then
deployed as a dependency-free WebAssembly module using `constriction`'s Rust API, have a
look at [The Linguistic Flux
Capacitor](https://robamler.github.io/linguistic-flux-capacitor).

## Project Status

We currently provide implementations of the following entropy coding algorithms:

- **Asymmetric Numeral Systems (ANS):** a highly efficient modern entropy coder with
  near-optimal compression effectiveness; ANS Coding operates as a stack ("last in first
  out") and is surjective, which enables advanced use cases like bits-back coding in
  hierarchical entropy models.
- **Range Coding:** a computationally efficent variant of Arithmetic Coding; it has
  essentially the same compression effectiveness as ANS Coding. When highly optimized
  entropy models are used, Range Coding can be a bit faster than ANS Coding for encoding but
  may be considerably slower for decoding. More importantly, Range Coding operates as a
  queue ("first in first out"), which makes it preferable for autoregressive entropy models.
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
codes. The library also proides adapters for turning custom probability distributions into
exactly invertible fixed-point arithmetic.

The provided implementations of entropy coding algorithms and probability distributions are
extensively tested and should be considered reliable (except for the still experimental
Chain Coder). However, their APIs may change in future versions of `constriction` if more
user experience reveals any shortcomings of the current APIs in terms of ergonomics. Please
[file an issue](https://github.com/bamler-lab/constriction/issues) if you run into a
scenario where the current APIs are suboptimal.

## Quick Start Guides And Examples in Python and Rust

TODO: actually provide a very short example in each language, then link to API
documentation.

See the [Python API Documentation](https://bamler-lab.github.io/constriction/apidoc/python/)
and the [Rust API Documentation](https://bamler-lab.github.io/constriction/apidoc/rust/),
respectively.

## Compiling From Source

Users of `constriction` typically don't need to manually compile the library from source.
Just install `constriction` via `pip` or `cargo` as described in the quick start guides of
the [Python API Documentation](https://bamler-lab.github.io/constriction/apidoc/python/) or
the [Rust API Documentation](https://bamler-lab.github.io/constriction/apidoc/rust/),
respectively. (TODO: replace by: as shown in the above quick start guides)

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

This work is licensed under the terms of the MIT license, Apache License Version 2.0, or
Boost Software License Version 1.0. You can choose between one of them if you use this work.
See the files whose name start with `LICENSE` in this directory. The compiled python
extension module is linked with a number of third party libraries. Binary distributions of
the `constriction` python extension module contain a list of all the license terms of its
dependencies (TODO: verify).

## What's With the Name?

Constriction is a library of compression primitives with bindings for Rust and
[Python](https://en.wikipedia.org/wiki/Python_(programming_language)).
[Pythons](https://en.wikipedia.org/wiki/Pythonidae) are a family of nonvenomous snakes that
subdue their prey by "compressing" it, a method known as
[constriction](https://en.wikipedia.org/wiki/Constriction).
