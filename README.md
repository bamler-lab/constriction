# Entropy Coding Primitives for Research and Production

The `constriction` library provides a set of composable entropy coding algorithms for both Rust and Python with a
focus on ease of use, flexibility, compression performance, and computational
efficiency. The goals of `constriction` are to three-fold:

1. **to facilitate research on novel lossless and lossy compression methods** by
   bridging the gap between declarative tools for data modeling from the machine
   learning community and imperative tools for algorithm design from the source coding
   literature;
2. **to simplify the transition from research code to production software** by providing
   both an easy-to-use Python API (for rapid iteration on research code) and a highly
   generic Rust API (for turning research code into optimized standalone binary programs or
   libraries with minimal dependencies); and
3. **to serve as a teaching resource** by providing a wide range of entropy coding
   algorithms within a single consistent framework, thus making the various algorithms
   easily discoverable and comparable on example models and data. [Additional teaching
   material](https://robamler.github.io/teaching/compress21/) will be made publicly
   available as a by-product of an upcoming university course on data compression with deep
   probabilistic models.

## Currently Supported Entropy Coding Algorithms

The `constriction` library currently supports the following algorithms:

- **Asymmetric Numeral Systems (ANS):** a highly efficient modern entropy coder with
  near-optimal compression performance that supports advanced use cases like bits-back
  coding;
  - A "split" variant of ANS coding is provided for advanced use cases in hierarchical
    models with cyclic dependencies.
- **Range Coding:** a variant of Arithmetic Coding that is optimized for realistic computing
  hardware; it has similar compression performance and almost the same computational
  performance as ANS. The main practical difference is that Range Coding is a queue (FIFO)
  while ANS is a stack (LIFO).
- **Huffman Coding:** a well-known symbol code, mainly provided here for teaching purpose;
  you'll usually want to use a stream code like ANS or Range Coding instead.

## Python API vs Rust API

The `constriction` library provides both a [Python API](TODO) and a [Rust API](TODO). This
provides the following advantages to users:

- **The Python API** allows data scientists to quickly iterate and experiment with new entropy
  models using their favorite Python packages for machine learning. The `constriction`
  Python package is implemented as a native extension module that is basically a thin
  wrapper around the Rust implementation, thus making it much faster than any pure Python
  implementation of the same algorithms could perform.
- **The Rust API** makes it easy to turn research code into efficient standalone software
  products with minimal dependencies (binaries, system libraries, or WebAssembly modules).
  You don't need to find a new library for entropy coding; you only have to port the entropy
  models and any orchestration code from Python to Rust (interfaces to other compiled
  languages with a compatible ABI like C or C++ are also possible albeit more difficult). To
  make the transition from research to production code seamless, the Rust API by default
  uses the exact identical settings for all entropy coding algorithms as the Python API, so
  your Python and Rust code are compatible with each other, i.e., and each one can decode
  compressed data encoded by the respective other.
- Some users need more **fine grained control** over details of the compression algorithms
  (such as word size, numeric precision, or rounding strategies). The Rust API makes it
  possible to deviate from the default settings in flexible ways while preventing erroneous
  usage and without sacrificing computational efficiency by exploiting Rust's powerful type
  system. You can use these advanced features to optimize production code for a specific
  target platform, or to expose specialized functionality to Python by copying parts of the provided Python bindings and modifying them slightly.

## Quick Starts And Examples in Python and Rust

See the [Python API documentation](TODO) and the [Rust API documentation](TODO), respectively.

## Compiling From Source

TODO

## Contributing

Pull requests and issue reports are welcome. TODO: mention license. There's no official
guide for contributions since nobody reads those anyway. Just be nice to each other and act
like a grown-up.

## License

TODO

## What's With the Name?

[Constriction](https://en.wikipedia.org/wiki/Constriction) is a method by which some snakes
subdue their prey through "compression". A well-known family of constriction snakes are
[pythons](https://en.wikipedia.org/wiki/Pythonidae).
