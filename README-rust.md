# Entropy Coders for Research and Production

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

## Quick Start

Add the following to your `Cargo.toml`:

```toml
[dependencies]
constriction = "0.3.0"
probability = "0.20.3" # Not strictly required but used in many examples.
```

### Encoding Example

In this example, we'll encode some symbols using a quantized Gaussian distribution as
entropy model. Each symbol will be modeled by a quantized Gaussian with a different
mean and standard deviation (so that the example is not too simplistic). We'll use the
`probability` crate for the Gaussian distributions, so make sure you have the following
dependency in your `Cargo.toml`:

```toml
probability = "0.17"
```

Now, let's encode (i.e., compress) some symbols. We'll use an Asymmetric Numeral Systems
(ANS) Coder here for its speed and compression performance. We'll discuss how you could
replace the ANS Coder with a Range Coder or a symbol code like Huffman Coding
[below](#user-content-exercise).

```rust
use constriction::stream::{stack::DefaultAnsCoder, model::DefaultLeakyQuantizer};
use probability::distribution::Gaussian;

fn encode_sample_data() -> Vec<u32> {
    // Create an empty ANS Coder with default word and state size:
    let mut coder = DefaultAnsCoder::new();

    // Some made up data and entropy models for demonstration purpose:
    let symbols = [23i32, -15, 78, 43, -69];
    let means = [35.2, -1.7, 30.1, 71.2, -75.1];
    let stds = [10.1, 25.3, 23.8, 35.4, 3.9];

    // Create an adapter that integrates 1-d probability density functions over bins
    // `[n - 0.5, n + 0.5)` for all integers `n` from `-100` to `100` using fixed point
    // arithmetic with default precision, guaranteeing a nonzero probability for each bin:
    let quantizer = DefaultLeakyQuantizer::new(-100..=100);

    // Encode the data (in reverse order, since ANS Coding operates as a stack):
    coder.encode_symbols_reverse(
        symbols.iter().zip(&means).zip(&stds).map(
            |((&sym, &mean), &std)| (sym, quantizer.quantize(Gaussian::new(mean, std)))
    )).unwrap();

    // Retrieve the compressed representation (filling it up to full words with zero bits).
    coder.into_compressed().unwrap()
}

assert_eq!(encode_sample_data(), [0x421C_7EC3, 0x000B_8ED1]);
```

### Decoding Example

Now, let's reconstruct the sample data from its compressed representation.

```rust
use constriction::stream::{stack::DefaultAnsCoder, model::DefaultLeakyQuantizer, Decode};
use probability::distribution::Gaussian;

fn decode_sample_data(compressed: Vec<u32>) -> Vec<i32> {
    // Create an ANS Coder with default word and state size from the compressed data:
    // (ANS uses the same type for encoding and decoding, which makes the method very flexible
    // and allows interleaving small encoding and decoding chunks, e.g., for bits-back coding.)
    let mut coder = DefaultAnsCoder::from_compressed(compressed).unwrap();

    // Same entropy models and quantizer we used for encoding:
    let means = [35.2, -1.7, 30.1, 71.2, -75.1];
    let stds = [10.1, 25.3, 23.8, 35.4, 3.9];
    let quantizer = DefaultLeakyQuantizer::new(-100..=100);

    // Decode the data:
    coder.decode_symbols(
        means.iter().zip(&stds).map(
            |(&mean, &std)| quantizer.quantize(Gaussian::new(mean, std))
    )).collect::<Result<Vec<_>, _>>().unwrap()
}

assert_eq!(decode_sample_data(vec![0x421C_7EC3, 0x000B_8ED1]), [23, -15, 78, 43, -69]);
```

## Exercise

Try out the above examples and verify that decoding reconstructs the original data. Then
see how easy `constriction` makes it to replace the ANS Coder with a Range Coder by
making the following substitutions:

**In the encoder,**

- replace `constriction::stream::stack::DefaultAnsCoder` with
  `constriction::stream::queue::DefaultRangeEncoder`; and
- replace `coder.encode_symbols_reverse` with `coder.encode_symbols` (you no longer need
  to encode symbols in reverse order since Range Coding operates as a queue, i.e.,
  first-in-first-out). You'll also have to add the line
  `use constriction::stream::Encode;` to the top of the file to bring the trait method
  `encode_symbols` into scope.

**In the decoder,**

- replace `constriction::stream::stack::DefaultAnsCoder` with
  `constriction::stream::queue::DefaultRangeDecoder` (note that Range Coding
  distinguishes between an encoder and a decoder type since the encoder writes to the
  back while the decoder reads from the front; by contrast, ANS Coding is a stack, i.e.,
  it reads and writes at the same position and allows interleaving reads and writes).

*Remark:* You could also use a symbol code like Huffman Coding (see module [`symbol`])
but that would have considerably worse compression performance, especially on large
files, since symbol codes always emit an integer number of bits per compressed symbol,
even if the information content of the symbol is a fractional number (stream codes like
ANS and Range Coding *effectively* emit a fractional number of bits per symbol since
they amortize over several symbols).

The above replacements should lead you to something like this:

```rust
use constriction::stream::{
    model::DefaultLeakyQuantizer,
    queue::{DefaultRangeEncoder, DefaultRangeDecoder},
    Encode, Decode,
};
use probability::distribution::Gaussian;

fn encode_sample_data() -> Vec<u32> {
    // Create an empty Range Encoder with default word and state size:
    let mut encoder = DefaultRangeEncoder::new();

    // Same made up data, entropy models, and quantizer as in the ANS Coding example above:
    let symbols = [23i32, -15, 78, 43, -69];
    let means = [35.2, -1.7, 30.1, 71.2, -75.1];
    let stds = [10.1, 25.3, 23.8, 35.4, 3.9];
    let quantizer = DefaultLeakyQuantizer::new(-100..=100);

    // Encode the data (this time in normal order, since Range Coding is a queue):
    encoder.encode_symbols(
        symbols.iter().zip(&means).zip(&stds).map(
            |((&sym, &mean), &std)| (sym, quantizer.quantize(Gaussian::new(mean, std)))
    )).unwrap();

    // Retrieve the (sealed up) compressed representation.
    encoder.into_compressed().unwrap()
}

fn decode_sample_data(compressed: Vec<u32>) -> Vec<i32> {
    // Create a Range Decoder with default word and state size from the compressed data:
    let mut decoder = DefaultRangeDecoder::from_compressed(compressed).unwrap();

    // Same entropy models and quantizer we used for encoding:
    let means = [35.2, -1.7, 30.1, 71.2, -75.1];
    let stds = [10.1, 25.3, 23.8, 35.4, 3.9];
    let quantizer = DefaultLeakyQuantizer::new(-100..=100);

    // Decode the data:
    decoder.decode_symbols(
        means.iter().zip(&stds).map(
            |(&mean, &std)| quantizer.quantize(Gaussian::new(mean, std))
    )).collect::<Result<Vec<_>, _>>().unwrap()
}

let compressed = encode_sample_data();

// We'll get a different compressed representation than in the ANS Coding
// example because we're using a different entropy coding algorithm ...
assert_eq!(compressed, [0x1C31EFEB, 0x87B430DA]);

// ... but as long as we decode with the matching algorithm we can still reconstruct the data:
assert_eq!(decode_sample_data(compressed), [23, -15, 78, 43, -69]);
```

## Where to Go Next?

If you already have an entropy model and you just want to encode and decode some
sequence of symbols then you can probably start by adjusting the above
examples to your needs. Or have a closer look at the `stream` module.

More examples and tutorials are linked from the [project website](https://bamler-lab.github.io/constriction).

If you're still new to the concept of entropy coding then check out the [teaching
material](https://robamler.github.io/teaching/compress21/).

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
