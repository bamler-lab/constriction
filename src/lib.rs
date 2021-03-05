//! Entropy Coding Primitives for Research and Production
//!
//! The `constriction` crate provides a set of composable entropy coding algorithms with a
//! focus on ease of use, flexibility, compression performance, and computational
//! efficiency. The goals of `constriction` are to three-fold:
//!
//! 1. **to facilitate research on novel lossless and lossy compression methods** by
//!    bridging the gap between declarative tools for data modeling from the machine
//!    learning community and imperative tools for algorithm design from the source coding
//!    literature;
//! 2. **to simplify the transition from research code to production software** by providing
//!    both an easy-to-use Python API (for rapid iteration on research code) and a highly
//!    generic Rust API (for turning research code into optimized standalone binary programs
//!    or libraries with minimal dependencies); and
//! 3. **to serve as a teaching resource** by providing a wide range of entropy coding
//!    algorithms within a single consistent framework, thus making the various algorithms
//!    easily discoverable and comparable on example models and data. [Additional teaching
//!    material](https://robamler.github.io/teaching/compress21/) will be made publicly
//!    available as a by-product of an upcoming university course on data compression with
//!    deep probabilistic models.
//!
//! # Currently Supported Entropy Coding Algorithms
//!
//! The `constriction` crate currently supports the following algorithms:
//!
//! - **Asymmetric Numeral Systems (ANS):** a highly efficient modern entropy coder with
//!   near-optimal compression performance that supports advanced use cases like bits-back
//!   coding;
//!   - A "split" variant of ANS coding is provided for advanced use cases in hierarchical
//!     models with cyclic dependencies.
//! - **Range Coding:** a variant of Arithmetic Coding that is optimized for realistic
//!   computing hardware; it has similar compression performance and almost the same
//!   computational performance as ANS. The main practical difference is that Range Coding
//!   is a queue (FIFO) while ANS is a stack (LIFO).
//! - **Huffman Coding:** a well-known symbol code, mainly provided here for teaching
//!   purpose; you'll usually want to use a stream code like ANS or Range Coding instead.
//!
//! # Quick Start With the Rust API
//!
//! You are currently reading the documentation of `constriction`'s Rust API. If Rust is not
//! your language of choice then head over to the [Python API
//! Documentation](https://bamler-lab.github.io/constriction/apidoc/python/). The Rust API
//! provides efficient and composable entropy coding primitives that can be adjusted to a
//! fine degree of detail using type parameters and const generics (type aliases with sane
//! defaults for all generic parameters are provided as a guidance). The Python API exposes
//! the most common use cases of these entropy coding primitives to an environment that
//! feels more natural to many data scientists.
//!
//! To use `constriction` in your Rust project, just add the following line to the
//! `[dependencies]` section of your `Cargo.toml`:
//!
//! ```toml
//! constriction = "0.2"
//! ```
//!
//! ## System Requirements
//!
//! `constriction` requires Rust version 1.51 or later for its use of the
//! `min_const_generics` feature. If you have an older version of Rust, update to the latest
//! version by running `rustup update stable`.
//!
//! ## Encoding Example
//!
//! In this example, we'll encode some symbols using a quantized Gaussian distribution as
//! entropy model, with a different mean and standard deviation of the Gaussian for each
//! symbol so that the example is not too simplistic. We'll use the `statrs` crate for the
//! Gaussian distributions, so also add the following dependency to your `Cargo.toml`:
//!
//! ```toml
//! statrs = "0.13"
//! ```
//!
//! Now, let's encode (i.e., compress) some symbols. We'll use an Asymmetric Numeral Systems
//! (ANS) coder here for its speed and compression performance. We'll discuss how to replace
//! the ANS coder with a range coder or a symbol code like Huffman coding
//! [below](#exercise).
//!
//! ```
//! use constriction::stream::{ans::DefaultAnsCoder, models::DefaultLeakyQuantizer};
//! use statrs::distribution::Normal;
//!
//! fn encode_sample_data() -> Vec<u32> {
//!     // Create an empty ANS Coder with default word and state size:
//!     let mut coder = DefaultAnsCoder::new();
//!
//!     // Some made up data and entropy models for demonstration purpose:
//!     let symbols = [23i32, -15, 78, 43, -69];
//!     let means = [35.2, -1.7, 30.1, 71.2, -75.1];
//!     let stds = [10.1, 25.3, 23.8, 35.4, 3.9];
//!
//!     // Create an adapter that maps 1-d probability distributions from floating point to fixed
//!     // point precision, and that guarantees nonzero probabilities on the range `-100..=100`:
//!     let quantizer = DefaultLeakyQuantizer::new(-100..=100);
//!
//!     // Encode the data (in reverse order, since ANS Coding operates as a stack):
//!     coder.encode_symbols_reverse(
//!         symbols.iter().zip(&means).zip(&stds).map(
//!             |((&sym, &mean), &std)| (sym, quantizer.quantize(Normal::new(mean, std).unwrap()))
//!     )).unwrap();
//!
//!     // Retrieve the compressed representation (filling it up to full words with zero bits).
//!     coder.into_compressed()
//! }
//!
//! assert_eq!(encode_sample_data(), [0x421C_7EC3, 0x000B_8ED1]);
//! ```
//!
//! ## Decoding Example
//!
//! Now let's reconstruct the sample data from its compressed representation.
//!
//! ```
//! use constriction::stream::{ans::DefaultAnsCoder, models::DefaultLeakyQuantizer, Decode};
//! use statrs::distribution::Normal;
//!
//! fn decode_sample_data(compressed: Vec<u32>) -> Vec<i32> {
//!     // Create an ANS Coder with default word and state size from the compressed data:
//!     // (ANS uses the same type for encoding and decoding, which makes the method very flexible
//!     // and allows interleaving small encoding and decoding chunks, e.g., for bits-back coding.)
//!     let mut coder = DefaultAnsCoder::from_compressed(compressed).unwrap();
//!
//!     // Same entropy models and quantizer we used for encoding:
//!     let means = [35.2, -1.7, 30.1, 71.2, -75.1];
//!     let stds = [10.1, 25.3, 23.8, 35.4, 3.9];
//!     let quantizer = DefaultLeakyQuantizer::new(-100..=100);
//!
//!     // Encode the data (in reverse order, since ANS is a stack):
//!     coder.decode_symbols(
//!         means.iter().zip(&stds).map(
//!             |(&mean, &std)| quantizer.quantize(Normal::new(mean, std).unwrap())
//!     )).collect::<Result<Vec<_>, _>>().unwrap()
//! }
//!
//! assert_eq!(decode_sample_data(vec![0x421C_7EC3, 0x000B_8ED1]), [23, -15, 78, 43, -69]);
//! ```
//!
//! ## Exercise
//!
//! Try out the above examples and verify that decoding reconstructs the original data. Then
//! see how easy `constriction` makes it to replace the ANS coder with a range coder by
//! making the following substitutions (TODO: verify that this is actually correct)
//!
//! **In the encoder,**
//!
//! - replace `constriction::stream::ans::DefaultAnsCoder` with
//!   `constriction::stream::range::DefaultRangeEncoder`; and
//! - replace `coder.encode_symbols_reverse` with `coder.encode_symbols` (you no longer need
//!   to encode symbols in reverse order since Range Coding operates as a queue, i.e.,
//!   first-in-first-out). You'll also have to add the line `use
//!   constriction::stream::Encode;` to bring the trait method `encode_symbols` into scope.
//!
//! **In the decoder,**
//!
//! - replace `constriction::stream::ans::DefaultAnsCoder` with
//!   `constriction::stream::range::DefaultRangeDecoder` (note that Range Coding
//!   distinguishes between an encoder and a decoder since the encoder writes to the back
//!   while the decoder reads from the front; by contrast, ANS Coding reads and writes at
//!   the same position and allows interleaving reads and writes).
//!
//! You could also use a symbol code like Huffman Coding (see module [`symbol`]) but that
//! would have considerably worse compression performance, especially on large files, since
//! symbol codes always emit an integer number of bits per compressed symbol, even if the
//! information content of the symbol is a fractional number (stream codes like ANS and
//! Range Coding *effectively* emit a fractional number of bits per symbol since they
//! amortize over several symbols).

#![no_std]
#![warn(rust_2018_idioms, missing_debug_implementations)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "pybindings")]
mod pybindings;

pub mod stream;
pub mod symbol;

use core::fmt::{Binary, Debug, LowerHex, UpperHex};

use num::{
    cast::AsPrimitive,
    traits::{WrappingAdd, WrappingSub},
    PrimInt, Unsigned,
};

/// Error type for [`constriction::Coder`]
///
/// [`constriction::Coder`]: struct.Coder.html
#[non_exhaustive]
#[derive(Debug)]
pub enum EncodingError {
    /// Tried to encode a symbol with zero probability under the used entropy model.
    ///
    /// This error can usually be avoided by using a "leaky" distribution, as the
    /// entropy model, i.e., a distribution that assigns a nonzero probability to all
    /// symbols within a finite domain. Leaky distributions can be constructed with,
    /// e.g., a [`LeakyQuantizer`](models/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_floating_point_probabilities`](
    /// models/struct.Categorical.html#method.from_floating_point_probabilities).
    ImpossibleSymbol,

    CapacityExceeded,
}

/// A trait for bit strings of fixed (and usually small) length.
///
/// Short fixed-length bit strings are fundamental building blocks of efficient
/// entropy coding algorithms. They are currently used for the following purposes:
/// - to represent the smallest unit of compressed data (see
///   [`Code::CompressedWord`]);
/// - to represent probabilities in fixed point arithmetic (see
///   [`EntropyModel::Probability`]); and
/// - the internal state of entropy coders (see [`Code::State`]) is typically
///   comprised of one or more `BitArray`s, although this is not a requirement.
///
/// This trait is implemented on all primitive unsigned integer types. There is
/// usually no reason to implement it on custom types since coders will assume, for
/// performance considerations, that `BitArray`s can be represented and manipulated
/// efficiently in hardware.
///
/// # Safety
///
/// This trait is marked `unsafe` so that entropy coders may rely on the assumption
/// that all `BitArray`s have precisely the same behavior as builtin unsigned
/// integer types, and that [`BitArray::BITS`] has the correct value.
pub unsafe trait BitArray:
    PrimInt
    + Unsigned
    + WrappingAdd
    + WrappingSub
    + Debug
    + LowerHex
    + UpperHex
    + Binary
    + Default
    + 'static
{
    /// The (fixed) length of the `BitArray` in bits.
    ///
    /// Defaults to `8 * core::mem::size_of::<Self>()`, which is suitable for all
    /// primitive unsigned integers.
    ///
    /// This could arguably be called `LEN` instead, but that may be confusing since
    /// "lengths" are typically not measured in bits in the Rust ecosystem.
    const BITS: usize = 8 * core::mem::size_of::<Self>();

    #[inline(always)]
    fn wrapping_pow2<const EXPONENT: usize>() -> Self {
        if EXPONENT >= Self::BITS {
            Self::zero()
        } else {
            Self::one() << EXPONENT
        }
    }
}

/// Constructs a `BitArray` from an iterator from most significant to least
/// significant chunks.
///
/// Terminates iteration as soon as either
/// - enough chunks have been read to specify all bits; or
/// - the provided iterator terminates; in this case, the provided chunks are used
///   to set the lower significant end of the return value and any remaining higher
///   significant bits are set to zero.
///
/// This method is the inverse of [`bit_array_to_chunks_truncated`](
/// #method.bit_array_to_chunks_truncated).
fn bit_array_from_chunks<Data, I>(chunks: I) -> Data
where
    Data: BitArray,
    I: IntoIterator,
    I::Item: BitArray + Into<Data>,
{
    let max_count = (Data::BITS + I::Item::BITS - 1) / I::Item::BITS;
    let mut result = Data::zero();
    for chunk in chunks.into_iter().take(max_count) {
        result = (result << I::Item::BITS) | chunk.into();
    }
    result
}

/// Iterates from most significant to least significant bits in chunks but skips any
/// initial zero chunks.
///
/// This method is one possible inverse of [`bit_array_from_chunks`].
fn bit_array_to_chunks_truncated<Data, Chunk>(
    data: Data,
) -> impl Iterator<Item = Chunk> + ExactSizeIterator + DoubleEndedIterator
where
    Data: BitArray + AsPrimitive<Chunk>,
    Chunk: BitArray,
{
    (0..(Data::BITS - data.leading_zeros() as usize))
        .step_by(Chunk::BITS)
        .rev()
        .map(move |shift| (data >> shift).as_())
}

/// Iterates from most significant to least significant bits in chunks without
/// skipping initial zero chunks.
///
/// This method is one possible inverse of [`bit_array_from_chunks`].
///
/// # Panics
///
/// Panics if `Self::BITS` is not an integer multiple of `Chunks::BITS`.
///
/// TODO: this will be turned into a compile time bound as soon as that's possible.
fn bit_array_to_chunks_exact<Data, Chunk>(
    data: Data,
) -> impl Iterator<Item = Chunk> + ExactSizeIterator + DoubleEndedIterator
where
    Data: BitArray + AsPrimitive<Chunk>,
    Chunk: BitArray,
{
    assert_eq!(Data::BITS % Chunk::BITS, 0);

    (0..Data::BITS)
        .step_by(Chunk::BITS)
        .rev()
        .map(move |shift| (data >> shift).as_())
}

unsafe impl BitArray for u8 {}
unsafe impl BitArray for u16 {}
unsafe impl BitArray for u32 {}
unsafe impl BitArray for u64 {}
unsafe impl BitArray for u128 {}
unsafe impl BitArray for usize {}
