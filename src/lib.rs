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
//! ## Crate Status
//!
//! The `constriction` crate currently provides solid implementations of the following
//! entropy coding algorithms:
//!
//! - **Asymmetric Numeral Systems (ANS):** a highly efficient modern entropy coder with
//!   near-optimal compression performance that supports advanced use cases like bits-back
//!   coding;
//!   - A "split" variant of ANS coding is provided for advanced use cases in hierarchical
//!     models with cyclic dependencies.
//! - **Range Coding:** a variant of Arithmetic Coding that is optimized for realistic
//!   computing hardware; it has similar compression performance and almost the same
//!   computational performance as ANS Coding. The main practical difference is that Range
//!   Coding is a queue (first-in-first-out) while ANS Coding is a stack
//!   (last-in-first-out), which makes Range Coding preferable for autoregressive models and
//!   ANS Coding preferable for hierarchical models (since bits-back coding is easier on a
//!   stack).
//! - **Huffman Coding:** a well-known symbol code, mainly provided here for teaching
//!   purpose; you'll usually want to use a stream code like ANS or Range Coding instead
//!   since they have better compression performance and are at least as fast.
//!
//! Further, `constriction` provides implementations of common probability distributions in
//! fixed-point arithmetic, which can be used as entropy models for all of the above entropy
//! coding algorithms.
//!
//! The provided implementations of entropy coding algorithms and probability distributions
//! are extensively tested and should be considered reliable. However, their APIs may change
//! in future versions of `constriction` if more user experience reveals any unnecessary
//! restrictions or repetitiveness of the current APIs. Please [file an
//! issue](https://github.com/bamler-lab/constriction/issues) if you run into a scenario
//! where the current APIs are suboptimal.
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
//! (ANS) coder here for its speed and compression performance. We'll discuss how you could
//! replace the ANS coder with a range coder or a symbol code like Huffman coding
//! [below](#exercise).
//!
//! ```
//! use constriction::stream::{stack::DefaultAnsCoder, models::DefaultLeakyQuantizer};
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
//!     // Create an adapter that integrates 1-d probability density functions over bins
//!     // `[n - 0.5, n + 0.5)` for all integers `n` from `-100` to `100` using fixed point
//!     // arithmetic with default precision, guaranteeing a nonzero probability for each bin:
//!     let quantizer = DefaultLeakyQuantizer::new(-100..=100);
//!
//!     // Encode the data (in reverse order, since ANS Coding operates as a stack):
//!     coder.encode_symbols_reverse(
//!         symbols.iter().zip(&means).zip(&stds).map(
//!             |((&sym, &mean), &std)| (sym, quantizer.quantize(Normal::new(mean, std).unwrap()))
//!     )).unwrap();
//!
//!     // Retrieve the compressed representation (filling it up to full words with zero bits).
//!     coder.into_compressed().unwrap()
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
//! use constriction::stream::{stack::DefaultAnsCoder, models::DefaultLeakyQuantizer, Decode};
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
//!     // Decode the data:
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
//! making the following substitutions.
//!
//! **In the encoder,**
//!
//! - replace `constriction::stream::stack::DefaultAnsCoder` with
//!   `constriction::stream::queue::DefaultRangeEncoder`; and
//! - replace `coder.encode_symbols_reverse` with `coder.encode_symbols` (you no longer need
//!   to encode symbols in reverse order since Range Coding operates as a queue, i.e.,
//!   first-in-first-out). You'll also have to add the line
//!   `use constriction::stream::Encode;` to bring the trait method `encode_symbols` into
//!   scope.
//!
//! **In the decoder,**
//!
//! - replace `constriction::stream::stack::DefaultAnsCoder` with
//!   `constriction::stream::queue::DefaultRangeDecoder` (note that Range Coding
//!   distinguishes between an encoder and a decoder type since the encoder writes to the
//!   back while the decoder reads from the front; by contrast, ANS Coding is a stack, i.e.,
//!   it reads and writes at the same position and allows interleaving reads and writes).
//!
//! You could also use a symbol code like Huffman Coding (see module [`symbol`]) but that
//! would have considerably worse compression performance, especially on large files, since
//! symbol codes always emit an integer number of bits per compressed symbol, even if the
//! information content of the symbol is a fractional number (stream codes like ANS and
//! Range Coding *effectively* emit a fractional number of bits per symbol since they
//! amortize over several symbols).
//!
//! The above replacements should lead you to something like this:
//!
//! ```
//! use constriction::stream::{
//!     models::DefaultLeakyQuantizer,
//!     queue::{DefaultRangeEncoder, DefaultRangeDecoder},
//!     Encode, Decode,
//! };
//! use statrs::distribution::Normal;
//!
//! fn encode_sample_data() -> Vec<u32> {
//!     // Create an empty Range Encoder with default word and state size:
//!     let mut encoder = DefaultRangeEncoder::new();
//!
//!     // Same made up data, entropy models, and quantizer as in the ANS Coding example above:
//!     let symbols = [23i32, -15, 78, 43, -69];
//!     let means = [35.2, -1.7, 30.1, 71.2, -75.1];
//!     let stds = [10.1, 25.3, 23.8, 35.4, 3.9];
//!     let quantizer = DefaultLeakyQuantizer::new(-100..=100);
//!
//!     // Encode the data (this time in normal order, since Range Coding is a queue):
//!     encoder.encode_symbols(
//!         symbols.iter().zip(&means).zip(&stds).map(
//!             |((&sym, &mean), &std)| (sym, quantizer.quantize(Normal::new(mean, std).unwrap()))
//!     )).unwrap();
//!
//!     // Retrieve the (sealed up) compressed representation.
//!     encoder.into_compressed().unwrap()
//! }
//!
//! fn decode_sample_data(compressed: Vec<u32>) -> Vec<i32> {
//!     // Create a Range Decoder with default word and state size from the compressed data:
//!     let mut decoder = DefaultRangeDecoder::from_compressed(compressed).unwrap();
//!
//!     // Same entropy models and quantizer we used for encoding:
//!     let means = [35.2, -1.7, 30.1, 71.2, -75.1];
//!     let stds = [10.1, 25.3, 23.8, 35.4, 3.9];
//!     let quantizer = DefaultLeakyQuantizer::new(-100..=100);
//!
//!     // Decode the data:
//!     decoder.decode_symbols(
//!         means.iter().zip(&stds).map(
//!             |(&mean, &std)| quantizer.quantize(Normal::new(mean, std).unwrap())
//!     )).collect::<Result<Vec<_>, _>>().unwrap()
//! }
//!
//! let compressed = encode_sample_data();
//!
//! // We'll get a different compressed representation than in the ANS Coding
//! // example because we're using a different entropy coding algorithm ...
//! assert_eq!(compressed, [0x1C31EFEC, 0x257DF32F]);
//!
//! // ... but as long as we decode with the matching algorithm we can still reconstruct the data:
//! assert_eq!(decode_sample_data(compressed), [23, -15, 78, 43, -69]);
//! ```
//!
//! # Where to Go Next?
//!
//! If you already have an entropy model and you just want to encode and decode some
//! sequence of symbols then you can probably start by adjusting the above
//! [examples](#encoding-example) to your needs. Or have a closer look at the [`stream`]
//! module.
//!
//! If you're still new to the concept of entropy coding then check out the [teaching
//! material (TBD)](https://robamler.github.io/teaching/compress21/).

#![no_std]
#![warn(rust_2018_idioms, missing_debug_implementations)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "pybindings")]
mod pybindings;

pub mod backends;
pub mod stream;
pub mod symbol;

use core::{
    convert::Infallible,
    fmt::{Binary, Debug, Display, LowerHex, UpperHex},
    num::{NonZeroU128, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize},
};

use num::{
    cast::AsPrimitive,
    traits::{WrappingAdd, WrappingSub},
    PrimInt, Unsigned,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoderError<FrontendError, BackendError> {
    FrontendError(FrontendError),
    BackendError(BackendError),
}

impl<BackendError: Display, FrontendError: Display> Display
    for CoderError<FrontendError, BackendError>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FrontendError(err) => write!(f, "Invalid compressed data: {}", err),
            Self::BackendError(err) => write!(f, "Error while reading compressed data: {}", err),
        }
    }
}

#[cfg(feature = "std")]
impl<FrontendError: std::error::Error + 'static, BackendError: std::error::Error + 'static>
    std::error::Error for CoderError<FrontendError, BackendError>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FrontendError(source) => Some(source),
            Self::BackendError(source) => Some(source),
        }
    }
}

impl<FrontendError, BackendError> From<BackendError> for CoderError<FrontendError, BackendError> {
    fn from(read_error: BackendError) -> Self {
        Self::BackendError(read_error)
    }
}

impl<FrontendError> CoderError<FrontendError, Infallible> {
    fn into_frontend_error(self) -> FrontendError {
        match self {
            CoderError::FrontendError(frontend_error) => frontend_error,
            CoderError::BackendError(infallible) => match infallible {},
        }
    }
}

type EncoderError<BackendError> = CoderError<EncoderFrontendError, BackendError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncoderFrontendError {
    /// Tried to encode a symbol with zero probability under the used entropy model.
    ///
    /// This error can usually be avoided by using a "leaky" distribution, as the
    /// entropy model, i.e., a distribution that assigns a nonzero probability to all
    /// symbols within a finite domain. Leaky distributions can be constructed with,
    /// e.g., a [`LeakyQuantizer`](models/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_floating_point_probabilities`](
    /// models/struct.Categorical.html#method.from_floating_point_probabilities).
    ImpossibleSymbol,
}

impl Display for EncoderFrontendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ImpossibleSymbol => write!(
                f,
                "Tried to encode symbol that has zero probability under the used entropy model."
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EncoderFrontendError {}

impl EncoderFrontendError {
    fn into_coder_error<BackendError>(self) -> EncoderError<BackendError> {
        EncoderError::FrontendError(self)
    }
}

/// A trait for bit strings of fixed (and usually small) length.
///
/// Short fixed-length bit strings are fundamental building blocks of efficient
/// entropy coding algorithms. They are currently used for the following purposes:
/// - to represent the smallest unit of compressed data (see
///   [`Code::Word`]);
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

    type NonZero: NonZeroBitArray<Base = Self>;

    #[inline(always)]
    fn into_nonzero(self) -> Option<Self::NonZero> {
        Self::NonZero::new(self)
    }

    #[inline(always)]
    unsafe fn into_nonzero_unchecked(self) -> Self::NonZero {
        Self::NonZero::new_unchecked(self)
    }
}

#[inline(always)]
fn wrapping_pow2<T: BitArray, const EXPONENT: usize>() -> T {
    if EXPONENT >= T::BITS {
        T::zero()
    } else {
        T::one() << EXPONENT
    }
}

pub unsafe trait NonZeroBitArray: Copy + Display + Debug {
    type Base: BitArray;

    fn new(n: Self::Base) -> Option<Self>;

    unsafe fn new_unchecked(n: Self::Base) -> Self;

    fn get(self) -> Self::Base;
}

/// Iterates from most significant to least significant bits in chunks but skips any
/// initial zero chunks.
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

macro_rules! unsafe_impl_bit_array {
    ($(($base:ty, $non_zero:ty)),+ $(,)?) => {
        $(
            unsafe impl BitArray for $base {
                type NonZero = $non_zero;
            }

            unsafe impl NonZeroBitArray for $non_zero {
                type Base = $base;

                #[inline(always)]
                fn new(n: Self::Base) -> Option<Self> {
                    Self::new(n)
                }

                #[inline(always)]
                unsafe fn new_unchecked(n: Self::Base) -> Self {
                    Self::new_unchecked(n)
                }

                #[inline(always)]
                fn get(self) -> Self::Base {
                    self.get()
                }
            }
        )+
    };
}

unsafe_impl_bit_array!(
    (u8, NonZeroU8),
    (u16, NonZeroU16),
    (u32, NonZeroU32),
    (u64, NonZeroU64),
    (u128, NonZeroU128),
    (usize, NonZeroUsize)
);

pub trait UnwrapInfallible<T> {
    fn unwrap_infallible(self) -> T;
}

impl<T> UnwrapInfallible<T> for Result<T, Infallible> {
    fn unwrap_infallible(self) -> T {
        match self {
            Ok(x) => x,
            Err(infallible) => match infallible {},
        }
    }
}

impl<T> UnwrapInfallible<T> for Result<T, CoderError<Infallible, Infallible>> {
    fn unwrap_infallible(self) -> T {
        match self {
            Ok(x) => x,
            Err(infallible) => match infallible {
                CoderError::BackendError(infallible) => match infallible {},
                CoderError::FrontendError(infallible) => match infallible {},
            },
        }
    }
}
