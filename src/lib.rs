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
//! your language of choice then head over to the [Python API Documentation](TODO). The Rust
//! API provides efficient and composable entropy coding primitives with sane default
//! settings that can be adjusted to a fine degree of detail using type parameters and const
//! generics. The Python API exposes the most common use cases of these entropy coding
//! primitives to an environment that feels more natural to many data scientists.
//!
//! ## System Requirements
//!
//! `constriction` requires Rust version 1.51 or later for its use of the
//! `min_const_generics` feature. If you have an older version of Rust, update to the latest
//! version by running `rustup update stable`.
//!
//! ## Example
//!
//! TODO

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
