//! Entropy Coding Primitives for Research and Production
//!
//! The `constriction` crate provides a set of composable entropy coding algorithms with a
//! focus on correctness, versatility, ease of use, compression performance, and
//! computational efficiency. The goals of `constriction` are to three-fold:
//!
//! 1. **to facilitate research on novel lossless and lossy compression methods** by
//!    providing a *composable* set of entropy coding primitives rather than a rigid
//!    implementation of a single preconfigured method;
//! 2. **to simplify the transition from research code to production software** by exposing
//!    the exact same functionality via both a Python API (for rapid prototyping on research
//!    code) and a Rust API (for turning successful prototypes into production); and
//! 3. **to serve as a teaching resource** by providing a wide range of entropy coding
//!    algorithms within a single consistent framework, thus making the various algorithms
//!    easily discoverable and comparable on example models and data. [Additional teaching
//!    material](https://robamler.github.io/teaching/compress21/) is being made publicly
//!    available as a by-product of an ongoing university course on data compression with
//!    deep probabilistic models.
//!
//! For an example of a compression codec that started as research code in Python and was
//! then deployed as a fast and dependency-free WebAssembly module using `constriction`'s
//! Rust API, have a look at [The Linguistic Flux
//! Capacitor](https://robamler.github.io/linguistic-flux-capacitor).
//!
//! # Project Status
//!
//! We currently provide implementations of the following entropy coding algorithms:
//!
//! - **Asymmetric Numeral Systems (ANS):** a fast modern entropy coder with near-optimal
//!   compression effectiveness that supports advanced use cases like bits-back coding.
//! - **Range Coding:** a computationally efficient variant of Arithmetic Coding, that has
//!   essentially the same compression effectiveness as ANS Coding but operates as a queue
//!   ("first in first out"), which makes it preferable for autoregressive models.
//! - **Chain Coding:** an experimental new entropy coder that combines the (net)
//!   effectiveness of stream codes with the locality of symbol codes; it is meant for
//!   experimental new compression approaches that perform joint inference, quantization,
//!   and bits-back coding in an end-to-end optimization. This experimental coder is mainly
//!   provided to prove to ourselves that the API for encoding and decoding, which is shared
//!   across all stream coders, is flexible enough to express complex novel tasks.
//! - **Huffman Coding:** a well-known symbol code, mainly provided here for teaching
//!   purpose; you'll usually want to use a stream code like ANS or Range Coding instead
//!   since symbol codes can have a considerable overhead on the bitrate, especially in the
//!   regime of low entropy per symbol, which is common in machine-learning based
//!   compression methods.
//!
//! Further, `constriction` provides implementations of common probability distributions in
//! fixed-point arithmetic, which can be used as entropy models in either of the above
//! stream codes. The crate also provides adapters for turning custom probability
//! distributions into exactly invertible fixed-point arithmetic.
//!
//! The provided implementations of entropy coding algorithms and probability distributions
//! are extensively tested and should be considered reliable (except for the still
//! experimental Chain Coder). However, their APIs may change in future versions of
//! `constriction` if more user experience reveals any shortcomings of the current APIs in
//! terms of ergonomics. Please [file an
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
//! ## Setup
//!
//! To use `constriction` in your Rust project, just add the following line to the
//! `[dependencies]` section of your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! constriction = "0.2.6"
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
//! entropy model. Each symbol will be modeled by a quantized Gaussian  with a different
//! mean and standard deviation (so that the example is not too simplistic). We'll use the
//! `probability` crate for the Gaussian distributions, so also add the following dependency
//! to your `Cargo.toml`:
//!
//! ```toml
//! probability = "0.17"
//! ```
//!
//! Now, let's encode (i.e., compress) some symbols. We'll use an Asymmetric Numeral Systems
//! (ANS) Coder here for its speed and compression performance. We'll discuss how you could
//! replace the ANS Coder with a Range Coder or a symbol code like Huffman Coding
//! [below](#exercise).
//!
//! ```
//! use constriction::stream::{stack::DefaultAnsCoder, model::DefaultLeakyQuantizer};
//! use probability::distribution::Gaussian;
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
//!             |((&sym, &mean), &std)| (sym, quantizer.quantize(Gaussian::new(mean, std)))
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
//! Now, let's reconstruct the sample data from its compressed representation.
//!
//! ```
//! use constriction::stream::{stack::DefaultAnsCoder, model::DefaultLeakyQuantizer, Decode};
//! use probability::distribution::Gaussian;
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
//!             |(&mean, &std)| quantizer.quantize(Gaussian::new(mean, std))
//!     )).collect::<Result<Vec<_>, _>>().unwrap()
//! }
//!
//! assert_eq!(decode_sample_data(vec![0x421C_7EC3, 0x000B_8ED1]), [23, -15, 78, 43, -69]);
//! ```
//!
//! ## Exercise
//!
//! Try out the above examples and verify that decoding reconstructs the original data. Then
//! see how easy `constriction` makes it to replace the ANS Coder with a Range Coder by
//! making the following substitutions:
//!
//! **In the encoder,**
//!
//! - replace `constriction::stream::stack::DefaultAnsCoder` with
//!   `constriction::stream::queue::DefaultRangeEncoder`; and
//! - replace `coder.encode_symbols_reverse` with `coder.encode_symbols` (you no longer need
//!   to encode symbols in reverse order since Range Coding operates as a queue, i.e.,
//!   first-in-first-out). You'll also have to add the line
//!   `use constriction::stream::Encode;` to the top of the file to bring the trait method
//!   `encode_symbols` into scope.
//!
//! **In the decoder,**
//!
//! - replace `constriction::stream::stack::DefaultAnsCoder` with
//!   `constriction::stream::queue::DefaultRangeDecoder` (note that Range Coding
//!   distinguishes between an encoder and a decoder type since the encoder writes to the
//!   back while the decoder reads from the front; by contrast, ANS Coding is a stack, i.e.,
//!   it reads and writes at the same position and allows interleaving reads and writes).
//!
//! *Remark:* You could also use a symbol code like Huffman Coding (see module [`symbol`])
//! but that would have considerably worse compression performance, especially on large
//! files, since symbol codes always emit an integer number of bits per compressed symbol,
//! even if the information content of the symbol is a fractional number (stream codes like
//! ANS and Range Coding *effectively* emit a fractional number of bits per symbol since
//! they amortize over several symbols).
//!
//! The above replacements should lead you to something like this:
//!
//! ```
//! use constriction::stream::{
//!     model::DefaultLeakyQuantizer,
//!     queue::{DefaultRangeEncoder, DefaultRangeDecoder},
//!     Encode, Decode,
//! };
//! use probability::distribution::Gaussian;
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
//!             |((&sym, &mean), &std)| (sym, quantizer.quantize(Gaussian::new(mean, std)))
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
//!             |(&mean, &std)| quantizer.quantize(Gaussian::new(mean, std))
//!     )).collect::<Result<Vec<_>, _>>().unwrap()
//! }
//!
//! let compressed = encode_sample_data();
//!
//! // We'll get a different compressed representation than in the ANS Coding
//! // example because we're using a different entropy coding algorithm ...
//! assert_eq!(compressed, [0x1C31EFEB, 0x87B430DA]);
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
//! material](https://robamler.github.io/teaching/compress21/).

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
    hash::Hash,
    num::{NonZeroU128, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroU8, NonZeroUsize},
};

use num_traits::{AsPrimitive, PrimInt, Unsigned, WrappingAdd, WrappingMul, WrappingSub};

// READ WRITE SEMANTICS =======================================================

/// A trait for marking how reading and writing order relate to each other.
///
/// This is currently only used in the [`backends`] module. Future versions of
/// `constriction` may expand its use to frontends.
pub trait Semantics: Default {}

/// Zero sized marker trait for last-in-first-out read/write [`Semantics`]
///
/// This type typically only comes up in advanced use cases that are generic over read/write
/// semantics. If you are looking for an entropy coder that operates as a stack, check out
/// the module [`stream::stack`].
#[derive(Debug, Default)]
pub struct Stack {}
impl Semantics for Stack {}

/// Zero sized marker trait for first-in-first-out read/write [`Semantics`]
///
/// This type typically only comes up in advanced use cases that are generic over read/write
/// semantics. If you are looking for an entropy coder that operates as a queue, check out
/// the module [`stream::queue`].
#[derive(Debug, Default)]
pub struct Queue {}
impl Semantics for Queue {}

// GENERIC ERROR TYPES ========================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoderError<FrontendError, BackendError> {
    Frontend(FrontendError),
    Backend(BackendError),
}

impl<FrontendError, BackendError> CoderError<FrontendError, BackendError> {
    pub fn map_frontend<E>(self, f: impl Fn(FrontendError) -> E) -> CoderError<E, BackendError> {
        match self {
            Self::Frontend(err) => CoderError::Frontend(f(err)),
            Self::Backend(err) => CoderError::Backend(err),
        }
    }

    pub fn map_backend<E>(self, f: impl Fn(BackendError) -> E) -> CoderError<FrontendError, E> {
        match self {
            Self::Backend(err) => CoderError::Backend(f(err)),
            Self::Frontend(err) => CoderError::Frontend(err),
        }
    }
}

impl<BackendError: Display, FrontendError: Display> Display
    for CoderError<FrontendError, BackendError>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Frontend(err) => write!(f, "Invalid compressed data: {}", err),
            Self::Backend(err) => write!(f, "Error while reading compressed data: {}", err),
        }
    }
}

#[cfg(feature = "std")]
impl<FrontendError: std::error::Error + 'static, BackendError: std::error::Error + 'static>
    std::error::Error for CoderError<FrontendError, BackendError>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Frontend(source) => Some(source),
            Self::Backend(source) => Some(source),
        }
    }
}

impl<FrontendError, BackendError> From<BackendError> for CoderError<FrontendError, BackendError> {
    fn from(read_error: BackendError) -> Self {
        Self::Backend(read_error)
    }
}

impl<FrontendError> CoderError<FrontendError, Infallible> {
    fn into_frontend_error(self) -> FrontendError {
        match self {
            CoderError::Frontend(frontend_error) => frontend_error,
            CoderError::Backend(infallible) => match infallible {},
        }
    }
}

type DefaultEncoderError<BackendError> = CoderError<DefaultEncoderFrontendError, BackendError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultEncoderFrontendError {
    /// Tried to encode a symbol with zero probability under the used entropy model.
    ///
    /// This error can usually be avoided by using a "leaky" distribution, as the
    /// entropy model, i.e., a distribution that assigns a nonzero probability to all
    /// symbols within a finite domain. Leaky distributions can be constructed with,
    /// e.g., a [`LeakyQuantizer`](models/struct.LeakyQuantizer.html) or with
    /// [`LeakyCategorical::from_floating_point_probabilities`](
    /// models/struct.LeakyCategorical.html#method.from_floating_point_probabilities).
    ImpossibleSymbol,
}

impl Display for DefaultEncoderFrontendError {
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
impl std::error::Error for DefaultEncoderFrontendError {}

impl DefaultEncoderFrontendError {
    #[inline(always)]
    const fn into_coder_error<BackendError>(self) -> DefaultEncoderError<BackendError> {
        DefaultEncoderError::Frontend(self)
    }
}

/// Trait for coders or backends that *might* implement [`Pos`] and/or [`Seek`]
///
/// If a type implements `PosSeek` then that doesn't necessarily mean that it also
/// implements [`Pos`] or [`Seek`]. Implementing `PosSeek` only fixes the common `Position`
/// type that is used *if* the type implements `Pos` and/or `Seek`.
pub trait PosSeek {
    type Position: Clone;
}

/// A trait for entropy coders that keep track of their current position within the
/// compressed data.
///
/// This is the counterpart of [`Seek`]. Call [`Pos::pos`] to record
/// "snapshots" of an entropy coder, and then call [`Seek::seek`] at a later time
/// to jump back to these snapshots. See examples in the documentations of [`Seek`]
/// and [`Seek::seek`].
pub trait Pos: PosSeek {
    /// Returns the position in the compressed data, in units of `Word`s.
    ///
    /// It is up to the entropy coder to define what constitutes the beginning and end
    /// positions within the compressed data (for example, a [`AnsCoder`] begins encoding
    /// at position zero but it begins decoding at position `ans.buf().len()`).
    ///
    /// [`AnsCoder`]: stream::stack::AnsCoder
    fn pos(&self) -> Self::Position;
}

/// A trait for entropy coders that support random access.
///
/// This is the counterpart of [`Pos`]. While [`Pos::pos`] can be used to
/// record "snapshots" of an entropy coder, [`Seek::seek`] can be used to jump to these
/// recorded snapshots.
///
/// Not all entropy coders that implement `Pos` also implement `Seek`. For example,
/// [`DefaultAnsCoder`] implements `Pos` but it doesn't implement `Seek` because it
/// supports both encoding and decoding and therefore always operates at the head. In
/// such a case one can usually obtain a seekable entropy coder in return for
/// surrendering some other property. For example, `DefaultAnsCoder` provides the methods
/// [`as_seekable_decoder`] and [`into_seekable_decoder`] that return a decoder which
/// implements `Seek` but which can no longer be used for encoding (i.e., it doesn't
/// implement [`Encode`]).
///
/// # Example
///
/// ```
/// use constriction::stream::{
///     model::DefaultContiguousCategoricalEntropyModel, stack::DefaultAnsCoder, Decode
/// };
/// use constriction::{Pos, Seek};
///
/// // Create a `AnsCoder` encoder and an entropy model:
/// let mut ans = DefaultAnsCoder::new();
/// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
/// let entropy_model = DefaultContiguousCategoricalEntropyModel
///     ::from_floating_point_probabilities(&probabilities).unwrap();
///
/// // Encode some symbols in two chunks and take a snapshot after each chunk.
/// let symbols1 = vec![8, 2, 0, 7];
/// ans.encode_iid_symbols_reverse(&symbols1, &entropy_model).unwrap();
/// let snapshot1 = ans.pos();
///
/// let symbols2 = vec![3, 1, 5];
/// ans.encode_iid_symbols_reverse(&symbols2, &entropy_model).unwrap();
/// let snapshot2 = ans.pos();
///
/// // As discussed above, `DefaultAnsCoder` doesn't impl `Seek` but we can get a decoder that does:
/// let mut seekable_decoder = ans.as_seekable_decoder();
///
/// // `seekable_decoder` is still a `AnsCoder`, so decoding would start with the items we encoded
/// // last. But since it implements `Seek` we can jump ahead to our first snapshot:
/// seekable_decoder.seek(snapshot1);
/// let decoded1 = seekable_decoder
///     .decode_iid_symbols(4, &entropy_model)
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
/// assert_eq!(decoded1, symbols1);
///
/// // We've reached the end of the compressed data ...
/// assert!(seekable_decoder.is_empty());
///
/// // ... but we can still jump to somewhere else and continue decoding from there:
/// seekable_decoder.seek(snapshot2);
///
/// // Creating snapshots didn't mutate the coder, so we can just decode through `snapshot1`:
/// let decoded_both = seekable_decoder.decode_iid_symbols(7, &entropy_model).map(Result::unwrap);
/// assert!(decoded_both.eq(symbols2.into_iter().chain(symbols1)));
/// assert!(seekable_decoder.is_empty()); // <-- We've reached the end again.
/// ```
///
/// [`Encode`]: stream::Encode
/// [`DefaultAnsCoder`]: stream::stack::DefaultAnsCoder
/// [`as_seekable_decoder`]: stream::stack::AnsCoder::as_seekable_decoder
/// [`into_seekable_decoder`]: stream::stack::AnsCoder::into_seekable_decoder
pub trait Seek: PosSeek {
    /// Jumps to a given position in the compressed data.
    ///
    /// The argument `pos` is the same pair of values returned by
    /// [`Pos::pos`], i.e., it is a tuple of the position in the compressed
    /// data and the `State` to which the entropy coder should be restored. Both values
    /// are absolute (i.e., seeking happens independently of the current state or
    /// position of the entropy coder). The position is measured in units of
    /// `Word`s (see second example below where we manipulate a position
    /// obtained from `Pos::pos` in order to reflect a manual reordering of
    /// the `Word`s in the compressed data).
    ///
    /// # Examples
    ///
    /// The method takes the position and state as a tuple rather than as independent
    /// method arguments so that one can simply pass in the tuple obtained from
    /// [`Pos::pos`] as sketched below:
    ///
    /// ```
    /// // Step 1: Obtain an encoder and encode some data (omitted for brevity) ...
    /// # use constriction::{stream::stack::DefaultAnsCoder, Pos, Seek};
    /// # let encoder = DefaultAnsCoder::new();
    ///
    /// // Step 2: Take a snapshot by calling `Pos::pos`:
    /// let snapshot = encoder.pos(); // <-- Returns a tuple `(pos, state)`.
    ///
    /// // Step 3: Encode some more data and then obtain a decoder (omitted for brevity) ...
    /// # let mut decoder = encoder.as_seekable_decoder();
    ///
    /// // Step 4: Jump to snapshot by calling `Seek::seek`:
    /// decoder.seek(snapshot); // <-- No need to deconstruct `snapshot` into `(pos, state)`.
    /// ```
    ///
    /// For more fine-grained control, one may want to assemble the tuple
    /// `pos` manually. For example, a [`DefaultAnsCoder`] encodes data from
    /// front to back and then decodes the data in the reverse direction from back to
    /// front. Decoding from back to front may be inconvenient in some use cases, so one
    /// might prefer to instead reverse the order of the `Word`s once encoding
    /// is finished, and then decode them in the more natural direction from front to
    /// back. Reversing the compressed data changes the position of each
    /// `Word`, and so any positions obtained from `Pos` need to be adjusted
    /// accordingly before they may be passed to `seek`, as in the following example:
    ///
    /// ```
    /// use constriction::{
    ///     stream::{model::LeakyQuantizer, stack::{DefaultAnsCoder, AnsCoder}, Decode},
    ///     Pos, Seek
    /// };
    ///
    /// // Construct a `DefaultAnsCoder` for encoding and an entropy model:
    /// let mut encoder = DefaultAnsCoder::new();
    /// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let entropy_model = quantizer.quantize(probability::distribution::Gaussian::new(0.0, 10.0));
    ///
    /// // Encode two chunks of symbols and take a snapshot in-between:
    /// encoder.encode_iid_symbols_reverse(-100..40, &entropy_model).unwrap();
    /// let (mut snapshot_pos, snapshot_state) = encoder.pos();
    /// encoder.encode_iid_symbols_reverse(50..101, &entropy_model).unwrap();
    ///
    /// // Obtain compressed data, reverse it, and create a decoder that reads it from front to back:
    /// let mut compressed = encoder.into_compressed().unwrap();
    /// compressed.reverse();
    /// snapshot_pos = compressed.len() - snapshot_pos; // <-- Adjusts the snapshot position.
    /// let mut decoder = AnsCoder::from_reversed_compressed(compressed).unwrap();
    ///
    /// // Since we chose to encode onto a stack, decoding yields the last encoded chunk first:
    /// assert_eq!(decoder.decode_symbol(&entropy_model).unwrap(), 50);
    /// assert_eq!(decoder.decode_symbol(&entropy_model).unwrap(), 51);
    ///
    /// // To jump to our snapshot, we have to use the adjusted `snapshot_pos`:
    /// decoder.seek((snapshot_pos, snapshot_state));
    /// assert!(decoder.decode_iid_symbols(140, &entropy_model).map(Result::unwrap).eq(-100..40));
    /// assert!(decoder.is_empty()); // <-- We've reached the end of the compressed data.
    /// ```
    ///
    /// [`DefaultAnsCoder`]: stream::stack::DefaultAnsCoder
    #[allow(clippy::result_unit_err)]
    fn seek(&mut self, pos: Self::Position) -> Result<(), ()>;
}

/// A trait for bit strings of fixed (and usually small) length.
///
/// Short fixed-length bit strings are fundamental building blocks of efficient entropy
/// coding algorithms. They are currently used for the following purposes:
/// - to represent the smallest unit of compressed data (see [`stream::Code::Word`]);
/// - to represent probabilities in fixed point arithmetic (see
///   [`stream::model::EntropyModel::Probability`]); and
/// - the internal state of entropy coders (see [`stream::Code::State`]) is typically comprised of
///   one or more `BitArray`s, although this is not a requirement.
///
/// This trait is implemented on all primitive unsigned integer types. It is not recommended
/// to implement this trait for custom types since coders will assume, for performance
/// considerations, that `BitArray`s can be represented and manipulated efficiently in
/// hardware.
///
/// # Safety
///
/// This trait is marked `unsafe` so that entropy coders may rely on the assumption that all
/// `BitArray`s have precisely the same behavior as builtin unsigned integer types, and that
/// [`BitArray::BITS`] has the correct value.
pub unsafe trait BitArray:
    PrimInt
    + Unsigned
    + WrappingAdd
    + WrappingSub
    + WrappingMul
    + LowerHex
    + UpperHex
    + Binary
    + Default
    + Copy
    + Display
    + Debug
    + Eq
    + Hash
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

    /// # Safety
    ///
    /// The provided value must be nonzero.
    #[inline(always)]
    unsafe fn into_nonzero_unchecked(self) -> Self::NonZero {
        Self::NonZero::new_unchecked(self)
    }
}

#[inline(always)]
fn wrapping_pow2<T: BitArray>(exponent: usize) -> T {
    if exponent >= T::BITS {
        T::zero()
    } else {
        T::one() << exponent
    }
}

/// A trait for bit strings like [`BitArray`] but with guaranteed nonzero values
///
/// # Safety
///
/// Must guarantee that the value is indeed nonzero. Failing to do so could, e.g., cause a
/// division by zero in entropy coders, which is undefined behavior.
pub unsafe trait NonZeroBitArray: Copy + Display + Debug + Eq + Hash + 'static {
    type Base: BitArray<NonZero = Self>;

    fn new(n: Self::Base) -> Option<Self>;

    /// # Safety
    ///
    /// The provided value `n` must be nonzero.
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
                    let non_zero = self.get();
                    unsafe {
                        // SAFETY: This is trivially safe because `non_zero` came from a
                        // `NonZero` type. We really shouldn't have to give the compiler
                        // this hint it turns out the compiler would otherwise emit an
                        // unnecessary check for div by zero. The unnecessary check used to
                        // have a significant impact on performance, but it doesn't seem to
                        // anymore as of rust version 1.58.0 (although the check itself is
                        // still there).
                        if non_zero == num_traits::zero::<Self::Base>() {
                            core::hint::unreachable_unchecked();
                        } else {
                            non_zero
                        }
                    }
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
    (usize, NonZeroUsize),
);

pub trait UnwrapInfallible<T> {
    fn unwrap_infallible(self) -> T;
}

impl<T> UnwrapInfallible<T> for Result<T, Infallible> {
    #[inline(always)]
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
                CoderError::Backend(infallible) => match infallible {},
                CoderError::Frontend(infallible) => match infallible {},
            },
        }
    }
}
