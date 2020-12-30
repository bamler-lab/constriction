//! Entropy coding primitives in Rust for Rust and/or Python.
//!
//! This crate provides high-performance generic lossless compression primitives.
//! Its main intended use case is for building more specialized lossless or lossy
//! compression methods on top of these primitives. For this reason, the library can
//! be used both as a standard Rust crate that one can include in other rust
//! projects, or one can (optionally) compile this crate as a [Python extension
//! module] (using `cargo build --release --feature pybindings`). This allows users
//! to quickly experiment in Python with different compression methods that build on
//! top of the primitives provided by this library. Once a promising compression
//! method has been developed, one can implement an optimized version of the method
//! in Rust building on the same entropy coding primitives provided by this crate.
//!
//! # Usage
//!
//! Rust users will likely want to start by encoding some data with a [`Coder`].
//!
//! Python users will likely want to install this library via `pip install
//! constriction`, then `import constriction` in their project and construct a
//! `constriction.Coder`.
//!
//! # A Primer on Entropy Coding
//!
//! Entropy coding is an approach to lossless compression that employs a
//! probabilistic model over the encoded data. This so called *entropy model* allows
//! an entropy coding algorithm to assign short codewords to data it will likely
//! see, at the cost of mapping unlikely data to longer codewords. The module
//! [`distributions`] provides tools to construct entropy models that you can use
//! with a [`Coder`] or a [`SeekableDecoder`].
//!
//! The information theoretically optimal (i.e., lowest possible) *expected* bitrate
//! for entropy coding lies within one bit of the [cross entropy] of the employed
//! entropy model relative to the true distribution of the encoded data. To achieve
//! this optimal expected bitrate, an optimal entropy coder has to map each possible
//! data point to a compressed bitstring whose length is the [information content]
//! of the data under the entropy model (rounded up to the nearest integer).
//!
//! The entropy coders provided by this library are *asymptotically optimal* in the
//! sense that they employ approximations to optimize for runtime performance, but
//! these approximations can be adjusted via generic parameters to get arbitrarily
//! close to the information theoretically optimal compression performance (in the
//! limit of large data). The documentation of `Coder` has a [section discussing the
//! involved tradeoffs](struct.Coder.html#guidance-for-choosing-w-and-precision).
//!
//! ## Streaming Entropy Coding
//!
//! The entropy coders provided in this library support what is called "streaming
//! entropy coding" over a sequence of symbols. This provides better compression
//! performance (lower bitrates) on large data than well-known alternative methods
//! such as [Huffman coding], which do not support streaming entropy coding.
//!
//! To understand the problem that streaming entropy coding solves, one has to look
//! at practical applications of entropy coding. Data compression is only useful if
//! one a large amount of data, typically represented as a long sequence of
//! *symbols* (such as a long sequence of characters when compressing a text
//! document). Conceptually, the entropy model is then a probability distribution
//! over the entire sequence of symbols (e.g, a probability distribution over all
//! possible text documents). In practice, however, dealing with probability
//! distributions over large amounts of data is difficult, and one therefore
//! typically factorizes the entropy model into individual models for each symbol
//! (such a factorization still allows for [modeling correlations](
//! #encoding-correlated-data) as discussed below).
//!
//! Such a factorization of the entropy model would lead to a significant overhead
//! (in bitrate) in some of the original entropy coding algorithms such as [Huffman
//! coding]. This is because Huffman coding compresses each symbol into its
//! individual bitstring. The compressed representation of the entire message is
//! then formed by concatenating the compressed bitstrings of each symbol (the
//! bitstrings are formed in a way that makes delimiters unnecessary). This has an
//! important drawback: each symbol contributes an integer number of bits to the
//! compressed data, even if its [information content] is not an integer number,
//! which is the common case (the information content of a symbol is the symbol's
//! contribution to the length of the compressed bitstring in an *optimal* entropy
//! coder). For example, consider the problem of compressing a sequence of one
//! million symbols, where the average information content of each symbol is
//! 0.1&nbsp;bits (this is a realistic scenario for neural compression methods).
//! While the information content of the entire message is only 1,000,000 × 0.1 =
//! 100,000&nbsp;bits, Huffman coding would need at least one full bit for each
//! symbol, resulting in a compressed bitstring that is at least ten times longer.
//!
//! The entropy coders provided by this library do not suffer from this overhead
//! because they support *streaming* entropy coding, which amortizes over the
//! information content of several symbols. In the above example, this library would
//! produce a compressed bitstring that is closer to 100,000&nbsp;bits in length
//! (the remaining overhead depends on details in the choice of entropy coder used,
//! which generally trades off compression performance against runtime and memory
//! performance). This amortization over symbols means that one can no longer map
//! each symbol to a span of bits in the compressed bitstring. Therefore, jumping
//! ("seeking") to a given position in the compressed bitstring in a
//! [`SeekableDecoder`] requires providing some small additional information aside
//! from the jump address. The additional information can be thought of as the
//! fractional (i.e., sub-bit) part of the jump address
//!
//! # Encoding Correlated Data
//!
//! The above mentioned factorization of the entropy model into individual models
//! for each encoded symbol may seem more restrictive than it actually is. It is not
//! to be confused with a "fully factorized" model, which is not required here. The
//! entropy model can still model correlations between different symbols in a
//! message.
//!
//! The most straight-forward way to encode correlations is by choosing the entropy
//! model of each symbol conditionally on other symbols ("autoregressive" entropy
//! models), as described in the example below. Another way to encode correlations
//! is via hierarchical probabilistic models and the bits-back algorithm, which can
//! naturally be implemented on top of a stack-based entropy coder such as the rANS
//! coder provided by this library.
//!
//! For an example of an autoregressive entropy model, consider the task of
//! compressing a message that consists of a sequence of three symbols
//! `[s1, s2, s3]`. The full entropy model for this message is a probability
//! distribution `p(s1, s2, s3)`. Such a distribution can, at least in principle,
//! always be factorized as `p(s1, s2, s3) = p1(s1) * p2(s2 | s1) *
//! p3(s3 | s1, s2)`. Here, `p1`, `p2`, and `p3` are entropy models for the
//! individual symbols `s1`, `s2`, and `s3`, respectively. In this notation, the bar
//! "`|`" separates the symbol on the left, which is the one that the given entropy
//! model describes, from the symbols on the right, which one must already know if
//! one wants to construct the entropy model.
//!
//! During encoding, we know the entire message `[s1, s2, s3]`, and so we can
//! construct all three entropy models `p1`, `p2`, and `p3`. We then use these
//! entropy models to encode the symbols `s1`, `s2`, and `s3` (in reverse order if a
//! stack-based entropy coding algorithm such as rANS is used). When decoding the
//! compressed bitstring, we do not know the symbols `s1`, `s2`, and `s3` upfront,
//! so we initially cannot construct `p2` or `p3`. But the entropy model `p1` of the
//! first symbol does not depend on any other symbols, so we can use it to decode
//! the first symbol `s1`. Using this decoded symbol, we can construct the entropy
//! model `p2` and use it to decode the second symbol `s2`. Finally, we use both
//! decoded symbols `s1` and `s2` to construct the entropy model `p3` and we decode
//! `s3`.
//!
//! This technique of autoregressive models can be scaled up to build very
//! expressive entropy models over complex data types. This is outside the scope of
//! this library, which only provides the primitive building blocks for constructing
//! [`distributions`] over individual symbols and for encoding and decoding data.
//!
//! [range Asymmetric Numeral Systems (rANS)]:
//! https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#Range_variants_(rANS)_and_streaming
//! [python extension module]: https://docs.python.org/3/extending/extending.html
//! [cross entropy]: https://en.wikipedia.org/wiki/Cross_entropy
//! [information content]: https://en.wikipedia.org/wiki/Information_content
//! [Huffman coding]: https://en.wikipedia.org/wiki/Huffman_coding

#![feature(min_const_generics)]
#![warn(rust_2018_idioms, missing_debug_implementations)]

#[cfg(feature = "pybindings")]
pub mod pybindings;

pub mod auryn;
pub mod distributions;
pub mod queue;
pub mod stack;

use std::{
    borrow::Borrow,
    error::Error,
    fmt::{Debug, LowerHex, UpperHex},
};

use distributions::DiscreteDistribution;
use num::{
    cast::AsPrimitive,
    traits::{WrappingAdd, WrappingSub},
    PrimInt, Unsigned,
};

pub trait Code {
    type CompressedWord: BitArray;
    type State: Clone;

    /// Returns the current internal state of the coder.
    ///
    /// This method is usually used together with [`Seek::seek`].
    fn state(&self) -> Self::State;

    /// Check if there might be no compressed data available.
    ///
    /// This method is useful to check for consistency, e.g., when decoding data with a
    /// [`Decode`]. This method returns `true` if there *might* not be any compressed
    /// data. This can have several causes, e.g.:
    /// - the method is called on a newly constructed empty encoder or decoder; or
    /// - the user is decoding in-memory data and called `maybe_empty` after decoding
    ///   all available compressed data; or
    /// - it is unknown whether there is any compressed data left.
    ///
    /// The last item in the above list deserves further explanation. It is not always
    /// possible to tell whether any compressed data is available. For example, when
    /// encoding data onto or decoding data from a stream (like a network socket), then
    /// the coder is not required to keep track of whether or not any compressed data
    /// has already been emitted or can still be received, respectively. In such a case,
    /// when it is not known whether any compressed data is available, `maybe_empty`
    /// *must* return `true`.
    ///
    /// The contrapositive of the above requirement is that, when `maybe_empty` returns
    /// `false`, then some compressed data is definitely available. Therefore,
    /// `maybe_empty` can used to check for data corruption: if the user of this library
    /// believes that they have decoded all available compressed data but `maybe_empty`
    /// returns `false`, then the decoded data must have been corrupted. However, the
    /// converse is not true: if `maybe_empty` returns `true` then this is not
    /// necessarily a particularly strong guarantee of data integrity.
    ///
    /// Note that it is always legal to call [`decode_symbol`] even on an empty
    /// [`Decode`]. Some decoder implementations may even always return and `Ok(_)`
    /// value (with an arbitrary but deterministically constructed wrapped symbol) even
    /// if the decoder is empty,
    ///
    /// # Implementation Guide
    ///
    /// It is always valid to return `true` from this method. By contrast, this method
    /// may return `false` only if there is definitely some compressed data available.
    /// Thus, when in doubt, return `true`.
    ///
    /// [`decode_symbol`]: Decode::decode_symbol
    fn maybe_empty(&self) -> bool;
}

pub trait Encode<const PRECISION: usize>: Code {
    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        distribution: D,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>;

    fn encode_symbols<S, D>(
        &mut self,
        symbols_and_distributions: impl IntoIterator<Item = (S, D)>,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        for (symbol, distribution) in symbols_and_distributions.into_iter() {
            self.encode_symbol(symbol, distribution)?;
        }

        Ok(())
    }

    fn try_encode_symbols<S, D, E>(
        &mut self,
        symbols_and_distributions: impl IntoIterator<Item = Result<(S, D), E>>,
    ) -> Result<(), TryCodingError<EncodingError, E>>
    where
        S: Borrow<D::Symbol>,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
        E: Error + 'static,
    {
        for symbol_and_distribution in symbols_and_distributions.into_iter() {
            let (symbol, distribution) =
                symbol_and_distribution.map_err(|err| TryCodingError::InvalidEntropyModel(err))?;
            self.encode_symbol(symbol, distribution)?;
        }

        Ok(())
    }

    fn encode_iid_symbols<S, D>(
        &mut self,
        symbols: impl IntoIterator<Item = S>,
        distribution: &D,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        self.encode_symbols(symbols.into_iter().map(|symbol| (symbol, distribution)))
    }
}

pub trait Decode<const PRECISION: usize>: Code {
    /// The error type for [`decode_symbol`].
    ///
    /// This is an associated type because, [`decode_symbol`] is infallible for some
    /// decoders (e.g., for a [`stack::Coder`]). These decoders set the `DecodingError`
    /// type to [`std::convert::Infallible`] so that the compiler can optimize away
    /// error checks.
    type DecodingError: Error + 'static;

    fn decode_symbol<D>(&mut self, distribution: D) -> Result<D::Symbol, Self::DecodingError>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>;

    /// TODO: This would be much nicer to denote as
    /// `fn decode_symbols(...) -> impl Iterator`
    /// but existential return types are currently not allowed in trait methods.
    fn decode_symbols<'s, I, D>(
        &'s mut self,
        distributions: I,
    ) -> DecodeSymbols<'s, Self, I::IntoIter, PRECISION>
    where
        I: IntoIterator<Item = D> + 's,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        DecodeSymbols {
            decoder: self,
            distributions: distributions.into_iter(),
        }
    }

    fn try_decode_symbols<'s, I, D, E>(
        &'s mut self,
        distributions: I,
    ) -> TryDecodeSymbols<'s, Self, I::IntoIter, PRECISION>
    where
        I: IntoIterator<Item = Result<D, E>> + 's,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        TryDecodeSymbols {
            decoder: self,
            distributions: distributions.into_iter(),
        }
    }

    /// Decode a sequence of symbols using a fixed entropy model.
    ///
    /// This is a convenience wrapper around [`decode_symbols`], and the inverse of
    /// [`encode_iid_symbols`]. See example in the latter.
    ///
    /// [`decode_symbols`]: #method.decode_symbols
    /// [`encode_iid_symbols`]: #method.encode_iid_symbols
    fn decode_iid_symbols<'s, D>(
        &'s mut self,
        amt: usize,
        distribution: &'s D,
    ) -> DecodeIidSymbols<'s, Self, D, PRECISION>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        DecodeIidSymbols {
            decoder: self,
            distribution,
            amt,
        }
    }
}

pub trait IntoDecoder<const PRECISION: usize>: Code + Sized {
    type Decoder: From<Self>
        + Code<CompressedWord = Self::CompressedWord, State = Self::State>
        + Decode<PRECISION>;

    fn into_decoder(self) -> Self::Decoder {
        self.into()
    }
}

pub trait Pos {
    fn pos(&self) -> usize;
}

pub trait Seek: Code {
    fn seek(&mut self, pos: usize, state: &Self::State) -> Result<(), ()>;
}

#[allow(missing_debug_implementations)] // Any useful debug output would have to mutate the decoder.
pub struct DecodeSymbols<'a, Decoder: ?Sized, I, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    distributions: I,
}

impl<'a, Decoder, I, D, const PRECISION: usize> Iterator
    for DecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = D>,
    D: DiscreteDistribution<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item =
        Result<<I::Item as DiscreteDistribution<PRECISION>>::Symbol, Decoder::DecodingError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.distributions
            .next()
            .map(|distribution| self.decoder.decode_symbol(distribution))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.distributions.size_hint()
    }
}

impl<'a, Decoder, I, D, const PRECISION: usize> ExactSizeIterator
    for DecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = D> + ExactSizeIterator,
    D: DiscreteDistribution<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
}

#[allow(missing_debug_implementations)] // Any useful debug output would have to mutate the decoder.
pub struct TryDecodeSymbols<'a, Decoder: ?Sized, I, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    distributions: I,
}

impl<'a, Decoder, I, D, E, const PRECISION: usize> Iterator
    for TryDecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = Result<D, E>>,
    D: DiscreteDistribution<PRECISION>,
    E: std::error::Error + 'static,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item = Result<D::Symbol, TryCodingError<Decoder::DecodingError, E>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.distributions.next().map(|distribution| {
            Ok(self.decoder.decode_symbol(
                distribution.map_err(|err| TryCodingError::InvalidEntropyModel(err))?,
            )?)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We don't terminate when we encounter an error, so the size doesn't change.
        self.distributions.size_hint()
    }
}

impl<'a, Decoder, I, D, E, const PRECISION: usize> ExactSizeIterator
    for TryDecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = Result<D, E>> + ExactSizeIterator,
    D: DiscreteDistribution<PRECISION>,
    E: std::error::Error + 'static,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
}

#[allow(missing_debug_implementations)] // Any useful debug output would have to mutate the decoder.
pub struct DecodeIidSymbols<'a, Decoder: ?Sized, D, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    distribution: &'a D,
    amt: usize,
}

impl<'a, Decoder, D, const PRECISION: usize> Iterator
    for DecodeIidSymbols<'a, Decoder, D, PRECISION>
where
    Decoder: Decode<PRECISION>,
    D: DiscreteDistribution<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item = Result<D::Symbol, Decoder::DecodingError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.amt != 0 {
            self.amt -= 1;
            Some(self.decoder.decode_symbol(self.distribution))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.amt, Some(self.amt))
    }
}

impl<'a, Decoder, D, const PRECISION: usize> ExactSizeIterator
    for DecodeIidSymbols<'a, Decoder, D, PRECISION>
where
    Decoder: Decode<PRECISION>,
    D: DiscreteDistribution<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
}

/// A trait for bit strings of fixed (and usually small) length.
///
/// Short fixed-length bit strings are fundamental building blocks of efficient
/// entropy coding algorithms. They are currently used for the following purposes:
/// - to represent the smallest unit of compressed data (see
///   [`Code::CompressedWord`]);
/// - to represent probabilities in fixed point arithmetic (see
///   [`DiscreteDistribution::Probability`]); and
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
/// integer types, and that [`BitArray::BITS has the correct value.
pub unsafe trait BitArray:
    PrimInt + Unsigned + WrappingAdd + WrappingSub + Debug + LowerHex + UpperHex + 'static
{
    /// The (fixed) length of the `BitArray` in bits.
    ///
    /// Defaults to `8 * std::mem::size_of::<Self>()`, which is suitable for all
    /// primitive unsigned integers.
    ///
    /// This could arguably be called `LEN` instead, but that may be confusing since
    /// "lengths" are typically not measured in bits in the Rust ecosystem.
    const BITS: usize = 8 * std::mem::size_of::<Self>();
}

/// Constructs a `BitArray` from an iterator over chunks of bits that starts at
/// the most significant chunk.
///
/// Terminates iteration as soon as either the provided iterator terminates or
/// enough chunks have been read to specify all bits.
///
/// This method is the inverse of [`chunks_reversed_truncated`](
/// #method.chunks_truncated) except that the two iterate in reverse direction
/// with respect to each other.
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

/// Iterates from most significant to most least significant bits in chunks without
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

/// Error type for [`constriction::Coder`]
///
/// [`constriction::Coder`]: struct.Coder.html
#[non_exhaustive]
#[derive(Debug)]
pub enum EncodingError {
    /// Tried to encode a symbol with zero probability under the used entropy model.
    ///
    /// This error can usually be avoided by using a "leaky" distribution, i.e., a
    /// distribution that assigns a nonzero probability to all symbols within a
    /// finite domain. Leaky distributions can be constructed with, e.g., a
    /// [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_floating_point_probabilities`](
    /// distributions/struct.Categorical.html#method.from_floating_point_probabilities).
    ImpossibleSymbol,
}

#[derive(Debug)]
pub enum TryCodingError<CodingError: Error + 'static, ModelError: Error + 'static> {
    /// The iterator provided to [`Coder::try_push_symbols`] or
    /// [`Coder::try_pop_symbols`] yielded `Err(_)`.
    ///
    /// The variant wraps the original error, which can also be retrieved via
    /// [`source`](#method.source).
    ///
    /// [`Coder::try_push_symbols`]: struct.Coder.html#method.try_push_symbols
    /// [`Coder::try_pop_symbols`]: struct.Coder.html#method.try_pop_symbols
    InvalidEntropyModel(ModelError),

    CodingError(CodingError),
}

impl std::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImpossibleSymbol => write!(
                f,
                "Tried to encode symbol that has zero probability under the used entropy model."
            ),
        }
    }
}

impl Error for EncodingError {}

impl<CodingError: Error + 'static, ModelError: Error + 'static> std::fmt::Display
    for TryCodingError<CodingError, ModelError>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidEntropyModel(err) => {
                write!(f, "Error while constructing entropy model or data: {}", err)
            }
            Self::CodingError(err) => {
                write!(f, "Error while entropy coding: {}", err)
            }
        }
    }
}

impl<CodingError: Error + 'static, ModelError: Error + 'static> Error
    for TryCodingError<CodingError, ModelError>
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidEntropyModel(source) => Some(source),
            Self::CodingError(source) => Some(source),
        }
    }
}

impl<CodingError: Error + 'static, ModelError: Error + 'static> From<CodingError>
    for TryCodingError<CodingError, ModelError>
{
    fn from(err: CodingError) -> Self {
        Self::CodingError(err)
    }
}
