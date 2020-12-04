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
//! streamcode`, then `import streamcode` in their project and construct a
//! `streamcode.Coder`.
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
#![warn(missing_docs, rust_2018_idioms, missing_debug_implementations)]

#[cfg(feature = "pybindings")]
pub mod pybindings;

pub mod distributions;
pub mod stack;

use std::{borrow::Borrow, error::Error, fmt::Debug};

use distributions::DiscreteDistribution;
use num::{
    cast::AsPrimitive,
    traits::{WrappingAdd, WrappingSub},
    PrimInt, Unsigned,
};

pub trait Encode {
    type Word: CompressedWord;

    fn encode_symbol<S, D: DiscreteDistribution<Word = Self::Word, Symbol = S>>(
        &mut self,
        symbol: impl Borrow<S>,
        distribution: D,
    ) -> Result<()>;

    fn encode_symbols<D, S, I>(&mut self, symbols_and_distributions: I) -> Result<()>
    where
        D: DiscreteDistribution<Word = Self::Word>,
        S: Borrow<D::Symbol>,
        I: IntoIterator<Item = (S, D)>,
    {
        for (symbol, distribution) in symbols_and_distributions.into_iter() {
            self.encode_symbol(symbol, distribution)?;
        }

        Ok(())
    }

    fn try_encode_symbols<E, D, S, I>(&mut self, symbols_and_distributions: I) -> Result<()>
    where
        E: Error + 'static,
        D: DiscreteDistribution<Word = Self::Word>,
        S: Borrow<D::Symbol>,
        I: IntoIterator<Item = std::result::Result<(S, D), E>>,
    {
        for symbol_and_distribution in symbols_and_distributions.into_iter() {
            let (symbol, distribution) =
                symbol_and_distribution.map_err(|err| CoderError::IterationError(Box::new(err)))?;
            self.encode_symbol(symbol, distribution)?;
        }

        Ok(())
    }

    fn encode_iid_symbols<D, S, I>(&mut self, symbols: I, distribution: &D) -> Result<()>
    where
        D: DiscreteDistribution<Word = Self::Word>,
        I: IntoIterator<Item = S>,
        S: Borrow<D::Symbol>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_symbols(symbols.into_iter().map(|symbol| (symbol, distribution)))
    }
}

pub trait Decode {
    type Word: CompressedWord;

    fn decode_symbol<D>(&mut self, distribution: D) -> D::Symbol
    where
        D: DiscreteDistribution<Word = Self::Word>;

    /// TODO: This would be much nicer to denote as
    /// `fn decode_symbols(...) -> impl Iterator`
    /// but existential return types are currently not allowed in trait methods.
    fn decode_symbols<'s, I>(&'s mut self, distributions: I) -> DecodeSymbols<'s, Self, I>
    where
        I: Iterator + 's,
        I::Item: DiscreteDistribution<Word = Self::Word>,
    {
        DecodeSymbols {
            decoder: self,
            distributions,
        }
    }

    fn try_decode_symbols<'s, E, D, I>(
        &'s mut self,
        distributions: I,
    ) -> TryDecodeSymbols<'s, Self, I>
    where
        E: Error + 'static,
        D: DiscreteDistribution<Word = Self::Word>,
        I: Iterator<Item = std::result::Result<D, E>> + 's,
    {
        TryDecodeSymbols {
            decoder: self,
            distributions,
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
    ) -> DecodeIidSymbols<'s, Self, D>
    where
        D: DiscreteDistribution<Word = Self::Word>,
    {
        DecodeIidSymbols {
            decoder: self,
            distribution,
            amt,
        }
    }
}

pub trait Seek {}

pub struct DecodeSymbols<'a, Decoder: ?Sized, I> {
    decoder: &'a mut Decoder,
    distributions: I,
}

impl<'a, Decoder: Decode, I: Iterator> Iterator for DecodeSymbols<'a, Decoder, I>
where
    I::Item: DiscreteDistribution<Word = Decoder::Word>,
{
    type Item = <I::Item as DiscreteDistribution>::Symbol;

    fn next(&mut self) -> Option<Self::Item> {
        self.distributions
            .next()
            .map(|distribution| self.decoder.decode_symbol(distribution))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.distributions.size_hint()
    }
}

impl<'a, Decoder: Decode, I: Iterator> ExactSizeIterator for DecodeSymbols<'a, Decoder, I>
where
    I::Item: DiscreteDistribution<Word = Decoder::Word>,
    I: ExactSizeIterator,
{
}

pub struct TryDecodeSymbols<'a, Decoder: ?Sized, I> {
    decoder: &'a mut Decoder,
    distributions: I,
}

impl<'a, Decoder: Decode, I, E, D> Iterator for TryDecodeSymbols<'a, Decoder, I>
where
    I: Iterator<Item = std::result::Result<D, E>>,
    D: DiscreteDistribution<Word = Decoder::Word>,
    E: std::error::Error + 'static,
{
    type Item = Result<D::Symbol>;

    fn next(&mut self) -> Option<Self::Item> {
        self.distributions
            .next()
            .map(|distribution| match distribution {
                Ok(distribution) => Ok(self.decoder.decode_symbol(distribution)),
                Err(err) => Err(CoderError::IterationError(Box::new(err))),
            })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We don't terminate when we encounter an error, so the size doesn't change.
        self.distributions.size_hint()
    }
}

impl<'a, Decoder: Decode, I, E, D> ExactSizeIterator for TryDecodeSymbols<'a, Decoder, I>
where
    I: Iterator<Item = std::result::Result<D, E>> + ExactSizeIterator,
    D: DiscreteDistribution<Word = Decoder::Word>,
    E: std::error::Error + 'static,
{
}

pub struct DecodeIidSymbols<'a, Decoder: ?Sized, D> {
    decoder: &'a mut Decoder,
    distribution: &'a D,
    amt: usize,
}

impl<'a, Decoder, D> Iterator for DecodeIidSymbols<'a, Decoder, D>
where
    Decoder: Decode,
    D: DiscreteDistribution<Word = Decoder::Word>,
{
    type Item = D::Symbol;

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

impl<'a, Decoder, D> ExactSizeIterator for DecodeIidSymbols<'a, Decoder, D>
where
    Decoder: Decode,
    D: DiscreteDistribution<Word = Decoder::Word>,
{
}

/// A trait for the smallest unit of compressed data in a [`Coder`].
///
/// This is the trait for the compressed data type `W` in a [`Coder`], and it is
/// also used to represent probabilities in fixed point arithmetic in a
/// [`DiscreteDistribution`].
///
/// The documentation of `Coder` has a [section that describes the meaning of the
/// compressed word type `W`](
/// struct.Coder.html#generic-parameters-compressed-word-type-w-and-precision).
pub unsafe trait CompressedWord:
    PrimInt + Unsigned + WrappingSub + WrappingAdd + 'static
{
    /// The type that holds the current head of the [`Coder`].
    ///
    /// Must be twice as large as `Self`, so that [`split_state`] can split it into two
    /// words.
    ///
    /// [`Coder`]: struct.Coder.html
    /// [`split_state`]: #method.split_state
    type State: PrimInt + From<Self> + AsPrimitive<Self>;

    /// Returns the number of compressed bits in a `CompressedWord`.
    ///
    /// Defaults to `8 * std::mem::size_of::<Self>()`, which is suitable for all
    /// primitive unsigned integers.
    ///
    /// This should really be a `const fn`, except that these aren't allowed on trait
    /// methods yet.
    fn bits() -> usize {
        8 * std::mem::size_of::<Self>()
    }

    /// Splits `state` into two `CompressedWord`s and returns `(low, high)`.
    ///
    /// Here, `low` holds the least significant bits and `high` the most significant
    /// bits of `state`. Inverse of [`compose_state`](#method.compose_state).
    ///
    /// # Example
    ///
    /// ```
    /// use ans::CompressedWord;
    ///
    /// let state: u64 = 0x0123_4567_89ab_cdef;
    /// let (low, high) = u32::split_state(state);
    /// assert_eq!(low, 0x89ab_cdef);
    /// assert_eq!(high, 0x0123_4567);
    ///
    /// let reconstructed = u32::compose_state(low, high);
    /// assert_eq!(reconstructed, state);
    /// ```
    fn split_state(state: Self::State) -> (Self, Self) {
        let high = (state >> Self::bits()).as_();
        let low = state.as_();
        (low, high)
    }

    /// Composes a `State` from two compressed words.
    ///
    /// Here, `low` holds the least significant bits and `high` the most significant
    /// bits of the returned `State`. See [`split_state`] for an example.
    ///
    /// [`split_state`]: #method.split_state
    fn compose_state(low: Self, high: Self) -> Self::State {
        (Self::State::from(high) << Self::bits()) | Self::State::from(low)
    }
}

unsafe impl CompressedWord for u8 {
    type State = u16;
}

unsafe impl CompressedWord for u16 {
    type State = u32;
}

unsafe impl CompressedWord for u32 {
    type State = u64;
}

unsafe impl CompressedWord for u64 {
    type State = u128;
}

/// Error type for [`ans::Coder`]
///
/// [`ans::Coder`]: struct.Coder.html
#[non_exhaustive]
#[derive(Debug)]
pub enum CoderError {
    /// Tried to encode a symbol with zero probability under the used entropy model.
    ///
    /// This error can usually be avoided by using a "leaky" distribution, i.e., a
    /// distribution that assigns a nonzero probability to all symbols within a
    /// finite domain. Leaky distributions can be constructed with, e.g., a
    /// [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_floating_point_probabilities`](
    /// distributions/struct.Categorical.html#method.from_floating_point_probabilities).
    ImpossibleSymbol,

    /// The iterator provided to [`Coder::try_push_symbols`] or
    /// [`Coder::try_pop_symbols`] yielded `Err(_)`.
    ///
    /// The variant wraps the original error, which can also be retrieved via
    /// [`source`](#method.source).
    ///
    /// [`Coder::try_push_symbols`]: struct.Coder.html#method.try_push_symbols
    /// [`Coder::try_pop_symbols`]: struct.Coder.html#method.try_pop_symbols
    IterationError(Box<dyn Error + 'static>),
}

impl Error for CoderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IterationError(ref source) => Some(&**source),
            _ => None,
        }
    }
}

impl std::fmt::Display for CoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error during entropy coding.")
    }
}

type Result<T> = std::result::Result<T, CoderError>;
