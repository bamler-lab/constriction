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
//! Rust users will likely want to start by encoding some data with a [`Stack`].
//!
//! Python users will likely want to install this library via `pip install
//! constriction`, then `import constriction` in their project and construct a
//! `constriction.Stack`.
//!
//! # A Primer on Entropy Coding
//!
//! Entropy coding is an approach to lossless compression that employs a
//! probabilistic model over the encoded data. This so called *entropy model* allows
//! an entropy coding algorithm to assign short codewords to data it will likely
//! see, at the cost of mapping unlikely data to longer codewords. The module
//! [`models`] provides tools to construct entropy models that you can use
//! with any coder that implements [`Encode`] or  [`Decode`].
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
//! ("seeking") to a given position in the compressed bitstring (see [`Seek::seek`])
//! requires providing some small additional information aside from the jump
//! address. The additional information can be thought of as the fractional (i.e.,
//! sub-bit) part of the jump address
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
//! [`models`] over individual symbols and for encoding and decoding data.
//!
//! [range Asymmetric Numeral Systems (rANS)]:
//! https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#Range_variants_(rANS)_and_streaming
//! [python extension module]: https://docs.python.org/3/extending/extending.html
//! [cross entropy]: https://en.wikipedia.org/wiki/Cross_entropy
//! [information content]: https://en.wikipedia.org/wiki/Information_content
//! [Huffman coding]: https://en.wikipedia.org/wiki/Huffman_coding
//! [`Stack`]: stack.Stack

#![feature(min_const_generics)]
#![warn(rust_2018_idioms, missing_debug_implementations)]

#[cfg(feature = "pybindings")]
pub mod pybindings;

pub mod models;
pub mod queue;
pub mod stack;

use std::{
    borrow::Borrow,
    error::Error,
    fmt::{Debug, LowerHex, UpperHex},
};

use models::{DecoderModel, EncoderModel, EntropyModel};
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
    /// The default implementation always returns `true` since this is always a *valid*
    /// (albeit not necessarily the most useful) return value. If you overwrite this
    /// method, you may return `false` only if there is definitely some compressed data
    /// available. When in doubt, return `true`.
    ///
    /// [`decode_symbol`]: Decode::decode_symbol
    fn maybe_empty(&self) -> bool {
        true
    }
}

pub trait Encode<const PRECISION: usize>: Code {
    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        model: D,
    ) -> Result<(), EncodingError>
    where
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>;

    fn encode_symbols<S, D>(
        &mut self,
        symbols_and_models: impl IntoIterator<Item = (S, D)>,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        for (symbol, model) in symbols_and_models.into_iter() {
            self.encode_symbol(symbol, model)?;
        }

        Ok(())
    }

    fn try_encode_symbols<S, D, E>(
        &mut self,
        symbols_and_models: impl IntoIterator<Item = Result<(S, D), E>>,
    ) -> Result<(), TryCodingError<EncodingError, E>>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
        E: Error + 'static,
    {
        for symbol_and_model in symbols_and_models.into_iter() {
            let (symbol, model) =
                symbol_and_model.map_err(|err| TryCodingError::InvalidEntropyModel(err))?;
            self.encode_symbol(symbol, model)?;
        }

        Ok(())
    }

    fn encode_iid_symbols<S, D>(
        &mut self,
        symbols: impl IntoIterator<Item = S>,
        model: &D,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        self.encode_symbols(symbols.into_iter().map(|symbol| (symbol, model)))
    }
}

pub trait Decode<const PRECISION: usize>: Code {
    /// The error type for [`decode_symbol`].
    ///
    /// This is an associated type because, [`decode_symbol`] is infallible for some
    /// decoders (e.g., for a [`Stack`]). These decoders set the `DecodingError`
    /// type to [`std::convert::Infallible`] so that the compiler can optimize away
    /// error checks.
    ///
    /// [`decode_symbol`]: #tymethod.decode_symbol
    /// [`Stack`]: stack.Stack
    type DecodingError: Error + 'static;

    fn decode_symbol<D>(&mut self, model: D) -> Result<D::Symbol, Self::DecodingError>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>;

    /// TODO: This would be much nicer to denote as
    /// `fn decode_symbols(...) -> impl Iterator`
    /// but existential return types are currently not allowed in trait methods.
    fn decode_symbols<'s, I, D>(
        &'s mut self,
        models: I,
    ) -> DecodeSymbols<'s, Self, I::IntoIter, PRECISION>
    where
        I: IntoIterator<Item = D> + 's,
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        DecodeSymbols {
            decoder: self,
            models: models.into_iter(),
        }
    }

    fn try_decode_symbols<'s, I, D, E>(
        &'s mut self,
        models: I,
    ) -> TryDecodeSymbols<'s, Self, I::IntoIter, PRECISION>
    where
        I: IntoIterator<Item = Result<D, E>> + 's,
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        TryDecodeSymbols {
            decoder: self,
            models: models.into_iter(),
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
        model: &'s D,
    ) -> DecodeIidSymbols<'s, Self, D, PRECISION>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        DecodeIidSymbols {
            decoder: self,
            model,
            amt,
        }
    }
}

/// A trait for conversion into matching decoder type.
///
/// This is useful for generic code that encodes some data with a user provided
/// encoder of generic type, and then needs to obtain a compatible decoder.
///
/// This trait is similar to [`AsDecoder`], except that the conversion takes
/// ownership of `self` (typically an encoder). This means that the calling
/// function may return the resulting decoder or put it on the heap since it will
/// (typically) be free of any references into the current stack frame.
///
/// If you don't have ownership of the original encoder, or you want to reuse the
/// original encoder once you no longer need the returned decoder, then consider
/// using [`AsDecoder`] instead.
///
/// # Example
///
/// To be able to convert an encoder of generic type `Encoder` into a decoder,
/// declare a trait bound `Encoder: IntoDecoder<PRECISION>`. In the following
/// example, differences to the example for [`AsDecoder`] are marked by `// <--`.
///
/// ```
/// # #![feature(min_const_generics)]
/// # use constriction::{
/// #     models::{EncoderModel, DecoderModel, LeakyQuantizer},
/// #     stack::DefaultStack,
/// #     Decode, Encode, IntoDecoder
/// # };
/// #
/// fn encode_and_decode<Encoder, D, const PRECISION: usize>(
///     mut encoder: Encoder, // <-- Needs ownership of `encoder`.
///     model: D
/// ) -> Encoder::IntoDecoder
/// where
///     Encoder: Encode<PRECISION> + IntoDecoder<PRECISION>, // <-- Different trait bound.
///     D: EncoderModel<PRECISION, Symbol=i32> + DecoderModel<PRECISION, Symbol=i32>,
///     D::Probability: Into<Encoder::CompressedWord>,
///     Encoder::CompressedWord: num::cast::AsPrimitive<D::Probability>
/// {
///     encoder.encode_symbol(137, &model);
///     let mut decoder = encoder.into_decoder();
///     let decoded = decoder.decode_symbol(&model).unwrap();
///     assert_eq!(decoded, 137);
///
///     // encoder.encode_symbol(42, &model); // <-- This would fail (we moved `encoder`).
///     decoder // <-- We can return `decoder` as it has no references to the current stack frame.
/// }
///
/// // Usage example:
/// let encoder = DefaultStack::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=200);
/// let model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 50.0).unwrap());
/// encode_and_decode(encoder, model);
/// ```
///
pub trait IntoDecoder<const PRECISION: usize>: Code + Sized {
    /// The target type of the conversion.
    ///
    /// This is the important part of the `IntoDecoder` trait. The actual conversion in
    /// [`into_decoder`] just delegates to `self.into()`. From a caller's perspective,
    /// the advantage of using the `IntoDecoder` trait rather than directly calling
    /// `self.into()` is that `IntoDecoder::IntoDecoder` defines a suitable return type
    /// so the caller doesn't need to specify one.
    ///
    /// [`into_decoder`]: Self::into_decoder
    type IntoDecoder: From<Self>
        + Code<CompressedWord = Self::CompressedWord, State = Self::State>
        + Decode<PRECISION>;

    /// Performs the conversion.
    ///
    /// The default implementation delegates to `self.into()`. There is usually no
    /// reason to overwrite the default implementation.
    fn into_decoder(self) -> Self::IntoDecoder {
        self.into()
    }
}

impl<Decoder: Decode<PRECISION>, const PRECISION: usize> IntoDecoder<PRECISION> for Decoder {
    type IntoDecoder = Self;
}

/// A trait for constructing a temporary matching decoder.
///
/// This is useful for generic code that encodes some data with a user provided
/// encoder of generic type, and then needs to obtain a compatible decoder.
///
/// This trait is similar to [`IntoDecoder`], but it has the following advantages
/// over it:
/// - it doesn't need ownership of `self` (typically an encoder); and
/// - `self` can be used again once the returned decoder is no longer used.
///
/// The disadvantage of `AsDecoder` compared to `IntoDecoder` is that the returned
/// decoder cannot outlive `self`, so it typically cannot be returned from the
/// calling function or put on the heap. If this would pose a problem for your use
/// case then use [`IntoDecoder`] instead.
///
/// # Example
///
/// To be able to temporarily convert an encoder of generic type `Encoder` into a
/// decoder, declare a trait bound `for<'a> Encoder: AsDecoder<'a, PRECISION>`. In
/// the following example, differences to the example for [`IntoDecoder`] are marked
/// by `// <--`.
///
/// ```
/// # #![feature(min_const_generics)]
/// # use constriction::{
/// #     models::{EncoderModel, DecoderModel, LeakyQuantizer},
/// #     stack::DefaultStack,
/// #     Decode, Encode, AsDecoder
/// # };
/// #
/// fn encode_decode_encode<Encoder, D, const PRECISION: usize>(
///     encoder: &mut Encoder, // <-- Doesn't need ownership of `encoder`
///     model: D
/// )
/// where
///     Encoder: Encode<PRECISION>,
///     for<'a> Encoder: AsDecoder<'a, PRECISION>, // <-- Different trait bound.
///     D: EncoderModel<PRECISION, Symbol=i32> + DecoderModel<PRECISION, Symbol=i32>,
///     D::Probability: Into<Encoder::CompressedWord>,
///     Encoder::CompressedWord: num::cast::AsPrimitive<D::Probability>
/// {
///     encoder.encode_symbol(137, &model);
///     let mut decoder = encoder.as_decoder();
///     let decoded = decoder.decode_symbol(&model).unwrap(); // (Doesn't mutate `encoder`.)
///     assert_eq!(decoded, 137);
///
///     std::mem::drop(decoder); // <-- We have to explicitly drop `decoder` ...
///     encoder.encode_symbol(42, &model); // <-- ... before we can use `encoder` again.
/// }
///
/// // Usage example:
/// let mut encoder = DefaultStack::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=200);
/// let model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 50.0).unwrap());
/// encode_decode_encode(&mut encoder, model);
/// ```
pub trait AsDecoder<'a, const PRECISION: usize>: Code + Sized + 'a {
    /// The target type of the conversion.
    ///
    /// This is the important part of the `AsDecoder` trait. The actual conversion in
    /// [`as_decoder`] just delegates to `self.into()`. From a caller's perspective, the
    /// advantage of using the `AsDecoder` trait rather than directly calling
    /// `self.into()` is that `AsDecoder::AsDecoder` defines a suitable return type so
    /// the caller doesn't need to specify one.
    ///
    /// [`as_decoder`]: Self::as_decoder
    type AsDecoder: From<&'a Self>
        + Code<CompressedWord = Self::CompressedWord, State = Self::State>
        + Decode<PRECISION>
        + 'a;

    /// Performs the conversion.
    ///
    /// The default implementation delegates to `self.into()`. There is usually no
    /// reason to overwrite the default implementation.
    fn as_decoder(&'a self) -> Self::AsDecoder {
        self.into()
    }
}

/// A trait for entropy coders that keep track of their current position within the
/// compressed data.
///
/// This is the counterpart of [`Seek`]. Call [`Pos::pos_and_state`] to record
/// "snapshots" of an entropy coder, and then call [`Seek::seek`] at a later time
/// to jump back to these snapshots. See examples in the documentations of [`Seek`]
/// and [`Seek::seek`].
pub trait Pos: Code {
    /// Returns the position in the compressed data, in units of `CompressedWord`s.
    ///
    /// It is up to the entropy coder to define what constitutes the beginning and end
    /// positions within the compressed data (for example, a [`Stack`] begins encoding
    /// at position zero but it begins decoding at position `stack.buf().len()`).
    ///
    /// [`Stack`]: stack::Stack
    fn pos(&self) -> usize;

    /// Convenience method that returns both parts of a snapshot expected by
    /// [`Seek::seek`].
    ///
    /// The default implementation just delegates to [`Pos::pos`] and [`Code::state`].
    /// See documentation of [`Seek::seek`] for usage examples.
    fn pos_and_state(&self) -> (usize, Self::State) {
        (self.pos(), self.state())
    }
}

/// A trait for entropy coders that support random access.
///
/// This is the counterpart of [`Pos`]. While [`Pos::pos_and_state`] can be used to
/// record "snapshots" of an entropy coder, [`Seek::seek`] can be used to jump to these
/// recorded snapshots.
///
/// Not all entropy coders that implement `Pos` also implement `Seek`. For example,
/// [`DefaultStack`] implements `Pos` but it doesn't implement `Seek` because it
/// supports both encoding and decoding and therefore always operates at the head. In
/// such a case one can usually obtain a seekable entropy coder in return for
/// surrendering some other property. For example, `DefaultStack` provides the methods
/// [`seekable_decoder`] and [`into_seekable_decoder`] that return a decoder which
/// implements `Seek` but which can no longer be used for encoding (i.e., it doesn't
/// implement [`Encode`]).
///
/// # Example
///
/// ```
/// use constriction::{models::Categorical, stack::DefaultStack, Decode, Pos, Seek};
///
/// // Create a `Stack` encoder and an entropy model:
/// let mut stack = DefaultStack::new();
/// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
/// let entropy_model = Categorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
///     .unwrap();
///
/// // Encode some symbols in two chunks and take a snapshot after each chunk.
/// let symbols1 = vec![8, 2, 0, 7];
/// stack.encode_iid_symbols_reverse(&symbols1, &entropy_model).unwrap();
/// let snapshot1 = stack.pos_and_state();
///
/// let symbols2 = vec![3, 1, 5];
/// stack.encode_iid_symbols_reverse(&symbols2, &entropy_model).unwrap();
/// let snapshot2 = stack.pos_and_state();
///
/// // As discussed above, `DefaultStack` doesn't impl `Seek` but we can get a decoder that does:
/// let mut seekable_decoder = stack.seekable_decoder();
///
/// // `seekable_decoder` is still a `Stack`, so decoding would start with the items we encoded
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
/// [`DefaultStack`]: stack::DefaultStack
/// [`seekable_decoder`]: stack::Stack::seekable_decoder
/// [`into_seekable_decoder`]: stack::Stack::into_seekable_decoder
pub trait Seek: Code {
    /// Jumps to a given position in the compressed data.
    ///
    /// The argument `pos_and_state` is the same pair of values returned by
    /// [`Pos::pos_and_state`], i.e., it is a tuple of the position in the compressed
    /// data and the `State` to which the entropy coder should be restored. Both values
    /// are absolute (i.e., seeking happens independently of the current state or
    /// position of the entropy coder). The position is measured in units of
    /// `CompressedWord`s (see second example below where we manipulate a position
    /// obtained from `Pos::pos_and_state` in order to reflect a manual reordering of
    /// the `CompressedWord`s in the compressed data).
    ///
    /// # Examples
    ///
    /// The method takes the position and state as a tuple rather than as independent
    /// method arguments so that one can simply pass in the tuple obtained from
    /// `Pos::pos_and_state` as sketched below:
    ///
    /// ```ignore
    /// // Encode some data ...
    /// let snapshot = encoder.pos_and_state(); // <-- Returns a tuple `(pos, state)`.
    /// // Encode some more data ...
    ///
    /// // Obtain a decoder, then jump to snapshot:
    /// decoder.seek(snapshot); // <-- No need to deconstruct the tuple `snapshot`.
    /// ```
    ///
    /// For more fine-grained control, one may want to assemble the tuple
    /// `pos_and_state` manually. For example, a [`DefaultStack`] encodes data from
    /// front to back and then decodes the data in the reverse direction from back to
    /// front. Decoding from back to front may be inconvenient in some use cases, so one
    /// might prefer to instead reverse the order of the `CompressedWord`s once encoding
    /// is finished, and then decode them in the more natural direction from front to
    /// back. Reversing the compressed data changes the position of each
    /// `CompressedWord`, and so any positions obtained from `Pos` need to be adjusted
    /// accordingly before they may be passed to `seek`, as in the following example:
    ///
    /// ```
    /// use constriction::{
    ///     models::LeakyQuantizer,
    ///     stack::{backend::ReadCursorForward, DefaultStack, Stack},
    ///     Decode, Pos, Seek
    /// };
    ///
    /// // Construct a `DefaultStack` for encoding and an entropy model:
    /// let mut encoder = DefaultStack::new();
    /// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let entropy_model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 10.0).unwrap());
    ///
    /// // Encode two chunks of symbols and take a snapshot in-between:
    /// encoder.encode_iid_symbols_reverse(-100..40, &entropy_model).unwrap();
    /// let (mut snapshot_pos, snapshot_state) = encoder.pos_and_state();
    /// encoder.encode_iid_symbols_reverse(50..101, &entropy_model).unwrap();
    ///
    /// // Obtain compressed data, reverse it, and create a decoder that reads it from front to back:
    /// let mut compressed = encoder.into_compressed();
    /// compressed.reverse();
    /// snapshot_pos = compressed.len() - snapshot_pos; // <-- Adjusts the snapshot position.
    /// let mut decoder = Stack::from_compressed(ReadCursorForward::new(compressed)).unwrap();
    ///
    /// // Decoding yields the last encoded chunk of symbols first:
    /// assert_eq!(decoder.decode_symbol(&entropy_model).unwrap(), 50);
    /// assert_eq!(decoder.decode_symbol(&entropy_model).unwrap(), 51);
    ///
    /// // But we can jump ahead:
    /// decoder.seek((snapshot_pos, snapshot_state)); // <-- Uses the adjusted `snapshot_pos`.
    /// let decoded = decoder.decode_iid_symbols(140, &entropy_model).map(|symbol| symbol.unwrap());
    /// assert!(decoded.eq(-100..40));
    /// assert!(decoder.is_empty()); // <-- We've reached the end of the compressed data.
    /// ```
    ///
    /// [`DefaultStack`]: stack::DefaultStack
    fn seek(&mut self, pos_and_state: (usize, Self::State)) -> Result<(), ()>;
}

#[allow(missing_debug_implementations)] // Any useful debug output would have to mutate the decoder.
pub struct DecodeSymbols<'a, Decoder: ?Sized, I, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    models: I,
}

impl<'a, Decoder, I, D, const PRECISION: usize> Iterator
    for DecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = D>,
    D: DecoderModel<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item = Result<<I::Item as EntropyModel<PRECISION>>::Symbol, Decoder::DecodingError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.models
            .next()
            .map(|model| self.decoder.decode_symbol(model))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.models.size_hint()
    }
}

impl<'a, Decoder, I, D, const PRECISION: usize> ExactSizeIterator
    for DecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = D> + ExactSizeIterator,
    D: DecoderModel<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
}

#[allow(missing_debug_implementations)] // Any useful debug output would have to mutate the decoder.
pub struct TryDecodeSymbols<'a, Decoder: ?Sized, I, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    models: I,
}

impl<'a, Decoder, I, D, E, const PRECISION: usize> Iterator
    for TryDecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = Result<D, E>>,
    D: DecoderModel<PRECISION>,
    E: std::error::Error + 'static,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item = Result<D::Symbol, TryCodingError<Decoder::DecodingError, E>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.models.next().map(|model| {
            Ok(self
                .decoder
                .decode_symbol(model.map_err(|err| TryCodingError::InvalidEntropyModel(err))?)?)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We don't terminate when we encounter an error, so the size doesn't change.
        self.models.size_hint()
    }
}

impl<'a, Decoder, I, D, E, const PRECISION: usize> ExactSizeIterator
    for TryDecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = Result<D, E>> + ExactSizeIterator,
    D: DecoderModel<PRECISION>,
    E: std::error::Error + 'static,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
}

#[derive(Debug)]
pub struct DecodeIidSymbols<'a, Decoder: ?Sized, D, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    model: &'a D,
    amt: usize,
}

impl<'a, Decoder, D, const PRECISION: usize> Iterator
    for DecodeIidSymbols<'a, Decoder, D, PRECISION>
where
    Decoder: Decode<PRECISION>,
    D: DecoderModel<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item = Result<D::Symbol, Decoder::DecodingError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.amt != 0 {
            self.amt -= 1;
            Some(self.decoder.decode_symbol(self.model))
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
    D: DecoderModel<PRECISION>,
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
            Self::CapacityExceeded => write!(f, "The encoder cannot accept any more symbols."),
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
