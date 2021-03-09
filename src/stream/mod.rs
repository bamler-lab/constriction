//! Stream Codes (entropy codes that amortize compressed bits over several symbols)
//!
//! This module provides implementations of stream codes and utilities for defining entropy
//! models for stream codes. Currently, two different stream codes are implemented (see
//! below for a [comparison](#comparison-of-the-implemented-algorithms)):
//!
//! - **Asymmetric Numeral Systems (ANS):** a modern stack-based stream code; see submodule
//!   [`ans`]; and
//! - **Range Coding:** a queue-based stream code that is a variant of Arithmetic Coding;
//!   see submodule [`range`].
//!
//! Both of these stream codes are provided through types that implement the [`Encode`] and
//! [`Decode`] traits defined in this module. To encode or decode a sequence of symbols, you
//! have to provide an [`EntropyModel`] for each symbol, for which the submodule [`models`]
//! provides the necessary utilities.
//!
//! # What's a Stream Code?
//!
//! A stream code is an entropy coding algorithm that encodes a sequence of symbols into a
//! compressed bit string, one symbol at a time, in a way that amortizes compressed bits
//! over several symbols, thus effectively using up a non-integer number of bits for each
//! symbol. This amortization allows stream codes to reach near-optimal compression
//! performance without giving up computational efficiency. In experiments, the default
//! variants of both provided implementations of stream codes showed about 0.1&nbsp;%
//! relative overhead over the theoretical bound on the bitrate (the overhead is due to
//! finite numerical precision and state size, which can even be increased further from
//! their default values via type parameters).
//!
//! The near-optimal compression performance of stream codes is to be seen in contrast to
//! symbol codes, which do not amortize over symbols, i.e., they map each symbol to an
//! integer number of bits, thus leading to a typical overhead of 0.5&nbsp;bits *per symbol*
//! in the best case, and reaching almost 1&nbsp;bit of overhead per symbol for entropy
//! models with very little (<&nbsp;1&nbsp;bit of) entropy per symbol (a common case in deep
//! learning based entropy models). The computational efficiency of stream codes is to be
//! seen in contrast to block codes (used in many popular compression codecs), which encode
//! a block of several symbols at once with computational cost exponential in the block
//! size.
//!
//! # Highly Customizable Implementations With Sane Defaults
//!
//! TODO:
//! - define term "word".
//! - rename `CompressedWord` to `Word` crate wide
//!
//! # Comparison of the Implemented Algorithms
//!
//! The list below compares several aspects of the two currently implemented stream codes
//! (ANS and Range Coding).
//!
//! **TL;DR:** Don't overthink it. For many practical use cases, both ANS and Range Coding
//! will probably do the job, and if you end up painting yourself into a corner then you can
//! still switch between the methods later relatively easily since most operations are
//! implemented in trait methods. If you want to use the bits-back trick or similar advanced
//! techniques, then definitely use ANS Coding because it is currently the only *surjective*
//! entropy coding method implemented in `constriction`. If you have an autoregressive
//! entropy model then you might prefer Range Coding. In any other case, a possibly biased
//! recommendation from the author of this paragraph is to use ANS Coding by default for its
//! simplicity.
//!
//! Here's the more detailed comparison:
//!
//! - **API:** The main practical difference between the two implemented algorithms is that
//!   ANS Coding operates as a stack (last-in-first-out) whereas Range Coding operates as a
//!   queue (first-in-first-out). This has a number of implications:
//!   - **Autoregressive entropy models: point for Range Coding.** If you use an ANS coder
//!     to encode a sequence of symbols then it is advisable to use the method
//!     [`AnsCoder::encode_symbols_reverse`]. As the name suggests, this method encodes the
//!     symbols onto the stack in *reverse* order so that, when you decode them at a later
//!     time, you'll recover them in original order. However, iterating over symbols and, in
//!     particular, over their associated entropy models in reverse order may be
//!     inconvenient for autoregressive models with long term memory. Using an ANS coder
//!     with such a model means that you may end up having to first build up the full
//!     autoregressive model in memory front-to-back and only then iterate over it
//!     back-to-front for encoding (the *decoder* can still iterate as normal,
//!     front-to-back). Range Coding makes this easier since both encoding and decoding can
//!     iterate in the natural front-to-back direction.
//!   - **Hierarchical entropy models: point for ANS Coding:** Sometimes a stack (as in ANS
//!     Coding) is just what you want because having only a single "open end" at the top of
//!     the stack makes things easier. By contrast, a queue (as in Range Coding) has two
//!     separate open ends for reading and writing, respectively. This leads to a number of
//!     corner cases when the two ends come close to each other (which can happen either
//!     within a single word or on two neighboring words). To avoid the computational
//!     overhead for checking for these corner cases on every encoded or decoded symbol, the
//!     provided Range Coder strictly distinguishes between a `RangeEncoder` and a
//!     `RangeDecoder` type. Switching from a `RangeEncoder` to a `RangeDecoder` requires
//!     "sealing" the write end and "opening up" the read end, both of which are
//!     irreversible operations, i.e., you can't switch back from a `RangeDecoder` with
//!     partially consumed data to a `RangeEncoder` to continue writing to it. By contrast,
//!     the provided ANS Coder just has a single `AnsCoder` type that supports arbitrary
//!     sequences of read and write operations. Being able to interleave reads and writes
//!     (i.e., decoding and encoding operations) is important for bits-back coding, a useful
//!     method for hierarchical entropy models.
//! - **Surjectivity: point for ANS Coding.** This is another property that is important for
//!   bits-back coding, and that only ANS Coding satisfies: an ANS Coder can decode any
//!   arbitrary bit string with any sequence of entropy models, even if the bit string was
//!   not generated by encoding a sequence of symbols with the same entropy models. The API
//!   expresses this property by the fact that the error type `AnsCoder::DecodingError` is
//!   declared as [`Infallible`]. If the bit string you decode is uniform random and if
//!   there is still at least about one word of compressed data left after decoding, then
//!   the decoded symbols are, to a good approximation, independent random draws from their
//!   respective entropy models. By contrast, Range Coding is not surjective. It has some
//!   (small) pockets of unrealizable bit strings for any given sequence of entropy models.
//!   Trying to decode such an unrealizable bit string would return the error variant
//!   [`queue::DecodingError::InvalidCompressedData`].
//! - **Compression performance: virtually no difference unless misused.** With the default
//!   settings, the compression performance of both ANS Coding and Range Coding is very
//!   close to the theoretical bound for entropy coding. The lack of surjectivity in Range
//!   Coding leaves a little bit of performance on the table, but it seems to be roughly
//!   countered by a slightly larger rounding overhead in ANS Coding. Keep in mind, however,
//!   that deviating from the default type parameter settings (see discusion
//!   [above](#highly-customizable-implementations-with-sane-defaults)) can considerably
//!   degrade compression performance if not done carefully. It is strongly recommended to
//!   always start with the `DefaultXxx` type aliases (for either ANS or Range Coding),
//!   which use well-tested settings, before experimenting with deviations from the
//!   defaults.
//! - **Computational efficiency: point for ANS Coding.** ANS Coding is a simpler algorithm
//!   than Range Coding, especially the decoding part. This manifests itself in fewer
//!   branches and a smaller internal coder state. Empirically, our decoding benchmarks in
//!   the file `benches/lookup.rs` run more than twice as fast with an `AnsCoder` than with
//!   a `RangeDecoder`. However, please note: (i) these benchmarks use the highly optimized
//!   entropy models from the [`models::lookup`] module; if you use other entropy models
//!   then these will likely be the computational bottleneck, not the coder; (ii) future
//!   versions of `constriction` may introduce further run-time optimizations; and (iii)
//!   while decoding is more than two times faster with ANS, *encoding* is somewhat
//!   (~&nbsp;10&nbsp;%) faster with Range Coding (this *might* be because the encoding
//!   algorithm for Range Coding is not quite as complicated as the decoding algorithm, and
//!   because encoding with ANS, unlike decoding, involves an integer division, which is a
//!   surprisingly slow operation on most hardware).
//! - **Random access decoding: minor point for ANS Coding.** Both ANS and Range Coding
//!   support decoding with random access via the [`Seek`] trait, but it comes at different
//!   costs. In order to jump ("[`seek`]") to a given location in a slice of compressed
//!   data, you have to provide an index for a position in the slice and an internal coder
//!   state (you can get these during encoding via [`Pos::pos_and_state`]). While the type
//!   of the internal coder state can be controlled through type parameters, its minimal
//!   (and also typical) size is only two words for ANS Coding but four words for Range
//!   Coding. Thus, if you're developing a container file format that contains a jump table
//!   for random access decoding, then choosing ANS Coding over Range Coding will allow you
//!   to reduce the memory footprint of the jump table. Whether or not this is significant
//!   depends on how fine grained your jump table is.
//! - **Serialization: minor point for ANS Coding.** The state of an ANS Coder is uniquely
//!   described by the compressed bit string. This means that you can interrupt encoding
//!   and/or decoding with an ANS Coder at any time, serialize the ANS Coder to a bit
//!   string, recover it at a later time by deserialization, and continue encoding and/or
//!   decoding. The serialized form of the ANS Coder is simply the string of compressed bits
//!   that the Coder has accumulated so far. By contrast, a Range Coder has some additional
//!   internal state that is not needed for decoding and therefore not part of the
//!   compressed bit string, but that you would need if you wanted to continue to append
//!   more symbols to a serialized-deserialized Range Coder.
//!
//! [`AnsCoder::encode_symbols_reverse`]: stack::AnsCoder::encode_symbols_reverse
//! [`Infallible`]: core::convert::Infallible
//! [`seek`]: Seek::seek

pub mod backends;
pub mod models;
pub mod queue;
pub mod stack;

#[cfg(feature = "std")]
use std::error::Error;

use core::{borrow::Borrow, fmt::Display};

use crate::{BitArray, CoderError, EncodingError};
use models::{DecoderModel, EncoderModel, EntropyModel};
use num::cast::AsPrimitive;

/// TODO:
/// - make `PRECISION` a generic parameter of `Code`. This makes it more consistent with the
///   fact that `CompressedWord` and `state` are also defined on `Code`.
/// - simplify `IntoDecoder`: has same `PRECISION` as the original Code.
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
    /// [`Decode`]. This method returns `true` if there *might* not be any compressed data.
    /// This can have several causes, e.g.:
    /// - the method is called on a newly constructed empty encoder or decoder; or
    /// - the user is decoding in-memory data and called `maybe_empty` after decoding all
    ///   available compressed data; or
    /// - it is unknown whether there is any compressed data left.
    ///
    /// The last item in the above list deserves further explanation. It is not always
    /// possible to tell whether any compressed data is available. For example, when
    /// encoding data onto or decoding data from a stream (like a network socket), then the
    /// coder is not required to keep track of whether or not any compressed data has
    /// already been emitted or can still be received, respectively. In such a case, when it
    /// is not known whether any compressed data is available, `maybe_empty` *must* return
    /// `true`.
    ///
    /// The contrapositive of the above requirement is that, when `maybe_empty` returns
    /// `false`, then some compressed data is definitely available. Therefore, `maybe_empty`
    /// can used to check for data corruption: if the user of this library believes that
    /// they have decoded all available compressed data but `maybe_empty` returns `false`,
    /// then the decoded data must have been corrupted. However, the converse is not true:
    /// if `maybe_empty` returns `true` then this is not necessarily a particularly strong
    /// guarantee of data integrity.
    ///
    /// Note that it is always legal to call [`decode_symbol`] even on an empty [`Decode`].
    /// Some decoder implementations may even always return and `Ok(_)` value (with an
    /// arbitrary but deterministically constructed wrapped symbol) even if the decoder is
    /// empty,
    ///
    /// # Implementation Guide
    ///
    /// The default implementation always returns `true` since this is always a *valid*
    /// (albeit not necessarily the most useful) return value. If you overwrite this method,
    /// you may return `false` only if there is definitely some compressed data available.
    /// When in doubt, return `true`.
    ///
    /// # TODO
    ///
    /// - split this into `maybe_exhausted` and `maybe_empty`, defined on `Decode` and
    ///   `Encode`, respectively, which may return different things on a queue.
    ///
    /// [`decode_symbol`]: Decode::decode_symbol
    fn maybe_empty(&self) -> bool {
        true
    }
}

/// TODO: rethink `PRECISION` parameter. Try again to encapsulate it in `Probability`
pub trait Encode<const PRECISION: usize>: Code {
    /// The error type for writing out encoded data.
    ///
    /// This will typically be the [`WriteError`] type of the of an underlying
    /// [`WriteBackend`], which is typically [`Infallible`] for automatically growing
    /// in-memory backends (such as `Vec`). But it may be an inhabitated error type if
    /// you're, e.g., encoding directly to a file or socket.
    #[cfg(not(feature = "std"))]
    type BackendError: Debug;

    /// The error type for writing out encoded data.
    ///
    /// This will typically be the [`WriteError`] type of the of an underlying
    /// [`WriteBackend`], which is typically [`Infallible`] for automatically growing
    /// in-memory backends (such as `Vec`). But it may be an inhabitated error type if
    /// you're, e.g., encoding directly to a file or socket.
    #[cfg(feature = "std")]
    type BackendError: Error;

    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        model: D,
    ) -> Result<(), EncodingError<Self::BackendError>>
    where
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>;

    fn encode_symbols<S, D>(
        &mut self,
        symbols_and_models: impl IntoIterator<Item = (S, D)>,
    ) -> Result<(), EncodingError<Self::BackendError>>
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
    ) -> Result<(), TryCodingError<EncodingError<Self::BackendError>, E>>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
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
    ) -> Result<(), EncodingError<Self::BackendError>>
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
    #[cfg(not(feature = "std"))]
    type BackendError: Debug;

    #[cfg(feature = "std")]
    type BackendError: Error;

    #[cfg(not(feature = "std"))]
    type FrontendError: Debug;

    #[cfg(feature = "std")]
    type FrontendError: Error;

    fn decode_symbol<D>(
        &mut self,
        model: D,
    ) -> Result<D::Symbol, CoderError<Self::FrontendError, Self::BackendError>>
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

    /// Decodes multiple symbols driven by a provided iterator.
    ///
    /// Iterates over all `x` in `iterator.into_iter()`. In each iteration, it decodes
    /// a symbol using the entropy model `model`, terminates on error and otherwise
    /// calls `callback(x, decoded_symbol)`.
    ///
    /// The default implementation literally just has a `for` loop over
    /// `iterator.into_iter()`, which contains a single line of code in its body. But,
    /// in a real-world example, it turns out to be significantly faster this way than
    /// if we would write the same `for` loop at the call site. This may hint at some
    /// issues with method inlining.
    fn map_decode_iid_symbols<'s, D, I>(
        &'s mut self,
        iterator: I,
        model: &'s D,
        mut callback: impl FnMut(I::Item, D::Symbol),
    ) -> Result<(), CoderError<Self::FrontendError, Self::BackendError>>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
        I: IntoIterator,
    {
        for x in iterator.into_iter() {
            callback(x, self.decode_symbol(model)?)
        }

        Ok(())
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
/// # use constriction::stream::{
/// #     models::{EncoderModel, DecoderModel, LeakyQuantizer},
/// #     stack::DefaultAnsCoder,
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
/// let encoder = DefaultAnsCoder::new();
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
    /// TODO: I no longer think we want to depend on `From` here.
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
/// # use constriction::stream::{
/// #     models::{EncoderModel, DecoderModel, LeakyQuantizer},
/// #     stack::DefaultAnsCoder,
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
/// let mut encoder = DefaultAnsCoder::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=200);
/// let model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 50.0).unwrap());
/// encode_decode_encode(&mut encoder, model);
/// ```
pub trait AsDecoder<'a, const PRECISION: usize>: Code + 'a {
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
    /// positions within the compressed data (for example, a [`AnsCoder`] begins encoding
    /// at position zero but it begins decoding at position `ans.buf().len()`).
    ///
    /// [`AnsCoder`]: stack::AnsCoder
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
/// [`DefaultAnsCoder`] implements `Pos` but it doesn't implement `Seek` because it
/// supports both encoding and decoding and therefore always operates at the head. In
/// such a case one can usually obtain a seekable entropy coder in return for
/// surrendering some other property. For example, `DefaultAnsCoder` provides the methods
/// [`seekable_decoder`] and [`into_seekable_decoder`] that return a decoder which
/// implements `Seek` but which can no longer be used for encoding (i.e., it doesn't
/// implement [`Encode`]).
///
/// # Example
///
/// ```
/// use constriction::stream::{models::Categorical, stack::DefaultAnsCoder, Decode, Pos, Seek};
///
/// // Create a `AnsCoder` encoder and an entropy model:
/// let mut ans = DefaultAnsCoder::new();
/// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
/// let entropy_model = Categorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
///     .unwrap();
///
/// // Encode some symbols in two chunks and take a snapshot after each chunk.
/// let symbols1 = vec![8, 2, 0, 7];
/// ans.encode_iid_symbols_reverse(&symbols1, &entropy_model).unwrap();
/// let snapshot1 = ans.pos_and_state();
///
/// let symbols2 = vec![3, 1, 5];
/// ans.encode_iid_symbols_reverse(&symbols2, &entropy_model).unwrap();
/// let snapshot2 = ans.pos_and_state();
///
/// // As discussed above, `DefaultAnsCoder` doesn't impl `Seek` but we can get a decoder that does:
/// let mut seekable_decoder = ans.seekable_decoder();
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
/// [`DefaultAnsCoder`]: stack::DefaultAnsCoder
/// [`seekable_decoder`]: stack::AnsCoder::seekable_decoder
/// [`into_seekable_decoder`]: stack::AnsCoder::into_seekable_decoder
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
    /// [`Pos::pos_and_state`] as sketched below:
    ///
    /// ```
    /// // Step 1: Obtain an encoder and encode some data (omitted for brevity) ...
    /// # use constriction::stream::{stack::DefaultAnsCoder, Pos, Seek};
    /// # let encoder = DefaultAnsCoder::new();
    ///
    /// // Step 2: Take a snapshot by calling `Pos::pos_and_state`:
    /// let snapshot = encoder.pos_and_state(); // <-- Returns a tuple `(pos, state)`.
    ///
    /// // Step 3: Encode some more data and then obtain a decoder (omitted for brevity) ...
    /// # let mut decoder = encoder.seekable_decoder();
    ///
    /// // Step 4: Jump to snapshot by calling `Seek::seek`:
    /// decoder.seek(snapshot); // <-- No need to deconstruct `snapshot` into `(pos, state)`.
    /// ```
    ///
    /// For more fine-grained control, one may want to assemble the tuple
    /// `pos_and_state` manually. For example, a [`DefaultAnsCoder`] encodes data from
    /// front to back and then decodes the data in the reverse direction from back to
    /// front. Decoding from back to front may be inconvenient in some use cases, so one
    /// might prefer to instead reverse the order of the `CompressedWord`s once encoding
    /// is finished, and then decode them in the more natural direction from front to
    /// back. Reversing the compressed data changes the position of each
    /// `CompressedWord`, and so any positions obtained from `Pos` need to be adjusted
    /// accordingly before they may be passed to `seek`, as in the following example:
    ///
    /// ```
    /// use constriction::stream::{
    ///     models::LeakyQuantizer, stack::{DefaultAnsCoder, AnsCoder}, Decode, Pos, Seek
    /// };
    ///
    /// // Construct a `DefaultAnsCoder` for encoding and an entropy model:
    /// let mut encoder = DefaultAnsCoder::new();
    /// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let entropy_model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 10.0).unwrap());
    ///
    /// // Encode two chunks of symbols and take a snapshot in-between:
    /// encoder.encode_iid_symbols_reverse(-100..40, &entropy_model).unwrap();
    /// let (mut snapshot_pos, snapshot_state) = encoder.pos_and_state();
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
    /// [`DefaultAnsCoder`]: stack::DefaultAnsCoder
    fn seek(&mut self, pos_and_state: (usize, Self::State)) -> Result<(), ()>;
}

#[derive(Debug)]
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
    type Item = Result<
        <I::Item as EntropyModel<PRECISION>>::Symbol,
        CoderError<Decoder::FrontendError, Decoder::BackendError>,
    >;

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
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item = Result<
        D::Symbol,
        TryCodingError<CoderError<Decoder::FrontendError, Decoder::BackendError>, E>,
    >;

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
    type Item = Result<D::Symbol, CoderError<Decoder::FrontendError, Decoder::BackendError>>;

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

#[derive(Debug)]
pub enum TryCodingError<CodingError, ModelError> {
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

impl<CodingError: Display, ModelError: Display> Display
    for TryCodingError<CodingError, ModelError>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

#[cfg(feature = "std")]
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

impl<CodingError, ModelError> From<CodingError> for TryCodingError<CodingError, ModelError> {
    fn from(err: CodingError) -> Self {
        Self::CodingError(err)
    }
}
