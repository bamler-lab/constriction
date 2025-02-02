//! Stream Codes (entropy codes that amortize over several symbols)
//!
//! # Module Overview
//!
//! This module provides implementations of stream codes and utilities for defining entropy
//! models for these stream codes. This module is the heart of the `constriction` crate.
//!
//! ## Provided Implementations of Stream Codes
//!
//! Currently, the following stream codes are provided (see below for a
//! [comparison](#which-stream-code-should-i-use)):
//!
//! - **Asymmetric Numeral Systems (ANS):** a highly efficient modern stream code with
//!   near-optimal compression effectiveness; ANS Coding operates as a stack ("last in first
//!   out") and is surjective, which enables advanced use cases like bits-back coding in
//!   hierarchical entropy models. See submodule [`stack`].
//! - **Range Coding:** a modern variant of Arithmetic Coding; it has essentially the same
//!   compression effectiveness as ANS Coding but operates as a queue ("first in first
//!   out"), which makes it preferable for autoregressive entropy models but less useful for
//!   hierarchical models (see [comparison](#which-stream-code-should-i-use)). See submodule
//!   [`queue`].
//! - **Chain Coding:** an experimental new entropy coder that combines the (net)
//!   effectiveness of stream codes with the locality of symbol codes; it is meant for
//!   experimental new compression approaches that perform joint inference, quantization,
//!   and bits-back coding in an end-to-end optimization. See submodule [`chain`].
//!
//! All of these stream codes are provided through types that implement the [`Encode`] and
//! [`Decode`] traits defined in this module.
//!
//! ## Provided Utilities for Entropy Models
//!
//! To encode or decode a sequence of symbols with one of the above stream codes, you have
//! to specify an [`EntropyModel`] for each symbol. The submodule [`model`] provides
//! utilities for defining `EntropyModel`s.
//!
//! # Examples
//!
//! See submodules [`stack`] and [`queue`].
//!
//! # Which Stream Code Should I Use?
//!
//! The list below shows a detailed comparison of the major two provided stream codes, ANS
//! Coding and Range Coding (most users won't want to use the Chain Coder).
//!
//! **TL;DR:** Don't overthink it. For many practical use cases, both ANS and Range Coding
//! will probably do the job, and if you end up painting yourself into a corner then you can
//! still switch between the methods later relatively easily since most operations are
//! implemented in trait methods. If you want to use the bits-back trick or similar advanced
//! techniques, then definitely use the ANS Coder because the provided Range Coder isn't
//! *surjective* (see below). If you have an autoregressive entropy model then you might
//! prefer Range Coding. In any other case, a possibly biased recommendation from the author
//! of this paragraph is to use ANS Coding by default for its simplicity and decoding speed.
//!
//! Here's the more detailed comparison between ANS Coding and Chain Coding:
//!
//! - **Read/write semantics:** The main practical difference between ANS and Range Coding
//!   is that ANS Coding operates as a stack ("last in first out") whereas Range Coding
//!   operates as a queue ("first in first out"). Whether a stack or a queue is better for
//!   you depends on your entropy model:
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
//!     Coding) is just what you need because having only a single "open end" at the top of
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
//!     sequences of (possibly interleaved) read and write operations. Being able to
//!     interleave reads and writes (i.e., decoding and encoding operations) is important
//!     for bits-back coding, a useful method for hierarchical entropy models.
//! - **Surjectivity: point for ANS Coding.** This is another property that is important for
//!   bits-back coding, and that only ANS Coding satisfies: encoding data with an ANS Coder
//!   and with a given sequence of entropy models can lead to any arbitrary sequence of
//!   words of compressed data. This means that decoding with an ANS Coder is invertible,
//!   even if the binary data that you decode is arbitrary. You can load any given binary
//!   data into an ANS Coder (with [`AnsCoder::from_binary`]) and then decode some part of
//!   the data into a sequence of symbols. Regardless of the source of the binary data,
//!   re-encoding the symbols will restore it. By contrast, Range Coding is not surjective,
//!   and decoding with the provided Range Coder is not always invertible. Due to rounding,
//!   Range Coding has some (small) pockets of unrealizable bit strings for any given
//!   sequence of entropy models. While the Range Decoder in `constriction` is currently
//!   lenient in this regard (it still outputs valid symbols when decoding an unrealizable
//!   bit string), re-encoding the symbols you obtain from decoding an unrealizable bit
//!   string will *not* restore the original (unrealizable) bit string. This means that the
//!   provided Range Coder cannot be used for bits-back coding in a reliable way.
//! - **Compression effectiveness: virtually no difference unless misused.** With the
//!   provided [presets](#presets)), the bitrates of both ANS Coding and Range Coding are
//!   both very close to the theoretical bound for lossless compression. The lack of
//!   surjectivity in Range Coding leads to a small overhead, but it seems to be roughly
//!   countered by a slightly larger rounding overhead in ANS Coding. Keep in mind, however,
//!   that [deviating from the presets](#customizations-for-advanced-use-cases) can
//!   considerably degrade compression effectiveness if not done carefully. It is strongly
//!   recommended to always start with the [`DefaultXxx` type aliases](#presets) (for either
//!   ANS or Range Coding), which use well-tested presets, before experimenting with
//!   deviations from these defaults.
//! - **Computational efficiency: point for ANS Coding.** ANS Coding is a simpler algorithm
//!   than Range Coding, especially the decoding part. This manifests itself in fewer
//!   branches and a smaller internal coder state. Empirically, our decoding benchmarks in
//!   the file `benches/lookup.rs` run more than twice as fast with an `AnsCoder` than with
//!   a `RangeDecoder`. However, please note that (i) these benchmarks use the highly
//!   optimized lookup models (see [`ContiguousLookupDecoderModel`] and
//!   [`NonContiguousLookupDecoderModel`]); if you use other entropy models then these will
//!   likely be the computational bottleneck, not the coder; (ii) future versions of
//!   `constriction` may introduce further run-time optimizations; and (iii) while
//!   *decoding* is more than two times faster with ANS, *encoding* is somewhat
//!   (~&nbsp;10&nbsp;%) faster with Range Coding (this *might* be because encoding with
//!   ANS, unlike decoding, involves an integer division, which is a surprisingly slow
//!   operation on most hardware).
//! - **Random access decoding: minor point for ANS Coding.** Both ANS and Range Coding
//!   support decoding with random access via the [`Seek`](crate::Seek) trait, but it comes
//!   at different costs. In order to jump ("[`seek`]") to a given location in a slice of
//!   compressed data, you have to provide an index for a position in the slice and an
//!   internal coder state (you can get both during encoding via [`Pos::pos`]). While the
//!   type of the internal coder state can be controlled through type parameters, its
//!   minimal (and also default) size is only two words for ANS Coding but four words for
//!   Range Coding. Thus, if you're developing a container file format that contains a jump
//!   table for random access decoding, then choosing ANS Coding over Range Coding will
//!   allow you to reduce the memory footprint of the jump table. Whether or not this is
//!   significant depends on how fine grained your jump table is.
//! - **Serialization: minor point for ANS Coding.** The state of an ANS Coder is entirely
//!   described by the compressed bit string. This means that you can interrupt encoding
//!   and/or decoding with an ANS Coder at any time, serialize the ANS Coder to a bit
//!   string, recover it at a later time by deserialization, and continue encoding and/or
//!   decoding. The serialized form of the ANS Coder is simply the string of compressed bits
//!   that the Coder has accumulated so far. By contrast, a Range Encoder has some
//!   additional internal state that is not needed for decoding and therefore not part of
//!   the compressed bit string, but that you would need if you wanted to continue to append
//!   more symbols to a serialized-deserialized Range Encoder.
//!
//! # What's a Stream Code?
//!
//! A stream code is an entropy coding algorithm that encodes a sequence of symbols into a
//! compressed bit string, one symbol at a time, in a way that amortizes compressed bits
//! over several symbols. This amortization allows stream codes to reach near-optimal
//! compression performance without giving up computational efficiency. In our experiments,
//! both the provided ANS Coder and the Range Coder implementations showed about 0.1&nbsp;%
//! relative overhead over the theoretical bound on the bitrate (using ["default"
//! presets](#presets)).
//!
//! *The near-optimal compression performance* of stream codes is to be seen in contrast to
//! symbol codes (see module [`symbol`](crate::symbol)), such as the well-known [Huffman
//! code](crate::symbol::huffman). Symbol codes do not amortize over symbols. Instead, they
//! map each symbol to a fixed sequence of bits of integer length (a "codeword"). This leads
//! to a typical overhead of 0.5&nbsp;bits *per symbol* in the best case, and to an overhead
//! of almost 1&nbsp;bit per symbol for entropy models with very low (≪&nbsp;1&nbsp;bit of)
//! entropy per symbol, which is common for deep learning based entropy models. Stream codes
//! do not suffer from this overhead.
//!
//! *The computational efficiency* of stream codes is to be seen in contrast to block codes.
//! Block codes are symbol codes that operate on blocks of several consecutive symbols at
//! once, which reduces the relative impact of the constant overhead of ~0.5&nbsp;bits per
//! block compared to regular symbol codes. Block codes typically require some method of
//! constructing the code books dynamically (e.g., [Deflate]) to avoid the computational
//! cost from growing exponentially in the block size. By contrast, stream codes amortize
//! over long sequences of symbols without ever constructing an explicit code book for the
//! entire sequence of symbols. This keeps their  computational cost linear in the number of
//! symbols.
//!
//! # Highly Customizable Implementations With Sane Presets
//!
//! Users who need precise control over the trade-off between compression effectiveness,
//! runtime performance, and memory consumption can fine tune the provided implementations
//! of stream codes and entropy models at compile time through type parameters discussed
//! [below](#customizations-for-advanced-use-cases). For most users, however, we recommend
//! using one of the following presets.
//!
//! ## Presets
//!
//! As finding optimal settings can be tricky, we provide the following type aliases for
//! useful presets:
//!
//! - **"Default" presets** are the recommended starting point for most users, and they are
//!   also used by `constriction`'s [Python
//!   API](https://bamler-lab.github.io/constriction/apidoc/python/). The "default" presets
//!   provide very near-optimal compression effectiveness for most conceivable applications
//!   and high runtime performance on typical (64&nbsp;bit) desktop computers. However, the
//!   "default" presets are *not* recommended for a [`ContiguousLookupDecoderModel`] or
//!   [`NonContiguousLookupDecoderModel`] as their high numerical precision would lead to
//!   enormeous lookup tables (~&nbsp;67&nbsp;MB), which would take a considerable time to
//!   build and likely leead to extremely poor cashing.
//!   - entropy *coders* with "default" presets: [`DefaultAnsCoder`],
//!     [`DefaultRangeEncoder`], [`DefaultRangeDecoder`], and [`DefaultChainCoder`];
//!   - entropy *models* with "default" presets: [`DefaultLeakyQuantizer`],
//!     [`DefaultContiguousCategoricalEntropyModel`],
//!     [`DefaultNonContiguousCategoricalEncoderModel`], and
//!     [`DefaultNonContiguousCategoricalDecoderModel`].
//! - **"Small" presets** may be considered when optimizing a final product for runtime
//!   efficiency and memory consumption. The "small" presets use a lower numerical precision
//!   and a smaller state and word size than the "default" presets. The lower numerical
//!   precision makes it possible to use the highly runtime efficient
//!   [`ContiguousLookupDecoderModel`] or [`NonContiguousLookupDecoderModel`], the smaller
//!   state size reduces the memory overhead of jump tables for random access, and the
//!   smaller word size may be advantageous on some embedded devices.
//!   - entropy *coders* with "small" presets: [`SmallAnsCoder`], [`SmallRangeEncoder`],
//!     [`SmallRangeDecoder`], and [`SmallChainCoder`];
//!   - entropy *models* with "small" presets: [`ContiguousLookupDecoderModel`],
//!     [`NonContiguousLookupDecoderModel`], [`SmallContiguousCategoricalEntropyModel`],
//!     [`SmallNonContiguousCategoricalEncoderModel`],
//!     [`SmallNonContiguousCategoricalDecoderModel`], and [`SmallLeakyQuantizer`].
//!
//! You'll usually want to use matching presets for entropy *coders* and entropy *models*.
//! However, it is legal to use entropy models with the "small" preset for an entropy coder
//! with the (larger) "default" preset (the opposite way is statically prohibited on purpose
//! by the type system). Using "small" models with a "default" coder may make sense if you
//! want to mix entropy models with "default" and "small" presets (e.g., if some of the
//! models are lookup tables but others aren't).
//!
//! ## Customizations for Advanced Use Cases
//!
//! Some advanced use cases may not be covered by the above presets. For such cases, the
//! entropy coders and entropy models can be adjusted type parameters listed below.
//!
//! **Warning:** Adjusting these parameters can severly impact compression effectiveness if
//! not done with care. We strongly recommend to always start from one of the above
//! [presets](#presets), to only deviate from them when necessary, and to measure the effect
//! of any deviations from the presets (on compression effectiveness, computational
//! performance, and memory consumption).
//!
//! ### Type Parameters of Entropy Coders
//!
//! - `Word`: a [`BitArray`] specifying the smallest unit of compressed data that the
//!   entropy coder emits or reads in at a time; an entropy coder's `Word` type has to be at
//!   least as large as the `Probability` type (see below) of any entropy model used with
//!   it.
//!   - The "default" preset sets `Word = u32`.
//!   - The "small" preset sets `Word = u16`.
//! - `State`: a [`BitArray`] that parameterizes the size of the internal coder state (the
//!   actual coder state [`Code::State`] may differ from `State`, e.g., it may contain
//!   several fields of type `State`, as in the case of a Range Coder). Typically, `State`
//!   has to be at least twice as large as `Word`. You'll want to use a small `State` if
//!   you're planning to perform random accesses via the [`Seek`](crate::Seek) trait because
//!   your container format will then typically contain some kind of jump table composed of
//!   indices and `Code::State`s (see [`PosSeek::Position`](crate::PosSeek::Position)).
//!   - The "default" preset sets `State = u64`.
//!   - The "small" preset sets `State = u32`.
//! - `Backend`: the source and/or sink of compressed data. See module [`backends`]. You'll
//!   rarely have to specify `Backend` explictily, it usually defaults to `Vec<Word>` for
//!   encoding and to either `Vec<Word>` or [`Cursor`] for decoding, depending on which
//!   entropy coder you use and who owns the compressed data. The [`ChainCoder`] has two
//!   type parameters for backends because it keeps track of compressed data and remainders
//!   separately.
//!
//! ### Type Parameters of Entropy Models
//!
//! - `Probability`: a [`BitArray`] specifying the type used to represent probabilities
//!   (smaller than one) in fixed point arithmetic; must hold at least `PRECISION` bits (see
//!   below) and must not be larger than the `Word` type of the entropy coder that uses the
//!   model.
//!   - The "default" preset sets `Probability = u32` (same as for `Word` above).
//!   - The "small" preset sets `Probability = u16` (same as for `Word` above).
//! - `PRECISION`: a `usize` const generic that defines the number of bits used when
//!   representing probabilities in fixed-point arithmetic. Must not be zero or larger than
//!   `Probability::BITS`. A small `PRECISION` will lead to compression overhead due to poor
//!   approximations of the true probability distribution. A large `PRECISION` will lead to
//!   a large memory overhead if you use a [`ContiguousLookupDecoderModel`] or
//!   [`NonContiguousLookupDecoderModel`], and it can make decoding with a
//!   [`LeakilyQuantizedDistribution`] slow.
//!   - The "default" preset sets `PRECISION = 24`.
//!   - The "small" preset sets `PRECISION = 12`.
//!
//! [`AnsCoder::encode_symbols_reverse`]: stack::AnsCoder::encode_symbols_reverse
//! [`Infallible`]: core::convert::Infallible
//! [`seek`]: crate::Seek::seek
//! [`Pos::pos`]: crate::Pos::pos
//! [`DefaultAnsCoder`]: stack::DefaultAnsCoder
//! [`DefaultRangeEncoder`]: queue::DefaultRangeEncoder
//! [`DefaultRangeDecoder`]: queue::DefaultRangeDecoder
//! [`DefaultChainCoder`]: chain::DefaultChainCoder
//! [`DefaultLeakyQuantizer`]: model::DefaultLeakyQuantizer
//! [`DefaultContiguousCategoricalEntropyModel`]:
//!     model::DefaultContiguousCategoricalEntropyModel
//! [`DefaultNonContiguousCategoricalEncoderModel`]:
//!     model::DefaultNonContiguousCategoricalEncoderModel
//! [`DefaultNonContiguousCategoricalDecoderModel`]:
//!     model::DefaultNonContiguousCategoricalDecoderModel
//! [`SmallAnsCoder`]: stack::SmallAnsCoder
//! [`SmallRangeEncoder`]: queue::SmallRangeEncoder
//! [`SmallRangeDecoder`]: queue::SmallRangeDecoder
//! [`SmallChainCoder`]: chain::SmallChainCoder
//! [`SmallLeakyQuantizer`]: model::SmallLeakyQuantizer
//! [`ContiguousLookupDecoderModel`]: model::ContiguousLookupDecoderModel
//! [`NonContiguousLookupDecoderModel`]: model::NonContiguousLookupDecoderModel
//! [`SmallContiguousCategoricalEntropyModel`]:
//!     model::SmallContiguousCategoricalEntropyModel
//! [`SmallNonContiguousCategoricalEncoderModel`]:
//!     model::SmallNonContiguousCategoricalEncoderModel
//! [`SmallNonContiguousCategoricalDecoderModel`]:
//!     model::SmallNonContiguousCategoricalDecoderModel
//! [`AnsCoder`]: stack::AnsCoder
//! [`AnsCoder::from_binary`]: stack::AnsCoder::from_binary
//! [`ChainCoder`]: chain::ChainCoder
//! [`Cursor`]: crate::backends::Cursor
//! [`backends`]: crate::backends
//! [Deflate]: https://en.wikipedia.org/wiki/Deflate
//! [`ContiguousLookupDecoderModel`]: model::ContiguousLookupDecoderModel
//! [`NonContiguousLookupDecoderModel`]: model::NonContiguousLookupDecoderModel
//! [`LeakilyQuantizedDistribution`]: model::LeakilyQuantizedDistribution

#![allow(clippy::type_complexity)]

pub mod chain;
pub mod model;
pub mod queue;
pub mod stack;

use core::{
    borrow::Borrow,
    fmt::{Debug, Display},
};

use crate::{BitArray, CoderError};
use model::{DecoderModel, EncoderModel, EntropyModel};
use num_traits::AsPrimitive;

/// Base trait for stream encoders and decoders
///
/// This trait has to be implemented by all stream encoders and decoders. In addition,
/// you'll likely want to implement at least one of the [`Encode`] and [`Decode`] trait.
/// While some stream coders may implement both `Encode` and `Decode` for the same type
/// (e.g., [`AnsCoder`] and [`ChainCoder`] if the default backends are used), others, like
/// the [`RangeEncoder`] and [`RangeDecoder`] provide different types that specialize for
/// encoding or decoding only.
///
/// This trait defines the [`Word`](Self::Word) and [`State`](Self::State) types, which
/// apply to both encoding and decoding.
///
/// # Naming Convention
///
/// This trait is deliberately called `Code` (as in the verb "to code") and not `Coder` so
/// that the term `Coder` can still be used for generic type parameters without leading to
/// confusion with trait bounds. For example, you may want to write `impl Wrapper<Coder>
/// where Coder: Code`. Using the verb for trait names and the noun for types has precedence
/// in the standard library: see, e.g., the [`BufRead`] trait, which is implemented by the
/// [`BufReader`] type.
///
/// [`BufRead`]: std::io::BufRead
/// [`BufReader`]: std::io::BufReader
/// [`AnsCoder`]: stack::AnsCoder
/// [`ChainCoder`]: chain::ChainCoder
/// [`RangeEncoder`]: queue::RangeEncoder
/// [`RangeDecoder`]: queue::RangeDecoder
pub trait Code {
    /// The smallest unit of compressed data that this coder can emit or read at once. Most
    /// coders guarantee that encoding emits at most one `Word` per symbol (plus a constant
    /// overhead).
    type Word: BitArray;

    /// The internal coder state, as returned by the method [`state`](Self::state).
    ///
    /// This internal coder state is relevant if the coder implements [`Pos`] and/or
    /// [`Seek`]. By convention, [`Pos::pos`] returns a tuple `(pos, state)` where `pos`
    /// represents the position in the backend(s) while `state` is the internal coder state.
    ///
    /// Most implementations of `Code` are generic over a type parameter `State`. Note that,
    /// while the associated type `Code::State` is typically related to the type parameter
    /// `State`, they do not necessarily need to be the same. For example, in a
    /// [`RangeEncoder`], the associated type `<RangeEncoder as Code>::State` is a struct of
    /// type [`RangeCoderState`], which has twice the size of the type parameter `State`
    /// (same for [`RangeDecoder`]).
    ///
    /// [`Pos`]: crate::Pos
    /// [`Seek`]: crate::Seek
    /// [`Pos::pos`]: crate::Pos::pos
    /// [`Seek::seek`]: crate::Seek::seek
    /// [`RangeEncoder`]: queue::RangeEncoder
    /// [`RangeCoderState`]: queue::RangeCoderState
    /// [`RangeDecoder`]: queue::RangeDecoder
    type State: Clone;

    /// Returns the current internal state of the coder.
    ///
    /// This method returns only the "frontend" state. If you also need the backend state,
    /// i.e., the current position of the coder in the stream of compressed data, then call
    /// [`Pos::pos`] (which is typically only available if the backend(s) implement
    /// [`Pos`]).
    ///
    /// [`Pos`]: crate::Pos
    /// [`Pos::pos`]: crate::Pos::pos
    fn state(&self) -> Self::State;

    /// Checks if there might not be any room to encode more data.
    ///
    /// This is a convenience forwarding method to simplify type annotations. In the default
    /// implementation, calling `encoder.encoder_maybe_full::<PRECISION>()` is equivalent to
    /// `Encode::<PRECISION>::maybe_full(&encoder)`. See [Encode::maybe_full].
    #[inline]
    fn encoder_maybe_full<const PRECISION: usize>(&self) -> bool
    where
        Self: Encode<PRECISION>,
    {
        self.maybe_full()
    }

    /// Checks if there might be no compressed data left for decoding.
    ///
    /// This is a convenience forwarding method to simplify type annotations. In the default
    /// implementation, calling `decoder.decoder_maybe_exhausted::<PRECISION>()` is
    /// equivalent to `Decode::<PRECISION>::maybe_exhausted(&decoder)`. See
    /// [Decode::maybe_exhausted].
    #[inline]
    fn decoder_maybe_exhausted<const PRECISION: usize>(&self) -> bool
    where
        Self: Decode<PRECISION>,
    {
        self.maybe_exhausted()
    }
}

/// A trait for stream encoders (i.e., compressors)
///
/// This trait defines methods for encoding a single symbol or a sequence of symbols.
///
/// # Naming Convention
///
/// This trait is deliberately called `Encode` and not `Encoder`. See corresponding comment
/// for the [`Code`] trait for the reasoning.
pub trait Encode<const PRECISION: usize>: Code {
    /// The error type for logical encoding errors.
    ///
    /// This is often a [`DefaultEncoderFrontendError`](super::DefaultEncoderFrontendError).
    ///
    /// Frontend errors are errors that relate to the logical encoding process, such as when
    /// a user tries to encode a symbol that has zero probability under the provided entropy
    /// model. They are to be distinguished from backend errors, which depend on the used
    /// sources and/or sinks of compressed data, and which can be errors like "out of space"
    /// or "I/O error".
    type FrontendError: Debug;

    /// The error type for writing out encoded data.
    ///
    /// If you stick with the default backend(s) and don't do anything fancy, then this will
    /// most likely be [`Infallible`], which means that the compiler can optimize away error
    /// checks. You can explicitly express that you expect an `Infallible` error type by
    /// calling [`.unwrap_infallible()`] on a `Result<T, Infallible>` or on a `Result<T,
    /// CoderError<Infallible, Infallible>>` (you'll have to bring the trait
    /// [`UnwrapInfallible`] into scope).
    ///
    /// If you use a custom backend, then the `BackendError` will typically be the
    /// [`WriteError`] type of the of the backend. For example, it could signal an I/O error
    /// if you're encoding directly to a file or socket rather than to an automatically
    /// growing in-memory buffer.
    ///
    /// [`WriteError`]: crate::backends::WriteWords::WriteError
    /// [`Infallible`]: core::convert::Infallible
    /// [`.unwrap_infallible()`]: crate::UnwrapInfallible::unwrap_infallible
    /// [`UnwrapInfallible`]: crate::UnwrapInfallible
    type BackendError: Debug;

    /// Encodes a single symbol with the given entropy model.
    ///
    /// This is the most basic encoding method. If you want to encode more than a single
    /// symbol then you may want to call [`encode_symbols`], [`try_encode_symbols`], or
    /// [`encode_iid_symbols`] instead.
    ///
    /// Note that:
    /// - the `symbol` can be passed either by value or by reference;
    /// - the `model` can also be passed by reference since, if `M` implements
    ///   `EncoderModel<PRECISION>`, then `&M` does so too; and
    /// - the `PRECISION` will typically be inferred from the entropy model, and it may not
    ///   be larger than `Word::BITS`; this is enforced by run-time assertions (that get
    ///   optimized away unless they fail) but it will be enforced at compile time in future
    ///   versions of `constriction` as soon as the type system allows this.
    ///
    /// # Errors
    ///
    /// Returns `Err(CoderError::Frontend(e))` if there was a logic error `e` during
    /// encoding (such as trying to encode a symbol with zero probability under the provided
    /// entropy model). Returns `Err(CoderError::Backend(e))` if writing compressed data
    /// lead to an I/O error `e`. Otherwise, returns `Ok(())`.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Encode};
    ///
    /// // Create an ANS Coder and an entropy model.
    /// let mut ans_coder = DefaultAnsCoder::new();
    /// let quantizer = DefaultLeakyQuantizer::new(-100i32..=100);
    /// let entropy_model = quantizer.quantize(probability::distribution::Gaussian::new(0.0, 10.0));
    ///
    /// // Encode a symbol, passing both the symbol and the entropy model by reference:
    /// ans_coder.encode_symbol(&12, &entropy_model).unwrap();
    ///
    /// // Encode a symbol, passing both the symbol and the entropy model by value:
    /// ans_coder.encode_symbol(-8, entropy_model).unwrap();
    ///
    /// // Get the compressed bitstring.
    /// let compressed = ans_coder.into_compressed();
    /// dbg!(compressed);
    /// ```
    ///
    /// [`encode_symbols`]: Self::encode_symbols
    /// [`try_encode_symbols`]: Self::try_encode_symbols
    /// [`encode_iid_symbols`]: Self::encode_iid_symbols
    fn encode_symbol<M>(
        &mut self,
        symbol: impl Borrow<M::Symbol>,
        model: M,
    ) -> Result<(), CoderError<Self::FrontendError, Self::BackendError>>
    where
        M: EncoderModel<PRECISION>,
        M::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<M::Probability>;

    /// Encodes a sequence of symbols, each with its individual entropy model.
    ///
    /// The provided iterator has to yield pairs `(symbol, entropy_model)`. The default
    /// implemnetation just calls [`encode_symbol`] for each item. You can overwrite the
    /// default implementation if your entropy coder can treat a sequence of symbols in a
    /// more efficient way.
    ///
    /// This method short-circuits as soon as encoding leads to an error (see discussion of
    /// error states for [`encode_symbol`]).
    ///
    /// This method encodes the symbols in the order in which they are yielded by the
    /// iterator. This is suitable for an encoder with "queue" semantics, like a
    /// [`RangeEncoder`]. If you're using an encoder with "stack" semantics, such as an
    /// [`AnsCoder`], then you may prefer encoding the symbols in reverse order (see
    /// [`AnsCoder::encode_symbols_reverse`]).
    ///
    /// Note that:
    /// - the `symbol`s can be yielded either by value or by reference;
    /// - the `model`s can also be yielded either by value or by reference since, if `M`
    ///   implements `EncoderModel<PRECISION>`, then `&M` does so too; and
    /// - the iterator has to yield models of a fixed type (unless you want to `Box` up each
    ///   model, which is most likely a bad idea); if you have entropy models of various
    ///   types then just call either this method or [`encode_symbol`] several times
    ///   manually.
    ///
    /// # See Also
    ///
    /// - [`try_encode_symbols`] if generating the entropy models may fail; and
    /// - [`encode_iid_symbols`] if all symbols use the same entropy model.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{model::DefaultLeakyQuantizer, queue::DefaultRangeEncoder, Encode};
    ///
    /// // Define the symbols we want to encode and the parameters of our entropy models.
    /// let quantizer = DefaultLeakyQuantizer::new(-100i32..=100);
    /// let symbols = [15, 3, -8, 2];
    /// let means = [10.2, 1.5, -3.9, 5.1];
    /// let stds = [7.1, 5.8, 10.9, 6.3];
    /// let entropy_model = quantizer.quantize(probability::distribution::Gaussian::new(0.0, 10.0));
    ///
    /// // Encode all symbols using range coding.
    /// let mut encoder1 = DefaultRangeEncoder::new();
    /// encoder1.encode_symbols(
    ///     symbols.iter().zip(&means).zip(&stds).map(
    ///         |((&symbol, &mean), &std)|
    ///             (symbol, quantizer.quantize(probability::distribution::Gaussian::new(mean, std)))
    ///     )
    /// ).unwrap();
    /// let compressed1 = encoder1.into_compressed();
    ///
    /// // The above is equivalent to:
    /// let mut encoder2 = DefaultRangeEncoder::new();
    /// for ((&symbol, &mean), &std) in symbols.iter().zip(&means).zip(&stds) {
    ///     let model = quantizer.quantize(probability::distribution::Gaussian::new(mean, std));
    ///     encoder2.encode_symbol(symbol, model).unwrap();
    /// }
    /// let compressed2 = encoder2.into_compressed();
    ///
    /// assert_eq!(compressed1, compressed2);
    /// ```
    ///
    /// [`encode_symbol`]: Self::encode_symbol
    /// [`try_encode_symbols`]: Self::try_encode_symbols
    /// [`encode_iid_symbols`]: Self::encode_iid_symbols
    /// [`RangeEncoder`]: queue::RangeEncoder
    /// [`AnsCoder`]: stack::AnsCoder
    /// [`AnsCoder::encode_symbols_reverse`]: stack::AnsCoder::encode_symbols_reverse
    #[inline]
    fn encode_symbols<S, M>(
        &mut self,
        symbols_and_models: impl IntoIterator<Item = (S, M)>,
    ) -> Result<(), CoderError<Self::FrontendError, Self::BackendError>>
    where
        S: Borrow<M::Symbol>,
        M: EncoderModel<PRECISION>,
        M::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<M::Probability>,
    {
        for (symbol, model) in symbols_and_models.into_iter() {
            self.encode_symbol(symbol, model)?;
        }

        Ok(())
    }

    /// Encodes a sequence of symbols from a fallible iterator.
    ///
    /// This method is equivalent to [`encode_symbols`](Self::encode_symbols), except that
    /// it takes a fallible iterator (i.e., an iterator that yields `Result`s). It encodes
    /// symbols as long as the iterator yields `Ok((symbol, entropy_model))` and encoding
    /// succeeds. The method short-circuits as soon as either
    /// - the iterator yields `Err(e)`, in which case it returns
    ///   `Err(TryCodingError::InvalidEntropyModel(e))` to the caller; or
    /// - encoding fails with `Err(e)`, in which case it returns
    ///   `Err(TryCodingError::CodingError(e))`.
    ///
    /// This method may be useful for parameterized entropy models whose parameters have to
    /// satisfy certain constraints (e.g., they have to be positive), but they come from an
    /// untrusted source they may violate the constraints.
    #[inline]
    fn try_encode_symbols<S, M, E>(
        &mut self,
        symbols_and_models: impl IntoIterator<Item = Result<(S, M), E>>,
    ) -> Result<(), TryCodingError<CoderError<Self::FrontendError, Self::BackendError>, E>>
    where
        S: Borrow<M::Symbol>,
        M: EncoderModel<PRECISION>,
        M::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<M::Probability>,
    {
        for symbol_and_model in symbols_and_models.into_iter() {
            let (symbol, model) = symbol_and_model.map_err(TryCodingError::InvalidEntropyModel)?;
            self.encode_symbol(symbol, model)?;
        }

        Ok(())
    }

    /// Encodes a sequence of symbols, all with the same entropy model.
    ///
    /// This method short-circuits as soon as encoding leads to an error (see discussion of
    /// error states for [`encode_symbol`]).
    ///
    /// While this method takes `model` formally by value, you'll typically want to pass the
    /// `EncoderModel` by reference (which is possible since any reference to an
    /// `EncoderModel` implements `EncoderModel` too). The bound on `M: Copy` prevents
    /// accidental misuse in this regard. We provide the ability to pass the `EncoderModel`
    /// by value as an opportunity for microoptimzations when dealing with models that can
    /// be cheaply copied (see, e.g.,
    /// [`ContiguousCategoricalEntropyModel::as_view`](model::ContiguousCategoricalEntropyModel::as_view)).
    ///
    /// Note that this method encodes the symbols in the order in which they are yielded by
    /// the iterator. This is suitable for an encoder with "queue" semantics, like a
    /// [`RangeEncoder`]. If you're using an encoder with "stack" semantics, such as an
    /// [`AnsCoder`], then you may prefer encoding the symbols in reverse order (see
    /// [`AnsCoder::encode_iid_symbols_reverse`]).
    ///
    /// If you want to encode each symbol with its individual entropy model, then consider
    /// calling [`encode_symbols`] instead. If you just want to encode a single symbol, then
    /// call [`encode_symbol`] instead.
    ///
    /// [`encode_symbol`]: Self::encode_symbol
    /// [`encode_symbols`]: Self::encode_symbols
    /// [`RangeEncoder`]: queue::RangeEncoder
    /// [`AnsCoder`]: stack::AnsCoder
    /// [`AnsCoder::encode_iid_symbols_reverse`]: stack::AnsCoder::encode_iid_symbols_reverse
    #[inline(always)]
    fn encode_iid_symbols<S, M>(
        &mut self,
        symbols: impl IntoIterator<Item = S>,
        model: M,
    ) -> Result<(), CoderError<Self::FrontendError, Self::BackendError>>
    where
        S: Borrow<M::Symbol>,
        M: EncoderModel<PRECISION> + Copy,
        M::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<M::Probability>,
    {
        self.encode_symbols(symbols.into_iter().map(|symbol| (symbol, model)))
    }

    /// Checks if there might not be any room to encode more data.
    ///
    /// If this method returns `false` then encoding one more symbol must not fail due to a
    /// full backend (it may still fail for other reasons). If this method returns `true`
    /// then it is unknown whether or not encoding one more symbol will overflow the
    /// backend.
    ///
    /// The default implementation always returns `true`, which is always correct albeit not
    /// particularly useful. Consider overwriting the default implementation and call
    /// [`WriteWords::maybe_full`] on your backend if appropriate.
    ///
    /// Calling this method can be awkward for entropy coders that implement
    /// `Encode<PRECISION>` for more than one value of `PRECISION`. The method
    /// [`Code::encoder_maybe_full`] is provided as a more convenient forwarding method.
    ///
    /// [`WriteWords::maybe_full`]: crate::backends::WriteWords::maybe_full
    #[inline(always)]
    fn maybe_full(&self) -> bool {
        true
    }
}

/// A trait for stream decoders (i.e., decompressors)
///
/// This trait defines methods for decoding a single symbol or a sequence of symbols.
///
/// # Naming Convention
///
/// This trait is deliberately called `Decode` and not `Decoder`. See corresponding comment
/// for the [`Code`] trait for the reasoning.
pub trait Decode<const PRECISION: usize>: Code {
    /// The error type for logical decoding errors.
    ///
    /// This may be [`Infallible`](core::convert::Infallible) for surjective coders, such as
    /// [`AnsCoder`](stack::AnsCoder).
    ///
    /// Frontend errors are errors that relate to the logical decoding process, such as when
    /// a user tries to decode a bitstring that is invalid under the used encoder. For
    /// decoding, this may include the case where there's no compressed data left if the
    /// decoder does not allow decoding in such a situation.
    type FrontendError: Debug;

    /// The error type for reading in encoded data.
    ///
    /// This does not include the case of running out of data, which is either represented
    /// by [`FrontendError`](Self::FrontendError), or which may even be allowed by the
    /// decoder.
    ///
    /// If you stick with the default backend(s) and don't do anything fancy, then the
    /// `BackendError` will most likely be [`Infallible`], which means that the compiler can
    /// optimize away error checks. You can explicitly express that you expect an
    /// `Infallible` error type by calling [`.unwrap_infallible()`] on a `Result<T,
    /// Infallible>` or on a `Result<T, CoderError<Infallible, Infallible>>` (you'll have to
    /// bring the trait [`UnwrapInfallible`] into scope).
    ///
    /// If you use a custom backend, then the `BackendError` will typically be the
    /// [`ReadError`] type of the of the backend. For example, it could signal an I/O error
    /// (other than "end of file") if you're decoding directly from a file or socket rather
    /// than from an in-memory buffer.
    ///
    /// [`ReadError`]: crate::backends::ReadWords::ReadError
    /// [`Infallible`]: core::convert::Infallible
    /// [`.unwrap_infallible()`]: crate::UnwrapInfallible::unwrap_infallible
    /// [`UnwrapInfallible`]: crate::UnwrapInfallible
    type BackendError: Debug;

    /// Decodes a single symbol using the given entropy model.
    ///
    /// This is the most basic decoding method. If you want to decode more than a single
    /// symbol then you may want to call [`decode_symbols`], [`try_decode_symbols`], or
    /// [`decode_iid_symbols`] instead.
    ///
    /// Note that:
    /// - the `model` can be passed either by value or by reference since, if `M` implements
    ///   `DecoderModel<PRECISION>`, then `&M` does so too; and
    /// - the `PRECISION` will typically be inferred from the entropy model, and it may not
    ///   be larger than `Word::BITS`; this is enforced by run-time assertions (that get
    ///   optimized away unless they fail) but it will be enforced at compile time in future
    ///   versions of `constriction` as soon as the type system allows this.
    ///
    /// # Errors
    ///
    /// Returns `Err(CoderError::Frontend(e))` if there was a logic error `e` during
    /// encoding (such as trying to decode an invalid compressed bitstring if your coder is
    /// not surjective, or running out of compressed data if your coder does not allow
    /// decoding in such a situation). Returns `Err(CoderError::Backend(e))` if reading
    /// compressed data lead to an I/O error `e` (other than "end of file"). Otherwise,
    /// returns `Ok(())`.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{
    ///     stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode},
    ///     UnwrapInfallible,
    /// };
    ///
    /// // Get some mock compressed data.
    /// let compressed = vec![0x1E34_22B0];
    /// // (calling this "compressed" is a bit of a misnomer since the compressed representation
    /// // of just two symbols is quite long due to the constant overhead.)
    ///
    /// // Create an ANS Coder from the compressed data and an entropy model.
    /// let mut ans_coder = DefaultAnsCoder::from_compressed(compressed).unwrap();
    /// let quantizer = DefaultLeakyQuantizer::new(-100i32..=100);
    /// let entropy_model = quantizer.quantize(probability::distribution::Gaussian::new(0.0, 10.0));
    ///
    /// // Decode a single symbol, passing the entropy model by reference:
    /// assert_eq!(ans_coder.decode_symbol(&entropy_model).unwrap_infallible(), -8);
    ///
    /// // Decode another symbol using the same entropy model, this time passing it by value:
    /// assert_eq!(ans_coder.decode_symbol(entropy_model).unwrap_infallible(), 12);
    ///
    /// // Verify that we've consumed all data on the coder.
    /// assert!(ans_coder.is_empty());
    /// ```
    ///
    /// [`decode_symbols`]: Self::decode_symbols
    /// [`try_decode_symbols`]: Self::try_decode_symbols
    /// [`decode_iid_symbols`]: Self::decode_iid_symbols
    fn decode_symbol<D>(
        &mut self,
        model: D,
    ) -> Result<D::Symbol, CoderError<Self::FrontendError, Self::BackendError>>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<D::Probability>;

    /// Decodes a sequence of symbols, using an individual entropy model for each symbol.
    ///
    /// This method is lazy: it doesn't actually decode anything until you iterate over the
    /// returned iterator.
    ///
    /// The return type implements `Iterator<Item = Result<M::Symbol, ...>>`, and it
    /// implements `ExactSizeIterator` if the provided iterator `models` implements
    /// `ExactSizeIterator`. The provided iterator `models` may yield entropy models either
    /// by value or by reference since, if `M` implements `DecoderModel<PRECISION>`, then
    /// `&M` does so too.
    ///
    /// This method does *not* short-circuit. If an error `e` occurs then the returned
    /// iterator will yield `Err(e)` but you can, in principle, continue to iterate over it.
    /// In practice, however, continuing to iterate after an error will likely yield either
    /// further errors or garbled symbols. Therefore, you'll usually want to short-circuit
    /// the returned iterator yourself, e.g., by applying the `?` operator on each yielded
    /// item or by `collect`ing the items from the returned iterator into a
    /// `Result<Container<_>, _>` as in the second example below.
    ///
    /// # Examples
    ///
    /// ## Iterate over the decoded symbols in a `for` loop
    ///
    /// ```
    /// use constriction::{
    ///     stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode},
    ///     UnwrapInfallible,
    /// };
    ///
    /// // Create a decoder from some mock compressed data and create some mock entropy models.
    /// let compressed = vec![0x2C63_D22E, 0x0000_0377];
    /// let mut decoder = DefaultAnsCoder::from_compressed(compressed).unwrap();
    /// let quantizer = DefaultLeakyQuantizer::new(-100i32..=100);
    /// let entropy_models = (0..5).map(
    ///     |i| quantizer.quantize(probability::distribution::Gaussian::new((i * 10) as f64, 10.0))
    /// );
    ///
    /// // Decode the symbols and iterate over them in a `for` loop:
    /// for symbol in decoder.decode_symbols(entropy_models) {
    ///     dbg!(symbol.unwrap_infallible()); // Prints the symbols `-3`, `12`, `19`, `28`, and `41`.
    /// }
    /// ```
    ///
    /// ## `collect` the decoded symbols into a `Result<Vec<M::Symbol>, _>`
    ///
    /// A useful trick when you have an iterator over `Result`s is that you can `collect` it
    /// into `Result<SomeContainer<_>, _>`:
    ///
    /// ```
    /// # use constriction::{
    /// #     stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode},
    /// #     UnwrapInfallible,
    /// # };
    /// # let compressed = vec![0x2C63_D22E, 0x0000_0377];
    /// # let mut decoder = DefaultAnsCoder::from_compressed(compressed).unwrap();
    /// # let quantizer = DefaultLeakyQuantizer::new(-100i32..=100);
    /// # let entropy_models = (0..5).map(
    /// #     |i| quantizer.quantize(probability::distribution::Gaussian::new((i * 10) as f64, 10.0))
    /// # );
    /// // Assume same setup as in previous example.
    /// let symbols = decoder.decode_symbols(entropy_models).collect::<Result<Vec<_>, _>>();
    /// assert_eq!(symbols.unwrap_infallible(), [-3, 12, 19, 28, 41]);
    /// ```
    ///
    /// This results in only a single memory allocation of the exact correct size, and it
    /// automatically short-circuits upon error (which is moot in this example because, for
    /// this particular entropy coder, both `FrontendError` and `BackendError` are
    /// `Infallible`, i.e., decoding cannot fail).
    ///
    /// # See Also
    ///
    /// - [`decode_symbol`] if you want to decode only   a single symbol;
    /// - [`try_decode_symbols`] if generating the entropy models may fail; and
    /// - [`decode_iid_symbols`] if all symbols use the same entropy model.
    ///
    /// [`decode_symbol`]: Self::decode_symbol
    /// [`try_decode_symbols`]: Self::try_decode_symbols
    /// [`decode_iid_symbols`]: Self::decode_iid_symbols
    #[inline(always)]
    fn decode_symbols<'s, I, M>(
        &'s mut self,
        models: I,
    ) -> DecodeSymbols<'s, Self, I::IntoIter, PRECISION>
    where
        I: IntoIterator<Item = M> + 's,
        M: DecoderModel<PRECISION>,
        M::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<M::Probability>,
    {
        DecodeSymbols {
            decoder: self,
            models: models.into_iter(),
        }
    }

    /// Decodes a sequence of symbols from a fallible iterator over entropy models.
    ///
    /// This method is equivalent to [`decode_symbols`], except that it takes a fallible
    /// iterator (i.e., an iterator that yields `Result`s).
    ///
    /// Just like [`decode_symbols`],
    /// - this method is lazy, i.e., it doesn't decode until you iterate over the returned
    ///   iterator;
    /// - the returned iterator implements `ExactSizeIterator` if `models` implements
    ///   `ExactSizeIterator`; and
    /// - you can, in principle, continue to decode after an error, but you'll likely rather
    ///   want to short-circuit the returned iterator. This applies to both decoding errors
    ///   and to errors from the supplied iterator `models`.
    ///
    /// The returned iterator yields
    /// - `Ok(symbol)` on success;
    /// - `Err(TryCodingError::InvalidEntropyModel(e))` if the iterator `models` yielded
    ///   `Err(e)`; and
    /// - `Err(TryCodingError::CodingError(e))` if decoding resulted in `Err(e)`.
    ///
    /// This method may be useful for parameterized entropy models whose parameters have to
    /// satisfy certain constraints (e.g., they have to be positive), but they come from an
    /// untrusted source they may violate the constraints.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{
    ///     stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode, TryCodingError},
    ///     CoderError,
    /// };
    /// use core::convert::Infallible;
    ///
    /// /// Helper function to decode symbols with Gaussian entropy models with untrusted parameters.
    /// fn try_decode_gaussian(compressed: Vec<u32>, means: &[f64], stds: &[f64])
    ///     -> Result<Vec<i32>, TryCodingError<CoderError<Infallible, Infallible>, &'static str>>
    /// {
    ///     let mut decoder = DefaultAnsCoder::from_compressed(compressed).unwrap();
    ///     let quantizer = DefaultLeakyQuantizer::new(-100i32..=100);
    ///     let entropy_models = means.iter().zip(stds).map(|(&mean, &std)| {
    ///         if std > 0.0 && mean.is_finite() {
    ///             Ok(quantizer.quantize(probability::distribution::Gaussian::new(mean, std)))
    ///         } else {
    ///             Err("argument error")
    ///         }
    ///     });
    ///     decoder.try_decode_symbols(entropy_models).collect::<Result<Vec<_>, _>>()
    /// }
    ///
    /// let compressed = vec![0x2C63_D22E, 0x0000_0377];
    /// let means = [0.0, 10.0, 20.0, 30.0, 40.0];
    /// let valid_stds = [10.0, 10.0, 10.0, 10.0, 10.0];
    /// let invalid_stds = [10.0, 10.0, -1.0, 10.0, 10.0]; // Invalid: negative standard deviation
    ///
    /// assert_eq!(
    ///     try_decode_gaussian(compressed.clone(), &means, &valid_stds),
    ///     Ok(vec![-3, 12, 19, 28, 41])
    /// );
    /// assert_eq!(
    ///     try_decode_gaussian(compressed, &means, &invalid_stds),
    ///     Err(TryCodingError::InvalidEntropyModel("argument error"))
    /// );
    /// ```
    ///
    /// [`decode_symbols`]: Self::decode_symbols
    #[inline(always)]
    fn try_decode_symbols<'s, I, M, E>(
        &'s mut self,
        models: I,
    ) -> TryDecodeSymbols<'s, Self, I::IntoIter, PRECISION>
    where
        I: IntoIterator<Item = Result<M, E>> + 's,
        M: DecoderModel<PRECISION>,
        M::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<M::Probability>,
    {
        TryDecodeSymbols {
            decoder: self,
            models: models.into_iter(),
        }
    }

    /// Decodes `amt` symbols using the same entropy model for all symbols.
    ///
    /// The return type implements `ExactSizeIterator<Item = Result<M::Symbol, ...>>`.
    ///
    /// Just like [`decode_symbols`],
    /// - this method is lazy, i.e., it doesn't decode until you iterate over the returned
    ///   iterator; and
    /// - you can, in principle, continue to decode after an error, but you'll likely rather
    ///   want to short-circuit the returned iterator.
    ///
    /// While this method takes `model` formally by value, you'll typically want to pass the
    /// `DecoderModel` by reference (which is possible since any reference to a
    /// `DecoderModel` implements `DecoderModel` too). The bound on `M: Copy` prevents
    /// accidental misuse in this regard. We provide the ability to pass the `DecoderModel`
    /// by value as an opportunity for microoptimzations when dealing with models that can
    /// be cheaply copied (see, e.g.,
    /// [`ContiguousLookupDecoderModel::as_view`](crate::stream::model::ContiguousLookupDecoderModel::as_view)).
    ///
    /// If you want to decode each symbol with its individual entropy model, then consider
    /// calling [`decode_symbols`] instead. If you just want to decode a single symbol, then
    /// call [`decode_symbol`] instead.
    ///
    /// [`decode_symbols`]: Self::decode_symbols
    /// [`decode_symbol`]: Self::decode_symbol
    #[inline(always)]
    fn decode_iid_symbols<M>(
        &mut self,
        amt: usize,
        model: M,
    ) -> DecodeIidSymbols<'_, Self, M, PRECISION>
    where
        M: DecoderModel<PRECISION> + Copy,
        M::Probability: Into<Self::Word>,
        Self::Word: AsPrimitive<M::Probability>,
    {
        DecodeIidSymbols {
            decoder: self,
            model,
            amt,
        }
    }

    /// Checks if there might be no compressed data left for decoding.
    ///
    /// If this method returns `false` then there must be additional data left to decode. If
    /// this method returns `true` then it is unknown whether or not the decoder is
    /// exhausted.
    ///
    /// The default implementation always returns `true`, which is always correct albeit not
    /// particularly useful. Consider overwriting the default implementation and call
    /// [`ReadWords::maybe_exhausted`] on your backend if appropriate.
    ///
    /// This method is useful for checking for consistency. If you have decoded all data
    /// that you expect to be on a decoder but this method still returns `false` then either
    /// your code has a bug or the compressed data was corrupted. If this method returns
    /// `true` then that's no guarantee that everything is correct, but at least there's no
    /// evidence of the contrary.
    ///
    /// Calling this method can be awkward for entropy coders that implement
    /// `Decode<PRECISION>` for more than one value of `PRECISION`. The method
    /// [`Code::decoder_maybe_exhausted`] is provided as a more convenient forwarding
    /// method.
    ///
    /// [`ReadWords::maybe_exhausted`]: crate::backends::ReadWords::maybe_exhausted
    fn maybe_exhausted(&self) -> bool {
        true
    }
}

/// A trait for permanent conversion into a matching decoder type.
///
/// This is useful for generic code that encodes some data with a user provided
/// encoder of generic type, and then needs to obtain a compatible decoder.
///
/// # See also
///
/// - [`AsDecoder`], if you want only a temporary decoder, or if you don't own the encoder.
///
/// # Example
///
/// To be able to convert an encoder of generic type `Encoder` into a decoder,
/// declare a trait bound `Encoder: IntoDecoder<PRECISION>`. In the following
/// example, differences to the example for [`AsDecoder`] are marked by `// <--`.
///
/// ```
/// use constriction::stream::{
///     model::{EncoderModel, DecoderModel, LeakyQuantizer},
///     stack::DefaultAnsCoder,
///     Decode, Encode, IntoDecoder
/// };
///
/// fn encode_and_decode<Encoder, D, const PRECISION: usize>(
///     mut encoder: Encoder, // <-- Needs ownership of `encoder`.
///     model: D
/// ) -> Encoder::IntoDecoder
/// where
///     Encoder: Encode<PRECISION> + IntoDecoder<PRECISION>, // <-- Different trait bound.
///     D: EncoderModel<PRECISION, Symbol=i32> + DecoderModel<PRECISION, Symbol=i32> + Copy,
///     D::Probability: Into<Encoder::Word>,
///     Encoder::Word: num_traits::AsPrimitive<D::Probability>
/// {
///     encoder.encode_symbol(137, model);
///     let mut decoder = encoder.into_decoder();
///     let decoded = decoder.decode_symbol(model).unwrap();
///     assert_eq!(decoded, 137);
///
///     // encoder.encode_symbol(42, model); // <-- This would fail (we moved `encoder`).
///     decoder // <-- We can return `decoder` as it has no references to the current stack frame.
/// }
///
/// // Usage example:
/// let encoder = DefaultAnsCoder::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=200);
/// let model = quantizer.quantize(probability::distribution::Gaussian::new(0.0, 50.0));
/// encode_and_decode(encoder, model);
/// ```
///
pub trait IntoDecoder<const PRECISION: usize>: Encode<PRECISION> {
    /// The target type of the conversion.
    type IntoDecoder: Decode<PRECISION, Word = Self::Word, State = Self::State>;

    /// Performs the conversion.
    fn into_decoder(self) -> Self::IntoDecoder;
}

/// A trait for temporary conversion into a matching decoder type.
///
/// This is useful for generic code that encodes some data with a user provided
/// encoder of generic type, and then needs to obtain a compatible decoder.
///
/// # See also
///
/// - [`IntoDecoder`], if you don't need the encoder anymore, or if you want to return the
///   resulting decoder from the current stack frame. Needs ownership of the encoder.
///
/// # Example
///
/// To be able to temporarily convert an encoder of generic type `Encoder` into a
/// decoder, declare a trait bound `for<'a> Encoder: AsDecoder<'a, PRECISION>`. In the
/// following example, differences to the example for [`IntoDecoder`] are marked by `//
/// <--`.
///
/// ```
/// use constriction::stream::{
///     model::{EncoderModel, DecoderModel, LeakyQuantizer},
///     stack::DefaultAnsCoder,
///     Decode, Encode, AsDecoder
/// };
///
/// fn encode_decode_encode<Encoder, D, const PRECISION: usize>(
///     encoder: &mut Encoder, // <-- Doesn't need ownership of `encoder`.
///     model: D
/// )
/// where
///     Encoder: Encode<PRECISION>,
///     for<'a> Encoder: AsDecoder<'a, PRECISION>, // <-- Different trait bound.
///     D: EncoderModel<PRECISION, Symbol=i32> + DecoderModel<PRECISION, Symbol=i32> + Copy,
///     D::Probability: Into<Encoder::Word>,
///     Encoder::Word: num_traits::AsPrimitive<D::Probability>
/// {
///     encoder.encode_symbol(137, model);
///     let mut decoder = encoder.as_decoder();
///     let decoded = decoder.decode_symbol(model).unwrap(); // (Doesn't mutate `encoder`.)
///     assert_eq!(decoded, 137);
///
///     std::mem::drop(decoder); // <-- We have to explicitly drop `decoder` ...
///     encoder.encode_symbol(42, model); // <-- ... before we can use `encoder` again.
/// }
///
/// // Usage example:
/// let mut encoder = DefaultAnsCoder::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=200);
/// let model = quantizer.quantize(probability::distribution::Gaussian::new(0.0, 50.0));
/// encode_decode_encode(&mut encoder, model);
/// ```
pub trait AsDecoder<'a, const PRECISION: usize>: Encode<PRECISION> + 'a {
    /// The target type of the conversion.
    type AsDecoder: Decode<PRECISION, Word = Self::Word, State = Self::State> + 'a;

    /// Performs the conversion.
    fn as_decoder(&'a self) -> Self::AsDecoder;
}

/// The iterator returned by [`Decode::decode_symbols`].
#[derive(Debug)]
pub struct DecodeSymbols<'a, Decoder: ?Sized, I, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    models: I,
}

impl<Decoder, I, D, const PRECISION: usize> Iterator for DecodeSymbols<'_, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = D>,
    D: DecoderModel<PRECISION>,
    Decoder::Word: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::Word>,
{
    type Item = Result<
        <I::Item as EntropyModel<PRECISION>>::Symbol,
        CoderError<Decoder::FrontendError, Decoder::BackendError>,
    >;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.models
            .next()
            .map(|model| self.decoder.decode_symbol(model))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.models.size_hint()
    }
}

impl<Decoder, I, D, const PRECISION: usize> ExactSizeIterator
    for DecodeSymbols<'_, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = D> + ExactSizeIterator,
    D: DecoderModel<PRECISION>,
    Decoder::Word: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::Word>,
{
}

/// The iterator returned by [`Decode::try_decode_symbols`].
#[derive(Debug)]
pub struct TryDecodeSymbols<'a, Decoder: ?Sized, I, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    models: I,
}

impl<Decoder, I, D, E, const PRECISION: usize> Iterator
    for TryDecodeSymbols<'_, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = Result<D, E>>,
    D: DecoderModel<PRECISION>,
    Decoder::Word: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::Word>,
{
    type Item = Result<
        D::Symbol,
        TryCodingError<CoderError<Decoder::FrontendError, Decoder::BackendError>, E>,
    >;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.models.next().map(|model| {
            Ok(self
                .decoder
                .decode_symbol(model.map_err(TryCodingError::InvalidEntropyModel)?)?)
        })
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // We don't terminate when we encounter an error, so the size doesn't change.
        self.models.size_hint()
    }
}

impl<Decoder, I, D, E, const PRECISION: usize> ExactSizeIterator
    for TryDecodeSymbols<'_, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = Result<D, E>> + ExactSizeIterator,
    D: DecoderModel<PRECISION>,
    Decoder::Word: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::Word>,
{
}

/// The iterator returned by [`Decode::decode_iid_symbols`].
#[derive(Debug)]
pub struct DecodeIidSymbols<'a, Decoder: ?Sized, M, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    model: M,
    amt: usize,
}

impl<Decoder, M, const PRECISION: usize> Iterator for DecodeIidSymbols<'_, Decoder, M, PRECISION>
where
    Decoder: Decode<PRECISION>,
    M: DecoderModel<PRECISION> + Copy,
    Decoder::Word: AsPrimitive<M::Probability>,
    M::Probability: Into<Decoder::Word>,
{
    type Item = Result<M::Symbol, CoderError<Decoder::FrontendError, Decoder::BackendError>>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.amt != 0 {
            self.amt -= 1;
            Some(self.decoder.decode_symbol(self.model))
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.amt, Some(self.amt))
    }
}

impl<Decoder, M, const PRECISION: usize> ExactSizeIterator
    for DecodeIidSymbols<'_, Decoder, M, PRECISION>
where
    Decoder: Decode<PRECISION>,
    M: DecoderModel<PRECISION> + Copy,
    Decoder::Word: AsPrimitive<M::Probability>,
    M::Probability: Into<Decoder::Word>,
{
}

/// The error type for [`Encode::try_encode_symbols`] and [`Decode::try_decode_symbols`].
#[derive(Debug, PartialEq, Eq, Hash)]
pub enum TryCodingError<CodingError, ModelError> {
    /// The iterator provided to [`Encode::try_encode_symbols`] or
    /// [`Decode::try_decode_symbols`] yielded `Err(_)`.
    ///
    /// The variant wraps the original error, which can also be retrieved via
    /// [`Error::source`] if both `ModelError` and `CodingError` implement
    /// [`std::error::Error`] and if not compiled in `no_std` mode.
    ///
    /// [`Error::source`]: std::error::Error::source
    InvalidEntropyModel(ModelError),

    /// The iterator provided to [`Encode::try_encode_symbols`] or
    /// [`Decode::try_decode_symbols`] yielded `Ok(_)` but encoding or decoding resulted in
    /// an error.
    ///
    /// The variant wraps the original error, which can also be retrieved via
    /// [`Error::source`] if both `ModelError` and `CodingError` implement
    /// [`std::error::Error`] and if not compiled in `no_std` mode.
    ///
    /// [`Error::source`]: std::error::Error::source
    CodingError(CodingError),
}

impl<CodingError: Display, ModelError: Display> Display
    for TryCodingError<CodingError, ModelError>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidEntropyModel(err) => {
                write!(f, "Error while constructing entropy model or data: {err}")
            }
            Self::CodingError(err) => {
                write!(f, "Error while entropy coding: {err}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl<CodingError: std::error::Error + 'static, ModelError: std::error::Error + 'static>
    std::error::Error for TryCodingError<CodingError, ModelError>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
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
