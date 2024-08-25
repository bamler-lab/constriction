//! Probability distributions that can be used as entropy models for stream codes.
//!
//! This module provides utilities for dealing with probabilistic models of data sources
//! ("entropy models") in exactly invertible fixed-point arithmetic so that no rounding
//! errors occur. As explained in the [motivation](#motivation) below, preventing rounding
//! errors is necessary for reliable entropy coding.
//!
//! The types defined in this module approximate arbitrary discrete (or quantized
//! one-dimensional continuous) probability distributions with a fixed-point representation.
//! The fixed-point representation has customizable numeric precision and can be either
//! explicit or implicit (i.e., lazy). While the conversion to the fixed-point approximation
//! itself generally involves rounding, once the fixed-point representation is obtained,
//! operations on it are exact. Therefore, the fixed-point representation can be used for
//! entropy coding.
//!
//! # Module Overview
//!
//! This module declares the base trait [`EntropyModel`] and its subtraits [`EncoderModel`]
//! and [`DecoderModel`], which specify the interfaces that entropy models provide and that
//! entropy coders in the sister modules can rely on.
//!
//! In addition, this module provides the following three utilities for constructing entropy
//! models:
//! - an adapter that converts parameterized discrete distributions (e.g., [`Binomial`]) or
//!   one-dimensional continuous probability distributions (e.g. [`Gaussian`]) from a
//!   representation in terms of float-valued functions to an (implicit) exactly invertible
//!   fixed-point representation; when provided with a continuous distribution (a
//!   probability density) then this adapter also quantizes the data space into bins. See
//!   [`DefaultLeakyQuantizer`] and [`SmallLeakyQuantizer`];
//! - types for representing arbitrary categorical distributions in an explicit fixed-point
//!   representation; these types are intended either as fallbacks for probability
//!   distributions that lack an efficiently evaluable analytic expression of the cumulative
//!   distribution function (and that therefore can't be handled by the above adaptor), or
//!   for efficient *encoding* of i.i.d. symbols by precalculating and tabularizing the
//!   fixed-point representation of each allowed symbol. See [`DefaultLeakyQuantizer`],
//!   [`DefaultContiguousCategoricalEntropyModel`],
//!   [`DefaultNonContiguousCategoricalEncoderModel`], and
//!   [`DefaultNonContiguousCategoricalDecoderModel`] (and their respective counterparts
//!   with the "Small" instead of "Default" preset); and
//! - types for high-performance "lookup tables" that enable efficient
//!   *decoding* of i.i.d. data; these types build up a lookup table with `2^PRECISION`
//!   entries (one entry per
//!   possible *quantile*) and are therefore only recommended to be used with relatively
//!   small `PRECISION`. See [`SmallContiguousLookupDecoderModel`] and
//!   [`SmallNonContiguousLookupDecoderModel`].
//!
//! # Examples
//!
//! See [`LeakyQuantizer`](LeakyQuantizer#examples), [`ContiguousCategoricalEntropyModel`],
//! [`NonContiguousCategoricalEncoderModel`]. [`NonContiguousCategoricalDecoderModel`], and
//! [`LookupDecoderModel`].
//!
//! TODO: direct links to "Examples" sections.
//!
//! # Motivation
//!
//! The general idea of entropy coding to find an optimal compression strategy by using a
//! *probabilistic model of the data source*. Ideally, all conceivable data points would be
//! compressed into a short bit string. However, short bit strings are a scarce commodity:
//! for any integer `N`, there are only `2^N - 1` distinct bit strings that are shorter than
//! `N` bits. For this reason, entropy coding algorithms assign the scarce short bit strings
//! to data points that are most probable to appear in practice, while assigning longer bit
//! strings to data points that may be possible in principle but that are extremely
//! improbable in practice. More precisely, entropy coding aims to minimize the expected bit
//! rate under a probabilistic model of the data source. We refer to this model as an
//! "entropy model".
//!
//! In contrast to many other use cases of probabilistic models in computing, entropy models
//! must be amenable to *exact* arithmetic operations. In particular, no rounding errors are
//! allowed when inverting the cumulative distribution function. Even a single arbitrarily
//! small rounding error could set off a chain reaction leading to arbitrarily large and
//! arbitrarily many errors when compressing and then decompressing a sequence of symbols
//! (see, e.g., the [motivating example for the `ChainCoder`](super::chain#motivation)).
//! This module provides utilities for defining entropy models that can be inverted exactly
//! without any rounding errors.
//!
//! # Zero Probability
//!
//! All entropy models provided in this module have a predictable support, i.e., it is
//! always easy to predict exactly which symbols have nonzero probability under the model.
//! This is an important property for entropy coding because trying to encode a symbol that
//! has zero probability under the used entropy model would fail.
//!
//! When constructing an entropy model then the caller always has to provide a support
//! (either as an integer range or as a list of symbols of arbitrary type). All entropy
//! models in this module enforce the following constraints:
//!
//! 1. all symbols within the user-provided support are assigned at least the smallest
//!    nonzero probability that is representable at the used fixed-point `PRECISION` (even
//!    if naive rounding would lead to zero probability);
//! 2. all symbols that are not in the user-provided support have probability zero;
//! 3. the probabilities add up to one (this even holds when, e.g., quantizing a continuous
//!    probability distribution on a finite support that is smaller than the continuous
//!    distribution's possibly unbounded support); and
//! 4. no single symbol has probability one, i.e., we disallow degenerate entropy models
//!    that put all probability mass on a single symbol, as such models can lead to problems
//!    in some entropy coders (if you don't know whether you may encounter degenerate
//!    entropy models for some symbols, just check for degeneracy and encode nothing in that
//!    case since the corresponding symbols can be trivially reconstructed).
//!
//! When entropy models are constructed from a floating-point representation of some
//! probability distribution then rounding is done in such a way that the above constraints
//! are satisfied. When entropy models are constructed by passing in probabilities that are
//! already in fixed-point representation, then the constructor verifies the above
//! constraints in an efficient way.
//!
//! While constraints (1) and (4) above are strictly enforced (for types defined in this
//! module), constraints (2) and (3) hold in practice but must not be relied on for memory
//! safety as they can technically be violated without the use of `unsafe` (by using a
//! [`LeakyQuantizer`] with an invalid [`Distribution`], i.e., one whose cumulative
//! distribution function either isn't monotonic or has an image that exceeds the interval
//! `[0, 1]`).
//!
//! [`stack`]: super::stack
//! [`queue`]: super::queue
//! [`Binomial`]: probability::distribution::Binomial
//! [`Gaussian`]: probability::distribution::Gaussian

/// Re-export of [`probability::distribution::Distribution`].
///
/// Most users will never have to interact with this trait directly. When a method requires
/// a type that implements `Distribution`, most users will likely use a predefined type from
/// the [`probability`] crate. You only need to implement this trait if you want to use a
/// probability distribution that is not (yet) provided by the `probability` crate.
///
/// # See Also
///
/// - [`Inverse`]
///
/// [`probability::distribution::Distribution`]:
///     https://docs.rs/probability/latest/probability/distribution/trait.Distribution.html
/// [`probability`]: https://docs.rs/probability/latest/probability/
pub use probability::distribution::Distribution;

/// Re-export of [`probability::distribution::Inverse`].
///
/// Most users will never have to interact with this trait directly. When a method requires
/// a type that implements `Inverse`, most users will likely use a predefined type from
/// the [`probability`] crate. You only need to implement this trait if you want to use a
/// probability distribution that is not (yet) provided by the `probability` crate.
///
/// # See Also
///
/// - [`Distribution`]
///
/// [`probability::distribution::Inverse`]:
///     https://docs.rs/probability/latest/probability/distribution/trait.Inverse.html
/// [`probability`]: https://docs.rs/probability/latest/probability/
pub use probability::distribution::Inverse;

mod categorical;
mod quantize;
mod uniform;

use core::{borrow::Borrow, hash::Hash};

use alloc::{boxed::Box, vec::Vec};

use num_traits::{float::FloatCore, AsPrimitive, One, Zero};

use crate::{BitArray, NonZeroBitArray};

/// Base trait for probabilistic models of a data source.
///
/// All entropy models (see [module level documentation](self)) that can be used for
/// encoding and/or decoding with stream codes must implement this trait and at least one of
/// [`EncoderModel`] and/or [`DecoderModel`]. This trait exposes the type of [`Symbol`]s
/// over which the entropy model is defined, the type that is used to represent a
/// [`Probability`] in fixed-point arithmetic, and the fixed point `PRECISION` (see
/// [discussion of type parameters](super#type-parameters-of-entropy-models)).
///
/// # Blanket Implementation for `&impl EntropyModel`
///
/// We provide the following blanket implementation for references to `EntropyModel`s:
///
/// ```ignore
/// impl<M, const PRECISION: usize> EntropyModel<PRECISION> for &M
/// where
///     M: EntropyModel<PRECISION> + ?Sized
/// { ... }
/// ```
///
/// This means that, if some type `M` implements `EntropyModel<PRECISION>` for some
/// `PRECISION`, then so does the reference type `&M`. Analogous blanket implementations are
/// provided for the traits [`EncoderModel`], [`DecoderModel`], and
/// [`IterableEntropyModel`]. The implementations simply delegate all calls to `M` (which is
/// possible since all methods only take an `&self` receiver). Therefore:
/// - you don't need to (and, in fact, currently can't) implement `EntropyModel`,
///   `EncoderModel`, or `DecoderModel` for reference types `&M`; just implement these
///   traits for "value types" `M` and you'll get the implementation for the corresponding
///   reference types for free.
/// - when you write a function or method that takes a generic entropy model as an argument,
///   always take the entropy model (formally) *by value* (i.e., declare your function as
///   `fn f(model: impl EntropyModel<PRECISION>)` or as `f<M:
///   EntropyModel<PRECISION>>(model: M)`). Since all references to `EntropyModel`s are also
///   `EntropyModel`s themselves, a function with one of these signatures can be called with
///   an entropy model passed in either by value or by reference. If your function or method
///   needs to pass out several copies of `model` then add an extra bound `M: Copy` (see,
///   e.g., [`Encode::encode_iid_symbols`](super::Encode::encode_iid_symbols)). This will
///   allow users to call your function either with a reference to an entropy model (all
///   shared references implement `Copy`), or with some cheaply copyable entropy model such
///   as a view to a lookup model (see [`LookupDecoderModel::as_view`]).
///
/// # See Also
///
/// - [`EncoderModel`]
/// - [`DecoderModel`]
///
/// [`Symbol`]: Self::Symbol
/// [`Probability`]: Self::Probability
pub trait EntropyModel<const PRECISION: usize> {
    /// The type of data over which the entropy model is defined.
    ///
    /// This is the type of an item of the *uncompressed* data.
    ///
    /// Note that, although any given `EntropyModel` has a fixed associated `Symbol` type,
    /// this doesn't prevent you from encoding heterogeneous sequences of symbols where each
    /// symbol has a different type. You can use a different `EntropyModel` with a different
    /// associated `Symbol` type for each symbol.
    type Symbol;

    /// The type used to represent probabilities, cumulatives, and quantiles.
    ///
    /// This is a primitive unsigned integer type that must hold at least `PRECISION` bits.
    /// An integer value `p: Probability` semantically represents the probability,
    /// cumulative, or quantile `p * 2.0^(-PRECISION)` (where `^` denotes exponentiation and
    /// `PRECISION` is a const generic parameter of the trait `EntropyModel`).
    ///
    /// In many places where `constriction`'s public API *returns* probabilities, they have
    /// already been verified to be nonzero. In such a case, the probability is returned as
    /// a `Probability::NonZero`, which denotes the corresponding non-zeroable type (e.g.,
    /// if `Probability` is `u32` then `Probability::NonZero` is
    /// [`NonZeroU32`](core::num::NonZeroU32)). The "bare" `Probability` type is mostly used
    /// for left-cumulatives and quantiles (i.e., for points on the y-axis in the graph of a
    /// cumulative distribution function).
    ///
    /// # Enforcing the Constraints
    ///
    /// Implementations of `EntropyModel` are encouraged to enforce the constraint
    /// `1 <= PRECISION <= Probability::BITS`. The simplest way to do so is by stating it as an
    /// assertion `assert!(1 <= PRECISION && PRECISION <= Probability::BITS)` at the beginning of
    /// relevant methods. This assertion has zero runtime cost because it can be
    /// trivially evaluated at compile time and therefore will be optimized out if it holds.
    /// As of `constriction` 0.4, implementations provided by `constriction` include a similar
    /// assertion that is checked at compile time using const evaluation tricks.
    ///
    /// # (Internal) Representation of Probability One
    ///
    /// The case of "probability one" is treated specially. This case does not come up in
    /// the public API since we disallow probability one for any individual symbol under any
    /// entropy model, and since all left-sided cumulatives always correspond to a symbol
    /// with nonzero probability. But the value "one" still comes up internally as the
    /// right-cumulative of the last allowed symbol for any model. Although our treatment of
    /// "probability one" can thus be considered an implementation detail, it is likely to
    /// become an issue in third-party implementations of `EntropyModel`, so it is worth
    /// documenting our recommended treatment.
    ///
    /// We internally represent "probability one" by its normal fixed-point representation
    /// of `p = 1 << PRECISION` (i.e., `p = 2^PRECISION` in mathematical notation) if this
    /// value fits into `Probability`, i.e., if `PRECISION != Probability::BITS`. In the
    /// (uncommon) case where `PRECISION == Probability::BITS`, we represent "probability
    /// one" as the integer zero (i.e., cutting off the overflowing bit). This means that
    /// any probability that is guaranteed to not be one can always be calculated by
    /// subtracting its left-sided cumulative from its right-sided cumulative in wrapping
    /// arithmetic. However, this convention means that one has to be careful not to confuse
    /// probability zero with probabilty one. In our implementations, these two possible
    /// interpretations of the integer `p = 0` always turned out to be easy to disambiguate
    /// statically.
    type Probability: BitArray;
}

/// A trait for [`EntropyModel`]s that can be used for encoding (compressing) data.
///
/// As discussed in the [module level documentation](self), all stream codes in
/// `constriction` use so-called [`EntropyModel`]s for encoding and/or decoding data. Some
/// of these `EntropyModel`s may be used only for encoding, only for decoding, or for both,
/// depending on their internal representation.
///
/// This `EncoderModel` trait is implemented for all entropy models that can be used for
/// *encoding* data. To encode data with an `EncoderModel`, construct an entropy coder that
/// implements the [`Encode`] trait and pass the data and the entropy model to one of the
/// methods of the [`Encode`] trait (or to an inherent method of the entropy coder, such as
/// [`AnsCoder::encode_symbols_reverse`]).
///
/// # Blanket Implementation for `&impl EncoderModel`
///
/// We provide the following blanket implementation for references to `EncoderModel`s:
///
/// ```ignore
/// impl<M, const PRECISION: usize> EncoderModel<PRECISION> for &M
/// where
///     M: EncoderModel<PRECISION> + ?Sized
/// { ... }
/// ```
///
/// This means that, if some type `M` implements `EncoderModel<PRECISION>` for some
/// `PRECISION`, then so does the reference type `&M`. Therefore, generic functions or
/// methods should never take a generic `EncoderModel` by reference. They should always take
/// the generic `EncoderModel` *by value* because this also covers the case of references
/// but is strictly more general. If your generic function needs to be able to cheaply copy
/// the `EncoderModel` (as it could with a shared reference) then it should still take the
/// generic `EncoderModel` formally by value and just add an additional `Copy` bound (see,
/// e.g., the method signature of [`Encode::encode_iid_symbols`]. For a more elaborate
/// explanation, please refer to the discussion of the analogous blanket implementation for
/// [`EntropyModel`].
///
/// # See Also
///
/// - base trait: [`EntropyModel`]
/// - sister trait: [`DecoderModel`]
/// - used with: [`Encode`]
///
/// [`Encode`]: super::Encode
/// [`AnsCoder::encode_symbols_reverse`]: super::stack::AnsCoder::encode_symbols_reverse
/// [`Encode::encode_iid_symbols`]: super::Encode::encode_iid_symbols
pub trait EncoderModel<const PRECISION: usize>: EntropyModel<PRECISION> {
    /// Looks up a symbol in the entropy model.
    ///
    /// Takes a `symbol` either by value or by reference and looks it up in the entropy
    /// model.
    /// - If `symbol` has a nonzero probability under the model, then this method returns
    ///   `Some((left_sided_cumulative, probability))`, where `probability` is the
    ///   probability in fixed-point representation (see
    ///   [discussion](EntropyModel::Probability)) and `left_sided_cumulative` is the sum of
    ///   the probabilities of all symbols up to but not including `symbol` (also in
    ///   fixed-point representation). Both `left_sided_cumulative` and `probability` are
    ///   guaranteed to be strictly smaller than `1 << PRECISION` (which would semantically
    ///   represent "probability one") because `probability` is nonzero and because we don't
    ///   support degenerate entropy models that put all probability mass on a single
    ///   symbol.
    /// - If `symbol` has zero probability under the model, then this method returns `None`.
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Self::Symbol>,
    ) -> Option<(Self::Probability, <Self::Probability as BitArray>::NonZero)>;

    /// Returns the probability of the given symbol in floating point representation.
    ///
    /// The trait bound `Self::Probability: Into<F>` guarantees that no rounding occurs in
    /// the conversion. You may have to specify the return type explicitly using "turbofish"
    /// notation `::<f64>(...)` or `::<f32>(...)`, see example below.
    ///
    /// Returns `0.0` if `symbol` is not in the support of the entropy model.
    ///
    /// This method is provided mainly as a convenience for debugging.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::model::{EncoderModel, DefaultNonContiguousCategoricalEncoderModel};
    ///
    /// let symbols = vec!['a', 'b', 'c', 'd'];
    /// let probabilities = vec![1u32 << 21, 1 << 23, 1 << 22, 1 << 21];
    /// let model = DefaultNonContiguousCategoricalEncoderModel // "Default" uses `PRECISION = 24`
    ///     ::from_symbols_and_nonzero_fixed_point_probabilities(
    ///         symbols.iter().copied(), &probabilities, false)
    ///     .unwrap();
    ///
    /// assert_eq!(model.floating_point_probability::<f64>('a'), 0.125);
    /// assert_eq!(model.floating_point_probability::<f64>('b'), 0.5);
    /// assert_eq!(model.floating_point_probability::<f64>('c'), 0.25);
    /// assert_eq!(model.floating_point_probability::<f64>('d'), 0.125);
    /// assert_eq!(model.floating_point_probability::<f64>('x'), 0.0);
    /// ```
    ///
    /// [`fixed_point_probabilities`]: #method.fixed_point_probabilities
    /// [`floating_point_probabilities_lossy`]: #method.floating_point_probabilities_lossy
    /// [`from_floating_point_probabilities`]: #method.from_floating_point_probabilities
    /// [`from_nonzero_fixed_point_probabilities`]:
    /// #method.from_nonzero_fixed_point_probabilities
    #[inline]
    fn floating_point_probability<F>(&self, symbol: Self::Symbol) -> F
    where
        F: FloatCore,
        Self::Probability: Into<F>,
    {
        // This gets compiled to a single floating point multiplication rather than a (slow)
        // division.
        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).into();
        let probability = self
            .left_cumulative_and_probability(symbol)
            .map_or(Self::Probability::zero(), |(_, p)| p.get());
        probability.into() / whole
    }
}

/// A trait for [`EntropyModel`]s that can be used for decoding (decompressing) data.
///
/// As discussed in the [module level documentation](self), all stream codes in
/// `constriction` use so-called [`EntropyModel`]s for encoding and/or decoding data. Some
/// of these `EntropyModel`s may be used only for encoding, only for decoding, or for both,
/// depending on their internal representation.
///
/// This `DecoderModel` trait is implemented for all entropy models that can be used for
/// *decoding* data. To decode data with a `DecoderModel`, construct an entropy coder that
/// implements the [`Decode`] trait and pass the entropy model to one of the methods of the
/// [`Decode`] trait.
///
/// # Blanket Implementation for `&impl DecoderModel`
///
/// We provide the following blanket implementation for references to `DecoderModel`s:
///
/// ```ignore
/// impl<M, const PRECISION: usize> DecoderModel<PRECISION> for &M
/// where
///     M: DecoderModel<PRECISION> + ?Sized
/// { ... }
/// ```
///
/// This means that, if some type `M` implements `DecoderModel<PRECISION>` for some
/// `PRECISION`, then so does the reference type `&M`. Therefore, generic functions or
/// methods should never take a generic `DecoderModel` by reference. They should always take
/// the generic `DecoderModel` *by value* because this also covers the case of references
/// but is strictly more general. If your generic function needs to be able to cheaply copy
/// the `DecoderModel` (as it could with a shared reference) then it should still take the
/// generic `DecoderModel` formally by value and just add an additional `Copy` bound (see,
/// e.g., the method signature of [`Decode::decode_iid_symbols`]. For a more elaborate
/// explanation, please refer to the discussion of the analogous blanket implementation for
/// [`EntropyModel`].
///
/// # See Also
///
/// - base trait: [`EntropyModel`]
/// - sister trait: [`EncoderModel`]
/// - used with: [`Decode`]
///
/// [`Decode`]: super::Decode
/// [`Decode::decode_iid_symbols`]: super::Encode::encode_iid_symbols
pub trait DecoderModel<const PRECISION: usize>: EntropyModel<PRECISION> {
    /// Looks up the symbol for a given quantile.
    ///
    /// The argument `quantile` represents a number in the half-open interval `[0, 1)` in
    /// fixed-point arithmetic, i.e., it must be strictly smaller than `1 << PRECISION`.
    /// Think of `quantile` as a point on the vertical axis of a plot of the cumulative
    /// distribution function of the probability model. This method evaluates the inverse of
    /// the cumulative distribution function, which is sometimes called the *quantile
    /// function*.
    ///
    /// Returns a tuple `(symbol, left_sided_cumulative, probability)` where `probability`
    /// is the probability of `symbol` under the entropy model (in fixed-point arithmetic)
    /// and `left_sided_cumulative` is the sum of the probabilities of all symbols up to and
    /// not including `symbol`. The returned `symbol` is the unique symbol that satisfies
    /// `left_sided_cumulative <= quantile < left_sided_cumulative + probability` (where the
    /// addition on the right-hand side is non-wrapping).
    ///
    /// Note that, in contrast to [`EncoderModel::left_cumulative_and_probability`], this
    /// method does *not* return an `Option`. This is because, as long as `quantile < 1 <<
    /// PRECISION`, a valid probability distribution always has a symbol for which the range
    /// `left_sided_cumulative..(left_sided_cumulative + quantile)` contains `quantile`, and
    /// the probability of this symbol is guaranteed to be nonzero because the probability
    /// is the size of the range, which contains at least the one element `quantile`.
    ///
    /// # Panics
    ///
    /// Implementations may panic if `quantile >= 1 << PRECISION`.
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (
        Self::Symbol,
        Self::Probability,
        <Self::Probability as BitArray>::NonZero,
    );
}

/// A trait for [`EntropyModel`]s that can be serialized into a common format.
///
/// The method [`symbol_table`] iterates over all symbols with nonzero probability under the
/// entropy. The iteration occurs in uniquely defined order of increasing left-sided
/// cumulative probability distribution of the symbols. All `EntropyModel`s for which such
/// iteration can be implemented efficiently should implement this trait. `EntropyModel`s
/// for which such iteration would require extra work (e.g., sorting symbols by left-sided
/// cumulative distribution) should *not* implement this trait so that callers can assume
/// that calling `symbol_table` is cheap.
///
/// The main advantage of implementing this trait is that it provides default
/// implementations of conversions to various other `EncoderModel`s and `DecoderModel`s, see
/// [`to_generic_encoder_model`], [`to_generic_decoder_model`], and
/// [`to_generic_lookup_decoder_model`].
///
/// [`symbol_table`]: Self::symbol_table
/// [`to_generic_encoder_model`]: Self::to_generic_encoder_model
/// [`to_generic_decoder_model`]: Self::to_generic_decoder_model
/// [`to_generic_lookup_decoder_model`]: Self::to_generic_lookup_decoder_model
pub trait IterableEntropyModel<'m, const PRECISION: usize>: EntropyModel<PRECISION> {
    /// Iterates over all symbols in the unique order that is consistent with the cumulative
    /// distribution.
    ///
    /// The iterator iterates in order of increasing cumulative.
    ///
    /// This method may be used, e.g., to export the model into a serializable format. It is
    /// also used internally by constructors that create a different but equivalent
    /// representation of the same entropy model (e.g., to construct a
    /// [`LookupDecoderModel`] from some `EncoderModel`).
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::model::{
    ///     IterableEntropyModel, SmallNonContiguousCategoricalDecoderModel
    /// };
    ///
    /// let symbols = vec!['a', 'b', 'x', 'y'];
    /// let probabilities = vec![0.125, 0.5, 0.25, 0.125]; // Can all be represented without rounding.
    /// let model = SmallNonContiguousCategoricalDecoderModel
    ///     ::from_symbols_and_floating_point_probabilities_fast(
    ///         symbols.iter().cloned(),
    ///         &probabilities,
    ///         None
    ///     ).unwrap();
    ///
    /// // Print a table representation of this entropy model (e.g., for debugging).
    /// dbg!(model.symbol_table().collect::<Vec<_>>());
    ///
    /// // Create a lookup model. This method is provided by the trait `IterableEntropyModel`.
    /// let lookup_decoder_model = model.to_generic_lookup_decoder_model();
    /// ```
    ///
    /// # See also
    ///
    /// - [`floating_point_symbol_table`](Self::floating_point_symbol_table)
    fn symbol_table(
        &'m self,
    ) -> impl Iterator<
        Item = (
            Self::Symbol,
            Self::Probability,
            <Self::Probability as BitArray>::NonZero,
        ),
    >;

    /// Similar to [`symbol_table`], but yields both cumulatives and probabilities in
    /// floating point representation.
    ///
    /// The conversion to floats is guaranteed to be lossless due to the trait bound `F:
    /// From<Self::Probability>`.
    ///
    /// [`symbol_table`]: Self::symbol_table
    ///
    /// TODO: test
    fn floating_point_symbol_table<F>(&'m self) -> impl Iterator<Item = (Self::Symbol, F, F)>
    where
        F: FloatCore + From<Self::Probability> + 'm,
        Self::Probability: Into<F>,
    {
        // This gets compiled into a constant, and the divisions by `whole` get compiled
        // into floating point multiplications rather than (slower) divisions.
        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).into();

        self.symbol_table()
            .map(move |(symbol, cumulative, probability)| {
                (
                    symbol,
                    cumulative.into() / whole,
                    probability.get().into() / whole,
                )
            })
    }

    /// Returns the entropy in units of bits (i.e., base 2).
    ///
    /// The entropy is the expected amortized bit rate per symbol of an optimal lossless
    /// entropy coder, assuming that the data is indeed distributed according to the model.
    ///
    /// Note that calling this method on a [`LeakilyQuantizedDistribution`] will return the
    /// entropy *after quantization*, not the differential entropy of the underlying
    /// continuous probability distribution.
    ///
    /// # See also
    ///
    /// - [`cross_entropy_base2`](Self::cross_entropy_base2)
    /// - [`reverse_cross_entropy_base2`](Self::reverse_cross_entropy_base2)
    /// - [`kl_divergence_base2`](Self::kl_divergence_base2)
    /// - [`reverse_kl_divergence_base2`](Self::reverse_kl_divergence_base2)
    fn entropy_base2<F>(&'m self) -> F
    where
        F: num_traits::Float + core::iter::Sum,
        Self::Probability: Into<F>,
    {
        let scaled_shifted = self
            .symbol_table()
            .map(|(_, _, probability)| {
                let probability = probability.get().into();
                probability * probability.log2() // probability is guaranteed to be nonzero.
            })
            .sum::<F>();

        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).into();
        F::from(PRECISION).unwrap() - scaled_shifted / whole
    }

    /// Returns the cross entropy between argument `p` and this model in units of bits
    /// (i.e., base 2).
    ///
    /// This is the expected amortized bit rate per symbol that an optimal coder will
    /// achieve when using this model on a data source that draws symbols from the provided
    /// probability distribution `p`.
    ///
    /// The cross entropy is defined as `H(p, self) = - sum_i p[i] * log2(self[i])` where
    /// `p` is provided as an argument and `self[i]` denotes the corresponding probabilities
    /// of the model. Note that `self[i]` is never zero for models in the `constriction`
    /// library, so the logarithm in the (forward) cross entropy can never be infinite.
    ///
    /// The argument `p` must yield a sequence of probabilities (nonnegative values that sum
    /// to 1) with the correct length and order to be compatible with the model.
    ///
    /// # See also
    ///
    /// - [`entropy_base2`](Self::entropy_base2)
    /// - [`reverse_cross_entropy_base2`](Self::reverse_cross_entropy_base2)
    /// - [`kl_divergence_base2`](Self::kl_divergence_base2)
    /// - [`reverse_kl_divergence_base2`](Self::reverse_kl_divergence_base2)
    fn cross_entropy_base2<F>(&'m self, p: impl IntoIterator<Item = F>) -> F
    where
        F: num_traits::Float + core::iter::Sum,
        Self::Probability: Into<F>,
    {
        let shift = F::from(PRECISION).unwrap();
        self.symbol_table()
            .zip(p)
            .map(|((_, _, probability), p)| {
                let probability = probability.get().into();
                // Perform the shift for each item individually so that the result is
                // reasonable even if `p` is not normalized.
                p * (shift - probability.log2()) // probability is guaranteed to be nonzero.
            })
            .sum::<F>()
    }

    /// Returns the cross entropy between this model and argument `p` in units of bits
    /// (i.e., base 2).
    ///
    /// This method is provided mostly for completeness. You're more likely to want to
    /// calculate [`cross_entropy_base2`](Self::cross_entropy_base2).
    ///
    /// The reverse cross entropy is defined as `H(self, p) = - sum_i self[i] * log2(p[i])`
    /// where `p` is provided as an argument and `self[i]` denotes the corresponding
    /// probabilities of the model.
    ///
    /// The argument `p` must yield a sequence of *nonzero* probabilities (that sum to 1)
    /// with the correct length and order to be compatible with the model.
    ///
    /// # See also
    ///
    /// - [`cross_entropy_base2`](Self::cross_entropy_base2)
    /// - [`entropy_base2`](Self::entropy_base2)
    /// - [`reverse_kl_divergence_base2`](Self::reverse_kl_divergence_base2)
    /// - [`kl_divergence_base2`](Self::kl_divergence_base2)
    fn reverse_cross_entropy_base2<F>(&'m self, p: impl IntoIterator<Item = F>) -> F
    where
        F: num_traits::Float + core::iter::Sum,
        Self::Probability: Into<F>,
    {
        let scaled = self
            .symbol_table()
            .zip(p)
            .map(|((_, _, probability), p)| {
                let probability = probability.get().into();
                probability * p.log2()
            })
            .sum::<F>();

        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).into();
        -scaled / whole
    }

    /// Returns Kullback-Leibler divergence `D_KL(p || self)`
    ///
    /// This is the expected *overhead* (due to model quantization) in bit rate per symbol
    /// that an optimal coder will incur when using this model on a data source that draws
    /// symbols from the provided probability distribution `p` (which this model is supposed
    /// to approximate).
    ///
    /// The KL-divergence is defined as `D_KL(p || self) = - sum_i p[i] * log2(self[i] /
    /// p[i])`, where `p` is provided as an argument and `self[i]` denotes the corresponding
    /// probabilities of the model. Any term in the sum where `p[i]` is *exactly* zero does
    /// not contribute (regardless of whether or not `self[i]` would also be zero).
    ///
    /// The argument `p` must yield a sequence of probabilities (nonnegative values that sum
    /// to 1) with the correct length and order to be compatible with the model.
    ///
    /// # See also
    ///
    /// - [`reverse_kl_divergence_base2`](Self::reverse_kl_divergence_base2)
    /// - [`entropy_base2`](Self::entropy_base2)
    /// - [`cross_entropy_base2`](Self::cross_entropy_base2)
    /// - [`reverse_cross_entropy_base2`](Self::reverse_cross_entropy_base2)
    fn kl_divergence_base2<F>(&'m self, p: impl IntoIterator<Item = F>) -> F
    where
        F: num_traits::Float + core::iter::Sum,
        Self::Probability: Into<F>,
    {
        let shifted = self
            .symbol_table()
            .zip(p)
            .map(|((_, _, probability), p)| {
                if p == F::zero() {
                    F::zero()
                } else {
                    let probability = probability.get().into();
                    p * (p.log2() - probability.log2())
                }
            })
            .sum::<F>();

        shifted + F::from(PRECISION).unwrap() // assumes that `p` is normalized
    }

    /// Returns reverse Kullback-Leibler divergence, i.e., `D_KL(self || p)`
    ///
    /// This method is provided mostly for completeness. You're more likely to want to
    /// calculate [`kl_divergence_base2`](Self::kl_divergence_base2).
    ///
    /// The reverse KL-divergence is defined as `D_KL(self || p) = - sum_i self[i] *
    /// log2(p[i] / self[i])` where `p`
    /// is provided as an argument and `self[i]` denotes the corresponding probabilities of
    /// the model.
    ///
    /// The argument `p` must yield a sequence of *nonzero* probabilities (that sum to 1)
    /// with the correct length and order to be compatible with the model.
    ///
    /// # See also
    ///
    /// - [`kl_divergence_base2`](Self::kl_divergence_base2)
    /// - [`entropy_base2`](Self::entropy_base2)
    /// - [`cross_entropy_base2`](Self::cross_entropy_base2)
    /// - [`reverse_cross_entropy_base2`](Self::reverse_cross_entropy_base2)
    fn reverse_kl_divergence_base2<F>(&'m self, p: impl IntoIterator<Item = F>) -> F
    where
        F: num_traits::Float + core::iter::Sum,
        Self::Probability: Into<F>,
    {
        let scaled_shifted = self
            .symbol_table()
            .zip(p)
            .map(|((_, _, probability), p)| {
                let probability = probability.get().into();
                probability * (probability.log2() - p.log2())
            })
            .sum::<F>();

        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).into();
        scaled_shifted / whole - F::from(PRECISION).unwrap()
    }

    /// Creates an [`EncoderModel`] from this `EntropyModel`
    ///
    /// This is a fallback method that should only be used if no more specialized
    /// conversions are available. It generates a [`NonContiguousCategoricalEncoderModel`]
    /// with the same probabilities and left-sided cumulatives as `self`. Note that a
    /// `NonContiguousCategoricalEncoderModel` is very generic and therefore not
    /// particularly optimized. Thus, before calling this method first check:
    /// - if the original `Self` type already implements `EncoderModel` (some types
    ///   implement *both* `EncoderModel` and `DecoderModel`); or
    /// - if the `Self` type has some inherent method with a name like `to_encoder_model`;
    ///   if it does, that method probably returns an implementation of `EncoderModel` that
    ///   is better optimized for your use case.
    #[inline(always)]
    fn to_generic_encoder_model(
        &'m self,
    ) -> NonContiguousCategoricalEncoderModel<Self::Symbol, Self::Probability, PRECISION>
    where
        Self::Symbol: Hash + Eq,
    {
        self.into()
    }

    /// Creates a [`DecoderModel`] from this `EntropyModel`
    ///
    /// This is a fallback method that should only be used if no more specialized
    /// conversions are available. It generates a [`NonContiguousCategoricalDecoderModel`]
    /// with the same probabilities and left-sided cumulatives as `self`. Note that a
    /// `NonContiguousCategoricalEncoderModel` is very generic and therefore not
    /// particularly optimized. Thus, before calling this method first check:
    /// - if the original `Self` type already implements `DecoderModel` (some types
    ///   implement *both* `EncoderModel` and `DecoderModel`); or
    /// - if the `Self` type has some inherent method with a name like `to_decoder_model`;
    ///   if it does, that method probably returns an implementation of `DecoderModel` that
    ///   is better optimized for your use case.
    #[inline(always)]
    fn to_generic_decoder_model(
        &'m self,
    ) -> NonContiguousCategoricalDecoderModel<
        Self::Symbol,
        Self::Probability,
        Vec<(Self::Probability, Self::Symbol)>,
        PRECISION,
    >
    where
        Self::Symbol: Clone,
    {
        self.into()
    }

    /// Creates a [`DecoderModel`] from this `EntropyModel`
    ///
    /// This is a fallback method that should only be used if no more specialized
    /// conversions are available. It generates a [`LookupDecoderModel`] that makes no
    /// assumption about contiguity of the support. Thus, before calling this method first
    /// check if the `Self` type has some inherent method with a name like
    /// `to_lookup_decoder_model`. If it does, that method probably returns a
    /// `LookupDecoderModel` that is better optimized for your use case.
    #[inline(always)]
    fn to_generic_lookup_decoder_model(
        &'m self,
    ) -> NonContiguousLookupDecoderModel<
        Self::Symbol,
        Self::Probability,
        Vec<(Self::Probability, Self::Symbol)>,
        Box<[Self::Probability]>,
        PRECISION,
    >
    where
        Self::Probability: Into<usize>,
        usize: AsPrimitive<Self::Probability>,
        Self::Symbol: Clone + Default,
    {
        self.into()
    }
}

impl<M, const PRECISION: usize> EntropyModel<PRECISION> for &M
where
    M: EntropyModel<PRECISION> + ?Sized,
{
    type Probability = M::Probability;
    type Symbol = M::Symbol;
}

impl<M, const PRECISION: usize> EncoderModel<PRECISION> for &M
where
    M: EncoderModel<PRECISION> + ?Sized,
{
    #[inline(always)]
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Self::Symbol>,
    ) -> Option<(Self::Probability, <Self::Probability as BitArray>::NonZero)> {
        (*self).left_cumulative_and_probability(symbol)
    }
}

impl<M, const PRECISION: usize> DecoderModel<PRECISION> for &M
where
    M: DecoderModel<PRECISION> + ?Sized,
{
    #[inline(always)]
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (
        Self::Symbol,
        Self::Probability,
        <Self::Probability as BitArray>::NonZero,
    ) {
        (*self).quantile_function(quantile)
    }
}

impl<'m, M, const PRECISION: usize> IterableEntropyModel<'m, PRECISION> for &'m M
where
    M: IterableEntropyModel<'m, PRECISION>,
{
    fn symbol_table(
        &'m self,
    ) -> impl Iterator<
        Item = (
            Self::Symbol,
            Self::Probability,
            <Self::Probability as BitArray>::NonZero,
        ),
    > {
        (*self).symbol_table()
    }

    fn entropy_base2<F>(&'m self) -> F
    where
        F: num_traits::Float + core::iter::Sum,
        Self::Probability: Into<F>,
    {
        (*self).entropy_base2()
    }

    #[inline(always)]
    fn to_generic_encoder_model(
        &'m self,
    ) -> NonContiguousCategoricalEncoderModel<Self::Symbol, Self::Probability, PRECISION>
    where
        Self::Symbol: Hash + Eq,
    {
        (*self).to_generic_encoder_model()
    }

    #[inline(always)]
    fn to_generic_decoder_model(
        &'m self,
    ) -> NonContiguousCategoricalDecoderModel<
        Self::Symbol,
        Self::Probability,
        Vec<(Self::Probability, Self::Symbol)>,
        PRECISION,
    >
    where
        Self::Symbol: Clone,
    {
        (*self).to_generic_decoder_model()
    }
}

pub use categorical::{
    contiguous::{
        ContiguousCategoricalEntropyModel, DefaultContiguousCategoricalEntropyModel,
        SmallContiguousCategoricalEntropyModel,
    },
    lazy_contiguous::{
        DefaultLazyContiguousCategoricalEntropyModel, LazyContiguousCategoricalEntropyModel,
        SmallLazyContiguousCategoricalEntropyModel,
    },
    lookup_contiguous::ContiguousLookupDecoderModel,
    lookup_noncontiguous::NonContiguousLookupDecoderModel,
    non_contiguous::{
        DefaultNonContiguousCategoricalDecoderModel, DefaultNonContiguousCategoricalEncoderModel,
        NonContiguousCategoricalDecoderModel, NonContiguousCategoricalEncoderModel,
        SmallNonContiguousCategoricalDecoderModel, SmallNonContiguousCategoricalEncoderModel,
    },
};
pub use quantize::{
    DefaultLeakyQuantizer, LeakilyQuantizedDistribution, LeakyQuantizer, SmallLeakyQuantizer,
};
pub use uniform::{DefaultUniformModel, SmallUniformModel, UniformModel};

#[cfg(test)]
mod tests {
    use probability::prelude::*;

    use super::*;

    #[test]
    fn entropy() {
        #[cfg(not(miri))]
        let (support, std_devs, means) = (-1000..=1000, [100., 200., 300.], [-10., 2.3, 50.1]);

        // We use different settings when testing on miri so that the test time stays reasonable.
        #[cfg(miri)]
        let (support, std_devs, means) = (-100..=100, [10., 20., 30.], [-1., 0.23, 5.01]);

        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(support);
        for &std_dev in &std_devs {
            for &mean in &means {
                let distribution = Gaussian::new(mean, std_dev);
                let model = quantizer.quantize(distribution);
                let entropy = model.entropy_base2::<f64>();
                let expected_entropy = 2.047095585180641 + std_dev.log2();
                assert!((entropy - expected_entropy).abs() < 0.01);
            }
        }
    }

    pub(super) fn test_entropy_model<'m, D, const PRECISION: usize>(
        model: &'m D,
        support: impl Clone + Iterator<Item = D::Symbol>,
    ) where
        D: IterableEntropyModel<'m, PRECISION>
            + EncoderModel<PRECISION>
            + DecoderModel<PRECISION>
            + 'm,
        D::Symbol: Copy + core::fmt::Debug + PartialEq,
        D::Probability: Into<u64>,
        u64: AsPrimitive<D::Probability>,
    {
        let mut sum = 0;
        for symbol in support.clone() {
            let (left_cumulative, prob) = model.left_cumulative_and_probability(symbol).unwrap();
            assert_eq!(left_cumulative.into(), sum);
            sum += prob.get().into();

            let expected = (symbol, left_cumulative, prob);
            assert_eq!(model.quantile_function(left_cumulative), expected);
            assert_eq!(model.quantile_function((sum - 1).as_()), expected);
            assert_eq!(
                model.quantile_function((left_cumulative.into() + prob.get().into() / 2).as_()),
                expected
            );
        }
        assert_eq!(sum, 1 << PRECISION);

        test_iterable_entropy_model(model, support);
    }

    pub(super) fn test_iterable_entropy_model<'m, D, const PRECISION: usize>(
        model: &'m D,
        support: impl Clone + Iterator<Item = D::Symbol>,
    ) where
        D: IterableEntropyModel<'m, PRECISION> + 'm,
        D::Symbol: Copy + core::fmt::Debug + PartialEq,
        D::Probability: Into<u64>,
        u64: AsPrimitive<D::Probability>,
    {
        let mut expected_cumulative = 0u64;
        let mut count = 0;
        for (expected_symbol, (symbol, left_sided_cumulative, probability)) in
            support.clone().zip(model.symbol_table())
        {
            assert_eq!(symbol, expected_symbol);
            assert_eq!(left_sided_cumulative.into(), expected_cumulative);
            expected_cumulative += probability.get().into();
            count += 1;
        }
        assert_eq!(count, support.size_hint().0);
        assert_eq!(expected_cumulative, 1 << PRECISION);
    }
}
