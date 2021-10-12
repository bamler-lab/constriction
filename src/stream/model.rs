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
//! [`LeakyQuantizer`] with an invalid
//! [`Distribution`](probability::distribution::Distribution), i.e., one whose cumulative
//! distribution function either isn't monotonic or has an image that exceeds the interval
//! `[0, 1]`).
//!
//! [`stack`]: super::stack
//! [`queue`]: super::queue
//! [`Binomial`]: probability::distribution::Binomial
//! [`Gaussian`]: probability::distribution::Gaussian

#[cfg(feature = "std")]
use std::collections::{
    hash_map::Entry::{Occupied, Vacant},
    HashMap,
};

#[cfg(not(feature = "std"))]
use hashbrown::hash_map::{
    Entry::{Occupied, Vacant},
    HashMap,
};

use alloc::{boxed::Box, vec::Vec};
use core::{borrow::Borrow, fmt::Debug, hash::Hash, marker::PhantomData, ops::RangeInclusive};
use num::{
    cast::AsPrimitive,
    traits::{WrappingAdd, WrappingSub},
    Float, One, PrimInt, Zero,
};

#[cfg(feature = "probability")]
use probability::distribution::{Distribution, Inverse};

/// Mock replacement for [`probability::distribution::Distribution`] in a no-std context
///
/// This trait is only exported if `constriction` is used in a no-std context (i.e., with
/// `default-features = false`). In this case, we can't use the `probability` crate because
/// it doesn't seems to be incompatible with no-std. However, for most things, we really
/// only need the trait definitions for `Distribution` and for [`Inverse`], so we copy them
/// here.
#[cfg(not(feature = "probability"))]
pub trait Distribution {
    /// The type of outcomes.
    type Value;

    /// Compute the cumulative distribution function.
    fn distribution(&self, x: f64) -> f64;
}

/// Mock replacement for [`probability::distribution::Distribution`] in a no-std context
///
/// This trait is only exported if `constriction` is used in a no-std context (i.e., with
/// `default-features = false`). In this case, we can't use the `probability` crate because
/// it doesn't seems to be incompatible with no-std. However, for most things, we really
/// only need the trait definitions for [`Distribution`] and for `Inverse`, so we copy them
/// here.
#[cfg(not(feature = "probability"))]
pub trait Inverse: Distribution {
    /// Compute the inverse of the cumulative distribution function.
    fn inverse(&self, p: f64) -> Self::Value;
}

use crate::{wrapping_pow2, BitArray, NonZeroBitArray};

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
///     M: EntropyModel<PRECISION>
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
    /// The constraint that `1 <= PRECISION <= Probability::BITS` currently isn't enforced
    /// statically since Rust does not yet allow const expressions in type bounds.
    /// Therefore, if your implementation of `EntropyModel` relies on this constraint at any
    /// point, it should state it as an assertion: `assert!(1 <= PRECISOIN && PRECISION <=
    /// Probability::BITS)`. This assertion has zero runtime cost because it can be
    /// trivially evaluated at compile time and therefore will be optimized out if it holds.
    /// The implementations provided by `constriction` strive to include this and related
    /// assertions wherever necessary.
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
    /// The type of the iterator returned by [`symbol_table`](Self::symbol_table).
    ///
    /// Each item is a tuple `(symbol, left_sided_cumulative, probability)`.
    type Iter: Iterator<
        Item = (
            Self::Symbol,
            Self::Probability,
            <Self::Probability as BitArray>::NonZero,
        ),
    >;

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
    ///     ::from_symbols_and_floating_point_probabilities(&symbols, &probabilities).unwrap();
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
    fn symbol_table(&'m self) -> Self::Iter;

    /// Similar to [`symbol_table`], but yields both cumulatives and probabilities in
    /// floating point representation.
    ///
    /// The conversion to floats is guaranteed to be lossless due to the trait bound `F:
    /// From<Self::Probability>`.
    ///
    /// [`symbol_table`]: Self::symbol_table
    fn floating_point_symbol_table<F>(
        &'m self,
    ) -> FloatingPointSymbolTable<F, Self::Iter, PRECISION>
    where
        F: From<Self::Probability>,
    {
        FloatingPointSymbolTable {
            inner: self.symbol_table(),
            phantom: PhantomData,
        }
    }

    /// Returns the entropy in units of bits (i.e., base 2).
    ///
    /// The entropy is the theoretical lower bound on the *expected* bit rate in any
    /// lossless entropy coder.
    ///
    /// Note that calling this method on a [`LeakilyQuantizedDistribution`] will return the
    /// entropy *after quantization*, not the differential entropy of the underlying
    /// continuous probability distribution.
    fn entropy_base2<F>(&'m self) -> F
    where
        F: Float + core::iter::Sum,
        Self::Probability: Into<F>,
    {
        let entropy_scaled = self
            .symbol_table()
            .into_iter()
            .map(|(_, _, probability)| {
                let probability = probability.get().into();
                probability * probability.log2() // probability is guaranteed to be nonzero.
            })
            .sum::<F>();

        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).into();
        F::from(PRECISION).unwrap() - entropy_scaled / whole
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
    ) -> LookupDecoderModel<
        Self::Symbol,
        Self::Probability,
        NonContiguousSymbolTable<Vec<(Self::Probability, Self::Symbol)>>,
        Box<[Self::Probability]>,
        PRECISION,
    >
    where
        Self::Probability: Into<usize>,
        usize: AsPrimitive<Self::Probability>,
        Self::Symbol: Copy + Default,
    {
        self.into()
    }
}

/// The iterator returned by [`IterableEntropyModel::floating_point_symbol_table`].
#[derive(Debug)]
pub struct FloatingPointSymbolTable<F, I, const PRECISION: usize> {
    inner: I,
    phantom: PhantomData<F>,
}

impl<F, Symbol, Probability, I, const PRECISION: usize> Iterator
    for FloatingPointSymbolTable<F, I, PRECISION>
where
    F: Float,
    Probability: BitArray + Into<F>,
    I: Iterator<Item = (Symbol, Probability, <Probability as BitArray>::NonZero)>,
{
    type Item = (Symbol, F, F);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (symbol, cumulative, prob) = self.inner.next()?;

        // This gets compiled into a constant, and the divisions by `whole` get compiled
        // into floating point multiplications rather than (slower) divisions.
        let whole = (F::one() + F::one()) * (Probability::one() << (PRECISION - 1)).into();
        Some((symbol, cumulative.into() / whole, prob.get().into() / whole))
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<F, Symbol, Probability, I, const PRECISION: usize> ExactSizeIterator
    for FloatingPointSymbolTable<F, I, PRECISION>
where
    F: Float,
    Probability: BitArray + Into<F>,
    I: ExactSizeIterator<Item = (Symbol, Probability, <Probability as BitArray>::NonZero)>,
{
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
///     M: EncoderModel<PRECISION>
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
        F: Float,
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
///     M: DecoderModel<PRECISION>
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

impl<M, const PRECISION: usize> EntropyModel<PRECISION> for &M
where
    M: EntropyModel<PRECISION>,
{
    type Probability = M::Probability;
    type Symbol = M::Symbol;
}

impl<'m, M, const PRECISION: usize> IterableEntropyModel<'m, PRECISION> for &'m M
where
    M: IterableEntropyModel<'m, PRECISION>,
{
    type Iter = M::Iter;

    fn symbol_table(&'m self) -> Self::Iter {
        (*self).symbol_table()
    }

    fn entropy_base2<F>(&'m self) -> F
    where
        F: Float + core::iter::Sum,
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

impl<M, const PRECISION: usize> EncoderModel<PRECISION> for &M
where
    M: EncoderModel<PRECISION>,
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
    M: DecoderModel<PRECISION>,
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

/// Quantizes probability distributions and represents them in fixed-point precision.
///
/// You will usually want to use this type through one of its type aliases,
/// [`DefaultLeakyQuantizer`] or [`SmallLeakyQuantizer`], see [discussion of
/// presets](super#presets).
///
/// # Examples
///
/// ## Quantizing Continuous Distributions
///
/// ```
/// use constriction::{
///     stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Encode, Decode},
///     UnwrapInfallible,
/// };
///
/// // Create a quantizer that supports integer symbols from -5 to 20 (inclusively),
/// // using the "default" preset for `Probability` and `PRECISION`.
/// let quantizer = DefaultLeakyQuantizer::new(-5..=20);
///
/// // Quantize a normal distribution with mean 8.3 and standard deviation 4.1.
/// let continuous_distribution1 = probability::distribution::Gaussian::new(8.3, 4.1);
/// let entropy_model1 = quantizer.quantize(continuous_distribution1);
///
/// // You can reuse the same quantizer for more than one distribution, and the distributions don't
/// // even have to be of the same type (e.g., one can be a `Gaussian` and another a `Laplace`).
/// let continuous_distribution2 = probability::distribution::Laplace::new(-1.4, 2.7);
/// let entropy_model2 = quantizer.quantize(continuous_distribution2);
///
/// // Use the entropy models with an entropy coder.
/// let mut ans_coder = DefaultAnsCoder::new();
/// ans_coder.encode_symbol(4, &entropy_model1).unwrap();
/// ans_coder.encode_symbol(-3, &entropy_model2).unwrap();
///
/// // Decode symbols (in reverse order, since the `AnsCoder` is a stack) and verify correctness.
/// assert_eq!(ans_coder.decode_symbol(entropy_model2).unwrap_infallible(), -3);
/// assert_eq!(ans_coder.decode_symbol(entropy_model1).unwrap_infallible(), 4);
/// assert!(ans_coder.is_empty());
/// ```
///
/// ## Quantizing a Discrete Distribution (That Has an Analytic Expression)
///
/// If you pass a discrete probability distribution to the method [`quantize`] then it no
/// longer needs to perform any quantization in the data space, but it will still perform
/// steps 2 and 3 in the list below, i.e., it will still convert to a "leaky" fixed-point
/// approximation that can be used by any of `constrictions`'s stream codes. In the
/// following example, we'll quantize a [`Binomial`](probability::distribution::Binomial)
/// distribution (as discussed [below](#dont-quantize-categorical-distributions-though), you
/// should *not* quantize a [`Categorical`](probability::distribution::Categorical)
/// distribution since there are more efficient specialized types for this use case).
///
/// ```
/// use constriction::stream::{
///     model::DefaultLeakyQuantizer, queue::DefaultRangeEncoder, Encode, Decode
/// };
///
/// let distribution = probability::distribution::Binomial::new(1000, 0.1); // arguments: `n, p`
/// let quantizer = DefaultLeakyQuantizer::new(0..=1000); // natural support is `0..=n`
/// let entropy_model = quantizer.quantize(distribution);
///
/// // Let's use a Range Coder this time, just for fun (we could as well use an ANS Coder again).
/// let mut range_encoder = DefaultRangeEncoder::new();
///
/// // Encode a "typical" symbol from the distribution (i.e., one with non-negligible probability).
/// range_encoder.encode_symbol(107, &entropy_model).unwrap();
///
/// // Due to the "leakiness" of the quantizer, the following still works despite the fact that
/// // the symbol `1000` has a ridiculously low probability under the binomial distribution.
/// range_encoder.encode_symbol(1000, &entropy_model).unwrap();
///
/// // Decode symbols (in forward order, since range coding operates as a queue) and verify.
/// let mut range_decoder = range_encoder.into_decoder().unwrap();
/// assert_eq!(range_decoder.decode_symbol(&entropy_model).unwrap(), 107);
/// assert_eq!(range_decoder.decode_symbol(&entropy_model).unwrap(), 1000);
/// assert!(range_decoder.maybe_exhausted());
/// ```
///
/// # Detailed Description
///
/// A `LeakyQuantizer` is a builder of [`LeakilyQuantizedDistribution`]s. It takes an
/// arbitrary probability distribution that implements the [`Distribution`] trait from the
/// crate [`probability`] and turns it into a [`LeakilyQuantizedDistribution`] by performing
/// the following three steps:
///
/// 1. **quantization**: lossless entropy coding can only be performed over *discrete* data.
///    Any continuous (real-valued) data has to be approximated by some discrete set of
///    points. If you provide a continuous distributions (i.e., a probability density
///    function) to this builder, then it will quantize the data space by rounding values to
///    the nearest integer. This step is optional, see
///    [below](#continuous-vs-discrete-probability-distributions).
/// 2. **approximation with fixed-point arithmetic**: an entropy model that is used for
///    compressing and decompressing has to be *exactly* invertible, so that its
///    [`EncoderModel`] implementation is compatible with its [`DecoderModel`]
///    implementation. The `LeakilyQuantizedDistribution`s that are built by this builder
///    represent probabilities and quantiles in fixed-point arithmetic with `PRECISION`
///    bits. This allows them to avoid rounding errors when inverting the model, so that
///    they can implement both `EncoderModel` and `DecoderModel` in such a way that one is
///    the *exact* inverse of the other.
/// 3. **introducing leakiness**: naively approximating a probability distribution with
///    fixed point arithmetic could lead to problems: it could round some very small
///    probabilities to zero. This would have the undesirable effect that the corresponding
///    symbol then could no longer be encoded. This builder ensures that the
///    `LeakilyQuantizedDistribution`s that it creates assign a nonzero probability to all
///    symbols within a user-defined range, so that these symbols can always be encoded,
///    even if their probabilities under the *original* probability distribution are very
///    low (or even zero).
///
/// # Continuous vs. Discrete Probability Distributions
///
/// The method [`quantize`] accepts both continuous probability distributions (i.e.,
/// probability density functions, such as [`Gaussian`]) and discrete distributions that are
/// defined only on (some) integers (i.e., probability mass functions, such as
/// [`Binomial`]). The resulting [`LeakilyQuantizedDistribution`] will always be a discrete
/// probability distribution. If the original probability distribution is continuous, then
/// the quantizer implicitly creates bins of size one by rounding to the nearest integer
/// (i.e., the bins range from `i - 0.5` to `i + 0.5` for each integer `i`). If the original
/// probability distribution is discrete then no rounding in the symbol space occurs, but
/// the quantizer still performs steps 2 and 3 above, i.e., it still rounds probabilities
/// and quantiles to fixed-point arithmetic in a way that ensures that all probabilities
/// within a user-defined range are nonzero.
///
/// ## Don't Quantize *Categorical* Distributions, Though.
///
/// Although you can use a `LeakyQuantizer` for *discrete* probability distributions, you
/// should *not* use it for probability distributions of the type
/// [`probability::distribution::Categorical`]. While this will technically work, it will
/// lead to poor computational performance (and also to *slightly* suboptimal compression
/// efficiency). If you're dealing with categorical distributions, use one of the dedicated
/// types [`ContiguousCategoricalEntropyModel`], [`NonContiguousCategoricalEncoderModel`],
/// [`NonContiguousCategoricalDecoderModel`], or [`LookupDecoderModel`] instead.
///
/// By contrast, *do* use a `LeakyQuantizer` if the underlying probability [`Distribution`]
/// can be described by some analytic function (e.g., the function `f(x) ∝ e^{-(x-\mu)^2/2}`
/// describing the bell curve of a Gaussian distribution, or the function `f_n(k) = (n
/// choose k) p^k (1-p)^{n-k}` describing the probability mass function of a binomial
/// distribution). For such parameterized distributions, both the cumulative distribution
/// function and its inverse can often be expressed as, or at least approximated by, some
/// analytic expression that can be evaluated in constant time, independent of the number of
/// possible symbols.
///
/// # Computational Efficiency
///
/// Two things should be noted about computational efficiency:
///
/// - **quantization is lazy:** both the constructor of a `LeakyQuantizer` and the method
///   [`quantize`] perform only a small constant amount of work, independent of the
///   `PRECISION` and the number of symbols on which the resulting entropy model will be
///   defined. The actual quantization is done once the resulting
///   [`LeakilyQuantizedDistribution`] is used for encoding and/or decoding, and it is only
///   done for the involved symbols.
/// - **quantization for decoding is more expensive than for encoding:** using a
///   `LeakilyQuantizedDistribution` as an [`EncoderModel`] only requires evaluating the
///   cumulative distribution function (CDF) of the underlying continuous probability
///   distribution a constant number of times (twice, to be precise). By contrast, using it
///   as a [`DecoderModel`] requires numerical inversion of the cumulative distribution
///   function. This numerical inversion starts by calling [`Inverse::inverse`] from the
///   crate [`probability`] on the underlying continuous probability distribution. But the
///   result of this method call then has to be refined by repeatedly probing the CDF in
///   order to deal with inevitable rounding errors in the implementation of
///   `Inverse::inverse`. The number of required iterations will depend on how accurate the
///   implementation of `Inverse::inverse` is.
///
/// The laziness means that it is relatively cheap to use a different
/// `LeakilyQuantizedDistribution` for each symbol of the message, which is a common
/// thing to do in machine-learning based compression methods. By contrast, if you want to
/// use the *same* entropy model for many symbols then a `LeakilyQuantizedDistribution` can
/// become unnecessarily expensive, especially for decoding, because you might end up
/// calculating the inverse CDF in the same region over and over again. If this is the case,
/// consider tabularizing the `LeakilyQuantizedDistribution` that you obtain from the method
/// [`quantize`] by calling [`to_generic_encoder_model`] or [`to_generic_decoder_model`] on
/// it (or, if you use a low `PRECISION`, you may even consider calling
/// [`to_generic_lookup_decoder_model`]). You'll have to bring the trait
/// [`IterableEntropyModel`] into scope to call these conversion methods (`use
/// constriction::stream::model::IterableEntropyModel`).
///
/// # Requirements for Correctness
///
/// The original distribution that you pass to the method [`quantize`] can only be an
/// approximation of a true (normalized) probability distribution because it represents
/// probabilities with finite (floating point) precision. Despite the possibility of
/// rounding errors in the underlying (floating point) distribution, a `LeakyQuantizer` is
/// guaranteed to generate a valid entropy model with exactly compatible implementations of
/// [`EncoderModel`] and [`DecoderModel`] as long as both of the following requirements are
/// met:
///
/// - The cumulative distribution function (CDF) [`Distribution::distribution`] is defined
///   on all finite non-NaN floating point numbers, monotonically nondecreasing, and its
///   values do not exceed the closed interval `[0.0, 1.0]`. It is OK if, due to rounding
///   errors, the CDF does not cover the entire interval from `0.0` to `1.0`.
/// - The quantile function or inverse CDF [`Inverse::inverse`] evaluates to a finite
///   non-NaN value everywhere on the open interval `(0.0, 1.0)`, and it is monotonically
///   nondecreasing on this interval. It does not have to be defined at the boundaries `0.0`
///   or `1.0` (more precisely, it only has to be defined on the closed interval
///   `[epsilon, 1.0 - epsilon]` where `epsilon := 2.0^{-(PRECISION+1)}` and `^` denotes
///   mathematical exponentiation). Further, the implementation of `Inverse::inverse` does
///   not actually have to be the inverse of `Distribution::distribution` because it is only
///   used as an initial hint where to start a search for the true inverse. It is OK if
///   `Inverse::inverse` is just some approximation of the true inverse CDF. Any deviations
///   between `Inverse::inverse` and the true inverse CDF will negatively impact runtime
///   performance but will otherwise have no observable effect.
///
/// [`quantize`]: Self::quantize
/// [`Gaussian`]: probability::distribution::Gaussian
/// [`Binomial`]: probability::distribution::Binomial
/// [`to_generic_encoder_model`]: IterableEntropyModel::to_generic_encoder_model
/// [`to_generic_decoder_model`]: IterableEntropyModel::to_generic_decoder_model
/// [`to_generic_lookup_decoder_model`]: IterableEntropyModel::to_generic_lookup_decoder_model
/// [`IterableEntropyModel`]: IterableEntropyModel
#[derive(Debug)]
pub struct LeakyQuantizer<F, Symbol, Probability, const PRECISION: usize> {
    min_symbol_inclusive: Symbol,
    max_symbol_inclusive: Symbol,
    free_weight: F,
    phantom: PhantomData<Probability>,
}

/// Type alias for a typical [`LeakyQuantizer`].
///
/// See:
/// - [`LeakyQuantizer`]
/// - [discussion of presets](super#presets)
pub type DefaultLeakyQuantizer<F, Symbol> = LeakyQuantizer<F, Symbol, u32, 24>;

/// Type alias for a [`LeakyQuantizer`] optimized for compatibility with lookup decoder
/// models.
///
/// See:
/// - [`LeakyQuantizer`]
/// - [discussion of presets](super#presets)
pub type SmallLeakyQuantizer<F, Symbol> = LeakyQuantizer<F, Symbol, u16, 12>;

impl<F, Symbol, Probability, const PRECISION: usize>
    LeakyQuantizer<F, Symbol, Probability, PRECISION>
where
    Probability: BitArray + Into<F>,
    Symbol: PrimInt + AsPrimitive<Probability> + WrappingSub + WrappingAdd,
    F: Float,
{
    /// Constructs a `LeakyQuantizer` with a finite support.
    ///
    /// The `support` is an inclusive range (which can be expressed with the `..=` notation,
    /// as in `-100..=100`). All [`LeakilyQuantizedDistribution`]s generated by this
    /// `LeakyQuantizer` are then guaranteed to assign a nonzero probability to all symbols
    /// within the `support`, and a zero probability to all symbols outside of the
    /// `support`. Having a known support is often a useful property of entropy models
    /// because it ensures that all symbols within the `support` can indeed be encoded, even
    /// if their probability under the underlying probability distribution is extremely
    /// small.
    ///
    /// This method takes `support` as a `RangeInclusive` because we want to support, e.g.,
    /// probability distributions over the `Symbol` type `u8` with full support `0..=255`.
    ///
    /// # Panics
    ///
    /// Panics if either of the following conditions is met:
    ///
    /// - `support` is empty; or
    /// - `support` contains only a single value (we do not support degenerate probability
    ///   distributions that put all probability mass on a single symbol); or
    /// - `support` is larger than `1 << PRECISION` (because in this case, assigning any
    ///   representable nonzero probability to all elements of `support` would exceed our
    ///   probability budge).
    ///
    /// [`quantize`]: #method.quantize
    pub fn new(support: RangeInclusive<Symbol>) -> Self {
        assert!(PRECISION > 0 && PRECISION <= Probability::BITS);

        // We don't support degenerate probability distributions (i.e., distributions that
        // place all probability mass on a single symbol).
        assert!(support.end() > support.start());

        let support_size_minus_one = support.end().wrapping_sub(support.start()).as_();
        let max_probability = Probability::max_value() >> (Probability::BITS - PRECISION);
        let free_weight = max_probability
            .checked_sub(&support_size_minus_one)
            .expect("The support is too large to assign a nonzero probability to each element.")
            .into();

        LeakyQuantizer {
            min_symbol_inclusive: *support.start(),
            max_symbol_inclusive: *support.end(),
            free_weight,
            phantom: PhantomData,
        }
    }

    /// Quantizes the given probability distribution and returns an [`EntropyModel`].
    ///
    /// See [struct documentation](Self) for details and code examples.
    ///
    /// Note that this method takes `self` only by reference, i.e., you can reuse
    /// the same `Quantizer` to quantize arbitrarily many distributions.
    pub fn quantize<D: Distribution>(
        &self,
        distribution: D,
    ) -> LeakilyQuantizedDistribution<'_, F, Symbol, Probability, D, PRECISION> {
        LeakilyQuantizedDistribution {
            inner: distribution,
            quantizer: self,
        }
    }

    /// Returns the exact range of symbols that have nonzero probability.
    ///
    /// The returned inclusive range is the same as the one that was passed in to the
    /// constructor [`new`](Self::new). All entropy models created by the method
    /// [`quantize`](Self::quantize) will assign a nonzero probability to all elements in
    /// the `support`, and they will assign a zero probability to all elements outside of
    /// the `support`. The support contains at least two and at most `1 << PRECISION`
    /// elements.
    #[inline]
    pub fn support(&self) -> RangeInclusive<Symbol> {
        self.min_symbol_inclusive..=self.max_symbol_inclusive
    }
}

/// An [`EntropyModel`] that approximates a parameterized probability [`Distribution`].
///
/// A `LeakilyQuantizedDistribution` can be created with a [`LeakyQuantizer`]. It can be
/// used for encoding and decoding with any of the stream codes provided by the
/// `constriction` crate (it can only be used for decoding if the underlying
/// [`Distribution`] implements the the trait [`Inverse`] from the [`probability`] crate).
///
/// # When Should I Use This Type of Entropy Model?
///
/// Use a `LeakilyQuantizedDistribution` when you have a probabilistic model that is defined
/// through some analytic expression (e.g., a mathematical formula for the probability
/// density function of a continuous probability distribution, or a mathematical formula for
/// the probability mass functions of some discrete probability distribution). Examples of
/// probabilistic models that lend themselves to being quantized are continuous
/// distributions such as [`Gaussian`], [`Laplace`], or [`Exponential`], as well as discrete
/// distributions with some analytic expression, such as [`Binomial`].
///
/// Do *not* use a `LeakilyQuantizedDistribution` if your probabilistic model can only be
/// presented as an explicit probability table. While you could, in principle, apply a
/// [`LeakyQuantizer`] to such a [`Categorical`] distribution, you will get better
/// computational performance (and also *slightly* better compression effectiveness) if you
/// instead use one of the dedicated types [`ContiguousCategoricalEntropyModel`],
/// [`NonContiguousCategoricalEncoderModel`], [`NonContiguousCategoricalDecoderModel`], or
/// [`LookupDecoderModel`].
///
/// # Examples
///
/// See [examples for `LeakyQuantizer`](LeakyQuantizer#examples).
///
/// # Computational Efficiency
///
/// See [discussion for `LeakyQuantizer`](LeakyQuantizer#computational-efficiency).
///
/// [`Gaussian`]: probability::distribution::Gaussian
/// [`Laplace`]: probability::distribution::Laplace
/// [`Exponential`]: probability::distribution::Exponential
/// [`Binomial`]: probability::distribution::Binomial
/// [`Categorical`]: probability::distribution::Categorical
#[derive(Debug, Clone, Copy)]
pub struct LeakilyQuantizedDistribution<'q, F, Symbol, Probability, D, const PRECISION: usize> {
    inner: D,
    quantizer: &'q LeakyQuantizer<F, Symbol, Probability, PRECISION>,
}

impl<'q, F, Symbol, Probability, D, const PRECISION: usize>
    LeakilyQuantizedDistribution<'q, F, Symbol, Probability, D, PRECISION>
where
    Probability: BitArray + Into<F>,
    Symbol: PrimInt + AsPrimitive<Probability> + WrappingSub + WrappingAdd,
    F: Float,
{
    /// Returns the quantizer that was used to create this entropy model.
    ///
    /// You may want to reuse this quantizer to quantize further probability distributions.
    #[inline]
    pub fn quantizer(&self) -> &'q LeakyQuantizer<F, Symbol, Probability, PRECISION> {
        self.quantizer
    }

    /// Returns a reference to the underlying (floating-point) probability [`Distribution`].
    ///
    /// Returns the floating-point probability distribution which this
    /// `LeakilyQuantizedDistribution` approximates in fixed-point arithmetic.
    ///
    /// # See also
    ///
    /// - [`inner_mut`](Self::inner_mut)
    /// - [`into_inner`](Self::into_inner)
    ///
    /// [`Distribution`]: probability::distribution::Distribution
    #[inline]
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Returns a mutable reference to the underlying (floating-point) probability
    /// [`Distribution`].
    ///
    /// You can use this method to mutate parameters of the underlying [`Distribution`]
    /// after it was already quantized. This is safe and cheap since quantization is done
    /// lazily anyway. Note that you can't mutate the [`support`](Self::support) since it is a
    /// property of the [`LeakyQuantizer`], not of the `Distribution`. If you want to modify
    /// the `support` then you have to create a new `LeakyQuantizer` with a different support.
    ///
    /// # See also
    ///
    /// - [`inner`](Self::inner)
    /// - [`into_inner`](Self::into_inner)
    ///
    /// [`Distribution`]: probability::distribution::Distribution
    #[inline]
    pub fn inner_mut(&mut self) -> &mut D {
        &mut self.inner
    }

    /// Consumes the entropy model and returns the underlying (floating-point) probability
    /// [`Distribution`].
    ///
    /// Returns the floating-point probability distribution which this
    /// `LeakilyQuantizedDistribution` approximates in fixed-point arithmetic.
    ///
    /// # See also
    ///
    /// - [`inner`](Self::inner)
    /// - [`inner_mut`](Self::inner_mut)
    ///
    /// [`Distribution`]: probability::distribution::Distribution
    #[inline]
    pub fn into_inner(self) -> D {
        self.inner
    }

    /// Returns the exact range of symbols that have nonzero probability.
    ///
    /// See [`LeakyQuantizer::support`].
    #[inline]
    pub fn support(&self) -> RangeInclusive<Symbol> {
        self.quantizer.support()
    }
}

#[inline(always)]
fn slack<Probability, Symbol>(symbol: Symbol, min_symbol_inclusive: Symbol) -> Probability
where
    Probability: BitArray,
    Symbol: AsPrimitive<Probability> + WrappingSub,
{
    // This whole `mask` business is only relevant if `Symbol` is a signed type smaller than
    // `Probability`, which should be very uncommon. In all other cases, this whole stuff
    // will be optimized away.
    let mask = wrapping_pow2::<Probability>(8 * core::mem::size_of::<Symbol>())
        .wrapping_sub(&Probability::one());
    symbol.borrow().wrapping_sub(&min_symbol_inclusive).as_() & mask
}

impl<'q, F, Symbol, Probability, D, const PRECISION: usize> EntropyModel<PRECISION>
    for LeakilyQuantizedDistribution<'q, F, Symbol, Probability, D, PRECISION>
where
    Probability: BitArray,
{
    type Probability = Probability;
    type Symbol = Symbol;
}

impl<'q, Symbol, Probability, D, const PRECISION: usize> EncoderModel<PRECISION>
    for LeakilyQuantizedDistribution<'q, f64, Symbol, Probability, D, PRECISION>
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + Into<f64> + WrappingSub,
    Probability: BitArray + Into<f64>,
    D: Distribution,
    D::Value: AsPrimitive<Symbol>,
{
    /// Performs (one direction of) the quantization.
    ///
    /// # Panics
    ///
    /// Panics if it detects some invalidity in the underlying probability distribution.
    /// This means that there is a bug in the implementation of [`Distribution`] for the
    /// distribution `D`: the cumulative distribution function is either not monotonically
    /// nondecreasing, returns NaN, or its values exceed the interval `[0.0, 1.0]` at some
    /// point.
    ///
    /// More precisely, this method panics if the quantization procedure leads to a zero
    /// probability despite the added leakiness (and despite the fact that the constructor
    /// checks that `min_symbol_inclusive < max_symbol_inclusive`, i.e., that there are at
    /// least two symbols with nonzero probability and therefore the probability of a single
    /// symbol should not be able to overflow).
    ///
    /// See [requirements for correctness](LeakyQuantizer#requirements-for-correctness).
    ///
    /// [`Distribution`]: probability::distribution::Distribution
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Symbol>,
    ) -> Option<(Probability, Probability::NonZero)> {
        let min_symbol_inclusive = self.quantizer.min_symbol_inclusive;
        let max_symbol_inclusive = self.quantizer.max_symbol_inclusive;
        let free_weight = self.quantizer.free_weight;

        if symbol.borrow() < &min_symbol_inclusive || symbol.borrow() > &max_symbol_inclusive {
            return None;
        };
        let slack = slack(*symbol.borrow(), min_symbol_inclusive);

        // Round both cumulatives *independently* to fixed point precision.
        let left_sided_cumulative = if symbol.borrow() == &min_symbol_inclusive {
            // Corner case: only makes a difference if we're cutting off a fairly significant
            // left tail of the distribution.
            Probability::zero()
        } else {
            let non_leaky: Probability =
                (free_weight * self.inner.distribution((*symbol.borrow()).into() - 0.5)).as_();
            non_leaky + slack
        };

        let right_sided_cumulative = if symbol.borrow() == &max_symbol_inclusive {
            // Corner case: make sure that the probabilities add up to one. The generic
            // calculation in the `else` branch may lead to a lower total probability
            // because we're cutting off the right tail of the distribution and we're
            // rounding down.
            wrapping_pow2(PRECISION)
        } else {
            let non_leaky: Probability =
                (free_weight * self.inner.distribution((*symbol.borrow()).into() + 0.5)).as_();
            non_leaky + slack + Probability::one()
        };

        let probability = right_sided_cumulative
            .wrapping_sub(&left_sided_cumulative)
            .into_nonzero()
            .expect("Invalid underlying continuous probability distribution.");

        Some((left_sided_cumulative, probability))
    }
}

impl<'q, Symbol, Probability, D, const PRECISION: usize> DecoderModel<PRECISION>
    for LeakilyQuantizedDistribution<'q, f64, Symbol, Probability, D, PRECISION>
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + Into<f64> + WrappingSub + WrappingAdd,
    Probability: BitArray + Into<f64>,
    D: Inverse,
    D::Value: AsPrimitive<Symbol>,
{
    fn quantile_function(
        &self,
        quantile: Probability,
    ) -> (Self::Symbol, Probability, Probability::NonZero) {
        let max_probability = Probability::max_value() >> (Probability::BITS - PRECISION);
        // This check should usually compile away in inlined and verifiably correct usages
        // of this method.
        assert!(quantile <= max_probability);

        let inverse_denominator = 1.0 / (max_probability.into() + 1.0);

        let min_symbol_inclusive = self.quantizer.min_symbol_inclusive;
        let max_symbol_inclusive = self.quantizer.max_symbol_inclusive;
        let free_weight = self.quantizer.free_weight;

        // Make an initial guess for the inverse of the leaky CDF.
        let mut symbol: Self::Symbol = self
            .inner
            .inverse((quantile.into() + 0.5) * inverse_denominator)
            .as_();

        let mut left_sided_cumulative = if symbol <= min_symbol_inclusive {
            // Corner case: we're in the left cut off tail of the distribution.
            symbol = min_symbol_inclusive;
            Probability::zero()
        } else {
            if symbol > max_symbol_inclusive {
                // Corner case: we're in the right cut off tail of the distribution.
                symbol = max_symbol_inclusive;
            }

            let non_leaky: Probability =
                (free_weight * self.inner.distribution(symbol.into() - 0.5)).as_();
            non_leaky + slack(symbol, min_symbol_inclusive)
        };

        // SAFETY: We have to ensure that all paths lead to a state where
        // `right_sided_cumulative != left_sided_cumulative`.
        let mut step = Self::Symbol::one(); // `diff` will always be a power of 2.
        let right_sided_cumulative = if left_sided_cumulative > quantile {
            // Our initial guess for `symbol` was too high. Reduce it until we're good.
            symbol = symbol - step;
            let mut found_lower_bound = false;

            loop {
                let old_left_sided_cumulative = left_sided_cumulative;

                if symbol == min_symbol_inclusive && step <= Symbol::one() {
                    left_sided_cumulative = Probability::zero();
                    // This can only be reached from a downward search, so `old_left_sided_cumulative`
                    // is the right sided cumulative since the step size is one.
                    // SAFETY: `old_left_sided_cumulative > quantile >= 0 = left_sided_cumulative`
                    break old_left_sided_cumulative;
                }

                let non_leaky: Probability =
                    (free_weight * self.inner.distribution(symbol.into() - 0.5)).as_();
                left_sided_cumulative = non_leaky + slack(symbol, min_symbol_inclusive);

                if left_sided_cumulative <= quantile {
                    found_lower_bound = true;
                    // We found a lower bound, so we're either done or we have to do a binary
                    // search now.
                    if step <= Symbol::one() {
                        let right_sided_cumulative = if symbol == max_symbol_inclusive {
                            wrapping_pow2(PRECISION)
                        } else {
                            let non_leaky: Probability =
                                (free_weight * self.inner.distribution(symbol.into() + 0.5)).as_();
                            (non_leaky + slack(symbol, min_symbol_inclusive))
                                .wrapping_add(&Probability::one())
                        };
                        // SAFETY: `old_left_sided_cumulative > quantile >= left_sided_cumulative`
                        break right_sided_cumulative;
                    } else {
                        step = step >> 1;
                        // The following addition can't overflow because we're in the binary search phase.
                        symbol = symbol + step;
                    }
                } else if found_lower_bound {
                    // We're in the binary search phase, so all following guesses will be within bounds.
                    if step > Symbol::one() {
                        step = step >> 1
                    }
                    symbol = symbol - step;
                } else {
                    // We're still in the downward search phase with exponentially increasing step size.
                    if step << 1 != Symbol::zero() {
                        step = step << 1;
                    }

                    symbol = loop {
                        let new_symbol = symbol.wrapping_sub(&step);
                        if new_symbol >= min_symbol_inclusive && new_symbol <= symbol {
                            // We can't reach this point if the subtraction wrapped because that would
                            // mean that `step = 1` and therefore the old `symbol` was `Symbol::min_value()`,
                            // so se would have ended up in the `left_sided_cumulative <= quantile` branch.
                            break new_symbol;
                        }
                        step = step >> 1;
                    };
                }
            }
        } else {
            // Our initial guess for `symbol` was either exactly right or too low.
            // Check validity of the right sided cumulative. If it isn't valid,
            // keep increasing `symbol` until it is.
            let mut found_upper_bound = false;

            loop {
                let right_sided_cumulative = if symbol == max_symbol_inclusive {
                    let right_sided_cumulative = max_probability.wrapping_add(&Probability::one());
                    if step <= Symbol::one() {
                        let non_leaky: Probability =
                            (free_weight * self.inner.distribution(symbol.into() - 0.5)).as_();
                        left_sided_cumulative = non_leaky + slack(symbol, min_symbol_inclusive);

                        // SAFETY: we have to manually check here.
                        if right_sided_cumulative == left_sided_cumulative {
                            panic!("Invalid underlying probability distribution.");
                        }

                        break right_sided_cumulative;
                    } else {
                        right_sided_cumulative
                    }
                } else {
                    let non_leaky: Probability =
                        (free_weight * self.inner.distribution(symbol.into() + 0.5)).as_();
                    (non_leaky + slack(symbol, min_symbol_inclusive))
                        .wrapping_add(&Probability::one())
                };

                if right_sided_cumulative > quantile
                    || right_sided_cumulative == Probability::zero()
                {
                    found_upper_bound = true;
                    // We found an upper bound, so we're either done or we have to do a binary
                    // search now.
                    if step <= Symbol::one() {
                        left_sided_cumulative = if symbol == min_symbol_inclusive {
                            Probability::zero()
                        } else {
                            let non_leaky: Probability =
                                (free_weight * self.inner.distribution(symbol.into() - 0.5)).as_();
                            non_leaky + slack(symbol, min_symbol_inclusive)
                        };

                        if left_sided_cumulative <= quantile || symbol == min_symbol_inclusive {
                            // SAFETY: we have `left_sided_cumulative <= quantile < right_sided_sided_cumulative`
                            break right_sided_cumulative;
                        }
                    } else {
                        step = step >> 1;
                    }
                    // The following subtraction can't overflow because we're in the binary search phase.
                    symbol = symbol - step;
                } else if found_upper_bound {
                    // We're in the binary search phase, so all following guesses will be within bounds.
                    if step > Symbol::one() {
                        step = step >> 1
                    }
                    symbol = symbol + step;
                } else {
                    // We're still in the upward search phase with exponentially increasing step size.
                    if step << 1 != Symbol::zero() {
                        step = step << 1;
                    }

                    symbol = loop {
                        let new_symbol = symbol.wrapping_add(&step);
                        if new_symbol <= max_symbol_inclusive && new_symbol >= symbol {
                            // We can't reach this point if the addition wrapped because that would
                            // mean that `step = 1` and therefore the old `symbol` was `Symbol::max_value()`,
                            // so se would have ended up in the `symbol == max_symbol_inclusive` branch.
                            break new_symbol;
                        }
                        step = step >> 1;
                    };
                }
            }
        };

        let probability = unsafe {
            // SAFETY: see above "SAFETY" comments on all paths that lead here.
            right_sided_cumulative
                .wrapping_sub(&left_sided_cumulative)
                .into_nonzero_unchecked()
        };
        (symbol, left_sided_cumulative, probability)
    }
}

impl<'m, 'q: 'm, Symbol, Probability, D, const PRECISION: usize> IterableEntropyModel<'m, PRECISION>
    for LeakilyQuantizedDistribution<'q, f64, Symbol, Probability, D, PRECISION>
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + AsPrimitive<usize> + Into<f64> + WrappingSub,
    Probability: BitArray + Into<f64>,
    D: Distribution + 'm,
    D::Value: AsPrimitive<Symbol>,
{
    type Iter = LeakilyQuantizedDistributionIter<Symbol, Probability, &'m Self, PRECISION>;

    fn symbol_table(&'m self) -> Self::Iter {
        LeakilyQuantizedDistributionIter {
            model: self,
            symbol: Some(self.quantizer.min_symbol_inclusive),
            left_sided_cumulative: Probability::zero(),
        }
    }
}

/// Iterator over the [`symbol_table`] of a [`LeakilyQuantizedDistribution`].
///
/// This type will become private once anonymous return types are allowed in trait methods.
/// Do not use it outside of the `constriction` library.
///
/// [`symbol_table`]: IterableEntropyModel::symbol_table
#[derive(Debug)]
pub struct LeakilyQuantizedDistributionIter<Symbol, Probability, M, const PRECISION: usize> {
    model: M,
    symbol: Option<Symbol>,
    left_sided_cumulative: Probability,
}

impl<'m, 'q, Symbol, Probability, D, const PRECISION: usize> Iterator
    for LeakilyQuantizedDistributionIter<
        Symbol,
        Probability,
        &'m LeakilyQuantizedDistribution<'q, f64, Symbol, Probability, D, PRECISION>,
        PRECISION,
    >
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + AsPrimitive<usize> + Into<f64> + WrappingSub,
    Probability: BitArray + Into<f64>,
    D: Distribution,
    D::Value: AsPrimitive<Symbol>,
{
    type Item = (Symbol, Probability, Probability::NonZero);

    fn next(&mut self) -> Option<Self::Item> {
        let symbol = self.symbol?;

        let right_sided_cumulative = if symbol == self.model.quantizer.max_symbol_inclusive {
            self.symbol = None;
            wrapping_pow2(PRECISION)
        } else {
            let next_symbol = symbol + Symbol::one();
            self.symbol = Some(next_symbol);
            let non_leaky: Probability = (self.model.quantizer.free_weight
                * self.model.inner.distribution((symbol).into() - 0.5))
            .as_();
            non_leaky + slack(next_symbol, self.model.quantizer.min_symbol_inclusive)
        };

        let probability = unsafe {
            // SAFETY: probabilities of
            right_sided_cumulative
                .wrapping_sub(&self.left_sided_cumulative)
                .into_nonzero_unchecked()
        };

        let left_sided_cumulative = self.left_sided_cumulative;
        self.left_sided_cumulative = right_sided_cumulative;

        Some((symbol, left_sided_cumulative, probability))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(symbol) = self.symbol {
            let len = slack::<usize, _>(symbol, self.model.quantizer.max_symbol_inclusive)
                .saturating_add(1);
            (len, None)
        } else {
            (0, Some(0))
        }
    }
}

/// A trait for internal representations of various forms of categorical entropy models.
///
/// This trait will become private once anonymous return types are allowed in trait methods.
/// Do not use it outside of the `constriction` library.
pub trait SymbolTable<Symbol, Probability: BitArray> {
    fn left_cumulative(&self, index: usize) -> Option<Probability>;

    fn support_size(&self) -> usize;

    /// # Safety
    ///
    /// Argument `index` must be strictly smaller than `1 << PRECISION` (for `PRECISION !=
    /// Probability::BITS`).
    unsafe fn left_cumulative_unchecked(&self, index: usize) -> Probability;

    /// # Safety
    ///
    /// Argument `symbol` must be in the support of the model.
    unsafe fn symbol_unchecked(&self, index: usize) -> Symbol;

    /// Bisects the symbol table to find the bin that contains `quantile`.
    fn quantile_function<const PRECISION: usize>(
        &self,
        quantile: Probability,
    ) -> (Symbol, Probability, Probability::NonZero) {
        assert!(PRECISION <= Probability::BITS);
        let max_probability = Probability::max_value() >> (Probability::BITS - PRECISION);
        assert!(quantile <= max_probability);

        let mut left = 0; // Smallest possible index.
        let mut right = self.support_size(); // One above largest possible index.

        // Bisect the symbol table to find the last entry whose left-sided cumulative is
        // `<= quantile`, exploiting the following facts:
        // - `self.as_ref.len() >= 2` (therefore, `left < right` initially)
        // - `cdf[0] == 0` (where `cdf[n] = self.left_cumulative_unchecked(n).0`)
        // - `quantile <= max_probability` (if this is violated then the method is still
        //   memory safe but will return the last bin; thus, memory safety doesn't hinge on
        //   `PRECISION` being correct).
        // - `cdf[self.as_ref().len() - 1] == max_probability.wrapping_add(1)`
        // - `cdf` is monotonically increasing except that it may wrap around only at the
        //   last entry (this happens iff `PRECISION == Probability::BITS`).
        //
        // The loop maintains the following two invariants:
        // (1) `0 <= left <= mid < right < self.as_ref().len()`
        // (2) `cdf[left] <= cdf[mid]`
        // (3) `cdf[mid] <= cdf[right]` unless `right == cdf.len() - 1`
        while left + 1 != right {
            let mid = (left + right) / 2;

            // SAFETY: safe by invariant (1)
            let pivot = unsafe { self.left_cumulative_unchecked(mid) };
            if pivot <= quantile {
                // Since `mid < right` and wrapping can occur only at the last entry,
                // `pivot` has not yet wrapped around
                left = mid;
            } else {
                right = mid;
            }
        }

        // SAFETY: invariant `0 <= left < right < self.as_ref().len()` still holds.
        let cdf = unsafe { self.left_cumulative_unchecked(left) };
        let symbol = unsafe { self.symbol_unchecked(left) };
        let next_cdf = unsafe { self.left_cumulative_unchecked(right) };

        let probability = unsafe {
            // SAFETY: The constructor ensures that all probabilities within bounds are
            // nonzero. (TODO)
            next_cdf.wrapping_sub(&cdf).into_nonzero_unchecked()
        };

        (symbol, cdf, probability)
    }
}

/// Internal representation of [`ContiguousCategoricalEntropyModel`].
///
/// This type will become private once anonymous return types are allowed in trait methods.
/// Do not use it outside of the `constriction` library.
#[derive(Debug, Clone, Copy)]
pub struct ContiguousSymbolTable<Table>(Table);

/// Internal representation of [`NonContiguousCategoricalEncoderModel`] and
/// [`NonContiguousCategoricalDecoderModel`].
///
/// This type will become private once anonymous return types are allowed in trait methods.
/// Do not use it outside of the `constriction` library.
#[derive(Debug, Clone, Copy)]
pub struct NonContiguousSymbolTable<Table>(Table);

impl<Symbol, Probability, Table> SymbolTable<Symbol, Probability> for ContiguousSymbolTable<Table>
where
    Probability: BitArray,
    Table: AsRef<[Probability]>,
    Symbol: BitArray + Into<usize>,
    usize: AsPrimitive<Symbol>,
{
    #[inline(always)]
    fn left_cumulative(&self, index: usize) -> Option<Probability> {
        self.0.as_ref().get(index).copied()
    }

    #[inline(always)]
    unsafe fn left_cumulative_unchecked(&self, index: usize) -> Probability {
        *self.0.as_ref().get_unchecked(index)
    }

    #[inline(always)]
    unsafe fn symbol_unchecked(&self, index: usize) -> Symbol {
        index.as_()
    }

    #[inline(always)]
    fn support_size(&self) -> usize {
        self.0.as_ref().len() - 1
    }
}

impl<Symbol, Probability, Table> SymbolTable<Symbol, Probability>
    for NonContiguousSymbolTable<Table>
where
    Probability: BitArray,
    Symbol: Clone,
    Table: AsRef<[(Probability, Symbol)]>,
{
    #[inline(always)]
    fn left_cumulative(&self, index: usize) -> Option<Probability> {
        self.0
            .as_ref()
            .get(index)
            .map(|(probability, _)| *probability)
    }

    #[inline(always)]
    unsafe fn left_cumulative_unchecked(&self, index: usize) -> Probability {
        self.0.as_ref().get_unchecked(index).0
    }

    #[inline(always)]
    unsafe fn symbol_unchecked(&self, index: usize) -> Symbol {
        self.0.as_ref().get_unchecked(index).1.clone()
    }

    #[inline(always)]
    fn support_size(&self) -> usize {
        self.0.as_ref().len() - 1
    }
}

/// Iterator over the [`symbol_table`] of various categorical distributions.
///
/// This type will become private once anonymous return types are allowed in trait methods.
///
/// [`symbol_table`]: IterableEntropyModel::symbol_table
#[derive(Debug)]
pub struct SymbolTableIter<Symbol, Probability, Table> {
    table: Table,
    index: usize,
    phantom: PhantomData<(Symbol, Probability)>,
}

impl<Symbol, Probability, Table> SymbolTableIter<Symbol, Probability, Table> {
    fn new(table: Table) -> Self {
        Self {
            table,
            index: 0,
            phantom: PhantomData,
        }
    }
}

impl<'a, Symbol, Probability, Table> Iterator for SymbolTableIter<Symbol, Probability, Table>
where
    Probability: BitArray,
    Table: SymbolTable<Symbol, Probability>,
{
    type Item = (Symbol, Probability, Probability::NonZero);

    fn next(&mut self) -> Option<Self::Item> {
        let old_index = self.index;
        if old_index == self.table.support_size() {
            None
        } else {
            let new_index = old_index + 1;
            self.index = new_index;
            unsafe {
                // SAFETY: TODO
                let left_cumulative = self.table.left_cumulative_unchecked(old_index);
                let symbol = self.table.symbol_unchecked(old_index);
                let right_cumulative = self.table.left_cumulative_unchecked(new_index);
                let probability = right_cumulative
                    .wrapping_sub(&left_cumulative)
                    .into_nonzero_unchecked();
                Some((symbol, left_cumulative, probability))
            }
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.table.support_size() - self.index;
        (len, Some(len))
    }
}

/// An entropy model for a categorical probability distribution over a contiguous range of
/// integers starting at zero.
///
/// You will usually want to use this type through one of its type aliases,
/// [`DefaultContiguousCategoricalEntropyModel`] or
/// [`SmallContiguousCategoricalEntropyModel`], see [discussion of presets](super#presets).
///
/// This entropy model implements both [`EncoderModel`] and [`DecoderModel`], which means
/// that it can be used for both encoding and decoding with any of the stream coders
/// provided by the `constriction` crate.
///
/// # Example
///
/// ```
/// use constriction::{
///     stream::{stack::DefaultAnsCoder, model::DefaultContiguousCategoricalEntropyModel, Decode},
///     UnwrapInfallible,
/// };
///
/// // Create a `ContiguousCategoricalEntropyModel` that approximates floating point probabilities.
/// let probabilities = [0.3, 0.0, 0.4, 0.1, 0.2]; // Note that `probabilities[1] == 0.0`.
/// let model = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities(
///     &probabilities
/// ).unwrap();
/// assert_eq!(model.support_size(), 5); // `model` supports the symbols `0..5usize`.
///
/// // Use `model` for entropy coding.
/// let message = vec![2, 0, 3, 1, 2, 4, 3, 2, 0];
/// let mut ans_coder = DefaultAnsCoder::new();
///
/// // We could pass `model` by reference but passing `model.as_view()` is slightly more efficient.
/// ans_coder.encode_iid_symbols_reverse(message.iter().cloned(), model.as_view()).unwrap();
/// // Note that `message` contains the symbol `1`, and that `probabilities[1] == 0.0`. However, we
/// // can still encode the symbol because the `ContiguousCategoricalEntropyModel` is "leaky", i.e.,
/// // it assigns a nonzero probability to all symbols in the range `0..model.support_size()`.
///
/// // Decode the encoded message and verify correctness.
/// let decoded = ans_coder
///     .decode_iid_symbols(9, model.as_view())
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap_infallible();
/// assert_eq!(decoded, message);
/// assert!(ans_coder.is_empty());
///
/// // The `model` assigns zero probability to any symbols that are not in the support
/// // `0..model.support_size()`, so trying to encode a message that contains such a symbol fails.
/// assert!(ans_coder.encode_iid_symbols_reverse(&[2, 0, 5, 1], model.as_view()).is_err())
/// // ERROR: symbol `5` is not in the support of `model`.
/// ```
///
/// # When Should I Use This Type of Entropy Model?
///
/// Use a `ContiguousCategoricalEntropyModel` for probabilistic models that can *only* be
/// represented as an explicit probability table, and not by some more compact analytic
/// expression. If you have a probability model that can be expressed by some analytical
/// expression (e.g., a [`Binomial`](probability::distribution::Binomial) distribution),
/// then use [`LeakyQuantizer`] instead (unless you want to encode lots of symbols with the
/// same entropy model, in which case the explicitly tabulated representation of a
/// categorical entropy model could improve runtime performance).
///
/// Further, a `ContiguousCategoricalEntropyModel` can only represent probability
/// distribution whose support (i.e., the set of symbols to which the model assigns a
/// non-zero probability) is a contiguous range of integers starting at zero. If the support
/// of your probability distribution has a more complicated structure (or if the `Symbol`
/// type is not an integer type), then you can use a
/// [`NonContiguousCategoricalEncoderModel`] or a [`NonContiguousCategoricalDecoderModel`],
/// which are strictly more general than a `ContiguousCategoricalEntropyModel` but which
/// have a larger memory footprint and slightly worse runtime performance.
///
/// If you want to *decode* lots of symbols with the same entropy model, and if reducing the
/// `PRECISION` to a moderate value is acceptable to you, then you may want to consider
/// using a [`LookupDecoderModel`] instead for even better runtime performance (at the cost
/// of a larger memory footprint and worse compression efficiency due to lower `PRECISION`).
///
/// # Computational Efficiency
///
/// For a probability distribution with a support of `N` symbols, a
/// `ContiguousCategoricalEntropyModel` has the following asymptotic costs:
///
/// - creation:
///   - runtime cost: `Θ(N)` when creating from fixed point probabilities, `Θ(N log(N))`
///     when creating from floating point probabilities;
///   - memory footprint: `Θ(N)`;
///   - both are cheaper by a constant factor than for a
///     [`NonContiguousCategoricalEncoderModel`] or a
///     [`NonContiguousCategoricalDecoderModel`].
/// - encoding a symbol (calling [`EncoderModel::left_cumulative_and_probability`]):
///   - runtime cost: `Θ(1)` (cheaper than for [`NonContiguousCategoricalEncoderModel`]
///     since it compiles to a simiple array lookup rather than a `HashMap` lookup)
///   - memory footprint: no heap allocations, constant stack space.
/// - decoding a symbol (calling [`DecoderModel::quantile_function`]):
///   - runtime cost: `Θ(log(N))` (both expected and worst-case; probably slightly cheaper
///     than for [`NonContiguousCategoricalDecoderModel`] due to better memory locality)
///   - memory footprint: no heap allocations, constant stack space.
///
/// [`EntropyModel`]: trait.EntropyModel.html
/// [`Encode`]: crate::Encode
/// [`Decode`]: crate::Decode
/// [`HashMap`]: std::hash::HashMap
#[derive(Debug, Clone, Copy)]
pub struct ContiguousCategoricalEntropyModel<Probability, Table, const PRECISION: usize> {
    /// Invariants:
    /// - `cdf.len() >= 2` (actually, we currently even guarantee `cdf.len() >= 3` but
    ///   this may be relaxed in the future)
    /// - `cdf[0] == 0`
    /// - `cdf` is monotonically increasing except that it may wrap around only at
    ///   the very last entry (this happens iff `PRECISION == Probability::BITS`).
    ///   Thus, all probabilities within range are guaranteed to be nonzero.
    cdf: ContiguousSymbolTable<Table>,

    phantom: PhantomData<Probability>,
}

/// An entropy model for a categorical probability distribution over arbitrary symbols, for
/// decoding only.
///
/// You will usually want to use this type through one of its type aliases,
/// [`DefaultNonContiguousCategoricalDecoderModel`] or
/// [`SmallNonContiguousCategoricalDecoderModel`], see [discussion of
/// presets](super#presets).
///
/// This type implements the trait [`DecoderModel`] but not the trait [`EncoderModel`].
/// Thus, you can use a `NonContiguousCategoricalDecoderModel` for *decoding* with any of
/// the stream decoders provided by the `constriction` crate, but not for encoding. If you
/// want to encode data, use a [`NonContiguousCategoricalEncoderModel`] instead. You can
/// convert a `NonContiguousCategoricalDecoderModel` to a
/// `NonContiguousCategoricalEncoderModel` by calling
/// [`to_generic_encoder_model`](IterableEntropyModel::to_generic_encoder_model) on it
/// (you'll have to bring the trait [`IterableEntropyModel`] into scope to do so: `use
/// constriction::stream::model::IterableEntropyModel`).
///
/// # Example
///
/// See [example for
/// `NonContiguousCategoricalEncoderModel`](NonContiguousCategoricalEncoderModel#example).
///
/// # When Should I Use This Type of Entropy Model?
///
/// Use a `NonContiguousCategoricalDecoderModel` for probabilistic models that can *only* be
/// represented as an explicit probability table, and not by some more compact analytic
/// expression. If you have a probability model that can be expressed by some analytical
/// expression (e.g., a [`Binomial`](probability::distribution::Binomial) distribution),
/// then use [`LeakyQuantizer`] instead (unless you want to encode lots of symbols with the
/// same entropy model, in which case the explicitly tabulated representation of a
/// categorical entropy model could improve runtime performance).
///
/// Further, if the *support* of your probabilistic model (i.e., the set of symbols to which
/// the model assigns a non-zero probability) is a contiguous range of integers starting at
/// zero, then it is better to use a [`ContiguousCategoricalEntropyModel`]. It has better
/// computational efficiency and it is easier to use since it supports both encoding and
/// decoding with a single type.
///
/// If you want to *decode* lots of symbols with the same entropy model, and if reducing the
/// `PRECISION` to a moderate value is acceptable to you, then you may want to consider
/// using a [`LookupDecoderModel`] instead for even better runtime performance (at the cost
/// of a larger memory footprint and worse compression efficiency due to lower `PRECISION`).
///
/// # Computational Efficiency
///
/// For a probability distribution with a support of `N` symbols, a
/// `NonContiguousCategoricalDecoderModel` has the following asymptotic costs:
///
/// - creation:
///   - runtime cost: `Θ(N)` when creating from fixed point probabilities, `Θ(N log(N))`
///     when creating from floating point probabilities;
///   - memory footprint: `Θ(N)`;
///   - both are more expensive by a constant factor than for a
///     [`ContiguousCategoricalEntropyModel`].
/// - encoding a symbol: not supported; use a [`NonContiguousCategoricalEncoderModel`].
/// - decoding a symbol (calling [`DecoderModel::quantile_function`]):
///   - runtime cost: `Θ(log(N))` (both expected and worst-case)
///   - memory footprint: no heap allocations, constant stack space.
///
/// [`EntropyModel`]: trait.EntropyModel.html
/// [`Encode`]: crate::Encode
/// [`Decode`]: crate::Decode
/// [`HashMap`]: std::hash::HashMap
#[derive(Debug, Clone, Copy)]
pub struct NonContiguousCategoricalDecoderModel<Symbol, Probability, Table, const PRECISION: usize>
{
    /// Invariants:
    /// - `cdf.len() >= 2` (actually, we currently even guarantee `cdf.len() >= 3` but
    ///   this may be relaxed in the future)
    /// - `cdf[0] == 0`
    /// - `cdf` is monotonically increasing except that it may wrap around only at
    ///   the very last entry (this happens iff `PRECISION == Probability::BITS`).
    ///   Thus, all probabilities within range are guaranteed to be nonzero.
    cdf: NonContiguousSymbolTable<Table>,

    phantom: PhantomData<(Symbol, Probability)>,
}

/// Type alias for a typical [`ContiguousCategoricalEntropyModel`].
///
/// See:
/// - [`ContiguousCategoricalEntropyModel`]
/// - [discussion of presets](super#presets)
pub type DefaultContiguousCategoricalEntropyModel<Table = Vec<u32>> =
    ContiguousCategoricalEntropyModel<u32, Table, 24>;

/// Type alias for a [`ContiguousCategoricalEntropyModel`] optimized for compatibility with
/// lookup decoder models.
///
/// See:
/// - [`ContiguousCategoricalEntropyModel`]
/// - [discussion of presets](super#presets)
pub type SmallContiguousCategoricalEntropyModel<Table = Vec<u16>> =
    ContiguousCategoricalEntropyModel<u16, Table, 12>;

/// Type alias for a typical [`NonContiguousCategoricalDecoderModel`].
///
/// See:
/// - [`NonContiguousCategoricalDecoderModel`]
/// - [discussion of presets](super#presets)
pub type DefaultNonContiguousCategoricalDecoderModel<Symbol, Table = Vec<(u32, Symbol)>> =
    NonContiguousCategoricalDecoderModel<Symbol, u32, Table, 24>;

/// Type alias for a [`NonContiguousCategoricalDecoderModel`] optimized for compatibility
/// with lookup decoder models.
///
/// See:
/// - [`NonContiguousCategoricalDecoderModel`]
/// - [discussion of presets](super#presets)
pub type SmallNonContiguousCategoricalDecoderModel<Symbol, Table = Vec<(u16, Symbol)>> =
    NonContiguousCategoricalDecoderModel<Symbol, u16, Table, 12>;

impl<Probability: BitArray, const PRECISION: usize>
    ContiguousCategoricalEntropyModel<Probability, Vec<Probability>, PRECISION>
{
    /// Constructs a leaky distribution whose PMF approximates given probabilities.
    ///
    /// The returned distribution will be defined for symbols of type `usize` from
    /// the range `0..probabilities.len()`.
    ///
    /// The argument `probabilities` is a slice of floating point values (`F` is
    /// typically `f64` or `f32`). All entries must be nonnegative and at least one
    /// entry has to be nonzero. The entries do not necessarily need to add up to
    /// one (the resulting distribution will automatically get normalized and an
    /// overall scaling of all entries of `probabilities` does not affect the
    /// result, up to effects due to rounding errors).
    ///
    /// The probability mass function of the returned distribution will approximate
    /// the provided probabilities as well as possible, subject to the following
    /// constraints:
    /// - probabilities are represented in fixed point arithmetic, where the const
    ///   generic parameter `PRECISION` controls the number of bits of precision.
    ///   This typically introduces rounding errors;
    /// - despite the possibility of rounding errors, the returned probability
    ///   distribution will be exactly normalized; and
    /// - each symbol in the support `0..probabilities.len()` gets assigned a strictly
    ///   nonzero probability, even if the provided probability for the symbol is zero or
    ///   below the threshold that can be resolved in fixed point arithmetic with
    ///   `PRECISION` bits. We refer to this property as the resulting distribution being
    ///   "leaky". The leakiness guarantees that all symbols within the support can be
    ///   encoded when this distribution is used as an entropy model.
    ///
    /// More precisely, the resulting probability distribution minimizes the cross
    /// entropy from the provided (floating point) to the resulting (fixed point)
    /// probabilities subject to the above three constraints.
    ///
    /// # Error Handling
    ///
    /// Returns an error if the provided probability distribution cannot be
    /// normalized, either because `probabilities` is of length zero, or because one
    /// of its entries is negative with a nonzero magnitude, or because the sum of
    /// its elements is zero, infinite, or NaN.
    ///
    /// Also returns an error if the probability distribution is degenerate, i.e.,
    /// if `probabilities` has only a single element, because degenerate probability
    /// distributions currently cannot be represented.
    ///
    /// TODO: should also return an error if support is too large to support leaky
    /// distribution
    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities<F>(probabilities: &[F]) -> Result<Self, ()>
    where
        F: Float + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        let slots = optimize_leaky_categorical::<_, _, PRECISION>(probabilities)?;
        Self::from_nonzero_fixed_point_probabilities(
            slots.into_iter().map(|slot| slot.weight),
            false,
        )
    }

    /// Constructs a distribution with a PMF given in fixed point arithmetic.
    ///
    /// This is a low level method that allows, e.g,. reconstructing a probability
    /// distribution previously exported with [`symbol_table`]. The more common way to
    /// construct a `LeakyCategorical` distribution is via
    /// [`from_floating_point_probabilities`].
    ///
    /// The items of `probabilities` have to be nonzero and smaller than `1 << PRECISION`,
    /// where `PRECISION` is a const generic parameter on the
    /// `ContiguousCategoricalEntropyModel`.
    ///
    /// If `infer_last_probability` is `false` then the items yielded by `probabilities`
    /// have to (logically) sum up to `1 << PRECISION`. If `infer_last_probability` is
    /// `true` then they must sum up to a value strictly smaller than `1 << PRECISION`, and
    /// the method will add an additional symbol at the end that takes the remaining
    /// probability mass.
    ///
    /// # Examples
    ///
    /// If `infer_last_probability` is `false`, the provided probabilities have to sum up to
    /// `1 << PRECISION`:
    ///
    /// ```
    /// use constriction::stream::model::{
    ///     DefaultContiguousCategoricalEntropyModel, IterableEntropyModel
    /// };
    ///
    /// let probabilities = vec![1u32 << 21, 1 << 22, 1 << 22, 1 << 22, 1 << 21];
    /// // `probabilities` sums up to `1 << PRECISION` as required:
    /// assert_eq!(probabilities.iter().sum::<u32>(), 1 << 24);
    ///
    /// let model = DefaultContiguousCategoricalEntropyModel
    ///     ::from_nonzero_fixed_point_probabilities(&probabilities, false).unwrap();
    /// let symbol_table = model.floating_point_symbol_table::<f64>().collect::<Vec<_>>();
    /// assert_eq!(
    ///     symbol_table,
    ///     vec![
    ///         (0, 0.0, 0.125),
    ///         (1, 0.125, 0.25),
    ///         (2, 0.375, 0.25),
    ///         (3, 0.625, 0.25),
    ///         (4, 0.875, 0.125),
    ///     ]
    /// );
    /// ```
    ///
    /// If `PRECISION` is set to the maximum value supported by the type `Probability`, then
    /// the provided probabilities still have to *logically* sum up to `1 << PRECISION`
    /// (i.e., the summation has to wrap around exactly once):
    ///
    /// ```
    /// use constriction::stream::model::{
    ///     ContiguousCategoricalEntropyModel, IterableEntropyModel
    /// };
    ///
    /// let probabilities = vec![1u32 << 29, 1 << 30, 1 << 30, 1 << 30, 1 << 29];
    /// // `probabilities` sums up to `1 << 32` (logically), i.e., it wraps around once.
    /// assert_eq!(probabilities.iter().fold(0u32, |accum, &x| accum.wrapping_add(x)), 0);
    ///
    /// let model = ContiguousCategoricalEntropyModel::<u32, Vec<u32>, 32>
    ///     ::from_nonzero_fixed_point_probabilities(&probabilities, false).unwrap();
    /// let symbol_table = model.floating_point_symbol_table::<f64>().collect::<Vec<_>>();
    /// assert_eq!(
    ///     symbol_table,
    ///     vec![
    ///         (0, 0.0, 0.125),
    ///         (1, 0.125, 0.25),
    ///         (2, 0.375, 0.25),
    ///         (3, 0.625, 0.25),
    ///         (4, 0.875, 0.125)
    ///     ]
    /// );
    /// ```
    ///
    /// Wrapping around twice fails:
    ///
    /// ```
    /// use constriction::stream::model::ContiguousCategoricalEntropyModel;
    /// let probabilities = vec![1u32 << 30, 1 << 31, 1 << 31, 1 << 31, 1 << 30];
    /// // `probabilities` sums up to `1 << 33` (logically), i.e., it would wrap around twice.
    /// assert!(
    ///     ContiguousCategoricalEntropyModel::<u32, Vec<u32>, 32>
    ///         ::from_nonzero_fixed_point_probabilities(&probabilities, false).is_err()
    /// );
    /// ```
    ///
    /// So does providing probabilities that just don't sum up to `1 << FREQUENCY`:
    ///
    /// ```
    /// use constriction::stream::model::ContiguousCategoricalEntropyModel;
    /// let probabilities = vec![1u32 << 21, 5 << 8, 1 << 22, 1 << 21];
    /// // `probabilities` sums up to `1 << 33` (logically), i.e., it would wrap around twice.
    /// assert!(
    ///     ContiguousCategoricalEntropyModel::<u32, Vec<u32>, 32>
    ///         ::from_nonzero_fixed_point_probabilities(&probabilities, false).is_err()
    /// );
    /// ```
    ///
    /// [`symbol_table`]: IterableEntropyModel::symbol_table
    /// [`fixed_point_probabilities`]: #method.fixed_point_probabilities
    /// [`from_floating_point_probabilities`]: #method.from_floating_point_probabilities
    #[allow(clippy::result_unit_err)]
    pub fn from_nonzero_fixed_point_probabilities<I>(
        probabilities: I,
        infer_last_probability: bool,
    ) -> Result<Self, ()>
    where
        I: IntoIterator,
        I::Item: Borrow<Probability>,
    {
        let probabilities = probabilities.into_iter();
        let mut cdf =
            Vec::with_capacity(probabilities.size_hint().0 + 1 + infer_last_probability as usize);
        accumulate_nonzero_probabilities::<_, _, _, _, _, PRECISION>(
            core::iter::repeat(()),
            probabilities,
            |(), left_sided_cumulative, _| {
                cdf.push(left_sided_cumulative);
                Ok(())
            },
            infer_last_probability,
        )?;
        cdf.push(wrapping_pow2(PRECISION));

        Ok(Self {
            cdf: ContiguousSymbolTable(cdf),
            phantom: PhantomData,
        })
    }
}

impl<Symbol, Probability: BitArray, const PRECISION: usize>
    NonContiguousCategoricalDecoderModel<Symbol, Probability, Vec<(Probability, Symbol)>, PRECISION>
where
    Symbol: Clone,
{
    /// Constructs a leaky distribution over the provided `symbols` whose PMF approximates
    /// given `probabilities`.
    ///
    /// The argument `probabilities` is a slice of floating point values (`F` is
    /// typically `f64` or `f32`). All entries must be nonnegative and at least one
    /// entry has to be nonzero. The entries do not necessarily need to add up to
    /// one (the resulting distribution will automatically get normalized and an
    /// overall scaling of all entries of `probabilities` does not affect the
    /// result, up to effects due to rounding errors).
    ///
    /// The probability mass function of the returned distribution will approximate
    /// the provided probabilities as well as possible, subject to the following
    /// constraints:
    /// - probabilities are represented in fixed point arithmetic, where the const
    ///   generic parameter `PRECISION` controls the number of bits of precision.
    ///   This typically introduces rounding errors;
    /// - despite the possibility of rounding errors, the returned probability
    ///   distribution will be exactly normalized; and
    /// - each symbol gets assigned a strictly nonzero probability, even if the provided
    ///   probability for the symbol is zero or below the threshold that can be resolved in
    ///   fixed point arithmetic with `PRECISION` bits. We refer to this property as the
    ///   resulting distribution being "leaky". The leakiness guarantees that a decoder can
    ///   in principle decode any of the provided symbols (if given appropriate compressed
    ///   data).
    ///
    /// More precisely, the resulting probability distribution minimizes the cross
    /// entropy from the provided (floating point) to the resulting (fixed point)
    /// probabilities subject to the above three constraints.
    ///
    /// # Error Handling
    ///
    /// Returns an error if `symbols.len() != probabilities.len()`.
    ///
    /// Also returns an error if the provided probability distribution cannot be normalized,
    /// either because `probabilities` is of length zero, or because one of its entries is
    /// negative with a nonzero magnitude, or because the sum of its elements is zero,
    /// infinite, or NaN.
    ///
    /// Also returns an error if the probability distribution is degenerate, i.e.,
    /// if `probabilities` has only a single element, because degenerate probability
    /// distributions currently cannot be represented.
    ///
    /// TODO: should also return an error if support is too large to support leaky
    /// distribution
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities<F>(
        symbols: &[Symbol],
        probabilities: &[F],
    ) -> Result<Self, ()>
    where
        F: Float + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        if symbols.len() != probabilities.len() {
            return Err(());
        };

        let slots = optimize_leaky_categorical::<_, _, PRECISION>(probabilities)?;
        Self::from_symbols_and_nonzero_fixed_point_probabilities(
            symbols.iter().cloned(),
            slots.into_iter().map(|slot| slot.weight),
            false,
        )
    }

    /// Constructs a distribution with a PMF given in fixed point arithmetic.
    ///
    /// This is a low level method that allows, e.g,. reconstructing a probability
    /// distribution previously exported with [`symbol_table`]. The more common way to
    /// construct a `NonContiguousCategoricalDecoderModel` distribution is via
    /// [`from_symbols_and_floating_point_probabilities`].
    ///
    /// The items of `probabilities` have to be nonzero and smaller than `1 << PRECISION`,
    /// where `PRECISION` is a const generic parameter on the
    /// `NonContiguousCategoricalDecoderModel`.
    ///
    /// If `infer_last_probability` is `false` then `probabilities` must yield the same
    /// number of items as `symbols` does and the items yielded by `probabilities` have to
    /// to (logically) sum up to `1 << PRECISION`. If `infer_last_probability` is `true`
    /// then `probabilities` must yield one fewer item than `symbols`, they items must sum
    /// up to a value strictly smaller than `1 << PRECISION`, and the method will assign the
    /// (nonzero) remaining probability to the last symbol.
    ///
    /// # Example
    ///
    /// Creating a `NonContiguousCategoricalDecoderModel` with inferred probability of the
    /// last symbol:
    ///
    /// ```
    /// use constriction::stream::model::{
    ///     DefaultNonContiguousCategoricalDecoderModel, IterableEntropyModel
    /// };
    ///
    /// let partial_probabilities = vec![1u32 << 21, 1 << 22, 1 << 22, 1 << 22];
    /// // `partial_probabilities` sums up to strictly less than `1 << PRECISION` as required:
    /// assert!(partial_probabilities.iter().sum::<u32>() < 1 << 24);
    ///
    /// let symbols = "abcde"; // Has one more entry than `probabilities`
    ///
    /// let model = DefaultNonContiguousCategoricalDecoderModel
    ///     ::from_symbols_and_nonzero_fixed_point_probabilities(
    ///         symbols.chars(), &partial_probabilities, true).unwrap();
    /// let symbol_table = model.floating_point_symbol_table::<f64>().collect::<Vec<_>>();
    /// assert_eq!(
    ///     symbol_table,
    ///     vec![
    ///         ('a', 0.0, 0.125),
    ///         ('b', 0.125, 0.25),
    ///         ('c', 0.375, 0.25),
    ///         ('d', 0.625, 0.25),
    ///         ('e', 0.875, 0.125), // Inferred last probability.
    ///     ]
    /// );
    /// ```
    ///
    /// For more related examples, see
    /// [`ContiguousCategoricalEntropyModel::from_nonzero_fixed_point_probabilities`].
    ///
    /// [`symbol_table`]: IterableEntropyModel::symbol_table
    /// [`fixed_point_probabilities`]: Self::fixed_point_probabilities
    /// [`from_symbols_and_floating_point_probabilities`]:
    ///     Self::from_symbols_and_floating_point_probabilities
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_nonzero_fixed_point_probabilities<S, P>(
        symbols: S,
        probabilities: P,
        infer_last_probability: bool,
    ) -> Result<Self, ()>
    where
        S: IntoIterator<Item = Symbol>,
        P: IntoIterator,
        P::Item: Borrow<Probability>,
    {
        let symbols = symbols.into_iter();
        let mut cdf = Vec::with_capacity(symbols.size_hint().0 + 1);
        let mut symbols = accumulate_nonzero_probabilities::<_, _, _, _, _, PRECISION>(
            symbols,
            probabilities.into_iter(),
            |symbol, left_sided_cumulative, _| {
                cdf.push((left_sided_cumulative, symbol));
                Ok(())
            },
            infer_last_probability,
        )?;
        cdf.push((
            wrapping_pow2(PRECISION),
            cdf.last().expect("`symbols` is not empty").1.clone(),
        ));

        if symbols.next().is_some() {
            Err(())
        } else {
            Ok(Self {
                cdf: NonContiguousSymbolTable(cdf),
                phantom: PhantomData,
            })
        }
    }

    /// Creates a `NonContiguousCategoricalDecoderModel` from any entropy model that
    /// implements [`IterableEntropyModel`].
    ///
    /// Calling `NonContiguousCategoricalDecoderModel::from_iterable_entropy_model(&model)`
    /// is equivalent to calling `model.to_generic_decoder_model()`, where the latter
    /// requires bringing [`IterableEntropyModel`] into scope.
    ///
    /// TODO: test
    pub fn from_iterable_entropy_model<'m, M>(model: &'m M) -> Self
    where
        M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
    {
        let symbol_table = model.symbol_table();
        let mut cdf = Vec::with_capacity(symbol_table.size_hint().0);
        for (symbol, left_sided_cumulative, _) in symbol_table {
            cdf.push((left_sided_cumulative, symbol));
        }
        cdf.push((
            wrapping_pow2(PRECISION),
            cdf.last().expect("`symbol_table` is not empty").1.clone(),
        ));

        Self {
            cdf: NonContiguousSymbolTable(cdf),
            phantom: PhantomData,
        }
    }
}

impl<Probability, Table, const PRECISION: usize>
    ContiguousCategoricalEntropyModel<Probability, Table, PRECISION>
where
    Probability: BitArray,
    Table: AsRef<[Probability]>,
{
    /// Returns the number of symbols supported by the model.
    ///
    /// The distribution is defined on the contiguous range of symbols from zero
    /// (inclusively) to `support_size()` (exclusively). All symbols within this range are
    /// guaranteed to have a nonzero probability, while all symbols outside of this range
    /// have a zero probability.
    #[inline(always)]
    pub fn support_size(&self) -> usize {
        SymbolTable::<usize, Probability>::support_size(&self.cdf)
    }

    /// Makes a very cheap shallow copy of the model that can be used much like a shared
    /// reference.
    ///
    /// The returned `ContiguousCategoricalEntropyModel` implements `Copy`, which is a
    /// requirement for some methods, such as [`Encode::encode_iid_symbols`] or
    /// [`Decode::decode_iid_symbols`]. These methods could also accept a shared reference
    /// to a `ContiguousCategoricalEntropyModel` (since all references to entropy models are
    /// also entropy models, and all shared references implement `Copy`), but passing a
    /// *view* instead may be slightly more efficient because it avoids one level of
    /// dereferencing.
    ///
    /// [`Encode::encode_iid_symbols`]: super::Encode::encode_iid_symbols
    /// [`Decode::decode_iid_symbols`]: super::Decode::decode_iid_symbols
    #[inline]
    pub fn as_view(
        &self,
    ) -> ContiguousCategoricalEntropyModel<Probability, &[Probability], PRECISION> {
        ContiguousCategoricalEntropyModel {
            cdf: ContiguousSymbolTable(self.cdf.0.as_ref()),
            phantom: PhantomData,
        }
    }

    /// Creates a [`LookupDecoderModel`] for efficient decoding of i.i.d. data
    ///
    /// While a `ContiguousCategoricalEntropyModel` can already be used for decoding (since
    /// it implements [`DecoderModel`]), you may prefer converting it to a
    /// `LookupDecoderModel` first for improved efficiency. Logically, the two will be
    /// equivalent.
    ///
    /// # Warning
    ///
    /// You should only call this method if both of the following conditions are satisfied:
    ///
    /// - `PRECISION` is relatively small (typically `PRECISION == 12`, as in the "Small"
    ///   [preset]) because the memory footprint of a `LookupDecoderModel` grows
    ///   exponentially in `PRECISION`; and
    /// - you're about to decode a relatively large number of symbols with the resulting
    ///   model; the conversion to a `LookupDecoderModel` bears a significant runtime and
    ///   memory overhead, so if you're going to use the resulting model only for a single
    ///   or a handful of symbols then you'll end up paying more than you gain.
    ///
    /// [preset]: super#presets
    #[inline(always)]
    pub fn to_lookup_decoder_model(
        &self,
    ) -> LookupDecoderModel<
        Probability,
        Probability,
        ContiguousSymbolTable<Vec<Probability>>,
        Box<[Probability]>,
        PRECISION,
    >
    where
        Probability: Into<usize>,
        usize: AsPrimitive<Probability>,
    {
        self.into()
    }
}

impl<Symbol, Probability, Table, const PRECISION: usize>
    NonContiguousCategoricalDecoderModel<Symbol, Probability, Table, PRECISION>
where
    Symbol: Clone,
    Probability: BitArray,
    Table: AsRef<[(Probability, Symbol)]>,
{
    /// Returns the number of symbols supported by the model, i.e., the number of symbols to
    /// which the model assigns a nonzero probability.
    #[inline(always)]
    pub fn support_size(&self) -> usize {
        self.cdf.support_size()
    }

    /// Makes a very cheap shallow copy of the model that can be used much like a shared
    /// reference.
    ///
    /// The returned `NonContiguousCategoricalDecoderModel` implements `Copy`, which is a
    /// requirement for some methods, such as [`Decode::decode_iid_symbols`]. These methods
    /// could also accept a shared reference to a `NonContiguousCategoricalDecoderModel`
    /// (since all references to entropy models are also entropy models, and all shared
    /// references implement `Copy`), but passing a *view* instead may be slightly more
    /// efficient because it avoids one level of dereferencing.
    ///
    /// [`Decode::decode_iid_symbols`]: super::Decode::decode_iid_symbols
    #[inline]
    pub fn as_view(
        &self,
    ) -> NonContiguousCategoricalDecoderModel<
        Symbol,
        Probability,
        &[(Probability, Symbol)],
        PRECISION,
    > {
        NonContiguousCategoricalDecoderModel {
            cdf: NonContiguousSymbolTable(self.cdf.0.as_ref()),
            phantom: PhantomData,
        }
    }
}

impl<Probability, Table, const PRECISION: usize> EntropyModel<PRECISION>
    for ContiguousCategoricalEntropyModel<Probability, Table, PRECISION>
where
    Probability: BitArray,
{
    type Symbol = usize;
    type Probability = Probability;
}

impl<Symbol, Probability, Table, const PRECISION: usize> EntropyModel<PRECISION>
    for NonContiguousCategoricalDecoderModel<Symbol, Probability, Table, PRECISION>
where
    Probability: BitArray,
{
    type Symbol = Symbol;
    type Probability = Probability;
}

impl<'m, Probability, Table, const PRECISION: usize> IterableEntropyModel<'m, PRECISION>
    for ContiguousCategoricalEntropyModel<Probability, Table, PRECISION>
where
    Probability: BitArray,
    Table: AsRef<[Probability]>,
{
    type Iter = SymbolTableIter<usize, Probability, ContiguousSymbolTable<&'m [Probability]>>;

    #[inline(always)]
    fn symbol_table(&'m self) -> Self::Iter {
        SymbolTableIter::new(self.as_view().cdf)
    }
}

impl<'m, Symbol, Probability, Table, const PRECISION: usize> IterableEntropyModel<'m, PRECISION>
    for NonContiguousCategoricalDecoderModel<Symbol, Probability, Table, PRECISION>
where
    Symbol: Clone + 'm,
    Probability: BitArray,
    Table: AsRef<[(Probability, Symbol)]>,
{
    type Iter =
        SymbolTableIter<Symbol, Probability, NonContiguousSymbolTable<&'m [(Probability, Symbol)]>>;

    #[inline(always)]
    fn symbol_table(&'m self) -> Self::Iter {
        SymbolTableIter::new(self.as_view().cdf)
    }
}

impl<Probability, Table, const PRECISION: usize> DecoderModel<PRECISION>
    for ContiguousCategoricalEntropyModel<Probability, Table, PRECISION>
where
    Probability: BitArray,
    Table: AsRef<[Probability]>,
{
    #[inline(always)]
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (usize, Probability, Probability::NonZero) {
        self.cdf.quantile_function::<PRECISION>(quantile)
    }
}

impl<Symbol, Probability, Table, const PRECISION: usize> DecoderModel<PRECISION>
    for NonContiguousCategoricalDecoderModel<Symbol, Probability, Table, PRECISION>
where
    Symbol: Clone,
    Probability: BitArray,
    Table: AsRef<[(Probability, Symbol)]>,
{
    #[inline(always)]
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Symbol, Probability, Probability::NonZero) {
        self.cdf.quantile_function::<PRECISION>(quantile)
    }
}

/// `EncoderModel` is only implemented for *contiguous* generic categorical models. To
/// decode encode symbols from a non-contiguous support, use an
/// `NonContiguousCategoricalEncoderModel`.
impl<Probability, Table, const PRECISION: usize> EncoderModel<PRECISION>
    for ContiguousCategoricalEntropyModel<Probability, Table, PRECISION>
where
    Probability: BitArray,
    Table: AsRef<[Probability]>,
{
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<usize>,
    ) -> Option<(Probability, Probability::NonZero)> {
        let index = *symbol.borrow();

        let (cdf, next_cdf) = unsafe {
            // SAFETY: we perform a single check if index is within bounds (we compare
            // `index >= len - 1` here and not `index + 1 >= len` because the latter could
            // overflow/wrap but `len` is guaranteed to be nonzero; once the check passes,
            // we know that `index + 1` doesn't wrap because `cdf.len()` can't be
            // `usize::max_value()` since that would mean that there's no space left even
            // for the call stack).
            if index >= self.support_size() {
                return None;
            }
            (
                SymbolTable::<usize, Probability>::left_cumulative_unchecked(&self.cdf, index),
                SymbolTable::<usize, Probability>::left_cumulative_unchecked(&self.cdf, index + 1),
            )
        };

        let probability = unsafe {
            // SAFETY: The constructors ensure that all probabilities within bounds are nonzero.
            next_cdf.wrapping_sub(&cdf).into_nonzero_unchecked()
        };

        Some((cdf, probability))
    }
}

impl<'m, Symbol, Probability, M, const PRECISION: usize> From<&'m M>
    for NonContiguousCategoricalDecoderModel<
        Symbol,
        Probability,
        Vec<(Probability, Symbol)>,
        PRECISION,
    >
where
    Symbol: Clone,
    Probability: BitArray,
    M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
{
    #[inline(always)]
    fn from(model: &'m M) -> Self {
        Self::from_iterable_entropy_model(model)
    }
}

/// An entropy model for a categorical probability distribution over arbitrary symbols, for
/// encoding only.
///
/// You will usually want to use this type through one of its type aliases,
/// [`DefaultNonContiguousCategoricalEncoderModel`] or
/// [`SmallNonContiguousCategoricalEncoderModel`], see [discussion of
/// presets](super#presets).
///
/// This type implements the trait [`EncoderModel`] but not the trait [`DecoderModel`].
/// Thus, you can use a `NonContiguousCategoricalEncoderModel` for *encoding* with any of
/// the stream encoders provided by the `constriction` crate, but not for decoding. If you
/// want to decode data, use a [`NonContiguousCategoricalDecoderModel`] instead.
///
/// # Example
///
/// ```
/// use constriction::{
///     stream::{stack::DefaultAnsCoder, Decode},
///     stream::model::DefaultNonContiguousCategoricalEncoderModel,
///     stream::model::DefaultNonContiguousCategoricalDecoderModel,
///     UnwrapInfallible,
/// };
///
/// // Create a `ContiguousCategoricalEntropyModel` that approximates floating point probabilities.
/// let alphabet = ['M', 'i', 's', 'p', '!'];
/// let probabilities = [0.09, 0.36, 0.36, 0.18, 0.0];
/// let encoder_model = DefaultNonContiguousCategoricalEncoderModel
///     ::from_symbols_and_floating_point_probabilities(alphabet.iter().cloned(), &probabilities)
///     .unwrap();
/// assert_eq!(encoder_model.support_size(), 5); // `encoder_model` supports 4 symbols.
///
/// // Use `encoder_model` for entropy coding.
/// let message = "Mississippi!";
/// let mut ans_coder = DefaultAnsCoder::new();
/// ans_coder.encode_iid_symbols_reverse(message.chars(), &encoder_model).unwrap();
/// // Note that `message` contains the symbol '!', which has zero probability under our
/// // floating-point model. However, we can still encode the symbol because the
/// // `NonContiguousCategoricalEntropyModel` is "leaky", i.e., it assigns a nonzero
/// // probability to all symbols that we provided to the constructor.
///
/// // Create a matching `decoder_model`, decode the encoded message, and verify correctness.
/// let decoder_model = DefaultNonContiguousCategoricalDecoderModel
///     ::from_symbols_and_floating_point_probabilities(&alphabet, &probabilities)
///     .unwrap();
///
/// // We could pass `decoder_model` by reference (like we did for `encoder_model` above) but
/// // passing `decoder_model.as_view()` is slightly more efficient.
/// let decoded = ans_coder
///     .decode_iid_symbols(12, decoder_model.as_view())
///     .collect::<Result<String, _>>()
///     .unwrap_infallible();
/// assert_eq!(decoded, message);
/// assert!(ans_coder.is_empty());
///
/// // The `encoder_model` assigns zero probability to any symbols that were not provided to its
/// // constructor, so trying to encode a message that contains such a symbol will fail.
/// assert!(ans_coder.encode_iid_symbols_reverse("Mix".chars(), &encoder_model).is_err())
/// // ERROR: symbol 'x' is not in the support of `encoder_model`.
/// ```
///
/// # When Should I Use This Type of Entropy Model?
///
/// Use a `NonContiguousCategoricalEncoderModel` for probabilistic models that can *only* be
/// represented as an explicit probability table, and not by some more compact analytic
/// expression. If you have a probability model that can be expressed by some analytical
/// expression (e.g., a [`Binomial`](probability::distribution::Binomial) distribution),
/// then use [`LeakyQuantizer`] instead (unless you want to encode lots of symbols with the
/// same entropy model, in which case the explicitly tabulated representation of a
/// categorical entropy model could improve runtime performance).
///
/// Further, if the *support* of your probabilistic model (i.e., the set of symbols to which
/// the model assigns a non-zero probability) is a contiguous range of integers starting at
/// zero, then it is better to use a [`ContiguousCategoricalEntropyModel`]. It has better
/// computational efficiency and it is easier to use since it supports both encoding and
/// decoding with a single type.
///
/// # Computational Efficiency
///
/// For a probability distribution with a support of `N` symbols, a
/// `NonContiguousCategoricalEncoderModel` has the following asymptotic costs:
///
/// - creation:
///   - runtime cost: `Θ(N)` when creating from fixed point probabilities, `Θ(N log(N))`
///     when creating from floating point probabilities;
///   - memory footprint: `Θ(N)`;
///   - both are more expensive by a constant factor than for a
///     [`ContiguousCategoricalEntropyModel`].
/// - encoding a symbol (calling [`EncoderModel::left_cumulative_and_probability`]):
///   - expected runtime cost: `Θ(1)` (worst case can be more expensive, uses a `HashMap`
///     under the hood).
///   - memory footprint: no heap allocations, constant stack space.
/// - decoding a symbol: not supported; use a [`NonContiguousCategoricalDecoderModel`].
///
/// [`EntropyModel`]: trait.EntropyModel.html
/// [`Encode`]: crate::Encode
/// [`Decode`]: crate::Decode
/// [`HashMap`]: std::hash::HashMap
#[derive(Debug, Clone)]
pub struct NonContiguousCategoricalEncoderModel<Symbol, Probability, const PRECISION: usize>
where
    Symbol: Hash,
    Probability: BitArray,
{
    table: HashMap<Symbol, (Probability, Probability::NonZero)>,
}

/// Type alias for a typical [`NonContiguousCategoricalEncoderModel`].
///
/// See:
/// - [`NonContiguousCategoricalEncoderModel`]
/// - [discussion of presets](super#presets)
pub type DefaultNonContiguousCategoricalEncoderModel<Symbol> =
    NonContiguousCategoricalEncoderModel<Symbol, u32, 24>;

/// Type alias for a [`NonContiguousCategoricalEncoderModel`] optimized for compatibility
/// with lookup decoder models.
///
/// See:
/// - [`NonContiguousCategoricalEncoderModel`]
/// - [discussion of presets](super#presets)
pub type SmallNonContiguousCategoricalEncoderModel<Symbol> =
    NonContiguousCategoricalEncoderModel<Symbol, u16, 12>;

impl<Symbol, Probability, const PRECISION: usize>
    NonContiguousCategoricalEncoderModel<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    /// Constructs a leaky distribution over the provided `symbols` whose PMF approximates
    /// given `probabilities`.
    ///
    /// This method operates logically identically to
    /// [`NonContiguousCategoricalDecoderModel::from_symbols_and_floating_point_probabilities`]
    /// except that it constructs an [`EncoderModel`] rather than a [`DecoderModel`].
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities<F>(
        symbols: impl IntoIterator<Item = Symbol>,
        probabilities: &[F],
    ) -> Result<Self, ()>
    where
        F: Float + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        let slots = optimize_leaky_categorical::<_, _, PRECISION>(probabilities)?;
        Self::from_symbols_and_nonzero_fixed_point_probabilities(
            symbols,
            slots.into_iter().map(|slot| slot.weight),
            false,
        )
    }

    /// Constructs a distribution with a PMF given in fixed point arithmetic.
    ///
    /// This method operates logically identically to
    /// [`NonContiguousCategoricalDecoderModel::from_symbols_and_nonzero_fixed_point_probabilities`]
    /// except that it constructs an [`EncoderModel`] rather than a [`DecoderModel`].
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_nonzero_fixed_point_probabilities<S, P>(
        symbols: S,
        probabilities: P,
        infer_last_probability: bool,
    ) -> Result<Self, ()>
    where
        S: IntoIterator<Item = Symbol>,
        P: IntoIterator,
        P::Item: Borrow<Probability>,
    {
        let symbols = symbols.into_iter();
        let mut table = HashMap::with_capacity(symbols.size_hint().0 + 1);
        let mut symbols = accumulate_nonzero_probabilities::<_, _, _, _, _, PRECISION>(
            symbols,
            probabilities.into_iter(),
            |symbol, left_sided_cumulative, probability| match table.entry(symbol) {
                Occupied(_) => Err(()),
                Vacant(slot) => {
                    let probability = probability.into_nonzero().ok_or(())?;
                    slot.insert((left_sided_cumulative, probability));
                    Ok(())
                }
            },
            infer_last_probability,
        )?;

        if symbols.next().is_some() {
            Err(())
        } else {
            Ok(Self { table })
        }
    }

    /// Creates a `NonContiguousCategoricalEncoderModel` from any entropy model that
    /// implements [`IterableEntropyModel`].
    ///
    /// Calling `NonContiguousCategoricalEncoderModel::from_iterable_entropy_model(&model)`
    /// is equivalent to calling `model.to_generic_encoder_model()`, where the latter
    /// requires bringing [`IterableEntropyModel`] into scope.
    ///
    /// TODO: test
    pub fn from_iterable_entropy_model<'m, M>(model: &'m M) -> Self
    where
        M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
    {
        let table = model
            .symbol_table()
            .map(|(symbol, left_sided_cumulative, probability)| {
                (symbol, (left_sided_cumulative, probability))
            })
            .collect::<HashMap<_, _>>();
        Self { table }
    }

    /// Returns the number of symbols in the support of the model.
    ///
    /// The support of the model is the set of all symbols that have nonzero probability.
    pub fn support_size(&self) -> usize {
        self.table.len()
    }

    /// Returns the entropy in units of bits (i.e., base 2).
    ///
    /// Similar to [`IterableEntropyModel::entropy_base2`], except that
    /// - this type doesn't implement `IterableEntropyModel` because it doesn't store
    ///   entries in a stable expected order;
    /// - because the order in which entries are stored will generally be different on each
    ///   program execution, rounding errors will be slightly different across multiple
    ///   program executions.
    pub fn entropy_base2<F>(&self) -> F
    where
        F: Float + core::iter::Sum,
        Probability: Into<F>,
    {
        let entropy_scaled = self
            .table
            .values()
            .map(|&(_, probability)| {
                let probability = probability.get().into();
                probability * probability.log2() // probability is guaranteed to be nonzero.
            })
            .sum::<F>();

        let whole = (F::one() + F::one()) * (Probability::one() << (PRECISION - 1)).into();
        F::from(PRECISION).unwrap() - entropy_scaled / whole
    }
}

impl<'m, Symbol, Probability, M, const PRECISION: usize> From<&'m M>
    for NonContiguousCategoricalEncoderModel<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
    M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
{
    #[inline(always)]
    fn from(model: &'m M) -> Self {
        Self::from_iterable_entropy_model(model)
    }
}

impl<Symbol, Probability: BitArray, const PRECISION: usize> EntropyModel<PRECISION>
    for NonContiguousCategoricalEncoderModel<Symbol, Probability, PRECISION>
where
    Symbol: Hash,
    Probability: BitArray,
{
    type Probability = Probability;
    type Symbol = Symbol;
}

impl<Symbol, Probability: BitArray, const PRECISION: usize> EncoderModel<PRECISION>
    for NonContiguousCategoricalEncoderModel<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    #[inline(always)]
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Self::Symbol>,
    ) -> Option<(Self::Probability, Probability::NonZero)> {
        self.table.get(symbol.borrow()).cloned()
    }
}

struct Slot<Probability> {
    original_index: usize,
    prob: f64,
    weight: Probability,
    win: f64,
    loss: f64,
}

/// Note: does not check if `symbols` is exhausted (this is so that you one can provide an
/// infinite iterator for `symbols` to optimize out the bounds check on it).
fn accumulate_nonzero_probabilities<Symbol, Probability, S, P, Op, const PRECISION: usize>(
    mut symbols: S,
    probabilities: P,
    mut operation: Op,
    infer_last_probability: bool,
) -> Result<S, ()>
where
    Probability: BitArray,
    S: Iterator<Item = Symbol>,
    P: Iterator,
    P::Item: Borrow<Probability>,
    Op: FnMut(Symbol, Probability, Probability) -> Result<(), ()>,
{
    assert!(PRECISION > 0);
    assert!(PRECISION <= Probability::BITS);

    // We accumulate all validity checks into single branches at the end in order to
    // keep the loop itself branchless.
    let mut laps_or_zeros = 0usize;
    let mut accum = Probability::zero();

    for probability in probabilities {
        let old_accum = accum;
        accum = accum.wrapping_add(probability.borrow());
        laps_or_zeros += (accum <= old_accum) as usize;
        let symbol = symbols.next().ok_or(())?;
        operation(symbol, old_accum, *probability.borrow())?;
    }

    let total = wrapping_pow2::<Probability>(PRECISION);

    if infer_last_probability {
        if accum >= total || laps_or_zeros != 0 {
            return Err(());
        }
        let symbol = symbols.next().ok_or(())?;
        let probability = total.wrapping_sub(&accum);
        operation(symbol, accum, probability)?;
    } else if accum != total || laps_or_zeros != (PRECISION == Probability::BITS) as usize {
        return Err(());
    }

    Ok(symbols)
}

fn optimize_leaky_categorical<Probability, F, const PRECISION: usize>(
    probabilities: &[F],
) -> Result<Vec<Slot<Probability>>, ()>
where
    F: Float + core::iter::Sum<F> + Into<f64>,
    Probability: BitArray + Into<f64> + AsPrimitive<usize>,
    f64: AsPrimitive<Probability>,
    usize: AsPrimitive<Probability>,
{
    assert!(PRECISION > 0 && PRECISION <= Probability::BITS);

    if probabilities.len() < 2 || probabilities.len() > Probability::max_value().as_() {
        return Err(());
    }

    // Start by assigning each symbol weight 1 and then distributing no more than
    // the remaining weight approximately evenly across all symbols.
    let mut remaining_free_weight =
        wrapping_pow2::<Probability>(PRECISION).wrapping_sub(&probabilities.len().as_());
    let normalization = probabilities.iter().map(|&x| x.into()).sum::<f64>();
    if !normalization.is_normal() || !normalization.is_sign_positive() {
        return Err(());
    }
    let scale = remaining_free_weight.into() / normalization;

    let mut slots = probabilities
        .iter()
        .enumerate()
        .map(|(original_index, &prob)| {
            if prob < F::zero() {
                return Err(());
            }
            let prob: f64 = prob.into();
            let current_free_weight = (prob * scale).as_();
            remaining_free_weight = remaining_free_weight - current_free_weight;
            let weight = current_free_weight + Probability::one();

            // How much the cross entropy would decrease when increasing the weight by one.
            let win = prob * (1.0f64 / weight.into()).ln_1p();

            // How much the cross entropy would increase when decreasing the weight by one.
            let loss = if weight == Probability::one() {
                f64::infinity()
            } else {
                -prob * (-1.0f64 / weight.into()).ln_1p()
            };

            Ok(Slot {
                original_index,
                prob,
                weight,
                win,
                loss,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Distribute remaining weight evenly among symbols with highest wins.
    while remaining_free_weight != Probability::zero() {
        // We can't use `sort_unstable_by` here because we want the result to be reproducible
        // even across updates of the standard library.
        slots.sort_by(|a, b| b.win.partial_cmp(&a.win).unwrap());
        let batch_size = core::cmp::min(remaining_free_weight.as_(), slots.len());
        for slot in &mut slots[..batch_size] {
            slot.weight = slot.weight + Probability::one(); // Cannot end up in `max_weight` because win would otherwise be -infinity.
            slot.win = slot.prob * (1.0f64 / slot.weight.into()).ln_1p();
            slot.loss = -slot.prob * (-1.0f64 / slot.weight.into()).ln_1p();
        }
        remaining_free_weight = remaining_free_weight - batch_size.as_();
    }

    loop {
        // Find slot where increasing its weight by one would incur the biggest win.
        let (buyer_index, &Slot { win: buyer_win, .. }) = slots
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.win.partial_cmp(&b.win).unwrap())
            .unwrap();
        // Find slot where decreasing its weight by one would incur the smallest loss.
        let (seller_index, seller) = slots
            .iter_mut()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.loss.partial_cmp(&b.loss).unwrap())
            .unwrap();

        if buyer_index == seller_index {
            // This can only happen due to rounding errors. In this case, we can't expect
            // to be able to improve further.
            break;
        }

        if buyer_win <= seller.loss {
            // We've found the optimal solution.
            break;
        }

        seller.weight = seller.weight - Probability::one();
        seller.win = seller.prob * (1.0f64 / seller.weight.into()).ln_1p();
        seller.loss = if seller.weight == Probability::one() {
            f64::infinity()
        } else {
            -seller.prob * (-1.0f64 / seller.weight.into()).ln_1p()
        };

        let buyer = &mut slots[buyer_index];
        buyer.weight = buyer.weight + Probability::one();
        buyer.win = buyer.prob * (1.0f64 / buyer.weight.into()).ln_1p();
        buyer.loss = -buyer.prob * (-1.0f64 / buyer.weight.into()).ln_1p();
    }

    slots.sort_unstable_by_key(|slot| slot.original_index);
    Ok(slots)
}

// LOOKUP TABLE ENTROPY MODELS (FOR FAST DECODING) ================================================

/// A tabularized [`DecoderModel`] that is optimized for fast decoding of i.i.d. symbols
///
/// You will usually want to use this type through one of the type aliases
/// [`SmallContiguousLookupDecoderModel`] or [`SmallNonContiguousLookupDecoderModel`]. See
/// these types for extended documentation and examples.
#[derive(Debug, Clone, Copy)]
pub struct LookupDecoderModel<Symbol, Probability, SymbolTable, LookupTable, const PRECISION: usize>
where
    Probability: BitArray,
{
    /// Satisfies invariant:
    /// `lookup_table.as_ref().len() == 1 << PRECISION`
    lookup_table: LookupTable,

    /// Satisfies invariant:
    /// `left_sided_cumulative_and_symbol.as_ref().len()
    /// == *lookup_table.as_ref().iter().max() as usize + 2`
    cdf: SymbolTable,

    phantom: PhantomData<(Probability, Symbol)>,
}

/// Type alias for a [`LookupDecoderModel`] over arbitrary symbols.
///
/// # Examples
///
/// TODO
///
/// # See also
///
/// - [`SmallNonContiguousLookupDecoderModel`]
pub type SmallNonContiguousLookupDecoderModel<
    Symbol,
    SymbolTable = Vec<(u16, Symbol)>,
    LookupTable = Box<[u16]>,
> = LookupDecoderModel<Symbol, u16, NonContiguousSymbolTable<SymbolTable>, LookupTable, 12>;

/// Type alias for a [`LookupDecoderModel`] over symbols `{0, 1, ..., n-1}` with sane settings.
///
/// This array lookup table can be used with a [`SmallAnsCoder`] or a [`SmallRangeDecoder`]
/// (as well as with a [`DefaultAnsCoder`] or a [`DefaultRangeDecoder`], since you can
/// always use a "bigger" coder on a "smaller" model).
///
/// # Example
///
/// Decoding a sequence of symbols with a [`SmallAnsCoder`], a [`DefaultAnsCoder`], a
/// [`SmallRangeDecoder`], and a [`DefaultRangeDecoder`], all using the same
/// `SmallContiguousLookupDecoderModel`.
///
/// ```
/// use constriction::stream::{
///     model::SmallContiguousLookupDecoderModel,
///     stack::{SmallAnsCoder, DefaultAnsCoder},
///     queue::{SmallRangeDecoder, DefaultRangeDecoder},
///     Decode, Code,
/// };
///
/// // Create a `SmallContiguousLookupDecoderModel` from a probability distribution that's already
/// // available in fixed point representation (e.g., because it was deserialized from a file).
/// // Alternatively, we could use `from_floating_point_probabilities_contiguous`.
/// let probabilities = [1489, 745, 1489, 373];
/// let decoder_model = SmallContiguousLookupDecoderModel
///     ::from_nonzero_fixed_point_probabilities_contiguous(&probabilities, false).unwrap();
///
/// let expected = [2, 1, 3, 0, 0, 2, 0, 2, 1, 0, 2];
///
/// let mut small_ans_coder = SmallAnsCoder::from_compressed(vec![0xDA86, 0x2949]).unwrap();
/// let reconstructed = small_ans_coder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
/// assert!(small_ans_coder.is_empty());
/// assert_eq!(reconstructed, expected);
///
/// let mut default_ans_decoder = DefaultAnsCoder::from_compressed(vec![0x2949DA86]).unwrap();
/// let reconstructed = default_ans_decoder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
/// assert!(default_ans_decoder.is_empty());
/// assert_eq!(reconstructed, expected);
///
/// let mut small_range_decoder = SmallRangeDecoder::from_compressed(vec![0xBCF8, 0x3ECA]).unwrap();
/// let reconstructed = small_range_decoder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
/// assert!(small_range_decoder.maybe_exhausted());
/// assert_eq!(reconstructed, expected);
///
/// let mut default_range_decoder = DefaultRangeDecoder::from_compressed(vec![0xBCF8733B]).unwrap();
/// let reconstructed = default_range_decoder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
/// assert!(default_range_decoder.maybe_exhausted());
/// assert_eq!(reconstructed, expected);
/// ```
///
/// # See also
///
/// - [`SmallNonContiguousLookupDecoderModel`]
///
/// [`SmallAnsCoder`]: super::stack::SmallAnsCoder
/// [`SmallRangeDecoder`]: super::queue::SmallRangeDecoder
/// [`DefaultAnsCoder`]: super::stack::DefaultAnsCoder
/// [`DefaultRangeDecoder`]: super::queue::DefaultRangeDecoder
pub type SmallContiguousLookupDecoderModel<SymbolTable = Vec<u16>, LookupTable = Box<[u16]>> =
    LookupDecoderModel<usize, u16, ContiguousSymbolTable<SymbolTable>, LookupTable, 12>;

impl<Symbol, Probability, const PRECISION: usize>
    LookupDecoderModel<
        Symbol,
        Probability,
        NonContiguousSymbolTable<Vec<(Probability, Symbol)>>,
        Box<[Probability]>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Symbol: Copy + Default,
{
    /// Create a `LookupDecoderModel` over arbitrary symbols.
    ///
    /// TODO: example
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities<F>(
        symbols: &[Symbol],
        probabilities: &[F],
    ) -> Result<Self, ()>
    where
        F: Float + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        if symbols.len() != probabilities.len() {
            return Err(());
        };

        let slots = optimize_leaky_categorical::<_, _, PRECISION>(probabilities)?;
        Self::from_symbols_and_nonzero_fixed_point_probabilities(
            symbols.iter().cloned(),
            slots.into_iter().map(|slot| slot.weight),
            false,
        )
    }

    /// Create a `LookupDecoderModel` over arbitrary symbols.
    ///
    /// TODO: example
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_nonzero_fixed_point_probabilities<S, P>(
        symbols: S,
        probabilities: P,
        infer_last_probability: bool,
    ) -> Result<Self, ()>
    where
        S: IntoIterator<Item = Symbol>,
        P: IntoIterator,
        P::Item: Borrow<Probability>,
    {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);
        assert!(PRECISION < <usize as BitArray>::BITS);

        let mut lookup_table = Vec::with_capacity(1 << PRECISION);
        let symbols = symbols.into_iter();
        let mut cdf =
            Vec::with_capacity(symbols.size_hint().0 + 1 + infer_last_probability as usize);
        let mut symbols = accumulate_nonzero_probabilities::<_, _, _, _, _, PRECISION>(
            symbols,
            probabilities.into_iter(),
            |symbol, _, probability| {
                let index = cdf.len().as_();
                cdf.push((lookup_table.len().as_(), symbol));
                lookup_table.resize(lookup_table.len() + probability.into(), index);
                Ok(())
            },
            infer_last_probability,
        )?;

        cdf.push((wrapping_pow2(PRECISION), Symbol::default()));

        if symbols.next().is_some() {
            Err(())
        } else {
            Ok(Self {
                lookup_table: lookup_table.into_boxed_slice(),
                cdf: NonContiguousSymbolTable(cdf),
                phantom: PhantomData,
            })
        }
    }

    /// TODO: test
    pub fn from_iterable_entropy_model<'m, M>(model: &'m M) -> Self
    where
        M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
    {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);
        assert!(PRECISION < <usize as BitArray>::BITS);

        let mut lookup_table = Vec::with_capacity(1 << PRECISION);
        let symbol_table = model.symbol_table();
        let mut cdf = Vec::with_capacity(symbol_table.size_hint().0 + 1);
        for (symbol, left_sided_cumulative, probability) in symbol_table {
            let index = cdf.len().as_();
            debug_assert_eq!(left_sided_cumulative, lookup_table.len().as_());
            cdf.push((lookup_table.len().as_(), symbol));
            lookup_table.resize(lookup_table.len() + probability.get().into(), index);
        }
        cdf.push((wrapping_pow2(PRECISION), Symbol::default()));

        Self {
            lookup_table: lookup_table.into_boxed_slice(),
            cdf: NonContiguousSymbolTable(cdf),
            phantom: PhantomData,
        }
    }
}

impl<Symbol, Probability, const PRECISION: usize>
    LookupDecoderModel<
        Symbol,
        Probability,
        ContiguousSymbolTable<Vec<Probability>>,
        Box<[Probability]>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Symbol: Copy + Default,
{
    /// Create a `LookupDecoderModel` over a contiguous range of symbols.
    ///
    /// TODO: example
    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities_contiguous<F>(probabilities: &[F]) -> Result<Self, ()>
    where
        F: Float + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        let slots = optimize_leaky_categorical::<_, _, PRECISION>(probabilities)?;
        Self::from_nonzero_fixed_point_probabilities_contiguous(
            slots.into_iter().map(|slot| slot.weight),
            false,
        )
    }

    /// Create a `LookupDecoderModel` over a contiguous range of symbols using fixed point arighmetic.
    ///
    /// # Example
    ///
    /// See [`SmallContiguousLookupDecoderModel`].
    #[allow(clippy::result_unit_err)]
    pub fn from_nonzero_fixed_point_probabilities_contiguous<I>(
        probabilities: I,
        infer_last_probability: bool,
    ) -> Result<Self, ()>
    where
        I: IntoIterator,
        I::Item: Borrow<Probability>,
    {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);
        assert!(PRECISION < <usize as BitArray>::BITS);

        let mut lookup_table = Vec::with_capacity(1 << PRECISION);
        let probabilities = probabilities.into_iter();
        let mut cdf =
            Vec::with_capacity(probabilities.size_hint().0 + 1 + infer_last_probability as usize);
        accumulate_nonzero_probabilities::<_, _, _, _, _, PRECISION>(
            core::iter::repeat(()),
            probabilities,
            |(), _, probability| {
                let index = cdf.len().as_();
                cdf.push(lookup_table.len().as_());
                lookup_table.resize(lookup_table.len() + probability.into(), index);
                Ok(())
            },
            infer_last_probability,
        )?;
        cdf.push(wrapping_pow2(PRECISION));

        Ok(Self {
            lookup_table: lookup_table.into_boxed_slice(),
            cdf: ContiguousSymbolTable(cdf),
            phantom: PhantomData,
        })
    }
}

impl<Probability, Table, LookupTable, const PRECISION: usize>
    LookupDecoderModel<
        Probability,
        Probability,
        ContiguousSymbolTable<Table>,
        LookupTable,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Table: AsRef<[Probability]>,
    LookupTable: AsRef<[Probability]>,
{
    /// Makes a very cheap shallow copy of the model that can be used much like a shared
    /// reference.
    ///
    /// The returned `LookupDecoderModel` implements `Copy`, which is a requirement for some
    /// methods, such as [`Decode::decode_iid_symbols`]. These methods could also accept a
    /// shared reference to a `NonContiguousCategoricalDecoderModel` (since all references
    /// to entropy models are also entropy models, and all shared references implement
    /// `Copy`), but passing a *view* instead may be slightly more efficient because it
    /// avoids one level of dereferencing.
    ///
    /// [`Decode::decode_iid_symbols`]: super::Decode::decode_iid_symbols
    pub fn as_view(
        &self,
    ) -> LookupDecoderModel<
        Probability,
        Probability,
        ContiguousSymbolTable<&[Probability]>,
        &[Probability],
        PRECISION,
    > {
        LookupDecoderModel {
            lookup_table: self.lookup_table.as_ref(),
            cdf: ContiguousSymbolTable(self.cdf.0.as_ref()),
            phantom: PhantomData,
        }
    }

    /// TODO: documentation
    pub fn as_contiguous_categorical(
        &self,
    ) -> ContiguousCategoricalEntropyModel<Probability, &[Probability], PRECISION> {
        ContiguousCategoricalEntropyModel {
            cdf: ContiguousSymbolTable(self.cdf.0.as_ref()),
            phantom: PhantomData,
        }
    }

    /// TODO: documentation
    pub fn into_contiguous_categorical(
        self,
    ) -> ContiguousCategoricalEntropyModel<Probability, Table, PRECISION> {
        ContiguousCategoricalEntropyModel {
            cdf: self.cdf,
            phantom: PhantomData,
        }
    }
}

impl<Symbol, Probability, Table, LookupTable, const PRECISION: usize>
    LookupDecoderModel<Symbol, Probability, NonContiguousSymbolTable<Table>, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Table: AsRef<[(Probability, Symbol)]>,
    LookupTable: AsRef<[Probability]>,
{
    /// Makes a very cheap shallow copy of the model that can be used much like a shared
    /// reference.
    ///
    /// The returned `LookupDecoderModel` implements `Copy`, which is a requirement for some
    /// methods, such as [`Decode::decode_iid_symbols`]. These methods could also accept a
    /// shared reference to a `NonContiguousCategoricalDecoderModel` (since all references
    /// to entropy models are also entropy models, and all shared references implement
    /// `Copy`), but passing a *view* instead may be slightly more efficient because it
    /// avoids one level of dereferencing.
    ///
    /// [`Decode::decode_iid_symbols`]: super::Decode::decode_iid_symbols
    pub fn as_view(
        &self,
    ) -> LookupDecoderModel<
        Symbol,
        Probability,
        NonContiguousSymbolTable<&[(Probability, Symbol)]>,
        &[Probability],
        PRECISION,
    > {
        LookupDecoderModel {
            lookup_table: self.lookup_table.as_ref(),
            cdf: NonContiguousSymbolTable(self.cdf.0.as_ref()),
            phantom: PhantomData,
        }
    }
}

impl<Symbol, Probability, Table, LookupTable, const PRECISION: usize> EntropyModel<PRECISION>
    for LookupDecoderModel<Symbol, Probability, Table, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
{
    type Symbol = Symbol;
    type Probability = Probability;
}

impl<Symbol, Probability, Table, LookupTable, const PRECISION: usize> DecoderModel<PRECISION>
    for LookupDecoderModel<Symbol, Probability, Table, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
    Table: SymbolTable<Symbol, Probability>,
    LookupTable: AsRef<[Probability]>,
    Symbol: Clone,
{
    #[inline(always)]
    fn quantile_function(
        &self,
        quantile: Probability,
    ) -> (Symbol, Probability, Probability::NonZero) {
        if Probability::BITS != PRECISION {
            // It would be nice if we could avoid this but we currently don't statically enforce
            // `quantile` to fit into `PRECISION` bits.
            assert!(PRECISION == Probability::BITS || quantile < Probability::one() << PRECISION);
        }

        let (left_sided_cumulative, symbol, next_cumulative) = unsafe {
            // SAFETY:
            // - `quantile_to_index` has length `1 << PRECISION` and we verified that
            //   `quantile` fits into `PRECISION` bits above.
            // - `left_sided_cumulative_and_symbol` has length
            //   `*quantile_to_index.as_ref().iter().max() as usize + 2`, so we can always
            //   access it at `index + 1` for `index` coming from `quantile_to_index`.
            let index = *self.lookup_table.as_ref().get_unchecked(quantile.into());
            let index = index.into();

            (
                self.cdf.left_cumulative_unchecked(index),
                self.cdf.symbol_unchecked(index),
                self.cdf.left_cumulative_unchecked(index + 1),
            )
        };

        let probability = unsafe {
            // SAFETY: The constructors ensure that `cdf` is strictly increasing (in
            // wrapping arithmetic) except at indices that can't be reached from
            // `quantile_to_index`).
            next_cumulative
                .wrapping_sub(&left_sided_cumulative)
                .into_nonzero_unchecked()
        };

        (symbol, left_sided_cumulative, probability)
    }
}

impl<'m, Symbol, Probability, M, const PRECISION: usize> From<&'m M>
    for LookupDecoderModel<
        Symbol,
        Probability,
        NonContiguousSymbolTable<Vec<(Probability, Symbol)>>,
        Box<[Probability]>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    Symbol: Copy + Default,
    usize: AsPrimitive<Probability>,
    M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
{
    #[inline(always)]
    fn from(model: &'m M) -> Self {
        Self::from_iterable_entropy_model(model)
    }
}

impl<'m, Probability, Table, const PRECISION: usize>
    From<&'m ContiguousCategoricalEntropyModel<Probability, Table, PRECISION>>
    for LookupDecoderModel<
        Probability,
        Probability,
        ContiguousSymbolTable<Vec<Probability>>,
        Box<[Probability]>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Table: AsRef<[Probability]>,
{
    fn from(model: &'m ContiguousCategoricalEntropyModel<Probability, Table, PRECISION>) -> Self {
        let cdf = model.cdf.0.as_ref().to_vec();
        let mut lookup_table = Vec::with_capacity(1 << PRECISION);
        for (symbol, &cumulative) in model.cdf.0.as_ref()[1..model.cdf.0.as_ref().len() - 1]
            .iter()
            .enumerate()
        {
            lookup_table.resize(cumulative.into(), symbol.as_());
        }
        lookup_table.resize(1 << PRECISION, (model.cdf.0.as_ref().len() - 2).as_());

        Self {
            lookup_table: lookup_table.into_boxed_slice(),
            cdf: ContiguousSymbolTable(cdf),
            phantom: PhantomData,
        }
    }
}

impl<'m, Probability, Table, LookupTable, const PRECISION: usize>
    IterableEntropyModel<'m, PRECISION>
    for LookupDecoderModel<
        Probability,
        Probability,
        ContiguousSymbolTable<Table>,
        LookupTable,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Table: AsRef<[Probability]>,
    LookupTable: AsRef<[Probability]>,
{
    type Iter = SymbolTableIter<Probability, Probability, ContiguousSymbolTable<&'m [Probability]>>;

    #[inline(always)]
    fn symbol_table(&'m self) -> Self::Iter {
        SymbolTableIter::new(self.as_view().cdf)
    }
}

impl<'m, Symbol, Probability, Table, LookupTable, const PRECISION: usize>
    IterableEntropyModel<'m, PRECISION>
    for LookupDecoderModel<
        Symbol,
        Probability,
        NonContiguousSymbolTable<Table>,
        LookupTable,
        PRECISION,
    >
where
    Symbol: Clone + 'm,
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Table: AsRef<[(Probability, Symbol)]>,
    LookupTable: AsRef<[Probability]>,
{
    type Iter =
        SymbolTableIter<Symbol, Probability, NonContiguousSymbolTable<&'m [(Probability, Symbol)]>>;

    #[inline(always)]
    fn symbol_table(&'m self) -> Self::Iter {
        SymbolTableIter::new(self.as_view().cdf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::{stack::DefaultAnsCoder, Decode};

    use alloc::{string::String, vec};
    use probability::distribution::{Binomial, Gaussian};

    #[test]
    fn leakily_quantized_normal() {
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-127..=127);
        for &std_dev in &[0.0001, 0.1, 3.5, 123.45, 1234.56] {
            for &mean in &[-300.6, -100.2, -5.2, 0.0, 50.3, 180.2, 2000.0] {
                let distribution = Gaussian::new(mean, std_dev);
                test_entropy_model(&quantizer.quantize(distribution), -127..128);
            }
        }
    }

    #[test]
    fn leakily_quantized_binomial() {
        for &n in &[1, 2, 10, 100, 1000, 10_000] {
            for &p in &[1e-30, 1e-20, 1e-10, 0.1, 0.4, 0.9] {
                if n < 1000 || p >= 0.1 {
                    // In the excluded situations, `<Binomial as Inverse>::inverse` currently doesn't terminate.
                    // TODO: file issue to `probability` repo.
                    let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=n as u32);
                    let distribution = Binomial::new(n, p);
                    test_entropy_model(&quantizer.quantize(distribution), 0..(n as u32 + 1));
                }
            }
        }
    }

    #[test]
    fn entropy() {
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-1000..=1000);
        for &std_dev in &[100., 200., 300.] {
            for &mean in &[-10., 2.3, 50.1] {
                let distribution = Gaussian::new(mean, std_dev);
                let model = quantizer.quantize(distribution);
                let entropy = model.entropy_base2::<f64>();
                let expected_entropy = 2.047095585180641 + std_dev.log2();
                assert!((entropy - expected_entropy).abs() < 0.01);
            }
        }
    }

    /// Test that `optimal_weights` reproduces the same distribution when fed with an
    /// already quantized model.
    #[test]
    fn trivial_optimal_weights() {
        let hist = [
            56319u32, 134860032, 47755520, 60775168, 75699200, 92529920, 111023616, 130420736,
            150257408, 169970176, 188869632, 424260864, 229548800, 236082432, 238252287, 234666240,
            1, 1, 227725568, 216746240, 202127104, 185095936, 166533632, 146508800, 126643712,
            107187968, 88985600, 72576000, 57896448, 45617664, 34893056, 26408448, 19666688,
            14218240, 10050048, 7164928, 13892864,
        ];
        assert_eq!(hist.iter().map(|&x| x as u64).sum::<u64>(), 1 << 32);

        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();
        let categorical =
            ContiguousCategoricalEntropyModel::<u32, _, 32>::from_floating_point_probabilities(
                &probabilities,
            )
            .unwrap();
        let weights: Vec<_> = categorical
            .symbol_table()
            .map(|(_, _, probability)| probability.get())
            .collect();

        assert_eq!(&weights[..], &hist[..]);
    }

    #[test]
    fn nontrivial_optimal_weights() {
        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        assert_ne!(hist.iter().map(|&x| x as u64).sum::<u64>(), 1 << 32);

        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();
        let categorical =
            ContiguousCategoricalEntropyModel::<u32, _, 32>::from_floating_point_probabilities(
                &probabilities,
            )
            .unwrap();
        let weights: Vec<_> = categorical
            .symbol_table()
            .map(|(_, _, probability)| probability.get())
            .collect();

        assert_eq!(weights.len(), hist.len());
        assert_eq!(weights.iter().map(|&x| x as u64).sum::<u64>(), 1 << 32);
        for &w in &weights {
            assert!(w > 0);
        }

        let mut weights_and_hist = weights
            .iter()
            .cloned()
            .zip(hist.iter().cloned())
            .collect::<Vec<_>>();

        // Check that sorting by weight is compatible with sorting by hist.
        weights_and_hist.sort_unstable();
        // TODO: replace the following with
        // `assert!(weights_and_hist.iter().map(|&(_, x)| x).is_sorted())`
        // when `is_sorted` becomes stable.
        let mut previous = 0;
        for (_, hist) in weights_and_hist {
            assert!(hist >= previous);
            previous = hist;
        }
    }

    #[test]
    fn contiguous_categorical() {
        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();

        let model =
            ContiguousCategoricalEntropyModel::<_, _, 32>::from_floating_point_probabilities(
                &probabilities,
            )
            .unwrap();
        test_entropy_model(&model, 0..probabilities.len());
    }

    #[test]
    fn non_contiguous_categorical() {
        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();
        let symbols = "QWERTYUIOPASDFGHJKLZXCVBNM 1234567890"
            .chars()
            .collect::<Vec<_>>();

        let model =
            NonContiguousCategoricalDecoderModel::<_,_, _, 32>::from_symbols_and_floating_point_probabilities(
                &symbols,
                &probabilities,
            )
            .unwrap();
        test_iterable_entropy_model(&model, symbols.iter().cloned());
    }

    fn test_entropy_model<'m, D, const PRECISION: usize>(
        model: &'m D,
        support: impl Clone + Iterator<Item = D::Symbol>,
    ) where
        D: IterableEntropyModel<'m, PRECISION, Probability = u32>
            + EncoderModel<PRECISION>
            + DecoderModel<PRECISION>
            + 'm,
        D::Symbol: Copy + core::fmt::Debug + PartialEq,
    {
        let mut sum = 0;
        for symbol in support.clone() {
            let (left_cumulative, prob) = model.left_cumulative_and_probability(symbol).unwrap();
            assert_eq!(left_cumulative as u64, sum);
            sum += prob.get() as u64;

            let expected = (symbol, left_cumulative, prob);
            assert_eq!(model.quantile_function(left_cumulative), expected);
            assert_eq!(model.quantile_function((sum - 1) as u32), expected);
            assert_eq!(
                model.quantile_function(left_cumulative + prob.get() / 2),
                expected
            );
        }
        assert_eq!(sum, 1 << PRECISION);

        test_iterable_entropy_model(model, support);
    }

    fn test_iterable_entropy_model<'m, D, const PRECISION: usize>(
        model: &'m D,
        support: impl Clone + Iterator<Item = D::Symbol>,
    ) where
        D: IterableEntropyModel<'m, PRECISION, Probability = u32> + 'm,
        D::Symbol: Copy + core::fmt::Debug + PartialEq,
    {
        let mut expected_cumulative = 0u64;
        let mut count = 0;
        for (expected_symbol, (symbol, left_sided_cumulative, probability)) in
            support.clone().zip(model.symbol_table())
        {
            assert_eq!(symbol, expected_symbol);
            assert_eq!(left_sided_cumulative as u64, expected_cumulative);
            expected_cumulative += probability.get() as u64;
            count += 1;
        }
        assert_eq!(count, support.size_hint().0);
        assert_eq!(expected_cumulative, 1 << PRECISION);
    }

    #[test]
    fn lookup_contiguous() {
        let probabilities = vec![3u8, 18, 1, 42];
        let model =
            ContiguousCategoricalEntropyModel::<_, _, 6>::from_nonzero_fixed_point_probabilities(
                probabilities,
                false,
            )
            .unwrap();
        let lookup_decoder_model = LookupDecoderModel::from_iterable_entropy_model(&model);

        // Verify that `decode(encode(x)) == x` and that `lookup_decode(encode(x)) == x`.
        for symbol in 0..4 {
            let (left_cumulative, probability) =
                model.left_cumulative_and_probability(symbol).unwrap();
            for quantile in left_cumulative..left_cumulative + probability.get() {
                assert_eq!(
                    model.quantile_function(quantile),
                    (symbol, left_cumulative, probability)
                );
                assert_eq!(
                    lookup_decoder_model.quantile_function(quantile),
                    (symbol, left_cumulative, probability)
                );
            }
        }

        // Verify that `encode(decode(x)) == x` and that `encode(lookup_decode(x)) == x`.
        for quantile in 0..1 << 6 {
            let (symbol, left_cumulative, probability) = model.quantile_function(quantile);
            assert_eq!(
                lookup_decoder_model.quantile_function(quantile),
                (symbol, left_cumulative, probability)
            );
            assert_eq!(
                model.left_cumulative_and_probability(symbol).unwrap(),
                (left_cumulative, probability)
            );
        }

        // Test encoding and decoding a few symbols.
        let symbols = vec![0, 3, 2, 3, 1, 3, 2, 0, 3];
        let mut ans = DefaultAnsCoder::new();
        ans.encode_iid_symbols_reverse(&symbols, &model).unwrap();
        assert!(!ans.is_empty());

        let mut ans2 = ans.clone();
        let decoded = ans
            .decode_iid_symbols(9, &model)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(decoded, symbols);
        assert!(ans.is_empty());

        let decoded = ans2
            .decode_iid_symbols(9, &lookup_decoder_model)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(decoded, symbols);
        assert!(ans2.is_empty());
    }

    #[test]
    fn lookup_noncontiguous() {
        let symbols = "axcy";
        let probabilities = vec![3u8, 18, 1, 42];
        let encoder_model = NonContiguousCategoricalEncoderModel::<_, u8, 6>::from_symbols_and_nonzero_fixed_point_probabilities(
            symbols.chars(),probabilities.iter(),false
        )
        .unwrap();
        let decoder_model = NonContiguousCategoricalDecoderModel::<_, _,_, 6>::from_symbols_and_nonzero_fixed_point_probabilities(
            symbols.chars(),probabilities.iter(),false
        )
        .unwrap();
        let lookup_decoder_model = LookupDecoderModel::from_iterable_entropy_model(&decoder_model);

        // Verify that `decode(encode(x)) == x` and that `lookup_decode(encode(x)) == x`.
        for symbol in symbols.chars() {
            let (left_cumulative, probability) = encoder_model
                .left_cumulative_and_probability(symbol)
                .unwrap();
            for quantile in left_cumulative..left_cumulative + probability.get() {
                assert_eq!(
                    decoder_model.quantile_function(quantile),
                    (symbol, left_cumulative, probability)
                );
                assert_eq!(
                    lookup_decoder_model.quantile_function(quantile),
                    (symbol, left_cumulative, probability)
                );
            }
        }

        // Verify that `encode(decode(x)) == x` and that `encode(lookup_decode(x)) == x`.
        for quantile in 0..1 << 6 {
            let (symbol, left_cumulative, probability) = decoder_model.quantile_function(quantile);
            assert_eq!(
                lookup_decoder_model.quantile_function(quantile),
                (symbol, left_cumulative, probability)
            );
            assert_eq!(
                encoder_model
                    .left_cumulative_and_probability(symbol)
                    .unwrap(),
                (left_cumulative, probability)
            );
        }

        // Test encoding and decoding a few symbols.
        let symbols = "axcxcyaac";
        let mut ans = DefaultAnsCoder::new();
        ans.encode_iid_symbols_reverse(symbols.chars(), &encoder_model)
            .unwrap();
        assert!(!ans.is_empty());
        let decoded = ans
            .decode_iid_symbols(9, &decoder_model)
            .collect::<Result<String, _>>()
            .unwrap();
        assert_eq!(decoded, symbols);
        assert!(ans.is_empty());
    }
}
