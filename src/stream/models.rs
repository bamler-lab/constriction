//! Probability distributions that can be used as entropy models for stream codes.
//!
//! This module provides utilities for dealing with probabilistic models of data sources
//! ("entropy models") in exactly invertible fixed-point arithmetic so as to avoid rounding
//! errors. As explained in the [motivation](#motivation) below, avoiding rounding errors is
//! necessary for reliable entropy coding.
//!
//! The types defined in this module approximate arbitrary discrete (or quantized
//! one-dimensional continuous) probability distributions with a fixed-point representation.
//! The fixed-point representation has customizable precision and can be either explicit or
//! implicit (i.e., lazy). While the approximation itself generally involves rounding, no
//! more rounding occurs when the resulting fixed-point representation is used to invert the
//! (cumulative distribution function of the) approximated probability distribution.
//!
//! # Module Overview
//!
//! This module declares the base trait [`EntropyModel`] and its subtraits [`EncoderModel`]
//! and [`DecoderModel`], which define the interfaces that entropy models provide and that
//! entropy coders can rely on.
//!
//! In addition, this module provides the following three utilities for constructing entropy
//! models:
//! - a generic adapter that turns parameterized discrete or one-dimensional continuous
//!   probability distributions to the fixed-point representation that the entropy coders in
//!   the sister modules expect; see [`LeakyQuantizer`].
//! - a type for representing arbitrary categorical distributions in fixed-point
//!   representation, intended as a fallback for probability distributions for which the
//!   above adapter can't be used because there's no closed-form analytic expression for
//!   them; see [`LeakyCategorical`]; and
//! - specialized implementations of high-performance "lookup tables", i.e, entropy models
//!   where the entire distribution is tabularized so that repeated evaluations of the model
//!   (for i.i.d. symbols) are fast; see submodule [`lookup`].
//!
//! # Examples
//!
//! See sister modules [`stack`] and [`queue`] for usage examples of these entropy models in
//! entropy coders.
//!
//! # Motivation
//!
//! The general idea of entropy coding is to use a probabilistic model of a data source to
//! find an optimal compression strategy for the type of data that one intends to compress.
//! Ideally, all conceivable data points would be compressed into a short bit string.
//! However, short bit strings are a scarce resource: for any integer `N`, there are only
//! `2^N - 1` distinct bit strings that are shorter than `N` bits. For this reason, entropy
//! coding takes into account that, for a typical data source (e.g., of natural images,
//! videos, audio, ...) many data points may be *possible* but are extremely *improbable* to
//! occur in practice. An entropy coder assigns longer bit strings to such improbable data
//! points so that it can use the scarce short bit strings for more probable data points.
//! More precisely, entropy coding aims to minimize the *expected* bit rate under the
//! probabilistic model of the data source.
//!
//! We refer to a probabilistic model of a data source in the context of entropy coding as
//! an "entropy model". In contrast to many other use cases of probabilistic models in
//! computing, entropy models must be amenable to *exact* arithmetic operations. In
//! particular, no rounding errors are allowed when inverting the cumulative distribution
//! function. Even arbitrarily small rounding errors could set off a chain reaction of
//! arbitrarily large and arbitrarily many errors when compressing and then decompressing a
//! sequence of symbols (see, e.g., the [motivating example for the
//! `ChainCoder`](super::chain#motivation)).
//!
//!
//! # Leakiness
//!
//! Several types in this module carry the term `Leaky` in their name. We call an entropy
//! model a "leaky" representation of some probability distribution if all valid symbols
//! within a user-defined domain (e.g., all integers within a given range) are guaranteed to
//! have a nonzero probability under the entropy model. This is often both a useful and a
//! nontrivial property of an entropy model. It is a useful property since a nonzero
//! probability means that all symbols from the domain *can* be encoded at a finite (albeit
//! potentially high) bit rate. It is a nontrivial property since entropy models typically
//! result from converting some floating-point representation of a probability distribution
//! to a fixed point representation, which involves rounding that can turn low but nonzero
//! probabilities into zero probabilities when done naively. If you use a "leaky" entropy
//! model then you don't have to worry about such cases.
//!
//! [`stack`]: super::stack
//! [`queue`]: super::queue

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
use probability::distribution::{Distribution, Inverse};

use crate::{wrapping_pow2, BitArray, NonZeroBitArray};

/// Base trait for probabilistic models of a data source.
///
/// All entropy models (see [module level documentation](self)) that can be used for
/// encoding and/or decoding with stream codes implement this trait, and at least one of
/// [`EncoderModel`] and/or [`DecoderModel`]. This trait exposes the type of [`Symbol`]s
/// over which the entropy model is defined, the type that is used to represent a
/// [`Probability`] in fixed-point arithmetic, and the fixed point `PRECISION`.
///
/// # Flaoting Point Precisoin
///
/// The const generic `PRECISION` specifies the number of bits that are used for
/// representing probabilities. It most not be zero and it must not be higher than
/// [`Probability`]::BITS. See documentation of the associated type [`Probability`] for a
/// discussion of further constraints.
///
/// # Blanket Implementation for `&impl EntropyModel`
///
/// We provide the following blanket implementation for references to `EntropyModel`s:
///
/// ```ignore
/// impl<M: EntropyModel<PRECISION>, const PRECISION: usize> EntropyModel<PRECISION> for &M { ... }
/// ```
///
/// This means that, if some type `M` implements `EntropyModel<PRECISION>` for some
/// `PRECISION`, then so does the reference type `&M`. Analogous blanket implementations are
/// provided for the traits [`EncoderModel`] and [`DecoderModel`]. The implementations
/// simply delegate all calls to `M` (which is possible since all methods only take an
/// `&self` receiver). Therefore:
/// - you don't need to implement `EntropyModel`, `EncoderModel`, or `DecoderModel` on
///   reference types `&M`; just implement these traits on "value types" `M` and you'll get
///   the implementation on the corresponding reference types for free.
/// - when you write a function or method that takes a generic entropy model as an argument,
///   always take the entropy model (formally) *by value* (i.e., declare your function as
///   `fn f<const PRECISION: usize>(model: impl EntropyModel<PRECISION>)`). Since all
///   references to `EntropyModel`s are also `EntropyModel`s themselves, a generic method
///   with this signature can be called with an entropy model passed in either by value or
///   by reference.
///
/// [`Symbol`]: Self::Symbol
/// [`Probability`]: Self::Probability
pub trait EntropyModel<const PRECISION: usize> {
    /// The type of data over which the entropy model is defined.
    ///
    /// This is the type of an item of the *uncompressed* data. Note that, when you encode a
    /// sequence of symbols, you may use a different entropy model with a different `Symbol`
    /// type for each symbol in the sequence.
    type Symbol;

    /// The type used to represent probabilities. Must hold at least PRECISION bits.
    ///
    /// We represent a probability `p ∈ [0, 1]` as the number `p * (1 << PRECISION)`, which
    /// must be a (nonegative) integer. The special case of `p = 1` comes up as, e.g., the
    /// right-cumulative of the last allowed symbol. It is represented as `0` in the
    /// (uncommon) setup where `PRECISION == Probability::BITS` (implementations of entropy
    /// models have to ensure that probability zero and probability one can never be
    /// confused). In the more common setups where `PRECISION < Probability::BITS`, the case
    /// `p = 1` is represented as `1 << PRECISION`, i.e., as the only allowed value of a
    /// `Probability` that has more then `PRECISION` valid bits.
    ///
    /// Neither the constraint that `1 <= PRECISION <= Probability::BITS` nor the above
    /// constraints on the valid bits of a `Probability` are currently enforced statically
    /// since Rust does not yet allow const expressions in type bounds. The constraints are,
    /// however, enforced at runtime whenever these runtime-checks are either guranteeed to
    /// get optimized out or in few unavoidable places where these checks are necessary to
    /// guarantee memory safety. Once Rust allows const expressions in type bounds, most of
    /// these constraints will be turned into statically checked bounds (which means that
    /// the API will technically become more restrictive, but it will only forbid usages
    /// that would panic at runtime today).
    type Probability: BitArray;
}

pub trait IterableEntropyModel<'m, const PRECISION: usize>: EntropyModel<PRECISION> {
    type Iter: Iterator<
        Item = (
            Self::Symbol,
            Self::Probability,
            <Self::Probability as BitArray>::NonZero,
        ),
    >;

    /// Iterates over symbols, the (left sided) cumulative distribution and the probability
    /// mass function, in fixed point arithmetic.
    ///
    /// This method may be used, e.g., to export
    /// the model into a serializable format, or to construct a different but equivalent representation of
    /// the same entropy model (e.g., to construct a [`LookupDecoderModel`] from some
    /// `EncoderModel`).
    ///
    /// # Example
    ///
    /// TODO: update the example
    ///
    /// ```
    /// use constriction::stream::models::LeakyCategorical;
    ///
    /// let probabilities = vec![0.125, 0.5, 0.25, 0.125]; // Can all be represented without rounding.
    /// let model = LeakyCategorical::<u32, 32>::from_floating_point_probabilities(&probabilities).unwrap();
    ///
    /// let pmf = model.fixed_point_probabilities().collect::<Vec<_>>();
    /// assert_eq!(pmf, vec![1 << 29, 1 << 31, 1 << 30, 1 << 29]);
    /// ```
    ///
    /// TODO: this a convenience wrapper that just calls `.into_iter()`. Its main purpose is
    /// to establish a convention that entropy models should implement `IntoIterator` when
    /// possible. It is used, e.g., for creating lookup table decoder models from arbitrary
    /// entropy models.
    ///
    /// The iterator must iterate in order of increasing cumulative.
    ///
    /// This is not implemented as a normal method because it may not be feasible to
    /// implement this method for all entroy models in an efficient way.
    fn symbol_table(&'m self) -> Self::Iter;

    /// Returns the entropy in units of bits (i.e., base 2).
    ///
    /// TODO: implement `entropy` as inherent method for
    /// `NonContiguousCategoricalEncoderwhich does not implement `IntoIterator` because it
    /// cannot guarantee a fixed order of the iteration.
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
                probability * probability.log2() // prob is guaranteed to be nonzero.
            })
            .sum::<F>();

        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).into();
        F::from(PRECISION).unwrap() - entropy_scaled / whole
    }

    #[inline(always)]
    fn to_generic_encoder_model(
        &'m self,
    ) -> NonContiguousCategoricalEncoderModel<Self::Symbol, Self::Probability, PRECISION>
    where
        Self::Symbol: Hash + Eq,
    {
        self.into()
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
        Self::Symbol: Clone + Default,
    {
        self.into()
    }
}

pub trait EncoderModel<const PRECISION: usize>: EntropyModel<PRECISION> {
    /// Returns `Ok((left_sided_cumulative, probability))` of the bin for the
    /// provided `symbol` if the symbol has nonzero probability.
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Self::Symbol>,
    ) -> Option<(Self::Probability, <Self::Probability as BitArray>::NonZero)>;

    /// TODO: update docs
    ///
    /// Returns the underlying probability mass function in floating point arithmetic.
    ///
    /// This method is similar to [`floating_point_probabilities_lossy`] except that it
    /// guarantees that the conversion from the internally used fixed point arithmetic
    /// to the requested floating point type `F` is lossless.
    ///
    /// This method is similar to [`fixed_point_probabilities`] except that it converts
    /// the probabilities to the desired floating point type `F`. If you use builtin
    /// integer and floating point types for `Probability` and `F`, respectively then the
    /// conversion is guaranteed to be lossless (because of the trait bound
    /// `Probability: Into<F>`). In this case, the yielded probabilities sum up *exactly* to one,
    /// and passing them to [`from_floating_point_probabilities`] will reconstruct the
    /// exact same model (although using [`fixed_point_probabilities`] and
    /// [`from_nonzero_fixed_point_probabilities`] would be cheaper for this purpose).
    ///
    /// The trait bound `Probability: Into<F>` is a conservative bound. In reality, whether or not
    /// the conversion can be guaranteed to be lossless depends on the const generic
    /// parameter `PRECISION`. However, there is currently no way to express the more
    /// precise bound based on `PRECISION` as a compile time check, so the method
    /// currently conservatively assumes `PRECISION` has the highest value allowed for
    /// type `Probability`.
    ///
    /// Note that the returned floating point probabilities will likely be slightly
    /// different than the ones you may have used to construct the
    /// `CategoricalDistribution` in [`from_floating_point_probabilities`]. This is because
    /// probabilities are internally represented in fixed-point arithmetic with `PRECISION`
    /// bits, and because of the constraint that each bin has a strictly nonzero
    /// probability.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::models::LeakyCategorical;
    ///
    /// let probabilities = vec![1u32 << 29, 1 << 31, 1 << 30, 1 << 29];
    /// let model = LeakyCategorical::<u32, 32>::from_nonzero_fixed_point_probabilities(&probabilities);
    ///
    /// let pmf = model.floating_point_probabilities().collect::<Vec<f64>>();
    /// assert_eq!(pmf, vec![0.125, 0.5, 0.25, 0.125]);
    /// ```
    ///
    /// [`fixed_point_probabilities`]: #method.fixed_point_probabilities
    /// [`floating_point_probabilities_lossy`]: #method.floating_point_probabilities_lossy
    /// [`from_floating_point_probabilities`]: #method.from_floating_point_probabilities
    /// [`from_nonzero_fixed_point_probabilities`]: #method.from_nonzero_fixed_point_probabilities
    #[inline]
    fn floating_point_probability<F>(&self, symbol: impl Borrow<Self::Symbol>) -> F
    where
        F: Float,
        Self::Probability: Into<F>,
    {
        // This will be compiled into a single floating point multiplication rather than a (slow)
        // division (it should actually be possible to avoid even that and instead just
        // manually compose the floating point number from mantissa and exponent).
        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).into();
        let probability = self
            .left_cumulative_and_probability(symbol)
            .map_or(Self::Probability::zero(), |(_, p)| p.get());
        probability.into() / whole
    }

    /// TODO: update docs
    ///
    /// Returns the underlying probability mass function in floating point arithmetic.
    ///
    /// This method is similar to [`floating_point_probabilities`] except that it does
    /// *not* guarantee that the conversion from the internally used fixed point
    /// arithmetic to the requested floating point type `F` is lossless.
    ///
    /// # Example
    ///
    /// The following call to [`floating_point_probabilities`] does not compile because
    /// the compiler cannot guarantee that the requested conversion from `u32` to `f32`
    /// be lossless (even though, for these particular values, it would be):
    ///
    /// ```compile_fail
    /// use constriction::stream::models::LeakyCategorical;
    /// let probabilities = vec![1u32 << 29, 1 << 31, 1 << 30, 1 << 29];
    /// let model = LeakyCategorical::<u32, 32>::from_nonzero_fixed_point_probabilities(&probabilities);
    ///
    /// let pmf = model.floating_point_probabilities().collect::<Vec<f32>>();
    /// //                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Compiler error: trait bound not satisfied
    /// ```
    ///
    /// This can either be fixed by replacing `f32` with `f64` (thus guaranteeing
    /// lossless conversion) or by replacing `floating_point_probabilities` with
    /// `floating_point_probabilities_lossy` as follows:
    ///
    /// ```
    /// use constriction::stream::models::LeakyCategorical;
    ///
    /// let probabilities = vec![1u32 << 29, 1 << 31, 1 << 30, 1 << 29];
    /// let model = LeakyCategorical::<u32, 32>::from_nonzero_fixed_point_probabilities(&probabilities);
    ///
    /// let pmf = model.floating_point_probabilities_lossy().collect::<Vec<f32>>();
    /// assert_eq!(pmf, vec![0.125, 0.5, 0.25, 0.125]);
    /// ```
    ///
    /// [`floating_point_probabilities`]: #method.floating_point_probabilities
    #[inline]
    fn floating_point_probability_lossy<F>(&self, symbol: impl Borrow<Self::Symbol>) -> F
    where
        F: Float + 'static,
        Self::Probability: AsPrimitive<F>,
    {
        // This will be compiled into a single floating point multiplication rather than a (slow)
        // division (it should actually be possible to avoid even that and instead just
        // manually compose the floating point number from mantissa and exponent).
        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).as_();
        let probability = self
            .left_cumulative_and_probability(symbol)
            .map_or(Self::Probability::zero(), |(_, p)| p.get());
        probability.as_() / whole
    }
}

pub trait DecoderModel<const PRECISION: usize>: EntropyModel<PRECISION> {
    /// Returns `(symbol, left_sided_cumulative, probability)` of the unique bin that
    /// satisfies `left_sided_cumulative <= quantile < left_sided_cumulative + probability`
    /// (where the addition on the right-hand side is non-wrapping).
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

impl<D, const PRECISION: usize> DecoderModel<PRECISION> for &D
where
    D: DecoderModel<PRECISION>,
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

/// Turns continuous distributions into discrete distributions (entropy models).
///
/// This is a builder of [`LeakilyQuantizedDistribution`]s.
///
/// Lossless entropy coding can only be performed over discrete data. Any continuous
/// (real-valued) data has to be approximate by some discrete set of points. This builder
/// allows taking continuous distributions (defined over `F`, which is typically `f64` or
/// `f32`), and approximating them by discrete distributions defined over the integer type
/// `Symbol` (which is typically something like `i32`) by rounding all values to the closest
/// integer. The resulting [`LeakilyQuantizedDistribution`]s can be used for entropy coding
/// with a coder that implements [`Encode`] or [`Decode`] because they implement
/// [`EntropyModel`].
///
/// This quantizer is a "leaky" quantizer. This means that the constructor [`new`] takes a
/// domain over the `Symbol` type as an argument. The resulting
/// [`LeakilyQuantizedDistribution`]s are guaranteed to assign a nonzero probability to
/// every integer within this domain. This is often a useful property of an entropy model
/// because it ensures that every integer within the chosen domain can in fact be encoded.
///
/// # Examples
///
/// Quantizing a Gaussian distribution:
///
/// ```
/// use constriction::stream::{models::LeakyQuantizer, stack::DefaultAnsCoder, Encode};
///
/// // Get a quantizer that supports integer symbols from -5 to 20 (inclusively),
/// // representing probabilities with 24 bit precision backed by `u32`s.
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-5..=20);
///
/// // Quantize a normal distribution with mean 8.3 and standard deviation 4.1.
/// let continuous_distribution1 = probability::distribution::Gaussian::new(8.3, 4.1);
/// let discrete_distribution1 = quantizer.quantize(continuous_distribution1);
///
/// // You can reuse the same quantizer for more than one distribution.
/// let continuous_distribution2 = probability::distribution::Gaussian::new(-1.4, 2.7);
/// let discrete_distribution2 = quantizer.quantize(continuous_distribution2);
///
/// // Use the discrete distributions with a `Code`.
/// let mut ans = DefaultAnsCoder::new();
/// ans.encode_symbol(4, discrete_distribution1);
/// ans.encode_symbol(-3, discrete_distribution2);
/// ```
///
/// You can use a `LeakyQuantizer` also for quantizing discrete probability distributions.
/// In this constext, the word "quantizing" only refers to the fact that *probability space*
/// is being quantized, i.e., that floating point probabilities are approximated with fixed
/// point precision. The quantization will again be "leaky", i.e., each symbol within the
/// specified domain will have a nonzero probability.
///
/// ```
/// use constriction::stream::{models::LeakyQuantizer, stack::DefaultAnsCoder, Encode, Decode};
///
/// let distribution = probability::distribution::Binomial::new(1000, 0.1); // arguments: `n, p`
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=1000); // natural domain is `0..=n`
/// let entropy_model = quantizer.quantize(distribution);
/// let mut ans = DefaultAnsCoder::new();
///
/// // Encode a "typical" symbol from the distribution (i.e., one with non-negligible probability).
/// ans.encode_symbol(107, &entropy_model);
///
/// // The following still works despite a ridiculously low probability of the symbol `1000`.
/// ans.encode_symbol(1000, &entropy_model);
///
/// // Decode symbols (in reverse order, since the `AnsCoder` is a stack) and verify correctness.
/// assert_eq!(ans.decode_symbol(&entropy_model), Ok(1000));
/// assert_eq!(ans.decode_symbol(&entropy_model), Ok(107));
/// assert!(ans.is_empty());
/// ```
///
/// # TODO
///
/// Implement non-leaky variant (implement a private type does both leaky and non-leaky
/// quantization depending on a const generic argument; then implement two public rapper
/// types around that; the field `free_weight` would probably only be needed for leaky
/// quantization, so it can sit outside the wrapper type and then be passed in to methods.)
///
/// [`Encode`]: crate::Encode [`Decode`]: crate::Decode [`new`]: #method.new
#[derive(Debug)]
pub struct LeakyQuantizer<F, Symbol, Probability, const PRECISION: usize> {
    min_symbol_inclusive: Symbol,
    max_symbol_inclusive: Symbol,
    free_weight: F,
    phantom: PhantomData<Probability>,
}

pub type DefaultLeakyQuantizer<F, Symbol> = LeakyQuantizer<F, Symbol, u32, 24>;
pub type SmallLeakyQuantizer<F, Symbol> = LeakyQuantizer<F, Symbol, u16, 12>;

impl<F, Symbol, Probability, const PRECISION: usize>
    LeakyQuantizer<F, Symbol, Probability, PRECISION>
where
    Probability: BitArray + Into<F>,
    Symbol: PrimInt + AsPrimitive<Probability> + WrappingSub + WrappingAdd,
    F: Float,
{
    /// Constructs a "leaky" quantizer defined on a finite domain.
    ///
    /// The `domain` is an inclusive range (which can be expressed with the `..=`
    /// notation such as `-10..=10`). All [`LeakilyQuantizedDistribution`]s generated
    /// with [`quantize`] are then guaranteed to assign a nonzero probability to all
    /// symbols within the `domain`. This is often a useful property for entropy coding
    /// because it ensures that all symbols within the `domain` can indeed be encoded.
    ///
    /// This method takes a `RangeInclusive` because we want to be able to support,
    /// e.g., probability distributions over the `Symbol` type `u8` with full
    /// `domain = 0..=255`.
    ///
    /// [`quantize`]: #method.quantize
    pub fn new(domain: RangeInclusive<Symbol>) -> Self {
        assert!(PRECISION > 0 && PRECISION <= Probability::BITS);

        // We don't support degenerate probability distributions (i.e., distributions that
        // place all probability mass on a single symbol).
        assert!(domain.end() > domain.start());

        let domain_size_minus_one = domain.end().wrapping_sub(&domain.start()).as_();
        let max_probability = Probability::max_value() >> (Probability::BITS - PRECISION);
        let free_weight = max_probability
            .checked_sub(&domain_size_minus_one)
            .expect("The domain is too large to assign a nonzero probability to each element.")
            .into();

        LeakyQuantizer {
            min_symbol_inclusive: *domain.start(),
            max_symbol_inclusive: *domain.end(),
            free_weight,
            phantom: PhantomData,
        }
    }
}

impl<F, Symbol, Probability, const PRECISION: usize>
    LeakyQuantizer<F, Symbol, Probability, PRECISION>
where
    Probability: BitArray + Into<F>,
    Symbol: PrimInt + AsPrimitive<Probability> + WrappingSub + WrappingAdd,
    F: Float,
{
    /// Quantizes the given continuous probability distribution.
    ///
    /// Note that this method takes `self` only by reference, i.e., you can reuse
    /// the same `Quantizer` to quantize arbitrarily many distributions. For an
    /// example, see [struct level documentation](struct.LeakyQuantizer.html).
    pub fn quantize<CD>(
        &self,
        distribution: CD,
    ) -> LeakilyQuantizedDistribution<'_, F, Symbol, Probability, CD, PRECISION>
    where
        CD: Inverse,
    {
        LeakilyQuantizedDistribution {
            inner: distribution,
            quantizer: self,
        }
    }
}

/// Wrapper that turns a continuous probability density into a
/// [`EntropyModel`].
///
/// Such a `LeakilyQuantizedDistribution` can be created with a [`LeakyQuantizer`].
/// It can be used for entropy coding since it implements [`EntropyModel`].
#[derive(Debug)]
pub struct LeakilyQuantizedDistribution<'q, F, Symbol, Probability, CD, const PRECISION: usize> {
    inner: CD,
    quantizer: &'q LeakyQuantizer<F, Symbol, Probability, PRECISION>,
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
    let mask = wrapping_pow2::<Probability>(8 * std::mem::size_of::<Symbol>())
        .wrapping_sub(&Probability::one());
    symbol.borrow().wrapping_sub(&min_symbol_inclusive).as_() & mask
}

impl<'q, F, Symbol, Probability, CD, const PRECISION: usize> EntropyModel<PRECISION>
    for LeakilyQuantizedDistribution<'q, F, Symbol, Probability, CD, PRECISION>
where
    Probability: BitArray,
{
    type Probability = Probability;
    type Symbol = Symbol;
}

impl<'q, Symbol, Probability, CD, const PRECISION: usize> EncoderModel<PRECISION>
    for LeakilyQuantizedDistribution<'q, f64, Symbol, Probability, CD, PRECISION>
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + Into<f64> + WrappingSub,
    Probability: BitArray + Into<f64>,
    CD: Distribution,
    CD::Value: AsPrimitive<Symbol>,
{
    /// Performs (one direction of) the quantization.
    ///
    /// # Panics
    ///
    /// If the underlying probability distribution is invalid, i.e., if the quantization
    /// leads to a zero probability despite the added leakiness (and despite the fact that
    /// the constructor checks that `min_symbol_inclusive < max_symbol_inclusive`, i.e.,
    /// that there are at least two symbols with nonzero probability and therefore the
    /// probability of a single symbol cannot overflow).
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

impl<'q, Symbol, Probability, CD, const PRECISION: usize> DecoderModel<PRECISION>
    for LeakilyQuantizedDistribution<'q, f64, Symbol, Probability, CD, PRECISION>
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + Into<f64> + WrappingSub + WrappingAdd,
    Probability: BitArray + Into<f64>,
    CD: Inverse,
    CD::Value: AsPrimitive<Symbol>,
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
        let right_sided_cumulative = if left_sided_cumulative > quantile {
            // Our initial guess for `symbol` was too high. Reduce it until we're good.
            let mut step = Self::Symbol::one(); // `diff` will always be a power of 2.
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
            let mut step = Self::Symbol::one(); // `diff` will always be a power of 2.
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

impl<'m, 'q: 'm, Symbol, Probability, CD, const PRECISION: usize>
    IterableEntropyModel<'m, PRECISION>
    for LeakilyQuantizedDistribution<'q, f64, Symbol, Probability, CD, PRECISION>
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + AsPrimitive<usize> + Into<f64> + WrappingSub,
    Probability: BitArray + Into<f64>,
    CD: Distribution + 'm,
    CD::Value: AsPrimitive<Symbol>,
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

#[derive(Debug)]
pub struct LeakilyQuantizedDistributionIter<Symbol, Probability, M, const PRECISION: usize> {
    model: M,
    symbol: Option<Symbol>,
    left_sided_cumulative: Probability,
}

impl<'m, 'q, Symbol, Probability, CD, const PRECISION: usize> Iterator
    for LeakilyQuantizedDistributionIter<
        Symbol,
        Probability,
        &'m LeakilyQuantizedDistribution<'q, f64, Symbol, Probability, CD, PRECISION>,
        PRECISION,
    >
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + AsPrimitive<usize> + Into<f64> + WrappingSub,
    Probability: BitArray + Into<f64>,
    CD: Distribution,
    CD::Value: AsPrimitive<Symbol>,
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

pub trait SymbolTable<Symbol, Probability: BitArray> {
    fn left_cumulative(&self, index: usize) -> Option<Probability>;

    fn num_symbols(&self) -> usize;

    unsafe fn left_cumulative_unchecked(&self, index: usize) -> Probability;

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
        let mut right = self.num_symbols(); // One above largest possible index.

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

#[derive(Debug, Clone, Copy)]
pub struct ContiguousSymbolTable<Table>(Table);

#[derive(Debug, Clone, Copy)]
pub struct NonContiguousSymbolTable<Table>(Table);

impl<Probability, Table> SymbolTable<usize, Probability> for ContiguousSymbolTable<Table>
where
    Probability: BitArray,
    Table: AsRef<[Probability]>,
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
    unsafe fn symbol_unchecked(&self, index: usize) -> usize {
        index
    }

    #[inline(always)]
    fn num_symbols(&self) -> usize {
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
    fn num_symbols(&self) -> usize {
        self.0.as_ref().len() - 1
    }
}

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
        if old_index == self.table.num_symbols() {
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
        let len = self.table.num_symbols() - self.index;
        (len, Some(len))
    }
}

/// A categorical distribution over a finite number of bins.
///
/// This distribution implements [`EntropyModel`], which means that it can be
/// used for entropy coding with a coder that implements [`Encode`] or [`Decode`].
///
/// Note: We currently don't provide a non-contiguous variant of this model. If you have
/// non-contiguous models, you can trivially map them to a contiguous range using a
/// [`HashMap`] for encoding or a simple array lookup for decoding. By contrast, we do
/// provide a specialized non-contiguous variant of lookup models because those models are
/// optimized for speed and the specialized non-contiguous variant allows us to remove one
/// indirection.
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

pub type DefaultContiguousCategorical<Table = Vec<u32>> =
    ContiguousCategoricalEntropyModel<u32, Table, 24>;
pub type SmallContiguousCategorical<Table = Vec<u16>> =
    ContiguousCategoricalEntropyModel<u16, Table, 12>;

pub type DefaultNonContiguousCategoricalDecoderModel<Symbol, Table = Vec<(u32, Symbol)>> =
    NonContiguousCategoricalDecoderModel<Symbol, u32, Table, 24>;
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
    /// - each symbol in the domain defined above gets assigned a strictly nonzero
    ///   probability, even if the provided probability for the symbol is zero or
    ///   below the threshold that can be resolved in fixed point arithmetic with
    ///   type `Probability`. We refer to this property as the resulting distribution
    ///   being "leaky". This probability ensures that all symbols within the domain
    ///   can be encoded when this distribution is used as an entropy model.
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
    /// TODO: should also return an error if domain is too large to support leaky
    /// distribution
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
    /// distribution previously exported with [`fixed_point_probabilities`]. The more common
    /// way to construct a `LeakyCategorical` distribution is via
    /// [`from_floating_point_probabilities`].
    ///
    /// The entries of `probabilities` have to be nonzero and (logically) sum up to
    /// `1 << PRECISION`, where `PRECISION` is a const generic parameter on the
    /// `LeakyCategorical` distribution. Further, all probabilities have to be nonzero (which is
    /// statically enforced by the trait bound on the iterator's items).
    ///
    /// # Panics
    ///
    /// If `probabilities` is not properly normalized (including if it is empty).
    ///
    /// # Examples
    ///
    /// The provided probabilities have to sum up to `1 << PRECISION`:
    ///
    /// ```
    /// use constriction::stream::models::LeakyCategorical;
    ///
    /// let probabilities = vec![1u32 << 21, 1 << 22, 1 << 22, 1 << 22, 1 << 21];
    /// // `probabilities` sums up to `1 << PRECISION` as required:
    /// assert_eq!(probabilities.iter().sum::<u32>(), 1 << 24);
    ///
    /// let model = LeakyCategorical::<u32, 24>::from_nonzero_fixed_point_probabilities(&probabilities);
    /// let pmf = model.floating_point_probabilities().collect::<Vec<f64>>();
    /// assert_eq!(pmf, vec![0.125, 0.25, 0.25, 0.25, 0.125]);
    /// ```
    ///
    /// If `PRECISION` is set to the maximum value supported by the `Word` type
    /// `Probability`, then the provided probabilities still have to *logically* sum up to
    /// `1 << PRECISION` (i.e., the summation has to wrap around exactly once):
    ///
    /// ```
    /// use constriction::stream::models::LeakyCategorical;
    ///
    /// let probabilities = vec![1u32 << 29, 1 << 30, 1 << 30, 1 << 30, 1 << 29];
    /// // `probabilities` sums up to `1 << 32` (logically), i.e., it wraps around once.
    /// assert_eq!(probabilities.iter().fold(0u32, |accum, &x| accum.wrapping_add(x)), 0);
    ///
    /// let model = LeakyCategorical::<u32, 32>::from_nonzero_fixed_point_probabilities(&probabilities);
    /// let pmf = model.floating_point_probabilities().collect::<Vec<f64>>();
    /// assert_eq!(pmf, vec![0.125, 0.25, 0.25, 0.25, 0.125]);
    /// ```
    ///
    /// Wrapping around twice panics:
    ///
    /// ```should_panic
    /// use constriction::stream::models::LeakyCategorical;
    /// let probabilities = vec![1u32 << 30, 1 << 31, 1 << 31, 1 << 31, 1 << 30];
    /// // `probabilities` sums up to `1 << 33` (logically), i.e., it would wrap around twice.
    /// let model = LeakyCategorical::<u32, 32>::from_nonzero_fixed_point_probabilities(&probabilities); // PANICS.
    /// ```
    ///
    /// So does providing probabilities that just don't sum up to `1 << FREQUENCY`:
    ///
    /// ```should_panic
    /// use constriction::stream::models::LeakyCategorical;
    /// let probabilities = vec![1u32 << 21, 5 << 8, 1 << 22, 1 << 21];
    /// let model = LeakyCategorical::<u32, 24>::from_nonzero_fixed_point_probabilities(&probabilities); // PANICS.
    /// ```
    ///
    /// [`fixed_point_probabilities`]: #method.fixed_point_probabilities
    /// [`from_floating_point_probabilities`]: #method.from_floating_point_probabilities
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
    Symbol: Clone + Default,
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
    /// - each symbol in the domain defined above gets assigned a strictly nonzero
    ///   probability, even if the provided probability for the symbol is zero or
    ///   below the threshold that can be resolved in fixed point arithmetic with
    ///   type `Probability`. We refer to this property as the resulting distribution
    ///   being "leaky". This probability ensures that all symbols within the domain
    ///   can be encoded when this distribution is used as an entropy model.
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
    /// TODO: should also return an error if domain is too large to support leaky
    /// distribution
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
    /// distribution previously exported with [`fixed_point_probabilities`]. The more common
    /// way to construct a `LeakyCategorical` distribution is via
    /// [`from_floating_point_probabilities`].
    ///
    /// The entries of `probabilities` have to be nonzero and (logically) sum up to
    /// `1 << PRECISION`, where `PRECISION` is a const generic parameter on the
    /// `LeakyCategorical` distribution. Further, all probabilities have to be nonzero (which is
    /// statically enforced by the trait bound on the iterator's items).
    ///
    /// # Panics
    ///
    /// If `probabilities` is not properly normalized (including if it is empty).
    ///
    /// # Examples
    ///
    /// The provided probabilities have to sum up to `1 << PRECISION`:
    ///
    /// ```
    /// use constriction::stream::models::LeakyCategorical;
    ///
    /// let probabilities = vec![1u32 << 21, 1 << 22, 1 << 22, 1 << 22, 1 << 21];
    /// // `probabilities` sums up to `1 << PRECISION` as required:
    /// assert_eq!(probabilities.iter().sum::<u32>(), 1 << 24);
    ///
    /// let model = LeakyCategorical::<u32, 24>::from_nonzero_fixed_point_probabilities(&probabilities);
    /// let pmf = model.floating_point_probabilities().collect::<Vec<f64>>();
    /// assert_eq!(pmf, vec![0.125, 0.25, 0.25, 0.25, 0.125]);
    /// ```
    ///
    /// If `PRECISION` is set to the maximum value supported by the `Word` type
    /// `Probability`, then the provided probabilities still have to *logically* sum up to
    /// `1 << PRECISION` (i.e., the summation has to wrap around exactly once):
    ///
    /// ```
    /// use constriction::stream::models::LeakyCategorical;
    ///
    /// let probabilities = vec![1u32 << 29, 1 << 30, 1 << 30, 1 << 30, 1 << 29];
    /// // `probabilities` sums up to `1 << 32` (logically), i.e., it wraps around once.
    /// assert_eq!(probabilities.iter().fold(0u32, |accum, &x| accum.wrapping_add(x)), 0);
    ///
    /// let model = LeakyCategorical::<u32, 32>::from_nonzero_fixed_point_probabilities(&probabilities);
    /// let pmf = model.floating_point_probabilities().collect::<Vec<f64>>();
    /// assert_eq!(pmf, vec![0.125, 0.25, 0.25, 0.25, 0.125]);
    /// ```
    ///
    /// Wrapping around twice panics:
    ///
    /// ```should_panic
    /// use constriction::stream::models::LeakyCategorical;
    /// let probabilities = vec![1u32 << 30, 1 << 31, 1 << 31, 1 << 31, 1 << 30];
    /// // `probabilities` sums up to `1 << 33` (logically), i.e., it would wrap around twice.
    /// let model = LeakyCategorical::<u32, 32>::from_nonzero_fixed_point_probabilities(&probabilities); // PANICS.
    /// ```
    ///
    /// So does providing probabilities that just don't sum up to `1 << FREQUENCY`:
    ///
    /// ```should_panic
    /// use constriction::stream::models::LeakyCategorical;
    /// let probabilities = vec![1u32 << 21, 5 << 8, 1 << 22, 1 << 21];
    /// let model = LeakyCategorical::<u32, 24>::from_nonzero_fixed_point_probabilities(&probabilities); // PANICS.
    /// ```
    ///
    /// [`fixed_point_probabilities`]: #method.fixed_point_probabilities
    /// [`from_floating_point_probabilities`]: #method.from_floating_point_probabilities
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
        cdf.push((wrapping_pow2(PRECISION), Symbol::default()));

        if symbols.next().is_some() {
            Err(())
        } else {
            Ok(Self {
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
        let last_entry = (wrapping_pow2(PRECISION), Symbol::default());
        let cdf = model
            .symbol_table()
            .map(|(symbol, left_sided_cumulative, _)| (left_sided_cumulative, symbol))
            .chain(core::iter::once(last_entry))
            .collect::<Vec<_>>();
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
    /// (inclusively) to `num_symbols()` (exclusively). All symbols within this range are
    /// guaranteed to have a nonzero probability, while all symbols outside of this range
    /// have a zero probability.
    #[inline(always)]
    pub fn num_symbols(&self) -> usize {
        self.cdf.num_symbols()
    }

    #[inline]
    pub fn as_view(
        &self,
    ) -> ContiguousCategoricalEntropyModel<Probability, &[Probability], PRECISION> {
        ContiguousCategoricalEntropyModel {
            cdf: ContiguousSymbolTable(self.cdf.0.as_ref()),
            phantom: PhantomData,
        }
    }
}

impl<Symbol, Probability, Table, const PRECISION: usize>
    NonContiguousCategoricalDecoderModel<Symbol, Probability, Table, PRECISION>
where
    Symbol: Clone,
    Probability: BitArray,
    Table: AsRef<[(Probability, Symbol)]>,
{
    /// Returns the number of symbols supported by the model.
    ///
    /// The distribution is defined on the contiguous range of symbols from zero
    /// (inclusively) to `num_symbols()` (exclusively). All symbols within this range are
    /// guaranteed to have a nonzero probability, while all symbols outside of this range
    /// have a zero probability.
    #[inline(always)]
    pub fn num_symbols(&self) -> usize {
        self.cdf.num_symbols()
    }

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
/// decode encode symbols from a non-contiguous domain, use an
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
            if index >= self.num_symbols() {
                return None;
            }
            (
                self.cdf.left_cumulative_unchecked(index),
                self.cdf.left_cumulative_unchecked(index + 1),
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
    Symbol: Clone + Default,
    Probability: BitArray,
    M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
{
    fn from(model: &'m M) -> Self {
        Self::from_iterable_entropy_model(model)
    }
}

#[derive(Debug, Clone)]
pub struct NonContiguousCategoricalEncoderModel<Symbol, Probability, const PRECISION: usize>
where
    Symbol: Hash,
    Probability: BitArray,
{
    table: HashMap<Symbol, (Probability, Probability::NonZero)>,
}

pub type DefaultNonContiguousCategoricalEncoderModel<Symbol> =
    NonContiguousCategoricalEncoderModel<Symbol, u32, 24>;
pub type SmallNonContiguousCategoricalEncoderModel<Symbol> =
    NonContiguousCategoricalEncoderModel<Symbol, u16, 12>;

impl<Symbol, Probability, const PRECISION: usize>
    NonContiguousCategoricalEncoderModel<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    pub fn from_symbols_and_floating_point_probabilities<F>(
        symbols: impl Iterator<Item = Symbol>,
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

    /// Returns the number of symbols supported by the model.
    ///
    /// The distribution is defined on the contiguous range of symbols from zero
    /// (inclusively) to `num_symbols()` (exclusively). All symbols within this range are
    /// guaranteed to have a nonzero probability, while all symbols outside of this range
    /// have a zero probability.
    pub fn num_symbols(&self) -> usize {
        self.table.len()
    }
}

impl<'m, Symbol, Probability, M, const PRECISION: usize> From<&'m M>
    for NonContiguousCategoricalEncoderModel<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
    M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
{
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
        laps_or_zeros = laps_or_zeros + (accum <= old_accum) as usize;
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
    } else {
        if accum != total || laps_or_zeros != (PRECISION == Probability::BITS) as usize {
            return Err(());
        }
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

/// Type alias for a [`LookupDecoderModel`] over symbols `{0, 1, ..., n-1}` with sane settings.
///
/// This array lookup table can be used with a [`SmallAnsCoder`] or a [`SmallRangeDecoder`]
/// (as well as with a [`DefaultAnsCoder`] or a [`DefaultRangeDecoder`]).
///
/// # Example
///
/// Let's decode the compressed bit strings we generated in the example for
/// [`SmallEncoderArrayLookupTable`].
///
/// ```
/// use constriction::stream::{
///     models::lookup::SmallDecoderIndexLookupTable,
///     stack::{SmallAnsCoder, DefaultAnsCoder},
///     queue::{SmallRangeDecoder, DefaultRangeDecoder},
///     Decode, Code,
/// };
///
/// let probabilities = [1489, 745, 1489, 373];
/// let decoder_model = SmallDecoderIndexLookupTable::from_probabilities(
///     probabilities.iter().cloned()
/// )
/// .unwrap();
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
/// - [`SmallDecoderGenericLookupTable`]
/// - [`SmallEncoderArrayLookupTable`]
///
/// [`SmallAnsCoder`]: super::super::stack::SmallAnsCoder
/// [`SmallRangeDecoder`]: super::super::queue::SmallRangeDecoder
/// [`DefaultAnsCoder`]: super::super::stack::DefaultAnsCoder
/// [`DefaultRangeDecoder`]: super::super::queue::DefaultRangeDecoder
pub type SmallNonContiguousLookupDecoderModel<
    Symbol,
    SymbolTable = Vec<(u16, Symbol)>,
    LookupTable = Box<[u16]>,
> = LookupDecoderModel<Symbol, u16, NonContiguousSymbolTable<SymbolTable>, LookupTable, 12>;

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
            probabilities.into_iter(),
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

    pub fn as_contiguous_categorical(
        &self,
    ) -> ContiguousCategoricalEntropyModel<Probability, &[Probability], PRECISION> {
        ContiguousCategoricalEntropyModel {
            cdf: ContiguousSymbolTable(self.cdf.0.as_ref()),
            phantom: PhantomData,
        }
    }

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

// TODO: implement `IterableEntropyModel` for lookup model.

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
        weights_and_hist.sort();
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
        domain: impl Clone + Iterator<Item = D::Symbol>,
    ) where
        D: IterableEntropyModel<'m, PRECISION, Probability = u32>
            + EncoderModel<PRECISION>
            + DecoderModel<PRECISION>
            + 'm,
        D::Symbol: Copy + core::fmt::Debug + PartialEq,
    {
        let mut sum = 0;
        for symbol in domain.clone() {
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

        test_iterable_entropy_model(model, domain);
    }

    fn test_iterable_entropy_model<'m, D, const PRECISION: usize>(
        model: &'m D,
        domain: impl Clone + Iterator<Item = D::Symbol>,
    ) where
        D: IterableEntropyModel<'m, PRECISION, Probability = u32> + 'm,
        D::Symbol: Copy + core::fmt::Debug + PartialEq,
    {
        let mut expected_cumulative = 0u64;
        let mut count = 0;
        for (expected_symbol, (symbol, left_sided_cumulative, probability)) in
            domain.clone().zip(model.symbol_table())
        {
            assert_eq!(symbol, expected_symbol);
            assert_eq!(left_sided_cumulative as u64, expected_cumulative);
            expected_cumulative += probability.get() as u64;
            count += 1;
        }
        assert_eq!(count, domain.size_hint().0);
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
