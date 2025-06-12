use core::{borrow::Borrow, hash::Hash, marker::PhantomData};

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
use num_traits::{float::FloatCore, AsPrimitive};

use crate::{wrapping_pow2, BitArray, NonZeroBitArray};

use super::{
    super::{DecoderModel, EncoderModel, EntropyModel, IterableEntropyModel},
    accumulate_nonzero_probabilities, fast_quantized_cdf, iter_extended_cdf,
    lookup_noncontiguous::NonContiguousLookupDecoderModel,
    perfectly_quantized_probabilities,
};

/// Type alias for a typical [`NonContiguousCategoricalEncoderModel`].
///
/// See:
/// - [`NonContiguousCategoricalEncoderModel`]
/// - [discussion of presets](crate::stream#presets)
pub type DefaultNonContiguousCategoricalEncoderModel<Symbol> =
    NonContiguousCategoricalEncoderModel<Symbol, u32, 24>;

/// Type alias for a [`NonContiguousCategoricalEncoderModel`] optimized for compatibility
/// with lookup decoder models.
///
/// See:
/// - [`NonContiguousCategoricalEncoderModel`]
/// - [discussion of presets](crate::stream#presets)
pub type SmallNonContiguousCategoricalEncoderModel<Symbol> =
    NonContiguousCategoricalEncoderModel<Symbol, u16, 12>;

/// Type alias for a typical [`NonContiguousCategoricalDecoderModel`].
///
/// See:
/// - [`NonContiguousCategoricalDecoderModel`]
/// - [discussion of presets](crate::stream#presets)
pub type DefaultNonContiguousCategoricalDecoderModel<Symbol, Cdf = Vec<(u32, Symbol)>> =
    NonContiguousCategoricalDecoderModel<Symbol, u32, Cdf, 24>;

/// Type alias for a [`NonContiguousCategoricalDecoderModel`] optimized for compatibility
/// with lookup decoder models.
///
/// See:
/// - [`NonContiguousCategoricalDecoderModel`]
/// - [discussion of presets](crate::stream#presets)
pub type SmallNonContiguousCategoricalDecoderModel<Symbol, Cdf = Vec<(u16, Symbol)>> =
    NonContiguousCategoricalDecoderModel<Symbol, u16, Cdf, 12>;

/// An entropy model for a categorical probability distribution over arbitrary symbols, for
/// decoding only.
///
/// You will usually want to use this type through one of its type aliases,
/// [`DefaultNonContiguousCategoricalDecoderModel`] or
/// [`SmallNonContiguousCategoricalDecoderModel`], see [discussion of
/// presets](crate::stream#presets).
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
/// expression.
///
/// - If you have a probability model that can be expressed by some analytical expression
///   (e.g., a [`Binomial`](probability::distribution::Binomial) distribution), then use
///   [`LeakyQuantizer`] instead (unless you want to encode lots of symbols with the same
///   entropy model, in which case the explicitly tabulated representation of a categorical
///   entropy model could improve runtime performance).
/// - If the *support* of your probabilistic model (i.e., the set of symbols to which the
///   model assigns a non-zero probability) is a contiguous range of integers starting at
///   zero, then it is better to use a [`ContiguousCategoricalEntropyModel`]. It has better
///   computational efficiency and it is easier to use since it supports both encoding and
///   decoding with a single type.
/// - If you want to decode only a few symbols with a given probability model, then use a
///   [`LazyContiguousCategoricalEntropyModel`], which will be faster (use an array to map
///   the decoded symbols from the contiguous range `0..N` to whatever noncontiguous
///   alphabet you have). This use case occurs, e.g., in autoregressive models, where each
///   individual model is often used for only exactly one symbol.
/// - If you want to decode lots of symbols with the same entropy model, and if reducing the
///   `PRECISION` to a moderate value is acceptable to you, then you may want to consider
///   using a [`NonContiguousLookupDecoderModel`] instead for even better runtime
///   performance (at the cost of a larger memory footprint and worse compression efficiency
///   due to lower `PRECISION`).
///
/// # Computational Efficiency
///
/// For a probability distribution with a support of `N` symbols, a
/// `NonContiguousCategoricalDecoderModel` has the following asymptotic costs:
///
/// - creation:
///   - runtime cost: `Θ(N log(N))` (when creating with the [`..._fast` constructor]);
///   - memory footprint: `Θ(N)`;
/// - encoding a symbol: not supported; use a [`NonContiguousCategoricalEncoderModel`]
///   instead.
/// - decoding a symbol (calling [`DecoderModel::quantile_function`]):
///   - runtime cost: `Θ(log(N))` (both expected and worst-case)
///   - memory footprint: no heap allocations, constant stack space.
///
/// [`EntropyModel`]: trait.EntropyModel.html
/// [`ContiguousCategoricalEntropyModel`]:
///     crate::stream::model::ContiguousCategoricalEntropyModel
/// [`NonContiguousLookupDecoderModel`]:
///     crate::stream::model::NonContiguousLookupDecoderModel
/// [`LeakyQuantizer`]: crate::stream::model::LeakyQuantizer
/// [`..._fast` constructor]: Self::from_symbols_and_floating_point_probabilities_fast
/// [`LazyContiguousCategoricalEntropyModel`]:
///     crate::stream::model::LazyContiguousCategoricalEntropyModel
#[derive(Debug, Clone, Copy)]
pub struct NonContiguousCategoricalDecoderModel<Symbol, Probability, Cdf, const PRECISION: usize> {
    /// Invariants:
    /// - `cdf.len() >= 2` (actually, we currently even guarantee `cdf.len() >= 3` but
    ///   this may be relaxed in the future)
    /// - `cdf[0] == 0`
    /// - `cdf` is monotonically increasing except that it may wrap around only at
    ///   the very last entry (this happens iff `PRECISION == Probability::BITS`).
    ///   Thus, all probabilities within range are guaranteed to be nonzero.
    pub(super) cdf: Cdf,

    pub(super) phantom: PhantomData<(Symbol, Probability)>,
}

impl<Symbol, Probability: BitArray, const PRECISION: usize>
    NonContiguousCategoricalDecoderModel<Symbol, Probability, Vec<(Probability, Symbol)>, PRECISION>
where
    Symbol: Clone,
{
    /// Constructs a leaky distribution (for decoding) over the provided `symbols` whose PMF
    /// approximates given `probabilities`.
    ///
    /// Semantics are analogous to
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast`],
    /// except that this constructor has an additional `symbols` argument to provide an
    /// iterator over the symbols in the alphabet (which has to yield exactly
    /// `probabilities.len()` symbols).
    ///
    /// # See also
    ///
    /// - [`from_symbols_and_floating_point_probabilities_perfect`], which can be
    ///   considerably slower but typically approximates the provided `probabilities` *very
    ///   slightly* better.
    ///
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast`]:
    ///     crate::stream::model::ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast
    /// [`from_symbols_and_floating_point_probabilities_perfect`]:
    ///     Self::from_symbols_and_floating_point_probabilities_perfect
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities_fast<F>(
        symbols: impl IntoIterator<Item = Symbol>,
        probabilities: &[F],
        normalization: Option<F>,
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + AsPrimitive<Probability>,
        Probability: AsPrimitive<usize>,
        usize: AsPrimitive<Probability> + AsPrimitive<F>,
    {
        let cdf = fast_quantized_cdf::<Probability, F, PRECISION>(probabilities, normalization)?;

        let mut extended_cdf = Vec::with_capacity(probabilities.len() + 1);
        extended_cdf.extend(cdf.zip(symbols));
        let last_symbol = extended_cdf.last().expect("`len` >= 2").1.clone();
        extended_cdf.push((wrapping_pow2(PRECISION), last_symbol));

        Ok(Self::from_extended_cdf(extended_cdf))
    }

    /// Slower variant of [`from_symbols_and_floating_point_probabilities_fast`].
    ///
    /// Similar to [`from_symbols_and_floating_point_probabilities_fast`], but the resulting
    /// (fixed-point precision) model typically approximates the provided floating point
    /// `probabilities` *very slightly* better. Only recommended if compression performance
    /// is *much* more important to you than runtime as this constructor can be
    /// significantly slower.
    ///
    /// See [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect`]
    /// for a detailed comparison between `..._fast` and `..._perfect` constructors of
    /// categorical entropy models.
    ///
    /// [`from_symbols_and_floating_point_probabilities_fast`]:
    ///     Self::from_symbols_and_floating_point_probabilities_fast
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect`]:
    ///     crate::stream::model::ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities_perfect<F>(
        symbols: impl IntoIterator<Item = Symbol>,
        probabilities: &[F],
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        let slots = perfectly_quantized_probabilities::<_, _, PRECISION>(probabilities)?;
        Self::from_symbols_and_nonzero_fixed_point_probabilities(
            symbols,
            slots.into_iter().map(|slot| slot.weight),
            false,
        )
    }

    /// Deprecated constructor.
    ///
    /// This constructor has been deprecated in constriction version 0.4.0, and it will be
    /// removed in constriction version 0.5.0.
    ///
    /// # Upgrade Instructions
    ///
    /// Most *new* use cases should call
    /// [`from_symbols_and_floating_point_probabilities_fast`] instead. Using that
    /// constructor (abbreviated as `..._fast` in the following) may lead to very slightly
    /// larger bit rates, but it runs considerably faster.
    ///
    /// However, note that the `..._fast` constructor breaks binary compatibility with
    /// `constriction` version <= 0.3.5. If you need to be able to exchange binary
    /// compressed data with a program that uses a categorical entropy model from
    /// `constriction` version <= 0.3.5, then call
    /// [`from_symbols_and_floating_point_probabilities_perfect`] instead (`..._perfect` for
    /// short). Another reason for using the `..._perfect` constructor could be if
    /// compression performance is *much* more important to you than runtime performance.
    /// See documentation of [`from_symbols_and_floating_point_probabilities_perfect`] for
    /// more information.
    ///
    /// # Compatibility Table
    ///
    /// (In the following table, "encoding" refers to
    /// [`NonContiguousCategoricalEncoderModel`])
    ///
    /// | constructor used for encoding → <br> ↓ constructor used for decoding ↓ | [legacy](NonContiguousCategoricalEncoderModel::from_symbols_and_floating_point_probabilities) |  [`..._perfect`](NonContiguousCategoricalEncoderModel::from_symbols_and_floating_point_probabilities_perfect) | [`..._fast`](NonContiguousCategoricalEncoderModel::from_symbols_and_floating_point_probabilities_fast) |
    /// | --------------------: | --------------- | --------------- | --------------- |
    /// | **legacy (this one)** | ✅ compatible   | ✅ compatible   | ❌ incompatible |
    /// | **[`..._perfect`]**   | ✅ compatible   | ✅ compatible   | ❌ incompatible |
    /// | **[`..._fast`]**      | ❌ incompatible | ❌ incompatible | ✅ compatible   |
    ///
    /// [`from_symbols_and_floating_point_probabilities_perfect`]:
    ///     Self::from_symbols_and_floating_point_probabilities_perfect
    /// [`..._perfect`]: Self::from_symbols_and_floating_point_probabilities_perfect
    /// [`from_symbols_and_floating_point_probabilities_fast`]:
    ///     Self::from_symbols_and_floating_point_probabilities_fast
    /// [`..._fast`]: Self::from_symbols_and_floating_point_probabilities_fast
    #[deprecated(
        since = "0.4.0",
        note = "Please use `from_symbols_and_floating_point_probabilities_fast` or \
        `from_symbols_and_floating_point_probabilities_perfect` instead. See documentation for \
        detailed upgrade instructions."
    )]
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities<F>(
        symbols: &[Symbol],
        probabilities: &[F],
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        Self::from_symbols_and_floating_point_probabilities_perfect(
            symbols.iter().cloned(),
            probabilities,
        )
    }

    /// Constructs a distribution with a PMF given in fixed point arithmetic.
    ///
    /// This is a low level method that allows, e.g,. reconstructing a probability
    /// distribution previously exported with [`symbol_table`]. The more common way to
    /// construct a `NonContiguousCategoricalDecoderModel` is via
    /// [`from_symbols_and_floating_point_probabilities_fast`].
    ///
    /// The items of `probabilities` have to be nonzero and smaller than `1 << PRECISION`,
    /// where `PRECISION` is a const generic parameter on the
    /// `NonContiguousCategoricalDecoderModel`.
    ///
    /// If `infer_last_probability` is `false` then `probabilities` must yield the same
    /// number of items as `symbols` does, and the items yielded by `probabilities` have to
    /// to (logically) sum up to `1 << PRECISION`. If `infer_last_probability` is `true`
    /// then `probabilities` must yield one fewer item than `symbols`, they must sum up to a
    /// value strictly smaller than `1 << PRECISION`, and the method will assign the
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
    /// [`from_symbols_and_floating_point_probabilities_fast`]:
    ///     Self::from_symbols_and_floating_point_probabilities_fast
    /// [`ContiguousCategoricalEntropyModel::from_nonzero_fixed_point_probabilities`]:
    ///     crate::stream::model::ContiguousCategoricalEntropyModel::from_nonzero_fixed_point_probabilities`
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
            Ok(Self::from_extended_cdf(cdf))
        }
    }

    #[inline(always)]
    fn from_extended_cdf(cdf: Vec<(Probability, Symbol)>) -> Self {
        Self {
            cdf,
            phantom: PhantomData,
        }
    }

    /// Creates a `NonContiguousCategoricalDecoderModel` from any entropy model that
    /// implements [`IterableEntropyModel`].
    ///
    /// Calling `NonContiguousCategoricalDecoderModel::from_iterable_entropy_model(&model)`
    /// is equivalent to calling `model.to_generic_decoder_model()`, where the latter
    /// requires bringing [`IterableEntropyModel`] into scope.
    pub fn from_iterable_entropy_model<'m, M>(model: &'m M) -> Self
    where
        M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
    {
        let symbol_table = model.symbol_table();
        let mut cdf = Vec::with_capacity(symbol_table.size_hint().0 + 1);
        cdf.extend(
            symbol_table.map(|(symbol, left_sided_cumulative, _)| (left_sided_cumulative, symbol)),
        );
        cdf.push((
            wrapping_pow2(PRECISION),
            cdf.last().expect("`symbol_table` is not empty").1.clone(),
        ));

        Self {
            cdf,
            phantom: PhantomData,
        }
    }
}

impl<Symbol, Probability, Cdf, const PRECISION: usize>
    NonContiguousCategoricalDecoderModel<Symbol, Probability, Cdf, PRECISION>
where
    Symbol: Clone,
    Probability: BitArray,
    Cdf: AsRef<[(Probability, Symbol)]>,
{
    /// Returns the number of symbols supported by the model, i.e., the number of symbols to
    /// which the model assigns a nonzero probability.
    #[inline(always)]
    pub fn support_size(&self) -> usize {
        self.cdf.as_ref().len() - 1
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
    /// [`Decode::decode_iid_symbols`]: crate::stream::Decode::decode_iid_symbols
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
            cdf: self.cdf.as_ref(),
            phantom: PhantomData,
        }
    }

    /// Creates a [`ContiguousLookupDecoderModel`] or [`NonContiguousLookupDecoderModel`] for efficient decoding of i.i.d. data
    ///
    /// While a `NonContiguousCategoricalEntropyModel` can already be used for decoding (since
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
    /// [preset]: crate::stream#presets
    /// [`ContiguousLookupDecoderModel`]: crate::stream::model::ContiguousLookupDecoderModel
    #[inline(always)]
    pub fn to_lookup_decoder_model(
        &self,
    ) -> NonContiguousLookupDecoderModel<
        Symbol,
        Probability,
        Vec<(Probability, Symbol)>,
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

impl<Symbol, Probability, Cdf, const PRECISION: usize> EntropyModel<PRECISION>
    for NonContiguousCategoricalDecoderModel<Symbol, Probability, Cdf, PRECISION>
where
    Probability: BitArray,
{
    type Symbol = Symbol;
    type Probability = Probability;
}

impl<'m, Symbol, Probability, Cdf, const PRECISION: usize> IterableEntropyModel<'m, PRECISION>
    for NonContiguousCategoricalDecoderModel<Symbol, Probability, Cdf, PRECISION>
where
    Symbol: Clone + 'm,
    Probability: BitArray,
    Cdf: AsRef<[(Probability, Symbol)]>,
{
    #[inline(always)]
    fn symbol_table(
        &'m self,
    ) -> impl Iterator<
        Item = (
            Self::Symbol,
            Self::Probability,
            <Self::Probability as BitArray>::NonZero,
        ),
    > {
        iter_extended_cdf(self.cdf.as_ref().iter().cloned())
    }

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

    fn entropy_base2<F>(&'m self) -> F
    where
        F: num_traits::Float + core::iter::Sum,
        Self::Probability: Into<F>,
    {
        let entropy_scaled = self
            .symbol_table()
            .map(|(_, _, probability)| {
                let probability = probability.get().into();
                probability * probability.log2() // probability is guaranteed to be nonzero.
            })
            .sum::<F>();

        let whole = (F::one() + F::one()) * (Self::Probability::one() << (PRECISION - 1)).into();
        F::from(PRECISION).unwrap() - entropy_scaled / whole
    }

    fn to_generic_encoder_model(
        &'m self,
    ) -> NonContiguousCategoricalEncoderModel<Self::Symbol, Self::Probability, PRECISION>
    where
        Self::Symbol: core::hash::Hash + Eq,
    {
        self.into()
    }

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

impl<Symbol, Probability, Cdf, const PRECISION: usize> DecoderModel<PRECISION>
    for NonContiguousCategoricalDecoderModel<Symbol, Probability, Cdf, PRECISION>
where
    Symbol: Clone,
    Probability: BitArray,
    Cdf: AsRef<[(Probability, Symbol)]>,
{
    #[inline(always)]
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Symbol, Probability, Probability::NonZero) {
        let cdf = self.cdf.as_ref();
        // SAFETY: `cdf` is not empty.
        let monotonic_part_of_cdf = unsafe { cdf.get_unchecked(..cdf.len() - 1) };
        let Err(next_index) = monotonic_part_of_cdf.binary_search_by(|(cumulative, _symbol)| {
            if *cumulative <= quantile {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Greater
            }
        }) else {
            // SAFETY: our search criterion never returns `Equal`, so the search cannot succeed.
            unsafe { core::hint::unreachable_unchecked() }
        };

        // SAFETY:
        // - `next_index < cdf.len()` because we searched only within `monotonic_part_of_cdf`, which
        //   is one element shorter than `cdf`. Thus `cdf.get_unchecked(next_index)` is sound.
        // - `next_index > 0` because `cdf[0] == 0` and our search goes right on equality; thus,
        //   `next_index - 1` does not wrap around, and so `next_index - 1` is also within bounds.
        let (right_cumulative, (left_cumulative, symbol)) = unsafe {
            (
                cdf.get_unchecked(next_index).0,
                cdf.get_unchecked(next_index - 1).clone(),
            )
        };

        // SAFETY: our constructors don't allow zero probabilities.
        let probability = unsafe {
            right_cumulative
                .wrapping_sub(&left_cumulative)
                .into_nonzero_unchecked()
        };

        (symbol, left_cumulative, probability)
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
/// presets](crate::stream#presets).
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
///     ::from_symbols_and_floating_point_probabilities_fast(
///         alphabet.iter().cloned(),
///         &probabilities,
///         None
///     )
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
///     ::from_symbols_and_floating_point_probabilities_fast(
///         &alphabet, &probabilities, None
///     )
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
/// expression.
///
/// Use a `NonContiguousCategoricalDecoderModel` for probabilistic models that can *only* be
/// represented as an explicit probability table, and not by some more compact analytic
/// expression.
///
/// - If you have a probability model that can be expressed by some analytical expression
///   (e.g., a [`Binomial`](probability::distribution::Binomial) distribution), then use
///   [`LeakyQuantizer`] instead (unless you want to encode lots of symbols with the same
///   entropy model, in which case the explicitly tabulated representation of a categorical
///   entropy model could improve runtime performance).
/// - If the *support* of your probabilistic model (i.e., the set of symbols to which the
///   model assigns a non-zero probability) is a contiguous range of integers starting at
///   zero, then it is better to use a [`ContiguousCategoricalEntropyModel`]. It has better
///   computational efficiency and it is easier to use since it supports both encoding and
///   decoding with a single type.
/// - If you want to encode only a few symbols with a given probability model, then use a
///   [`LazyContiguousCategoricalEntropyModel`], which will be faster (use a [`HashMap`] to
///   first map from your noncontiguous support to indices in a contiguous range `0..N`,
///   where `N` is the size of your support). This use case occurs, e.g., in autoregressive
///   models, where each individual model is often used for only exactly one symbol.
///
/// # Computational Efficiency
///
/// For a probability distribution with a support of `N` symbols, a
/// `NonContiguousCategoricalEncoderModel` has the following asymptotic costs:
///
/// - creation:
///   - runtime cost: `Θ(N log(N))` (when creating with the [`..._fast` constructor]);
///   - memory footprint: `Θ(N)`;
/// - encoding a symbol (calling [`EncoderModel::left_cumulative_and_probability`]):
///   - expected runtime cost: `Θ(1)` (worst case can be more expensive, uses a `HashMap`
///     under the hood).
///   - memory footprint: no heap allocations, constant stack space.
/// - decoding a symbol: not supported; use a [`NonContiguousCategoricalDecoderModel`].
///
/// [`EntropyModel`]: trait.EntropyModel.html
/// [`ContiguousCategoricalEntropyModel`]: crate::stream::model::ContiguousCategoricalEntropyModel
/// [`LeakyQuantizer`]: crate::stream::model::LeakyQuantizer
/// [`..._fast` constructor]: Self::from_symbols_and_floating_point_probabilities_fast
/// [`LazyContiguousCategoricalEntropyModel`]:
///     crate::stream::model::LazyContiguousCategoricalEntropyModel
#[derive(Debug, Clone)]
pub struct NonContiguousCategoricalEncoderModel<Symbol, Probability, const PRECISION: usize>
where
    Symbol: Hash,
    Probability: BitArray,
{
    table: HashMap<Symbol, (Probability, Probability::NonZero)>,
}

impl<Symbol, Probability, const PRECISION: usize>
    NonContiguousCategoricalEncoderModel<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    /// Constructs a leaky distribution (for encoding) over the provided `symbols` whose PMF
    /// approximates given `probabilities`.
    ///
    /// Semantics are analogous to
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast`],
    /// except that this constructor has an additional `symbols` argument to provide an
    /// iterator over the symbols in the alphabet (which has to yield exactly
    /// `probabilities.len()` symbols).
    ///
    /// # See also
    ///
    /// - [`from_symbols_and_floating_point_probabilities_perfect`], which can be
    ///   considerably slower but typically approximates the provided `probabilities` *very
    ///   slightly* better.
    ///
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast`]:
    ///     crate::stream::model::ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast
    /// [`from_symbols_and_floating_point_probabilities_perfect`]:
    ///     Self::from_symbols_and_floating_point_probabilities_perfect
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities_fast<F>(
        symbols: impl IntoIterator<Item = Symbol>,
        probabilities: &[F],
        normalization: Option<F>,
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + AsPrimitive<Probability>,
        Probability: AsPrimitive<usize>,
        usize: AsPrimitive<Probability> + AsPrimitive<F>,
    {
        let cdf = fast_quantized_cdf::<Probability, F, PRECISION>(probabilities, normalization)?;
        Self::from_symbols_and_cdf(symbols, cdf)
    }

    /// Slower variant of [`from_symbols_and_floating_point_probabilities_fast`].
    ///
    /// Similar to [`from_symbols_and_floating_point_probabilities_fast`], but the resulting
    /// (fixed-point precision) model typically approximates the provided floating point
    /// `probabilities` *very slightly* better. Only recommended if compression performance
    /// is *much* more important to you than runtime as this constructor can be
    /// significantly slower.
    ///
    /// See [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect`]
    /// for a detailed comparison between `..._fast` and `..._perfect` constructors of
    /// categorical entropy models.
    ///
    /// [`from_symbols_and_floating_point_probabilities_fast`]:
    ///     Self::from_symbols_and_floating_point_probabilities_fast
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect`]:
    ///     crate::stream::model::ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities_perfect<F>(
        symbols: impl IntoIterator<Item = Symbol>,
        probabilities: &[F],
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        let slots = perfectly_quantized_probabilities::<_, _, PRECISION>(probabilities)?;
        Self::from_symbols_and_nonzero_fixed_point_probabilities(
            symbols,
            slots.into_iter().map(|slot| slot.weight),
            false,
        )
    }

    /// Deprecated constructor.
    ///
    /// This constructor has been deprecated in constriction version 0.4.0, and it will be
    /// removed in constriction version 0.5.0.
    ///
    /// # Upgrade Instructions
    ///
    /// Most *new* use cases should call
    /// [`from_symbols_and_floating_point_probabilities_fast`] instead. Using that
    /// constructor (abbreviated as `..._fast` in the following) may lead to very slightly
    /// larger bit rates, but it runs considerably faster.
    ///
    /// However, note that the `..._fast` constructor breaks binary compatibility with
    /// `constriction` version <= 0.3.5. If you need to be able to exchange binary
    /// compressed data with a program that uses a categorical entropy model from
    /// `constriction` version <= 0.3.5, then call
    /// [`from_symbols_and_floating_point_probabilities_perfect`] instead (`..._perfect` for
    /// short). Another reason for using the `..._perfect` constructor could be if
    /// compression performance is *much* more important to you than runtime performance.
    /// See documentation of [`from_symbols_and_floating_point_probabilities_perfect`] for
    /// more information.
    ///
    /// # Compatibility Table
    ///
    /// (In the following table, "encoding" refers to
    /// [`NonContiguousCategoricalDecoderModel`])
    ///
    /// | constructor used for encoding → <br> ↓ constructor used for decoding ↓ | legacy <br> (this one) |  [`..._perfect`] | [`..._fast`] |
    /// | --------------------: | --------------- | --------------- | --------------- |
    /// | **[legacy](NonContiguousCategoricalDecoderModel::from_symbols_and_floating_point_probabilities)** | ✅ compatible   | ✅ compatible   | ❌ incompatible |
    /// | **[`..._perfect`](NonContiguousCategoricalDecoderModel::from_symbols_and_floating_point_probabilities_perfect)**   | ✅ compatible   | ✅ compatible   | ❌ incompatible |
    /// | **[`..._fast`](NonContiguousCategoricalDecoderModel::from_symbols_and_floating_point_probabilities_fast)**      | ❌ incompatible | ❌ incompatible | ✅ compatible   |
    ///
    /// [`from_symbols_and_floating_point_probabilities_perfect`]:
    ///     Self::from_symbols_and_floating_point_probabilities_perfect
    /// [`..._perfect`]: Self::from_symbols_and_floating_point_probabilities_perfect
    /// [`from_symbols_and_floating_point_probabilities_fast`]:
    ///     Self::from_symbols_and_floating_point_probabilities_fast
    /// [`..._fast`]: Self::from_symbols_and_floating_point_probabilities_fast
    #[deprecated(
        since = "0.4.0",
        note = "Please use `from_symbols_and_floating_point_probabilities_fast` or \
        `from_symbols_and_floating_point_probabilities_perfect` instead. See documentation for \
        detailed upgrade instructions."
    )]
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities<F>(
        symbols: impl IntoIterator<Item = Symbol>,
        probabilities: &[F],
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        Self::from_symbols_and_floating_point_probabilities_perfect(symbols, probabilities)
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
        let mut table =
            HashMap::with_capacity(symbols.size_hint().0 + infer_last_probability as usize);
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

    #[allow(clippy::result_unit_err)]
    fn from_symbols_and_cdf<S, P>(symbols: S, cdf: P) -> Result<Self, ()>
    where
        S: IntoIterator<Item = Symbol>,
        P: IntoIterator<Item = Probability>,
    {
        let mut symbols = symbols.into_iter();
        let mut cdf = cdf.into_iter();
        let mut table = HashMap::with_capacity(symbols.size_hint().0);

        let mut left_cumulative = cdf.next().ok_or(())?;
        for right_cumulative in cdf {
            let symbol = symbols.next().ok_or(())?;
            match table.entry(symbol) {
                Occupied(_) => return Err(()),
                Vacant(slot) => {
                    let probability = (right_cumulative - left_cumulative)
                        .into_nonzero()
                        .ok_or(())?;
                    slot.insert((left_cumulative, probability));
                }
            }
            left_cumulative = right_cumulative;
        }

        let last_symbol = symbols.next().ok_or(())?;
        let right_cumulative = wrapping_pow2::<Probability>(PRECISION);
        match table.entry(last_symbol) {
            Occupied(_) => return Err(()),
            Vacant(slot) => {
                let probability = right_cumulative
                    .wrapping_sub(&left_cumulative)
                    .into_nonzero()
                    .ok_or(())?;
                slot.insert((left_cumulative, probability));
            }
        }

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
        F: num_traits::Float + core::iter::Sum,
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

#[cfg(test)]
mod tests {
    use super::super::super::tests::{test_iterable_entropy_model, verify_iterable_entropy_model};

    use super::*;

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

        let fast =
            NonContiguousCategoricalDecoderModel::<_,u32, _, 32>::from_symbols_and_floating_point_probabilities_fast(
                symbols.iter().cloned(),
                &probabilities,
                None
            )
            .unwrap();
        test_iterable_entropy_model(&fast, symbols.iter().cloned());
        let kl_fast = verify_iterable_entropy_model(&fast, &hist, 1e-8);

        let perfect =
            NonContiguousCategoricalDecoderModel::<_,u32, _, 32>::from_symbols_and_floating_point_probabilities_perfect(
                symbols.iter().cloned(),
                &probabilities,
            )
            .unwrap();
        test_iterable_entropy_model(&perfect, symbols.iter().cloned());
        let kl_perfect = verify_iterable_entropy_model(&perfect, &hist, 1e-8);

        assert!(kl_perfect < kl_fast);
    }
}
