use core::{borrow::Borrow, marker::PhantomData};

use alloc::vec::Vec;
use num_traits::{float::FloatCore, AsPrimitive};

use crate::{generic_static_asserts, wrapping_pow2, BitArray};

use super::super::{DecoderModel, EncoderModel, EntropyModel};

/// Type alias for a typical [`LazyContiguousCategoricalEntropyModel`].
///
/// See:
/// - [`LazyContiguousCategoricalEntropyModel`]
/// - [discussion of presets](crate::stream#presets)
pub type DefaultLazyContiguousCategoricalEntropyModel<F = f32, Pmf = Vec<F>> =
    LazyContiguousCategoricalEntropyModel<u32, F, Pmf, 24>;

/// Type alias for a [`LazyContiguousCategoricalEntropyModel`] that can be used with coders that use
/// `u16` for their word size.
///
/// Note that, unlike the other type aliases with the `Small...` prefix, creating a lookup table for
/// a *lazy* categorical model is rarely useful. Lazy models are optimized for applications where a
/// model gets used only a few times (e.g., as a part of an autoregressive model) whereas lookup
/// tables are useful if you use the same model lots of times.
///
/// See:
/// - [`LazyContiguousCategoricalEntropyModel`]
/// - [discussion of presets](crate::stream#presets)
pub type SmallLazyContiguousCategoricalEntropyModel<F = f32, Pmf = Vec<F>> =
    LazyContiguousCategoricalEntropyModel<u16, F, Pmf, 12>;

/// Lazily constructed variant of [`ContiguousCategoricalEntropyModel`]
///
/// This type is similar to [`ContiguousCategoricalEntropyModel`], and data encoded with
/// either of the two models can be decoded with either of the two models (provided the both
/// models are constructed with constructors with the same name; see [compatibility table
/// for `ContiguousCategoricalEntropyModel`]).
///
/// The difference between this type and `ContiguousCategoricalEntropyModel` is that this
/// type is lazy, i.e., it delays most of the calculation necessary for approximating a
/// given floating-point probability mass function into fixed-point precision to encoding or
/// decoding time (and then only does the work necessary for the models that actually get
/// encoded or decoded).
///
/// # When Should I Use This Type of Entropy Model?
///
/// - Use this type if you want to encode or decode only a few (or even just a single)
///   symbol with the same categorical distribution.
/// - Use [`ContiguousCategoricalEntropyModel`], [`NonContiguousCategoricalEncoderModel`],
///   or [`NonContiguousCategoricalDecoderModel`] if you want to encode several symbols with
///   the same categorical distribution. These models precalculate the fixed-point
///   approximation of the entire cumulative distribution function at model construction, so
///   that the calculation doesn't have to be done at every encoding/decoding step.
/// - Use [`ContiguousLookupDecoderModel`] or [`NonContiguousLookupDecoderModel`] (together
///   with a small `Probability` data type, see [discussion of presets]) for decoding a
///   *very* large number of i.i.d. symbols if runtime is more important to you than
///   near-optimal bit rate. These models create a lookup table that maps all `2^PRECISION`
///   possible quantiles to the corresponding symbol, thus eliminating the need for a binary
///   search over the CDF at decoding time.
///
/// # Computational Efficiency
///
/// For a probability distribution with a support of `N` symbols, a
/// `LazyContiguousCategoricalEntropyModel` has the following asymptotic costs:
///
/// - creation:
///   - runtime cost: `Θ(1)` if the normalization constant is known and provided, `O(N)`
///     otherwise (but still faster by a constant factor than creating a
///     [`ContiguousCategoricalEntropyModel`] from floating point probabilities);
///   - memory footprint: `Θ(N)`;
///   - both are cheaper by a constant factor than for a
///     [`NonContiguousCategoricalEncoderModel`] or a
///     [`NonContiguousCategoricalDecoderModel`].
/// - encoding a symbol (calling [`EncoderModel::left_cumulative_and_probability`]):
///   - runtime cost: `Θ(1)` (cheaper than for [`NonContiguousCategoricalEncoderModel`]
///     since it compiles to a simple array lookup rather than a `HashMap` lookup)
///   - memory footprint: no heap allocations, constant stack space.
/// - decoding a symbol (calling [`DecoderModel::quantile_function`]):
///   - runtime cost: `Θ(log(N))` (both expected and worst-case; probably slightly cheaper
///     than for [`NonContiguousCategoricalDecoderModel`] due to better memory locality)
///   - memory footprint: no heap allocations, constant stack space.
///
/// # Why is there no `NonContiguous` variant of this model?
///
/// In contrast to `NonContiguousCategorical{En, De}coderModel`, there is no `NonContiguous`
/// variant of this type. A `NonContiguous` variant of this type would offer no improvement
/// in runtime performance compared to using this type
/// (`LazyContiguousCategoricalEntropyModel`) together with a HashMap or Array (for encoding
/// or decoding, respectively) to map between a non-contiguous alphabet and a contiguous set
/// of indices. (This is different for `NonContiguousCategorical{En, De}coderModel`, which
/// avoid an otherwise additional array lookup).
///
/// [`ContiguousCategoricalEntropyModel`]:
///     crate::stream::model::ContiguousCategoricalEntropyModel
/// [`NonContiguousCategoricalEncoderModel`]:
///     crate::stream::model::NonContiguousCategoricalEncoderModel
/// [`NonContiguousCategoricalDecoderModel`]:
///     crate::stream::model::NonContiguousCategoricalDecoderModel
/// [`ContiguousLookupDecoderModel`]: crate::stream::model::ContiguousLookupDecoderModel
/// [`NonContiguousLookupDecoderModel`]:
///     crate::stream::model::NonContiguousLookupDecoderModel
/// [compatibility table for `ContiguousCategoricalEntropyModel`]:
///     crate::stream::model::ContiguousCategoricalEntropyModel#compatibility-table
/// [discussion of presets]: crate::stream#presets
#[derive(Debug, Clone, Copy)]
pub struct LazyContiguousCategoricalEntropyModel<Probability, F, Pmf, const PRECISION: usize> {
    /// Invariants:
    /// - `pmf.len() >= 2`
    pmf: Pmf,
    scale: F,
    phantom: PhantomData<Probability>,
}

impl<Probability, F, Pmf, const PRECISION: usize>
    LazyContiguousCategoricalEntropyModel<Probability, F, Pmf, PRECISION>
where
    Probability: BitArray,
    F: FloatCore + core::iter::Sum<F>,
    Pmf: AsRef<[F]>,
{
    /// Lazily constructs a leaky distribution whose PMF approximates given probabilities.
    ///
    /// Equivalent (and binary compatible to) the [constructor for
    /// `ContiguousCategoricalEntropyModel` with the same
    /// name](crate::stream::model::ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast).
    /// However, this constructor is lazy, i.e., it delays most of the calculation necessary
    /// for approximating the given `probabilities` into fixed-point precision to encoding
    /// or decoding time (and then only does the work necessary for the models that actually
    /// get encoded or decoded). See [struct documentation](Self).
    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities_fast(
        probabilities: Pmf,
        normalization: Option<F>,
    ) -> Result<Self, ()>
    where
        F: AsPrimitive<Probability>,
        Probability: AsPrimitive<usize>,
        usize: AsPrimitive<Probability> + AsPrimitive<F>,
    {
        generic_static_asserts!(
            (Probability: BitArray; const PRECISION: usize);
            PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
            PRECISION_MUST_BE_NONZERO: PRECISION > 0;
        );

        let probs = probabilities.as_ref();

        if probs.len() < 2 || probs.len() >= wrapping_pow2::<usize>(PRECISION).wrapping_sub(1) {
            return Err(());
        }

        let remaining_free_weight =
            wrapping_pow2::<Probability>(PRECISION).wrapping_sub(&probs.len().as_());
        let normalization =
            normalization.unwrap_or_else(|| probabilities.as_ref().iter().copied().sum::<F>());
        if !normalization.is_normal() || !normalization.is_sign_positive() {
            return Err(());
        }

        let scale = AsPrimitive::<F>::as_(remaining_free_weight.as_()) / normalization;

        Ok(Self {
            pmf: probabilities,
            scale,
            phantom: PhantomData,
        })
    }

    /// Returns the number of symbols supported by the model.
    ///
    /// The distribution is defined on the contiguous range of symbols from zero
    /// (inclusively) to `support_size()` (exclusively). All symbols within this range are
    /// guaranteed to have a nonzero probability, while all symbols outside of this range
    /// have a zero probability.
    #[inline(always)]
    pub fn support_size(&self) -> usize {
        self.pmf.as_ref().len()
    }

    /// Makes a very cheap shallow copy of the model that can be used much like a shared
    /// reference.
    ///
    /// The returned `LazyContiguousCategoricalEntropyModel` implements `Copy`, which is a
    /// requirement for some methods, such as [`Encode::encode_iid_symbols`] or
    /// [`Decode::decode_iid_symbols`]. These methods could also accept a shared reference
    /// to a `LazyContiguousCategoricalEntropyModel` (since all references to entropy models are
    /// also entropy models, and all shared references implement `Copy`), but passing a
    /// *view* instead may be slightly more efficient because it avoids one level of
    /// dereferencing.
    ///
    /// Note that `LazyContiguousCategoricalEntropyModel` is optimized for models that are used
    /// only rarely (often just a single time). Thus, if you find yourself handing out lots of
    /// views to the same `LazyContiguousCategoricalEntropyModel` then you'd likely be better off
    /// using a [`ContiguousCategoricalEntropyModel`] instead.
    ///
    /// [`Encode::encode_iid_symbols`]: crate::stream::Encode::encode_iid_symbols
    /// [`Decode::decode_iid_symbols`]: crate::stream::Decode::decode_iid_symbols
    /// [`ContiguousCategoricalEntropyModel`]: crate::stream::model::ContiguousCategoricalEntropyModel
    #[inline]
    pub fn as_view(
        &self,
    ) -> LazyContiguousCategoricalEntropyModel<Probability, F, &[F], PRECISION> {
        LazyContiguousCategoricalEntropyModel {
            pmf: self.pmf.as_ref(),
            scale: self.scale,
            phantom: PhantomData,
        }
    }
}

impl<Probability, F, Pmf, const PRECISION: usize> EntropyModel<PRECISION>
    for LazyContiguousCategoricalEntropyModel<Probability, F, Pmf, PRECISION>
where
    Probability: BitArray,
{
    type Symbol = usize;
    type Probability = Probability;
}

impl<Probability, F, Pmf, const PRECISION: usize> EncoderModel<PRECISION>
    for LazyContiguousCategoricalEntropyModel<Probability, F, Pmf, PRECISION>
where
    Probability: BitArray,
    F: FloatCore + core::iter::Sum<F> + AsPrimitive<Probability>,
    usize: AsPrimitive<Probability>,
    Pmf: AsRef<[F]>,
{
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Self::Symbol>,
    ) -> Option<(Self::Probability, <Self::Probability as BitArray>::NonZero)> {
        let symbol = *symbol.borrow();
        let pmf = self.pmf.as_ref();
        let probability_float = *pmf.get(symbol)?;

        // SAFETY: when we initialized `probability_float`, we checked if `symbol` is out of bounds.
        let left_side = unsafe { pmf.get_unchecked(..symbol) };
        let left_cumulative_float = left_side.iter().copied().sum::<F>();
        let left_cumulative = (left_cumulative_float * self.scale).as_() + symbol.as_();

        // It may seem easier to calculate `probability` directly from `probability_float` but
        // this could pick up different rounding errors, breaking guarantees of `EncoderModel`.
        let right_cumulative_float = left_cumulative_float + probability_float;
        let right_cumulative: Probability = if symbol == pmf.len() - 1 {
            // We have to treat the last symbol as a special case since standard treatment could
            // lead to an inaccessible last quantile due to rounding errors.
            wrapping_pow2(PRECISION)
        } else {
            (right_cumulative_float * self.scale).as_() + symbol.as_() + Probability::one()
        };
        let probability = right_cumulative
            .wrapping_sub(&left_cumulative)
            .into_nonzero()
            .expect("leakiness should guarantee nonzero probabilities.");

        Some((left_cumulative, probability))
    }
}

impl<Probability, F, Pmf, const PRECISION: usize> DecoderModel<PRECISION>
    for LazyContiguousCategoricalEntropyModel<Probability, F, Pmf, PRECISION>
where
    F: FloatCore + core::iter::Sum<F> + AsPrimitive<Probability>,
    usize: AsPrimitive<Probability>,
    Probability: BitArray + AsPrimitive<F>,
    Pmf: AsRef<[F]>,
{
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (
        Self::Symbol,
        Self::Probability,
        <Self::Probability as BitArray>::NonZero,
    ) {
        // We avoid division completely and float-to-int conversion as much as possible here
        // because they are slow.

        let mut left_cumulative_float = F::zero();
        let mut right_cumulative_float = F::zero();

        // First, skip any symbols where we can conclude even without any expensive float-to-int
        // conversions that are too early. We slightly over-estimate `self.scale` so that any
        // mismatch in rounding errors can only make our bound more conservative.
        let enlarged_scale = (F::one() + F::epsilon() + F::epsilon()) * self.scale;
        let lower_bound =
            quantile.saturating_sub(self.pmf.as_ref().len().as_()).as_() / enlarged_scale;

        let mut iter = self.pmf.as_ref().iter();
        let mut next_symbol = 0usize;
        for &next_probability in &mut iter {
            next_symbol = next_symbol.wrapping_add(1);
            left_cumulative_float = right_cumulative_float;
            right_cumulative_float = right_cumulative_float + next_probability;
            if right_cumulative_float >= lower_bound {
                break;
            }
        }

        // Then search for the correct `symbol` using the same float-to-int conversions as in
        // `EncoderModel::left_cumulative_and_probability`.
        let mut left_cumulative =
            (left_cumulative_float * self.scale).as_() + next_symbol.wrapping_sub(1).as_();

        for &next_probability in &mut iter {
            let right_cumulative = (right_cumulative_float * self.scale).as_() + next_symbol.as_();
            if right_cumulative > quantile {
                let probability = right_cumulative
                    .wrapping_sub(&left_cumulative)
                    .into_nonzero()
                    .expect("leakiness should guarantee nonzero probabilities.");
                return (next_symbol.wrapping_sub(1), left_cumulative, probability);
            }

            left_cumulative = right_cumulative;

            right_cumulative_float = right_cumulative_float + next_probability;
            next_symbol = next_symbol.wrapping_add(1);
        }

        // We have to treat the last symbol as a special case since standard treatment could
        // lead to an inaccessible last quantile due to rounding errors.
        let right_cumulative = wrapping_pow2::<Probability>(PRECISION);
        let probability = right_cumulative
            .wrapping_sub(&left_cumulative)
            .into_nonzero()
            .expect("leakiness should guarantee nonzero probabilities.");

        (next_symbol.wrapping_sub(1), left_cumulative, probability)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_contiguous_categorical() {
        #[allow(clippy::excessive_precision)]
        let unnormalized_probs: [f32; 30] = [
            4.22713972, 1e-20, 0.22221771, 0.00927659, 1.58383270, 0.95804675, 0.78104103,
            0.81518454, 0.75206966, 0.58559047, 0.00024284, 1.81382388, 3.22535052, 0.77940434,
            0.24507986, 0.07767093, 0.0, 0.11429778, 0.00179474, 0.30613952, 0.72192056,
            0.00778274, 0.18957551, 10.2402638, 3.36959484, 0.02624742, 1.85103708, 0.25614601,
            0.09754817, 0.27998250,
        ];
        let normalization = 33.538302;

        const PRECISION: usize = 32;
        let model =
            LazyContiguousCategoricalEntropyModel::<u32, _,_, PRECISION>::from_floating_point_probabilities_fast(
                &unnormalized_probs,
                None,
            ).unwrap();

        let mut sum: u64 = 0;
        for (symbol, &unnormalized_prob) in unnormalized_probs.iter().enumerate() {
            let (left_cumulative, prob) = model.left_cumulative_and_probability(symbol).unwrap();
            assert_eq!(left_cumulative as u64, sum);
            let float_prob = prob.get() as f32 / (1u64 << PRECISION) as f32;
            assert!((float_prob - unnormalized_prob / normalization).abs() < 1e-6);
            sum += prob.get() as u64;

            let expected = (symbol, left_cumulative, prob);
            assert_eq!(model.quantile_function(left_cumulative), expected);
            assert_eq!(model.quantile_function((sum - 1).as_()), expected);
            assert_eq!(
                model.quantile_function((left_cumulative as u64 + prob.get() as u64 / 2) as u32),
                expected
            );
        }
        assert_eq!(sum, 1 << PRECISION);
    }
}
