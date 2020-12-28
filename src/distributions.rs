//! Probability distributions that can be used as entropy models for compression.
//!
//! See documentation of [`Coder`] for an example how to use these distributions for
//! data compression or decompression.
//!
//! [`Coder`]: crate::Coder

use num::{cast::AsPrimitive, traits::WrappingSub, Float, PrimInt};
use statrs::distribution::{InverseCDF, Univariate};
use std::{borrow::Borrow, fmt::Debug, marker::PhantomData, ops::RangeInclusive};

use super::BitArray;

/// A trait for probability distributions that can be used as entropy models.
///
/// The type parameter `Self::Probability` controls the fixed-point precision at which probabilities
/// are represented. The type system enforces that this matches the type of
/// compressed words of the [`Coder`] or [`SeekableDecoder`].
///
/// [`Coder`]: crate::Coder
/// [`SeekableDecoder`]: crate::SeekableDecoder
pub trait DiscreteDistribution {
    type Probability: BitArray;
    const PRECISION: usize;

    /// The type of data over which the probability distribution is defined.
    ///
    /// When the `DiscreteDistribution` is used as an entropy model, this is the
    /// type of an item of the *uncompressed* data. Note that a [`Coder`] can encode
    /// several symbols with different entropy models, and each entropy model may
    /// have a different `Symbol` type. Only the type parameter `Self::Probability` (which controls
    /// the fixed-point precision of the entropy models) must be the same across all
    /// entropy models used with a given `Coder` (and this is enforced by the
    /// compiler).
    ///
    /// [`Coder`]: crate::Coder
    /// [`CompressedWord`]: crate::CompressedWord
    type Symbol;

    /// Returns `Ok((left_sided_cumulative, probability))` of the bin for the
    /// provided `symbol` if the symbol has nonzero probability.
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Self::Symbol>,
    ) -> Result<(Self::Probability, Self::Probability), ()>;

    /// Returns `(symbol, left_sided_cumulative, probability)` of the unique bin
    /// that satisfies `left_sided_cumulative <= quantile < right_sided_cumulative`.
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Self::Symbol, Self::Probability, Self::Probability);
}

impl<D: DiscreteDistribution> DiscreteDistribution for &D {
    type Probability = D::Probability;
    const PRECISION: usize = D::PRECISION;
    type Symbol = D::Symbol;

    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Self::Symbol>,
    ) -> Result<(Self::Probability, Self::Probability), ()> {
        (*self).left_cumulative_and_probability(symbol)
    }

    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Self::Symbol, Self::Probability, Self::Probability) {
        (*self).quantile_function(quantile)
    }
}

/// Turns continuous distributions into discrete distributions.
///
/// This is a builder of [`LeakilyQuantizedDistribution`]s.
///
/// Lossless entropy coding can only be performed over discrete data. Any continuous
/// (real-valued) data has to be approximate by some discrete set of points. This
/// builder allows taking continuous distributions (defined over `F`, which is
/// typically `f64` or `f32`), and approximating them by discrete distributions
/// defined over the integer type `Symbol` (which is typically something like `i32`) by
/// rounding all values to the closest integer. The resulting
/// [`LeakilyQuantizedDistribution`]s can be used for entropy coding with a
/// [`Coder`] or a [`SeekableDecoder`] because they implement
/// [`DiscreteDistribution`].
///
/// This quantizer is a "leaky" quantizer. This means that the constructor [`new`]
/// takes a domain in `Symbol` as an argument. The resulting
/// [`LeakilyQuantizedDistribution`]s are guaranteed to assign a nonzero probability
/// to every integer within this domain. This is often a useful property of an
/// entropy model since it ensures that every integer within the chosen domain can
/// in fact be encoded.
///
/// # Example
///
/// ```
/// use constriction::{distributions::LeakyQuantizer, stack::DefaultStack, Encode};
///
/// // Get a quantizer that supports integer symbols from -5 to 20 (inclusively),
/// // representing probabilities with 24 bit precision backed by `u32`s.
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-5..=20);
///
/// // Quantize a normal distribution with mean 8.3 and standard deviation 4.1.
/// let continuous_distribution1 = statrs::distribution::Normal::new(8.3, 4.1).unwrap();
/// let discrete_distribution1 = quantizer.quantize(continuous_distribution1);
///
/// // You can reuse the same quantizer for more than one distribution.
/// let continuous_distribution2 = statrs::distribution::Normal::new(-1.4, 2.7).unwrap();
/// let discrete_distribution2 = quantizer.quantize(continuous_distribution2);
///
/// // Use the discrete distributions with a `Coder`.
/// let mut coder = DefaultStack::new();
/// coder.encode_symbol(4, discrete_distribution1);
/// coder.encode_symbol(-3, discrete_distribution2);
/// ```
///
/// # TODO
///
/// Implement non-leaky variant once minimal const generics are stable
/// (est. February 2021).
///
/// [`Coder`]: crate::Coder
/// [`SeekableDecoder`]: crate::SeekableDecoder
/// [`new`]: #method.new
#[derive(Debug)]
pub struct LeakyQuantizer<F, Symbol, Probability, const PRECISION: usize> {
    min_symbol_inclusive: Symbol,
    max_symbol_inclusive: Symbol,
    free_weight: F,
    phantom: PhantomData<Probability>,
}

impl<F, Symbol, Probability, const PRECISION: usize>
    LeakyQuantizer<F, Symbol, Probability, PRECISION>
where
    Probability: BitArray + Into<F>,
    Symbol: PrimInt + AsPrimitive<Probability> + WrappingSub,
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
    Symbol: PrimInt + AsPrimitive<Probability> + WrappingSub,
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
        CD: Univariate<F, F> + InverseCDF<F>,
    {
        LeakilyQuantizedDistribution {
            inner: distribution,
            quantizer: self,
        }
    }
}

/// Wrapper that turns a continuous probability density into a
/// [`DiscreteDistribution`].
///
/// Such a `LeakilyQuantizedDistribution` can be created with a [`LeakyQuantizer`].
/// It can be used for entropy coding since it implements [`DiscreteDistribution`].
#[derive(Debug)]
pub struct LeakilyQuantizedDistribution<'a, F, Symbol, Probability, CD, const PRECISION: usize> {
    inner: CD,
    quantizer: &'a LeakyQuantizer<F, Symbol, Probability, PRECISION>,
}

impl<'a, F, Symbol, Probability, CD, const PRECISION: usize> DiscreteDistribution
    for LeakilyQuantizedDistribution<'a, F, Symbol, Probability, CD, PRECISION>
where
    Symbol: PrimInt + AsPrimitive<Probability> + Into<F> + WrappingSub,
    F: Float + AsPrimitive<Symbol> + AsPrimitive<Probability>,
    Probability: BitArray + Into<F>,
    CD: Univariate<F, F> + InverseCDF<F>,
{
    type Probability = Probability;
    const PRECISION: usize = PRECISION;
    type Symbol = Symbol;

    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Symbol>,
    ) -> Result<(Self::Probability, Self::Probability), ()> {
        let half = F::one() / (F::one() + F::one());

        let min_symbol_inclusive = self.quantizer.min_symbol_inclusive;
        let max_symbol_inclusive = self.quantizer.max_symbol_inclusive;
        let free_weight = self.quantizer.free_weight;

        if symbol.borrow() < &min_symbol_inclusive || symbol.borrow() > &max_symbol_inclusive {
            return Err(());
        };
        let slack = symbol.borrow().wrapping_sub(&min_symbol_inclusive).as_();

        // Round both cumulatives *independently* to fixed point precision.
        let left_sided_cumulative = if symbol.borrow() == &min_symbol_inclusive {
            // Corner case: only makes a difference if we're cutting off a fairly significant
            // left tail of the distribution.
            Self::Probability::zero()
        } else {
            let non_leaky: Self::Probability =
                (free_weight * self.inner.cdf((*symbol.borrow()).into() - half)).as_();
            non_leaky + slack
        };

        let right_sided_cumulative = if symbol.borrow() == &max_symbol_inclusive {
            // Corner case: make sure that the probabilities add up to one. The generic
            // calculation in the `else` branch may lead to a lower total probability
            // because we're cutting off the right tail of the distribution and we're
            // rounding down.
            let max_probability =
                Self::Probability::max_value() >> (Self::Probability::BITS - Self::PRECISION);
            max_probability.wrapping_add(&Self::Probability::one())
        } else {
            let non_leaky: Self::Probability =
                (free_weight * self.inner.cdf((*symbol.borrow()).into() + half)).as_();
            non_leaky + slack + Self::Probability::one()
        };

        let probability = right_sided_cumulative.wrapping_sub(&left_sided_cumulative);

        Ok((left_sided_cumulative, probability))
    }

    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Self::Symbol, Self::Probability, Self::Probability) {
        let max_probability =
            Self::Probability::max_value() >> (Self::Probability::BITS - Self::PRECISION);
        // This check should usually compile away in inlined and verifiably correct usages
        // of this method.
        assert!(quantile <= max_probability);

        let half = F::one() / (F::one() + F::one());
        let inverse_denominator = F::one() / (max_probability.into() + F::one());

        let min_symbol_inclusive = self.quantizer.min_symbol_inclusive;
        let max_symbol_inclusive = self.quantizer.max_symbol_inclusive;
        let free_weight = self.quantizer.free_weight;

        // Make an initial guess for the inverse of the leaky CDF.
        let mut symbol: Self::Symbol = self
            .inner
            .inverse_cdf((quantile.into() + half) * inverse_denominator)
            .as_();

        let mut left_sided_cumulative = if symbol <= min_symbol_inclusive {
            // Corner case: we're in the left cut off tail of the distribution.
            symbol = min_symbol_inclusive;
            Self::Probability::zero()
        } else {
            if symbol > max_symbol_inclusive {
                // Corner case: we're in the right cut off tail of the distribution.
                symbol = max_symbol_inclusive;
            }

            let non_leaky: Self::Probability =
                (free_weight * self.inner.cdf(symbol.into() - half)).as_();
            non_leaky + symbol.wrapping_sub(&min_symbol_inclusive).as_()
        };

        let right_sided_cumulative = if left_sided_cumulative > quantile {
            // Our initial guess for `symbol` was too high. Reduce it until we're good.
            loop {
                let right_sided_cumulative = left_sided_cumulative;
                symbol = symbol - Self::Symbol::one();

                if symbol == min_symbol_inclusive {
                    left_sided_cumulative = Self::Probability::zero();
                    break right_sided_cumulative;
                }

                let non_leaky: Self::Probability =
                    (free_weight * self.inner.cdf(symbol.into() - half)).as_();
                left_sided_cumulative =
                    non_leaky + symbol.wrapping_sub(&min_symbol_inclusive).as_();
                if left_sided_cumulative <= quantile {
                    break right_sided_cumulative;
                }
            }
        } else {
            // Our initial guess for `symbol` was either exactly right or too low.
            // Check validity of the right sided cumulative. If it isn't valid,
            // keep increasing `symbol` until it is.
            loop {
                if symbol == max_symbol_inclusive {
                    let max_probability = Self::Probability::max_value()
                        >> (Self::Probability::BITS - Self::PRECISION);
                    break max_probability.wrapping_add(&Self::Probability::one());
                }

                let non_leaky: Self::Probability =
                    (free_weight * self.inner.cdf(symbol.into() + half)).as_();
                let right_sided_cumulative = non_leaky
                    + symbol.wrapping_sub(&min_symbol_inclusive).as_()
                    + Self::Probability::one();
                if right_sided_cumulative > quantile {
                    break right_sided_cumulative;
                }

                left_sided_cumulative = right_sided_cumulative;
                symbol = symbol + Self::Symbol::one();
            }
        };

        (
            symbol,
            left_sided_cumulative,
            right_sided_cumulative.wrapping_sub(&left_sided_cumulative),
        )
    }
}

/// A categorical distribution over a finite number of bins.
///
/// This distribution implements [`DiscreteDistribution<Self::Probability>`], which means that it
/// can be used for entropy coding with a [`Coder<Self::Probability>`] or [`SeekableDecoder<Self::Probability>`].
///
/// [`DiscreteDistribution<Self::Probability>`]: trait.DiscreteDistribution.html
/// [`Coder<Self::Probability>`]: crate::Coder
/// [`SeekableDecoder<Self::Probability>`]: crate::SeekableDecoder
pub struct Categorical<Probability, const PRECISION: usize> {
    /// Invariants:
    /// - `cdf.len() >= 2` (actually, we currently even guarantee `cdf.len() >= 3` but
    ///   this may be relaxed in the future)
    /// - `cdf[0] == 0`
    /// - `cdf` is monotonically nondecreasing except that it may wrap around only at
    ///   the very last entry (this happens iff `PRECISION == Probability::BITS`).
    cdf: Vec<Probability>,
}

impl<Probability, const PRECISION: usize> Debug for Categorical<Probability, PRECISION>
where
    Probability: BitArray + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.fixed_point_probabilities())
            .finish()
    }
}

impl<Probability: BitArray, const PRECISION: usize> Categorical<Probability, PRECISION> {
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
    ///   type `Probability`. We refer to this property as the resulting distribution being
    ///   "leaky". This probability ensures that all symbols within the domain can
    ///   be encoded when this distribution is used as an entropy model.
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
        F: Float + std::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        struct Slot<Probability> {
            original_index: usize,
            prob: f64,
            weight: Probability,
            win: f64,
            loss: f64,
        }

        assert!(PRECISION > 0 && PRECISION <= Probability::BITS);

        if probabilities.len() < 2 || probabilities.len() > Probability::max_value().as_() {
            return Err(());
        }

        // Start by assigning each symbol weight 1 and then distributing no more than
        // the remaining weight approximately evenly across all symbols.
        let max_probability = Probability::max_value() >> (Probability::BITS - Self::PRECISION);
        let mut remaining_free_weight = max_probability
            .wrapping_add(&Probability::one())
            .wrapping_sub(&probabilities.len().as_());
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
            slots.sort_by(|a, b| b.win.partial_cmp(&a.win).unwrap());
            let batch_size = std::cmp::min(remaining_free_weight.as_(), slots.len());
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

        slots.sort_by_key(|slot| slot.original_index);

        Ok(Self::from_fixed_point_probabilities(
            slots.into_iter().map(|slot| slot.weight),
        ))
    }

    /// Constructs a distribution with a PMF given in fixed point arithmetic.
    ///
    /// This is a low level method that allows, e.g,. reconstructing a probability
    /// distribution previously exported with [`fixed_point_probabilities`]. The
    /// more common way to construct a `Categorical` distribution is via
    /// [`from_floating_point_probabilities`].
    ///
    /// The entries of `probabilities` have to (logically) sum up to `1 << PRECISION`,
    /// where `PRECISION` is a const generic parameter on the `Categorical`
    /// distribution.
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
    /// let probabilities = vec![1u32 << 21, 1 << 22, 1 << 22, 1 << 22, 1 << 21];
    /// // `probabilities` sums up to `1 << PRECISION` as required:
    /// assert_eq!(probabilities.iter().sum::<u32>(), 1 << 24);
    ///
    /// let distribution =
    ///     constriction::distributions::Categorical::<u32, 24>::from_fixed_point_probabilities(&probabilities);
    /// let pmf = distribution.floating_point_probabilities().collect::<Vec<f64>>();
    /// assert_eq!(pmf, vec![0.125, 0.25, 0.25, 0.25, 0.125]);
    /// ```
    ///
    /// If `PRECISION` is set to the maximum value supported by the `CompressedWord`
    /// type `Probability`, then the provided probabilities still have to *logically* sum up to
    /// `1 << PRECISION` (i.e., the summation has to wrap around exactly once):
    ///
    /// ```
    /// let probabilities = vec![1u32 << 29, 1 << 30, 1 << 30, 1 << 30, 1 << 29];
    /// // `probabilities` sums up to `1 << 32` (logically), i.e., it wraps around once.
    /// assert_eq!(probabilities.iter().fold(0u32, |accum, &x| accum.wrapping_add(x)), 0);
    ///
    /// let distribution =
    ///     constriction::distributions::Categorical::<u32, 32>::from_fixed_point_probabilities(&probabilities);
    /// let pmf = distribution.floating_point_probabilities().collect::<Vec<f64>>();
    /// assert_eq!(pmf, vec![0.125, 0.25, 0.25, 0.25, 0.125]);
    /// ```
    ///
    /// Wrapping around twice panics:
    ///
    /// ```should_panic
    /// let probabilities = vec![1u32 << 30, 1 << 31, 1 << 31, 1 << 31, 1 << 30];
    /// // `probabilities` sums up to `1 << 33` (logically), i.e., it would wrap around twice.
    /// let distribution = // PANICS
    ///     constriction::distributions::Categorical::<u32, 32>::from_fixed_point_probabilities(&probabilities);
    /// ```
    ///
    /// So does providing probabilities that just don't sum up to `1 << FREQUENCY`:
    ///
    /// ```should_panic
    /// let probabilities = vec![1u32 << 21, 5 << 8, 1 << 22, 1 << 21];
    /// let distribution = // PANICS
    ///     constriction::distributions::Categorical::<u32, 24>::from_fixed_point_probabilities(&probabilities);
    /// ```
    ///
    /// [`fixed_point_probabilities`]: #method.fixed_point_probabilities
    /// [`from_floating_point_probabilities`]: #method.from_floating_point_probabilities
    pub fn from_fixed_point_probabilities<I>(probabilities: I) -> Self
    where
        I: IntoIterator,
        I::Item: Borrow<Probability>,
    {
        assert!(PRECISION > 0 && PRECISION <= Probability::BITS);

        let mut laps: usize = 0;
        let mut accum = Probability::zero();
        let mut fingerprint = Probability::zero();

        let mut cdf = std::iter::once(Probability::zero())
            .chain(probabilities.into_iter().map(|prob| {
                let old_accum = accum;
                accum = accum.wrapping_add(prob.borrow());
                laps += (accum < old_accum) as usize; // branchless check if we've wrapped around
                fingerprint = fingerprint | *prob.borrow(); // branchless check for degenerateness
                accum
            }))
            .collect::<Vec<_>>();

        if Self::PRECISION == Probability::BITS {
            assert_eq!(laps, 1);
            assert!(cdf.last() == Some(&Probability::zero()));

            // Truncate trailing zero probabilities to ensure that `cdf` is monotonically
            // nondecreasing except at the very last entry.
            let (last, _) = cdf
                .iter()
                .enumerate()
                .rfind(|(_, &x)| x != Probability::zero())
                .unwrap();
            cdf.resize(last + 2, Probability::zero()); // Should never make cdf larger.
        } else {
            assert_eq!(laps, 0);
            let expected_last = Probability::one() << Self::PRECISION;
            assert!(fingerprint != expected_last);
            assert!(cdf.last() == Some(&expected_last));
        }

        Self { cdf }
    }

    /// Returns the underlying probability mass function in fixed point arithmetic.
    ///
    /// This method may be used together with [`domain`](#method.domain) to export
    /// the distribution into a format that will be stable across minor version
    /// changes of this library. The distribution can then be reconstructed via
    /// [`from_fixed_point_probabilities`](#method.from_fixed_point_probabilities).
    ///
    /// The entries of the returned iterator add up to `Probability::max_value() + 1`
    /// (logically).
    ///
    /// To get the probabilities in a more interpretable representation, consider
    /// [`floating_point_probabilities`](#method.floating_point_probabilities) or
    /// [`floating_point_probabilities_lossy`](
    /// #method.floating_point_probabilities_lossy).
    ///
    /// # Example
    ///
    /// ```
    /// let probabilities = vec![0.125, 0.5, 0.25, 0.125]; // Can all be represented without rounding.
    /// let distribution =
    ///     constriction::distributions::Categorical::<u32, 32>::from_floating_point_probabilities(
    ///         &probabilities
    ///     )
    ///     .unwrap();
    ///
    /// let pmf = distribution.fixed_point_probabilities().collect::<Vec<_>>();
    /// assert_eq!(pmf, vec![1 << 29, 1 << 31, 1 << 30, 1 << 29]);
    /// ```
    pub fn fixed_point_probabilities(
        &self,
    ) -> impl Iterator<Item = Probability> + ExactSizeIterator + '_ {
        let mut previous = Probability::zero();
        self.cdf.iter().skip(1).map(move |&current| {
            let probability = current.wrapping_sub(&previous);
            previous = current;
            probability
        })
    }

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
    /// exact same distribution (although using [`fixed_point_probabilities`] and
    /// [`from_fixed_point_probabilities`] would be cheaper for this purpose).
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
    /// let probabilities = vec![1u32 << 29, 1 << 31, 1 << 30, 1 << 29];
    /// let distribution =
    ///     constriction::distributions::Categorical::<u32, 32>::from_fixed_point_probabilities(&probabilities);
    ///
    /// let pmf = distribution.floating_point_probabilities().collect::<Vec<f64>>();
    /// assert_eq!(pmf, vec![0.125, 0.5, 0.25, 0.125]);
    /// ```
    ///
    /// [`fixed_point_probabilities`]: #method.fixed_point_probabilities
    /// [`floating_point_probabilities_lossy`]: #method.floating_point_probabilities_lossy
    /// [`from_floating_point_probabilities`]: #method.from_floating_point_probabilities
    /// [`from_fixed_point_probabilities`]: #method.from_fixed_point_probabilities
    pub fn floating_point_probabilities<'s, F>(
        &'s self,
    ) -> impl Iterator<Item = F> + ExactSizeIterator + 's
    where
        F: Float + 's,
        Probability: Into<F>,
    {
        let half = F::one() / (F::one() + F::one());
        let scale = half / (Probability::one() << (Self::PRECISION - 1)).into();
        self.fixed_point_probabilities()
            .map(move |x| scale * x.into())
    }

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
    /// let probabilities = vec![1u32 << 29, 1 << 31, 1 << 30, 1 << 29];
    /// let distribution =
    ///     constriction::distributions::Categorical::<u32, 32>::from_fixed_point_probabilities(&probabilities);
    ///
    /// let pmf = distribution.floating_point_probabilities().collect::<Vec<f32>>();
    /// //                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Compiler error: trait bound not satisfied
    /// ```
    ///
    /// This can either be fixed by replacing `f32` with `f64` (thus guaranteeing
    /// lossless conversion) or by replacing `floating_point_probabilities` with
    /// `floating_point_probabilities_lossy` as follows:
    ///
    /// ```
    /// let probabilities = vec![1u32 << 29, 1 << 31, 1 << 30, 1 << 29];
    /// let distribution =
    ///     constriction::distributions::Categorical::<u32, 32>::from_fixed_point_probabilities(&probabilities);
    ///
    /// let pmf = distribution.floating_point_probabilities_lossy().collect::<Vec<f32>>();
    /// assert_eq!(pmf, vec![0.125, 0.5, 0.25, 0.125]);
    /// ```
    ///
    /// [`floating_point_probabilities`]: #method.floating_point_probabilities
    pub fn floating_point_probabilities_lossy<'s, F>(
        &'s self,
    ) -> impl Iterator<Item = F> + ExactSizeIterator + 's
    where
        F: Float + 'static,
        Probability: AsPrimitive<F>,
    {
        let half_denominator: F = (Probability::one() << (Self::PRECISION - 1)).as_();
        let denominator = half_denominator + half_denominator;
        self.fixed_point_probabilities()
            .map(move |x| x.as_() / denominator)
    }

    /// Returns the size of the domain of the distribution
    ///
    /// The distribution is defined on symbols ranging from 0 (inclusively) to the
    /// value returned by this method (exclusively). Any symbol larger than or equal to
    /// the value returned by this method is guaranteed to have zero probability
    /// under the distribution.
    ///
    /// Note that symbols within the above domain may also have zero probability
    /// unless the probability distribution is "leaky" (use
    /// [`from_floating_point_probabilities`](
    /// #method.from_floating_point_probabilities) to construct a *leaky*
    /// categorical distribution).
    pub fn domain_size(&self) -> usize {
        self.cdf.len() - 1
    }

    /// Returns the entropy in units of bits (i.e., base 2).
    pub fn entropy<F>(&self) -> F
    where
        F: Float + std::iter::Sum,
        Probability: Into<F>,
    {
        let entropy_scaled = self
            .cdf
            .iter()
            .skip(1)
            .scan(Probability::zero(), |previous, &cdf| {
                let prob = cdf.wrapping_sub(previous);
                *previous = cdf;
                Some(if prob.is_zero() {
                    F::zero()
                } else {
                    let prob = prob.into();
                    prob * prob.log2()
                })
            })
            .sum::<F>();

        let max_probability = Probability::max_value() >> (Probability::BITS - Self::PRECISION);
        F::from(Self::PRECISION).unwrap() - entropy_scaled / (max_probability.into() + F::one())
    }
}

impl<Probability: BitArray, const PRECISION: usize> DiscreteDistribution
    for Categorical<Probability, PRECISION>
{
    type Probability = Probability;
    const PRECISION: usize = PRECISION;
    type Symbol = usize;

    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<usize>,
    ) -> Result<(Self::Probability, Self::Probability), ()> {
        let index: usize = *symbol.borrow();

        let (cdf, next_cdf) = unsafe {
            // SAFETY: we perform a single check if index is within bounds.
            if index as usize + 1 >= self.cdf.len() {
                return Err(());
            }
            (
                *self.cdf.get_unchecked(index as usize),
                *self.cdf.get_unchecked(index as usize + 1),
            )
        };

        Ok((cdf, next_cdf.wrapping_sub(&cdf)))
    }

    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Self::Symbol, Self::Probability, Self::Probability) {
        let max_probability =
            Self::Probability::max_value() >> (Self::Probability::BITS - Self::PRECISION);
        // This check should usually compile away in inlined and verifiably correct usages
        // of this method.
        assert!(quantile <= max_probability);

        let mut left = 0; // Smallest possible index.
        let mut right = self.cdf.len() - 1; // One above largest possible index.

        // Binary search for the last entry of `self.cdf` that is <= quantile,
        // exploiting the following facts:
        // - `self.cdf.len() >= 2` (therefore, `left < right` initially)
        // - `self.cdf[0] == 0`
        // - `quantile <= max_probability`
        // - `*self.cdf.last().unwrap() == max_probability.wrapping_add(1)`
        // - `self.cdf` is monotonically nondecreasing except that it may wrap around
        //   only at the very last entry (this happens iff `PRECISION == Self::Probability::BITS`).
        //
        // The loop maintains the following two invariants:
        // (1) `0 <= left <= mid < right < self.cdf.len()`
        // (2) `cdf[left] <= cdf[mid]`
        // (3) `cdf[mid] <= cdf[right]` unless `right == cdf.len() - 1`
        while left + 1 != right {
            let mid = (left + right) / 2;

            // SAFETY: safe by invariant (1)
            let pivot = unsafe { *self.cdf.get_unchecked(mid) };
            if pivot <= quantile {
                // Since `mid < right` and wrapping can occur only at the last entry,
                // `pivot` has not yet wrapped around
                left = mid;
            } else {
                right = mid;
            }
        }

        // SAFETY: invariant `0 <= left < right < self.cdf.len()` still holds.
        let cdf = unsafe { *self.cdf.get_unchecked(left) };
        let next_cdf = unsafe { *self.cdf.get_unchecked(right) };

        (left, cdf, next_cdf.wrapping_sub(&cdf))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use statrs::distribution::Normal;

    #[test]
    fn leaky_quantized_normal() {
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-127..=127);

        for &std_dev in &[0.0001, 0.1, 3.5, 123.45, 1234.56] {
            for &mean in &[-300.6, -100.2, -5.2, 0.0, 50.3, 180.2, 2000.0] {
                let continuous_distribution = Normal::new(mean, std_dev).unwrap();
                test_discrete_distribution(quantizer.quantize(continuous_distribution), -127..128);
            }
        }
    }

    /// Test that `optimal_weights` reproduces the same distribution when fed with an
    /// already quantized distribution.
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
            Categorical::<_, 32>::from_floating_point_probabilities(&probabilities).unwrap();
        let weights: Vec<u32> = categorical.fixed_point_probabilities().collect();

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
            Categorical::<_, 32>::from_floating_point_probabilities(&probabilities).unwrap();
        let weights: Vec<u32> = categorical.fixed_point_probabilities().collect();

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
    fn categorical() {
        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();

        let distribution =
            Categorical::<_, 32>::from_floating_point_probabilities(&probabilities).unwrap();
        test_discrete_distribution(distribution, 0..probabilities.len());
    }

    fn test_discrete_distribution<D: DiscreteDistribution<Probability = u32>>(
        distribution: D,
        domain: std::ops::Range<D::Symbol>,
    ) where
        D::Symbol: Copy + std::fmt::Debug + PartialEq,
        std::ops::Range<D::Symbol>: Iterator<Item = D::Symbol>,
    {
        let mut sum = 0;

        for symbol in domain {
            let (left_cumulative, prob) = distribution
                .left_cumulative_and_probability(symbol)
                .unwrap();
            assert_eq!(left_cumulative as u64, sum);
            sum += prob as u64;

            let expected = (symbol, left_cumulative, prob);
            assert_eq!(distribution.quantile_function(left_cumulative), expected);
            assert_eq!(distribution.quantile_function((sum - 1) as u32), expected);
            assert_eq!(
                distribution.quantile_function(left_cumulative + prob / 2),
                expected
            );
        }

        assert_eq!(sum, 1 << D::PRECISION);
    }
}
