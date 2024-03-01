//! Experimental quantization methods.
//!
//! This module is a work in progress that will provide fast quantization methods that go beyond
//! simple rounding to a uniform grid. Quantization is mostly necessary for lossy compression of
//! continuous (i.e., floating point) data. However, quantization can also arise in lossless
//! compression for discrete data if one uses a latent variable model with continuous latents. Also,
//! the quantization methods provided in this module are generic over data types and not restricted
//! to floating point data as long as a distortion metric exists.
//!
//! ## The Quantization Problem
//!
//! Quantizing a collection of points involves solving three optimization problems:
//! 1. finding a good discrete set of *grid points* to which we quantize the original points;
//! 2. mapping each point to a grid point that is close (in some distortion metric) to the original
//!    point; and
//! 3. making sure the collection of all quantized points has low entropy.
//!
//! For optimal performance, all three quantization techniques should ideally be optimized jointly.
//! However, this is difficult to do in practice, and various quantization techniques make different
//! tradeoffs. For example, the simplest form of quantization rounds to the nearest neighbor in a
//! uniform grid, which neglects optimization problems 1 and 3. On the other hand, various vector
//! quantization methods focus on optimization problem 1, but they often neglect problem 3, and they
//! often use a rather simple distortion metric (e.g., Euclidean distance) for optimization
//! problem 2. [Variational Bayesian Quantization (VBQ)] uses a more advanced distortion metric, and
//! it addresses optimization problem 3 through a rate/distortion optimization, where the "rate"
//! term is a somewhat heuristic approximation of the entropy of the quantized points. However, the
//! original version of VBQ addresses optimization problem 1 rather poorly, especially in the
//! low-bit rate regime.
//!
//! ## Module Structure
//!
//! The core of this module is the type [`EmpiricalDistribution`], which is a dynamic data structure
//! that can be used to represent the empirical frequencies of values in a collection of points that
//! one wants to quantize. Quantization methods use an `EmpiricalDistribution` to inform the
//! (original) positioning of grid points (optimization problem 1 above), and to quickly estimate
//! how many points out of the collection will be mapped to each grid point (for optimization
//! problem 3 above). The actual APIs are more generic in that they accept not only an
//! `EmpiricalDistribution` but instead any type that implements, depending on context, either the
//! trait [`UnnormalizedDistribution`] or [`UnnormalizedInverse`]. These traits are both implemented
//! by `EmpiricalDistribution`, but also by parametric distributions defined in the [`probability`]
//! crate (see traits [`Distribution`](crate::stream::model::Distribution) and
//! [`Inverse`](crate::stream::model::Inverse), which are the same traits used for entropy coding
//! with a [LeakyQuantizer](crate::stream::model::LeakyQuantizer)).
//!
//! Currently, the only implemented quantization method is Variational Bayesian Quantization (see
//! type [`Vbq`]). But the plan for future versions of `constriction` is to add other
//! quantization methods, and to design them modular enough so that they can be combined in order
//! to address all three of the above optimization problems. How to achieve this modularity is an
//! ongoing research question, which is why the APIs in this module are less stable than the APIs in
//! the [`stream`](crate::stream) and [`symbol`](crate::symbol) modules, and will likely change
//! significantly in future versions of `constriction` (in accordance with SemVer guarantees).
//!
//! [Variational Bayesian Quantization (VBQ)]: http://proceedings.mlr.press/v119/yang20a/yang20a.pdf

#[cfg(not(feature = "benchmark-internals"))]
mod augmented_btree;

#[cfg(feature = "benchmark-internals")]
pub mod augmented_btree;

use core::{
    borrow::Borrow,
    convert::TryFrom,
    fmt::{Debug, Display},
    hash::Hash,
    ops::{Add, Mul, Sub},
};

use alloc::{collections::BTreeMap, vec::Vec};
use num_traits::{AsPrimitive, Bounded, One, Zero};

use crate::{UnwrapInfallible, F32};

use self::augmented_btree::AugmentedBTree;

/// A trait for unnormalized distributions over ordered data
///
/// This trait is similar to the [`Distribution`](crate::stream::model::Distribution) trait
/// re-exported from the [`probability`] crate, but more generic. Firstly, a distribution that
/// implements this trait does not need to be (explicitly) normalized (i.e., the cumulative
/// distribution method [`cdf`] may range from zero to any positive value, not necessarily one).
/// Secondly, because the distribution does not need to be normalized, quantiles (i.e., return
/// values of the method `cdf`) are not restricted to floating point numbers; they may be integers
/// if more appropriate for the use case (see, e.g., [`EmpiricalDistribution`]).
///
/// # Compatibility With Normalized Distributions
///
/// We provide a [blanket implementation](#impl-UnnormalizedDistribution-for-T) for (normalized)
/// distributions from the [`probability`] crate.
///
/// # See also
///
/// - [`UnnormalizedInverse`]
///
/// [`cdf`]: UnnormalizedDistribution::cdf
pub trait UnnormalizedDistribution {
    /// The type of outcomes, analogous to [`probability::distribution::Distribution::Value`].
    type Value: Copy;

    /// The type used to represent unnormalized (probability) mass.
    ///
    /// This type is called "`Count`" since [`EmpiricalDistribution`] uses this type to represent
    /// empirical counts of values, typically using an integer type. But `Count` is not limited to
    /// integer types (e.g., the [blanket implementation](#impl-UnnormalizedDistribution-for-T) for
    /// normalized distributions from the `probability` crate sets `Count = f64` and uses it to
    /// represent probabilities and quantiles in the interval `[0, 1]`).
    type Count: Copy + num_traits::Num;

    /// Returns the normalization constant.
    ///
    /// This has to be larger or equal to the largest value that [`cdf`] can return (if it is larger
    /// than the largest possible return value of `cdf`, then difference between the two is the
    /// (unnormalized) mass that the distribution puts on the highest value representable by
    /// `Self::Value`).
    ///
    /// [`cdf`]: UnnormalizedDistribution::cdf
    fn total(&self) -> Self::Count;

    /// Calculates the cumulative distribution function (CDF).
    ///
    /// For discrete distributions, it is recommended to return the *left-sided* CDF, i.e., the
    /// density or mass *strictly below* `x`. However, adapters of externally implemented
    /// distributions may not be able to control whether this method returns the left-sided or
    /// right-sided CDF, or something in-between. Therefore, algorithms must not rely on the
    /// assumption that this method returns the left-sided CDF (especially not for memory safety),
    /// and they should still work reasonably well if this method returns the right-sided CDF.
    /// (What "reasonably well" means in this context depends on the algorithm; for example, the
    /// type [`Vbq`] still works exactly as expected if `cdf` returns the right-sided CDF as
    /// long as [`UnnormalizedInverse::ppf`] behaves accordingly; and if `cdf` and `ppf` follow
    /// opposite conventions then `vbq` still terminates and quantizes to a grid point from the
    /// provided `prior`, but the quantization might be biased towards rounding up or down).
    ///
    /// # Example
    ///
    /// See [example for `EmpiricalDistribution`](EmpiricalDistribution#lookup).
    fn cdf(&self, x: Self::Value) -> Self::Count;
}

impl<T> UnnormalizedDistribution for T
where
    T: probability::distribution::Distribution,
    T::Value: num_traits::AsPrimitive<f64>,
{
    type Value = T::Value;
    type Count = f64;

    fn total(&self) -> Self::Count {
        1.0
    }

    /// Note for discrete distributions that this might not be the *left-sided* cdf, see discussion
    /// in the [trait documentation](UnnormalizedDistribution::cdf).
    fn cdf(&self, x: Self::Value) -> Self::Count {
        self.distribution(x.as_())
    }
}

/// A trait for unnormalized distributions over ordered data whose CDF can be inverted
///
/// This trait is similar to the [`Inverse`](crate::stream::model::Inverse) trait re-exported from
/// the [`probability`] crate, but more generic (see discussion in [`UnnormalizedDistribution`]).
///
/// # Compatibility With Normalized Distributions
///
/// We provide a [blanket implementation](#impl-UnnormalizedInverse-for-T) for (normalized)
/// distributions from the [`probability`] crate.
pub trait UnnormalizedInverse: UnnormalizedDistribution {
    /// Returns the inverse of [`cdf`](UnnormalizedDistribution::cdf).
    ///
    /// For discrete distributions, it is recommended to return the right-sided inverse of the
    /// left-sided CDF, i.e., the right-most position where the left-sided CDF is smaller than or
    /// equal to the argument `cum`. However, adapters of externally implemented distributions may
    /// not be able to guarantee this convention, see discussion in
    /// [`cdf`](UnnormalizedDistribution::cdf).
    ///
    /// Returns `None` if `cum < 0` or `cum > self.total()`. If this method follows the precise
    /// recommended convention described above (i.e., if it is the right-sided inverse of the
    /// left-sided CDF), then it should return `None` also for `cum == self.total()`. In all of
    /// these cases, there is no clear right-most position where the left-sided CDF is smaller than
    /// or equal to the argument `cum`, either because (i) no position satisfies the criterion (in
    /// the case of `cum < 0`) or (ii) because all positions satisfy the criterion (in the case of
    /// `cum >= self.total()`), and so the right-most one would just be the largest number that can
    /// be represented by `Self::Value` (e.g., `f32::INFINITY`), which is probably not what callers
    /// would expect.
    ///
    /// # Precise Relation to [`cdf`](UnnormalizedDistribution::cdf)
    ///
    /// For discrete (unnormalized) distributions that follow the recommended convention, and for
    /// continuous (unnormalized) distributions, the following two relations hold :
    ///
    /// - `self.cdf(self.ppf(self.cdf(pos)).unwrap()) == self.cdf(pos)` (assuming that `.unwrap()`
    ///   succeeds).
    /// - `self.ppf(self.cdf(self.ppf(cum).unwrap())).unwrap()) == self.ppf(cum).unwrap()`
    ///   (assuming that the inner `.unwrap()` on the left-hand side succeeds—in which case the
    ///   outer `.unwrap()` is guaranteed to succeed).
    ///
    /// # Example
    ///
    /// See [example for `EmpiricalDistribution`](EmpiricalDistribution#lookup).
    fn ppf(&self, cum: Self::Count) -> Option<Self::Value>;
}

impl<T> UnnormalizedInverse for T
where
    T: probability::distribution::Inverse,
    T::Value: num_traits::AsPrimitive<f64>,
{
    /// Note for discrete distributions that this might not follow the recommended convention
    /// described in the [trait documentation](UnnormalizedInverse::ppf).
    fn ppf(&self, cum: Self::Count) -> Option<Self::Value> {
        if (0.0..=1.0).contains(&cum) {
            Some(self.inverse(cum))
        } else {
            None
        }
    }
}

/// A trait for distributions and grids that can be created from an empirical collection of points.
pub trait FromPoints<V, C>
where
    Self: Sized,
    V: Copy,
    C: Copy,
{
    /// Creates a distribution based on a collection of points.
    ///
    /// If the distribution is over floating point values, then you might be better off with
    /// [`try_from_points`](Self::try_from_points).
    fn from_points<I>(points: I) -> Self
    where
        I: IntoIterator,
        <I as IntoIterator>::Item: Borrow<V>,
    {
        Self::try_from_points(points.into_iter().map(|point| *point.borrow())).unwrap_infallible()
    }

    /// Fallible variant of [`from_points`](Self::from_points).
    ///
    /// This is mostly intended for `EmpiricalDistribution`s over floating point values, which have
    /// to be converted to [`NonNanFloat`](crate::NonNanFloat) before being inserted. See example in
    /// the documentation of [`EmpiricalDistribution`].
    fn try_from_points<F>(
        points: impl IntoIterator<Item = F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>;

    /// Reconstructs a distribution from a compact representation
    ///
    /// This can, e.g., be used for serialization / deserialization of a type that also implements
    /// [`ToPointsAndCounts`]. In this case, the argument `points_and_counts` can be an iterator
    /// returned by [`ToPointsAndCounts::points_and_counts_iter`] (but it does not have to iterate
    /// over points in the same order that that iterator would).
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{F32, quant::{EmpiricalDistribution, FromPoints}};
    ///
    /// let points = [0.1, -0.3, 1.5, -0.3, 4.2, 1.5, 1.5];
    /// let original_distribution = EmpiricalDistribution::<F32, u32>::try_from_points(
    ///     points.iter().copied()
    /// ).unwrap();
    ///
    /// let original_entropy = original_distribution.entropy_base2::<f32>();
    /// assert!((original_entropy - 1.842371).abs() < 1e-6);
    ///
    /// let (points, counts): (Vec<F32>, Vec<u32>) = original_distribution.iter().unzip();
    ///
    /// // ... save `points` and `counts` to a file and load them back later ...
    ///
    /// let reconstructed_distribution = EmpiricalDistribution::<F32, u32>::from_points_and_counts(
    ///     points.iter().zip(&counts)
    /// );
    /// assert_eq!(reconstructed_distribution.entropy_base2::<f32>(), original_entropy);
    /// ```
    ///
    /// ## See also
    ///
    /// - [`Self::try_from_points_and_counts`]
    fn from_points_and_counts<I, VV, CC>(points_and_counts: I) -> Self
    where
        I: IntoIterator<Item = (VV, CC)>,
        VV: Borrow<V>,
        CC: Borrow<C>,
    {
        Self::try_from_points_and_counts(
            points_and_counts
                .into_iter()
                .map(|(point, count)| (*point.borrow(), *count.borrow())),
        )
        .unwrap_infallible()
    }

    /// Fallible variant of [`from_points_and_counts`](Self::from_points_and_counts).
    ///
    /// This is mostly intended for `EmpiricalDistribution`s over floating point values, which have
    /// to be converted to [`NonNanFloat`](crate::NonNanFloat) before being inserted.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{F32, quant::{EmpiricalDistribution, FromPoints}};
    ///
    /// let points = [0.1, -0.3, 1.5, -0.3, 4.2, 1.5, 1.5];
    /// let original_distribution = EmpiricalDistribution::<F32, u32>::try_from_points(
    ///     points.iter().copied()
    /// ).unwrap();
    ///
    /// let original_entropy = original_distribution.entropy_base2::<f32>();
    /// assert!((original_entropy - 1.842371).abs() < 1e-6);
    ///
    /// let (points, counts): (Vec<f32>, Vec<u32>) = original_distribution
    ///     .iter()
    ///     .map(|(point, count)| (point.get(), count))
    ///     .unzip();
    ///
    /// // ... save `points` and `counts` to a file and load them back later ...
    ///
    /// let reconstructed_distribution = EmpiricalDistribution::<F32, u32>::try_from_points_and_counts(
    ///     points.iter().copied().zip(counts.iter().copied())
    /// ).unwrap();
    /// assert_eq!(reconstructed_distribution.entropy_base2::<f32>(), original_entropy);
    /// ```
    fn try_from_points_and_counts<F>(
        points_and_counts: impl IntoIterator<Item = (F, C)>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>;
}

pub trait ToPointsAndCounts<V, C> {
    fn points_and_counts_iter(&self) -> impl Iterator<Item = (V, C)> + '_;
}

pub trait DynamicDistribution<V, C> {
    type InsertSuccess;
    type InsertError;

    /// Inserts `count` points with the provided `value` into the distribution.
    fn insert(&mut self, value: V, count: C) -> Result<Self::InsertSuccess, Self::InsertError>;

    /// Removes `count` points with the provided `value`.
    ///
    /// `count` should be nonzero. If `count` is zero, then the distribution remains unchanged but
    /// whether the method returns `Ok(..)` or `Err(..)` is unspecified and may depend not only of
    /// the contents but even of the history of the distribution.
    ///
    /// On success, returns `Ok(original_count)`, where `original_count >= 1` is the number of
    /// points with the provided `value` that were present *before* the removal (thus, once the
    /// method returns, there are only `original_count - 1` points with the provided `value` left).
    ///
    /// Returns `Err(NotFoundError)` if removal fails because there aren't enough points with the
    /// provided `value` (in this case, the `distribution remains unchanged).
    fn remove(&mut self, value: V, count: C) -> Result<C, NotFoundError>;

    /// Removes *all* points with the provided `value`.
    ///
    /// Returns the number of removed points (which can be zero).
    fn remove_all(&mut self, value: V) -> C;
}

/// A trait for quantization methods that quantize to a given `Grid` type.
///
/// The `Grid` type can be either an explicitly defined grid such as [`RatedGrid`], or it can be a
/// type that provides the quantization method with the required information to implicitly define a
/// quantization grid distribution (e.g., a distribution that implements [`UnnormalizedInverse`] so
/// it can be used as a prior in [`Vbq`] to implicitly define the placement of grid points).
pub trait QuantizationMethod<Grid, V, L> {
    /// Quantize a given `unquantized` point to the provided `grid`.
    ///
    /// The quantization method should minimize the rate/distortion tradeoff
    ///
    /// `distortion(unquantized, quantized) + bit_penalty * rate_estimate(quantized)`
    ///
    /// where `rate_estimate` is some estimate of the information content of the candidate
    /// quantization point. How this estimate is defined depends on the implementor, and the range
    /// of `quantized` points over which the optimization runs depends on the provided `grid`.
    fn quantize(
        &self,
        unquantized: V,
        grid: &Grid,
        distortion: impl Fn(V, V) -> L,
        bit_penalty: L,
    ) -> V;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CountWrapper<C>(C);

impl<C: Display> Display for CountWrapper<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl<C: num_traits::Num> Default for CountWrapper<C> {
    #[inline(always)]
    fn default() -> Self {
        Self(C::zero())
    }
}

impl<C: PartialOrd> PartialOrd for CountWrapper<C> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<C: Ord> Ord for CountWrapper<C> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<C: Add<Output = C>> Add for CountWrapper<C> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        CountWrapper(self.0.add(rhs.0))
    }
}

impl<C: Sub<Output = C>> Sub for CountWrapper<C> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        CountWrapper(self.0.sub(rhs.0))
    }
}

/// Dynamic data structure for frequencies of values within a finite set of points.
///
/// An `EmpiricalDistribution` counts how many times each value appears within a finite set of
/// points, and it provides efficient access to the cumulative and inverse cumulative distribution
/// functions (by implementing [`UnnormalizedDistribution`] and [`UnnormalizedInverse`],
/// respectively). `EmpiricalDistribution` is a dynamic data structure, i.e., it provides methods to
/// efficiently [remove](Self::remove) existing points and [insert](Self::insert) new points.
///
/// # Examples
///
/// ## Construction And Iteration
///
/// The following example constructs an `EmpiricalDistribution` over a set of floating point
/// numbers. Since the value type `V` has to be totally ordered (i.e., it has to implement [`Ord`]),
/// we have to wrap floating point numbers in a [`NonNanFloat`](crate::NonNanFloat), which fails if
/// we encounter `NaN`.
///
/// ```
/// use constriction::{F32, quant::{EmpiricalDistribution, FromPoints}};
///
/// let points = [0.1, -0.3, 1.5, -0.3, 4.2, 1.5, 1.5];
/// let distribution = EmpiricalDistribution::<F32, u32>::try_from_points(
///     points.iter().copied()
/// ).unwrap();
///
/// // Iterate over entries sorted by value:
/// let points_and_counts = distribution
///     .iter()
///     .map(|(point, count)| (point.get(), count))
///     .collect::<Vec<_>>();
/// assert_eq!(points_and_counts, [(-0.3, 2), (0.1, 1), (1.5, 3), (4.2, 1)]);
///
/// // The following fails because `points2` contains `NaN`:
/// let points2 = [0.1, -0.3, 0.0 / 0.0 /* <-- NaN */, -0.3, 4.2, 1.5, 1.5];
/// let distribution2 = EmpiricalDistribution::<F32, u32>::try_from_points(points2.iter().copied());
/// assert_eq!(distribution2.unwrap_err(), constriction::NanError);
/// ```
///
/// ## Lookup
///
/// ```
/// use constriction::{
///     F32,
///     quant::{EmpiricalDistribution, FromPoints, UnnormalizedDistribution, UnnormalizedInverse}
/// };
///
/// let points = [0.1, -0.3, 1.5, -0.3, 4.2, 1.5, 1.5];
/// let distribution = EmpiricalDistribution::<F32, u32>::try_from_points(
///     points.iter().copied()
/// ).unwrap();
/// // `distribution` contains: -0.3 (2x), 0.1 (1x), 1.5 (3x), and 4.2 (1x)
///
/// assert_eq!(distribution.total(), 7); // Total number of (not necessarily distinct) entries.
///
/// // Look up values in the cumulative distribution function (CDF):
/// assert_eq!(distribution.cdf(F32::new(-1.0).unwrap()), 0);
/// assert_eq!(distribution.cdf(F32::new(1.0).unwrap()), 3);
/// assert_eq!(distribution.cdf(F32::new(1.5).unwrap()), 3); // CDF is left-sided
/// assert_eq!(distribution.cdf(F32::new(1.50001).unwrap()), 6); // CDF is left-sided
/// assert_eq!(distribution.cdf(F32::new(100.0).unwrap()), 7);
///
/// // Look up values in the inverse of the CDF (also known as percent point function, PPF):
/// assert_eq!(distribution.ppf(0), Some(F32::new(-0.3).unwrap()));
/// assert_eq!(distribution.ppf(1), Some(F32::new(-0.3).unwrap()));
/// assert_eq!(distribution.ppf(2), Some(F32::new(0.1).unwrap()));
/// assert_eq!(distribution.ppf(6), Some(F32::new(4.2).unwrap()));
/// assert_eq!(distribution.ppf(7), None); // See documentation of `UnnormalizedInverse::ppf`.
/// ```
///
/// # Runtime Complexity
///
/// For an `EmpiricalDistribution<V, C, CAP>` over `n` *distinct* entries, we have:
///
/// - lookups for both the [cumulative distribution function (CDF)](UnnormalizedDistribution::cdf)
///   and the [inverse CDF](UnnormalizedInverse::ppf) take `O(log n)` time; and
/// - [inserts](Self::insert) and [removals](Self::remove) take `O(CAP + log n)` time (where
///   `CAP=64` by default).
///
/// # Details
///
/// ## Const Generic Parameter `CAP`
///
/// The const generic `CAP: usize` controls the size of contiguous memory allocations. We statically
/// enforce `CAP >= 4`. The current implementation uses an augmented B-tree whose nodes have a
/// capacity to hold at most `CAP` entries (and which never hold fewer than `CAP / 2` entries except
/// if `n < CAP / 2`). The default value for `CAP` was chosen by optimizing for speed in relatively
/// simple benchmarks, but it was not tuned very heavily since performance appeared relatively
/// stable over a range of values for CAP. Changing `CAP` hardly affects total memory consumption
/// (except if `n < CAP`) but it affects the number of memory allocations, memory fragmentation, and
/// the overhead for inserting and removing entries, see runtime complexities stated above.
///
/// ## Floating Point Edge Cases
///
/// `EmpiricalDistribution<V, C, CAP>` requires the value type `V` to be totally ordered (i.e., `V`
/// has to implement [`Ord`]). Unfortunately, the floating point types `f32` and `f64` only
/// implement [`PartialOrd`] but not `Ord` because the possibility of having `NaN` values breaks
/// total order. To use an `EmpiricalDistribution` over float types, the floats have to be wrapped
/// in a [`NonNanFloat`](crate::NonNanFloat), see [example above](Self#construction-and-iteration).
/// The wrapper `NonNanFloat` implements `Ord` because it ensures that it never wraps a `NaN` value.
///
/// In addition to `NaN`, there are a few more edge cases with floating point numbers:
/// - **zero:** like the comparison operators in rust, `EmpiricalDistribution` considers positive
///   and negative zero equal. So if we insert both positive and a negative zero values into an
///   `EmpiricalDistribution`, the distribution lumps them together and keeps track of only their
///   total count. We can [`remove`](Self::remove) zeros by providing either positive or negative
///   zero as a key, regardless of which variant was inserted how many times. See code below.
/// - **subnormal numbers:** no special treatment, see code below.
/// - **infinity:** technically not treated special, but infinities might still lead to surprising
///   behavior (e.g., if an `EmpiricalDistribution` contains some positive infinite values, then,
///   by definition of the left-sided CDF, [`UnnormalizedDistribution::cdf`] always returns a
///   cumulative that is strictly smaller than [`UnnormalizedDistribution::total`]).
///
/// Examples of edge cases involving floating point zeros and subnormal numbers:
///
/// ```
/// use constriction::{
///     F32, quant::{DynamicDistribution, EmpiricalDistribution, UnnormalizedDistribution,
///     UnnormalizedInverse}
/// };
///
/// let positive_zero = 0.0_f32;
/// let negative_zero = -0.0_f32;
/// assert!(positive_zero.to_bits() != negative_zero.to_bits()); // They have different bit patterns
/// assert!(positive_zero == negative_zero); // ... but are still specified to compare as equal.
/// assert!(!positive_zero.is_subnormal());  // Zero is *not* considered subnormal.
/// assert!(!negative_zero.is_subnormal());
///
/// let positive_subnormal1 = 1.0e-40_f32;
/// let positive_subnormal2 = 4.0e-40_f32;
/// let negative_subnormal = -1.0e-40_f32;
/// assert!(positive_subnormal1.is_subnormal());
/// assert!(positive_subnormal2.is_subnormal());
/// assert!(negative_subnormal.is_subnormal());
///
/// // Insert positive and negative zero and some subnormal numbers into an `EmpiricalDistribution`:
/// let mut distribution = EmpiricalDistribution::<F32, u32>::new();
/// distribution.insert(F32::new(positive_zero).unwrap(), 1);
/// distribution.insert(F32::new(negative_zero).unwrap(), 1);
/// distribution.insert(F32::new(positive_subnormal1).unwrap(), 1);
/// distribution.insert(F32::new(positive_subnormal2).unwrap(), 1);
/// distribution.insert(F32::new(negative_subnormal).unwrap(), 1);
///
/// // Inspect how the inserted values are stored in the `EmpiricalDistribution`:
/// let values_and_counts = distribution.iter().collect::<Vec<_>>();
/// assert_eq!(
///     values_and_counts,
///     [
///         (F32::new(negative_subnormal).unwrap(), 1),
///         (F32::new(0.0).unwrap(), 2), // Positive and negative zero get lumped together.
///         (F32::new(positive_subnormal1).unwrap(), 1), // But different subnormal numbers
///         (F32::new(positive_subnormal2).unwrap(), 1), // are distinguished.
///     ]
/// );
///
/// // We got back the variant of zero that was inserted first:
/// assert_eq!(values_and_counts[1].0.get().to_bits(), positive_zero.to_bits());
///
/// // Lookups in the CDF don't distinguish between positive and negative zero:
/// assert_eq!(distribution.cdf(F32::new(negative_zero).unwrap()), 1);
/// assert_eq!(distribution.cdf(F32::new(positive_zero).unwrap()), 1);
///
/// // But they do distinguish the tiny differences between subnormal numbers:
/// assert_eq!(distribution.cdf(F32::new(positive_subnormal1).unwrap()), 3);
/// assert_eq!(distribution.cdf(F32::new(2e-40_f32).unwrap()), 4);
/// assert_eq!(distribution.cdf(F32::new(positive_subnormal2).unwrap()), 4);
/// assert_eq!(distribution.cdf(F32::new(5e-40_f32).unwrap()), 5);
///
/// // Lookups in the PPF return the variant of zero that was inserted first.
/// assert_eq!(distribution.ppf(0).unwrap().get(), negative_subnormal);
/// assert_eq!(distribution.ppf(1).unwrap().get().to_bits(), positive_zero.to_bits());
/// assert_eq!(distribution.ppf(2).unwrap().get().to_bits(), positive_zero.to_bits());
/// assert_eq!(distribution.ppf(3).unwrap().get(), positive_subnormal1);
/// assert_eq!(distribution.ppf(4).unwrap().get(), positive_subnormal2);
/// assert_eq!(distribution.ppf(5), None);
///
/// // When removing a zero value, we can use either positive or negative zero as the key:
/// assert_eq!(distribution.remove(F32::new(positive_zero).unwrap(), 1), Ok(2));
/// assert_eq!(distribution.remove(F32::new(negative_zero).unwrap(), 1), Ok(1));
/// assert!(distribution.remove(F32::new(positive_zero).unwrap(), 1).is_err());
/// ```
pub struct EmpiricalDistribution<V = F32, C = u32, const CAP: usize = 64>(
    AugmentedBTree<V, CountWrapper<C>, CAP>,
);

impl<V, C, const CAP: usize> EmpiricalDistribution<V, C, CAP>
where
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num,
{
    /// Creates an empty `EmpiricalDistribution`.
    ///
    /// This currently allocates one (empty) tree node (which is different from how
    /// [`Vec::new`](alloc::vec::Vec::new) operates).
    pub fn new() -> Self {
        Self(AugmentedBTree::new())
    }

    /// Iterate over unique values in sorted order.
    ///
    /// For each unique value in the distribution, the iteration yields a pair `(value, count)`,
    /// where `count` is the number of times that `value` appears in the distribution. See
    /// [example in the struct-level documentation](Self#construction-and-iteration).
    pub fn iter(&self) -> impl Iterator<Item = (V, C)> + '_ {
        self.0.iter().scan(C::zero(), |prev_accum, (value, accum)| {
            let count = accum.0 - *prev_accum;
            *prev_accum = accum.0;
            Some((value, count))
        })
    }

    /// Calculates the entropy of the (corresponding normalized) distribution.
    ///
    /// The entropy is defined as
    ///
    /// `entropy = - sum_i( (count_i / total) * log_2(count_i / total) )`
    ///
    /// where `log_2` is the logarithm to base 2, the sum runs over all *distinct* values in the
    /// distribution, and `(count_i / total)` is the frequency with which the `i`th value appears in
    /// the distribution.
    ///
    /// The calculation (including the logarithm) is performed in the number space `F`, which may
    /// need to be specified explicitly if it cannot be inferred by the return type, see example
    /// below.
    ///
    /// # Examples
    ///
    /// A distribution over two values where each value occurs exactly half of the time has an
    /// entropy of 1 bit:
    ///
    /// ```
    /// use constriction::{F32, quant::{EmpiricalDistribution, FromPoints}};
    ///
    /// let points1 = [0.2, 3.5, 3.5, 0.2, 3.5, 0.2];
    /// let distribution1 = EmpiricalDistribution::<F32, u32>::try_from_points(
    ///     points1.iter().copied()
    /// ).unwrap();
    /// assert!((distribution1.entropy_base2::<f32>() - 1.0).abs() < 1e-6);
    /// ```
    ///
    /// More generally, a uniform distribution over `2^n` values has an entropy of `n` bit. If the
    /// distribution is only approximately uniform then its entropy is slightly lower:
    ///
    /// ```
    /// # use constriction::{F32, quant::{EmpiricalDistribution, FromPoints}};
    /// let points2 = [0.1, 0.2, 0.3, 0.4, 0.1, 0.3, 0.4]; // 4 distinct values, approximately uniform.
    /// let distribution2 = EmpiricalDistribution::<F32, u32>::try_from_points(
    ///     points2.iter().copied()
    /// ).unwrap();
    /// assert!((distribution2.entropy_base2::<f32>() - 1.9502121).abs() < 1e-6); // almost 2 bit.
    /// ```
    ///
    /// If the distribution is very non-uniform (i.e., strongly peaked) then its entropy can be
    /// significantly lower. For very peaked distributions, the entropy can even drop below 1 bit,
    /// and this can happen even with distributions that have support on  more than two distinct
    /// values:
    ///
    /// ```
    /// # use constriction::{F32, quant::{EmpiricalDistribution, FromPoints}};
    /// let points3 = [0.2, 0.2, 0.2, 0.2, 10.5, 0.2, 0.2, 0.2, 0.2, -20.8, 0.2, 0.2];
    /// let distribution3 = EmpiricalDistribution::<F32, u32>::try_from_points(
    ///     points3.iter().copied()
    /// ).unwrap();
    /// assert!((distribution3.entropy_base2::<f32>() - 0.8166891).abs() < 1e-6); // less than 1 bit.
    /// ```
    pub fn entropy_base2<R>(&self) -> R
    where
        R: num_traits::real::Real + 'static,
        C: num_traits::AsPrimitive<R>,
    {
        let mut sum_count_log_count = R::zero();
        for (_value, count) in self.iter() {
            let count = count.as_();
            sum_count_log_count = sum_count_log_count + count * count.log2()
        }

        let total = self.total().as_();
        total.log2() - sum_count_log_count / total
    }

    pub fn static_rated_grid<R>(&self) -> StaticRatedGrid<V, R>
    where
        R: num_traits::real::Real + 'static,
        C: num_traits::AsPrimitive<R>,
    {
        self.into()
    }

    pub fn dynamic_rated_grid<R>(&self) -> DynamicRatedGrid<V, R, C>
    where
        R: num_traits::real::Real + 'static,
        C: num_traits::AsPrimitive<R>,
    {
        self.into()
    }

    /// Quantizes a given value using [Variational Bayesian Quantization (VBQ)].
    ///
    /// This is a convenience method that just forwards to `Vbq::quantize`, using `self` as the
    /// `grid` argument. See documentation of [`Vbq`].
    ///
    /// [Variational Bayesian Quantization (VBQ)]: http://proceedings.mlr.press/v119/yang20a/yang20a.pdf
    pub fn vbq<L>(&self, unquantized: V, distortion: impl Fn(V, V) -> L, bit_penalty: L) -> V
    where
        C: PartialOrd + Bounded + 'static,
        L: Copy + PartialOrd + Zero + Bounded,
    {
        Vbq.quantize(unquantized, self, distortion, bit_penalty)
    }
}

impl<V, C, const CAP: usize> FromPoints<V, C> for EmpiricalDistribution<V, C, CAP>
where
    Self: Default,
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num,
{
    fn try_from_points<F>(
        points: impl IntoIterator<Item = F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>,
    {
        let mut this = Self::default();
        for point in points {
            this.0.insert(V::try_from(point)?, CountWrapper(C::one()))
        }
        Ok(this)
    }

    fn try_from_points_and_counts<F>(
        points_and_counts: impl IntoIterator<Item = (F, C)>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>,
    {
        let mut this = Self::default();
        for (point, count) in points_and_counts {
            this.0.insert(V::try_from(point)?, CountWrapper(count))
        }
        Ok(this)
    }
}

impl<V, C, const CAP: usize> ToPointsAndCounts<V, C> for EmpiricalDistribution<V, C, CAP>
where
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num,
{
    fn points_and_counts_iter(&self) -> impl Iterator<Item = (V, C)> + '_ {
        self.iter()
    }
}

impl<V, C, const CAP: usize> DynamicDistribution<V, C> for EmpiricalDistribution<V, C, CAP>
where
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num,
{
    type InsertSuccess = ();
    type InsertError = core::convert::Infallible;

    /// If the distribution already has some point(s) with the same `value`, then no allocation
    /// is required and only the count for `value` is increased. Otherwise, a new entry with the
    /// provided `count` is inserted.
    ///
    /// If `count` is zero then this is a noop.
    fn insert(&mut self, value: V, count: C) -> Result<(), Self::InsertError> {
        self.0.insert(value, CountWrapper(count));
        Ok(())
    }

    fn remove(&mut self, value: V, count: C) -> Result<C, NotFoundError> {
        self.0
            .remove(value, CountWrapper(count))
            .map(|c| c.0)
            .ok_or(NotFoundError)
    }

    fn remove_all(&mut self, value: V) -> C {
        self.0.remove_all(value).0
    }
}

impl<V, C, const CAP: usize> UnnormalizedDistribution for EmpiricalDistribution<V, C, CAP>
where
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num + 'static,
{
    type Value = V;
    type Count = C;

    fn total(&self) -> Self::Count {
        self.0.total().0
    }

    /// This is the *left-sided* CDF, as recommended in the
    /// [trait documentation](UnnormalizedDistribution::cdf).
    fn cdf(&self, x: Self::Value) -> Self::Count {
        self.0.left_cumulative(x).0
    }
}

impl<V, C, const CAP: usize> UnnormalizedInverse for EmpiricalDistribution<V, C, CAP>
where
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num + 'static,
{
    /// This method follows the recommended convention described in the
    /// [trait documentation](UnnormalizedInverse::ppf).
    fn ppf(&self, cum: C) -> Option<V> {
        self.0.quantile_function(CountWrapper(cum))
    }
}

impl<V: Display, C: Display + Debug, const CAP: usize> Debug for EmpiricalDistribution<V, C, CAP> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("DynamicEmpiricalDistribution")
            .field(&self.0)
            .finish()
    }
}

impl<V: Copy, C: Copy, const CAP: usize> Clone for EmpiricalDistribution<V, C, CAP> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<V: Copy, C: Copy, const CAP: usize> Default for EmpiricalDistribution<V, C, CAP>
where
    V: Copy + Ord,
    C: Ord + Copy + num_traits::Num,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Generalization of [`StaticRatedGrid`] and [`DynamicRatedGrid`]
pub trait RatedGrid<V, R> {
    /// Must not return an empty iterator.
    fn points_and_unnormalized_rates(&self) -> impl Iterator<Item = (V, R)>;
}

/// A finite grid of (not necessarily one-dimensional) points, each weighted with a "bit rate".
///
/// This type implements the trait [`FromPoints`], which means that it can be constructed from a
/// collection of (already quantized) points. Alternatively, a `RatedGrid` can be constructed from
/// other grid types, e.g., from an [`EmpiricalDistribution`] that has been updated while
/// performing quantization with it with [`Vbq`] so that it now tracks the distribution of quantized
/// points. In either case, the "weight" of each grid point `x` the `RatedGrid` will be set to the
/// bit rate `-log_2(f(x))` where `f(x)` is the empirical frequency of `x` in the collection of
/// points or in the distribution. The weight is thus an accurate estimate of the (amortized) bit
/// rate that each point `x` would contribute to the total bit rate if we were to compress the
/// collection of points with a [stream code](crate::stream) and an optimal entropy model.
///
/// A `RatedGrid` can be used in a [`RateDistortionQuantization`] to re-quantize the original
/// (unquantized) data to the same grid, but with potentially different mappings to grid points that
/// takes into account an estimate of the resulting rate.
///
/// # See Also
///
/// - [`DynamicRatedGrid`]
///
/// # Example
///
/// The following example constructs a `RatedGrid` over a set of floating point numbers. Since the
/// value type `V` has to be totally ordered (so we can disambiguate ties in a deterministic way) we
/// have to wrap floating point numbers in a [`NonNanFloat`](crate::NonNanFloat), which fails if we
/// encounter `NaN`.
///
/// ```
/// use constriction::{F32, quant::{StaticRatedGrid, FromPoints}};
///
/// let points = [4.2, -0.3, 1.5, -0.3, 0.1, 1.5, 1.5];
/// let grid = StaticRatedGrid::<F32, f32>::try_from_points::<_, u32>(
///     points.iter().copied()
/// ).unwrap();
///
/// // Iterate over entries sorted by rate:
/// let expected_rates = [1.222392, 1.807355, 2.807355, 2.807355]; // Sorted in ascending order.
/// let expected_grid_points = [1.5, -0.3, 0.1, 4.2]; // Sorted where rates are tied (last two).
/// for ((&(point, rate), &expected_point), &expected_rate) in grid.points_and_rates()
///     .iter().zip(&expected_grid_points).zip(&expected_rates)
/// {
///     assert_eq!(point.get(), expected_point);
///     assert!((rate - expected_rate).abs() < 1e-6);
/// }
///
/// // The following fails because `points2` contains `NaN`:
/// let points2 = [0.1, -0.3, 0.0 / 0.0 /* <-- NaN */, -0.3, 4.2, 1.5, 1.5];
/// let grid2 = StaticRatedGrid::<F32, f32>::try_from_points::<_, u32>(points2.iter().copied());
/// assert_eq!(grid2.unwrap_err(), constriction::NanError);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticRatedGrid<V = F32, R = f32> {
    /// Nonempty Vec of tuples `(grid_point, rate)`, sorted by increasing rate.
    grid: Vec<(V, R)>,
    entropy: R,
}

/// TODO: documentation with code examples
impl<V, R> StaticRatedGrid<V, R> {
    /// Shortcut for `<Self as FromPoints<V, C>>::from_points` that makes it easier to
    /// resolve the type for `C`.
    ///
    /// The type `C` is an integer type that is used temporarily to count the number of occurrences
    /// of each distinct value that occurs in the iterator `points`. It should be able to represent
    /// all counts that can occur without overflowing. Once all occurrences are counted, the counts
    /// are used to calculate the empirical frequencies `f(x)` of each value `x`, and from that the
    /// information contents `-log_2 (f(x))`, which is used as the "rate" for each grid point.
    pub fn from_points<C, I>(points: I) -> Self
    where
        Self: Sized,
        V: Copy + Ord,
        C: Copy + Ord + AsPrimitive<R> + num_traits::Num,
        R: num_traits::real::Real + 'static,
        I: IntoIterator,
        <I as IntoIterator>::Item: Borrow<V>,
    {
        <Self as FromPoints<V, C>>::from_points(points)
    }

    /// Shortcut for `<Self as FromPoints<V, C>>::try_from_points` that makes it easier
    /// to resolve the type for `C`.
    ///
    /// The integer type `C` is used temporarily to count the number of occurrences of each distinct
    /// value in the iterator `points`, see documentation of [`Self::from_points`].
    ///
    /// # Example
    ///
    /// See [struct level documentation](Self).
    pub fn try_from_points<F, C>(
        points: impl IntoIterator<Item = F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        Self: Sized,
        V: Copy + Ord + TryFrom<F>,
        C: Copy + Ord + AsPrimitive<R> + num_traits::Num,
        R: num_traits::real::Real + 'static,
    {
        <Self as FromPoints<V, C>>::try_from_points(points)
    }

    /// Returns the grid points and their associated bit rates, sorted by increasing rate.
    ///
    /// Any ties are broken by sorting by increasing grid point value.
    pub fn points_and_rates(&self) -> &[(V, R)] {
        &self.grid
    }

    /// Analogous to [`points_and_rates`](Self::points_and_rates), but consumes the `RatedGrid` and
    /// returns an owned representation.
    pub fn into_points_and_rates(self) -> Vec<(V, R)> {
        self.grid
    }

    /// Returns the entropy corresponding to the bit rates.
    ///
    /// The entropy could technically be calculated as follows,
    ///
    /// `entropy = - sum_i( 2^{-rate} * rate )`.
    ///
    /// But the current implementation instead pre-calculates it when calculating the rates in the
    /// various constructors as most of the operations to do this have to be performed there anyway.
    /// So calling `RatedGrid::entropy_base2` is cheap.
    pub fn entropy_base2(&self) -> R
    where
        R: num_traits::real::Real + 'static,
    {
        self.entropy
    }

    /// Quantizes a given value to a grid point by optimizing a rate/distortion-tradeoff.
    ///
    /// The quantization method minimizes the following objective:
    ///
    /// `quantized = arg_min[ distortion(unquantized, quantized) - bit_penalty * rate(quantized) ]`
    ///
    /// where `distortion` and `bit_penalty` are provided by the caller, `rate(quantized)` is the
    /// rate that the `RatedGrid` associates with the grid point `quantized`, and the minimization
    /// runs over all grid points `quantized`.
    ///
    /// The `distortion` function must be nonnegative for all arguments, but it does not need to be
    /// zero for `quantized == unquantized`, and it does not have to be unimodal.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{F32, quant::{StaticRatedGrid, FromPoints}};
    ///
    /// let points = [4.2, -0.3, 1.5, -0.3, 0.1, 1.5, 1.5];
    /// let grid = StaticRatedGrid::<F32, f32>::try_from_points::<_, u32>(
    ///     points.iter().copied()
    /// ).unwrap();
    ///
    /// let unquantized = F32::new(3.0).unwrap();
    /// let quantized = grid.quantize(unquantized, |x, y| ((x - y) * (x - y)).get(), 1.0);
    /// assert_eq!(quantized.get(), 1.5);
    /// ```
    ///
    /// This method can also be called in generic code through [`QuantizationMethod::quantize`].
    /// Thus, the above method call is equal to the following trait method call:
    ///
    /// ```
    /// # use constriction::{F32, quant::{StaticRatedGrid, FromPoints}};
    /// #
    /// # let points = [4.2, -0.3, 1.5, -0.3, 0.1, 1.5, 1.5];
    /// # let grid = StaticRatedGrid::<F32, f32>::try_from_points::<_, u32>(
    /// #     points.iter().copied()
    /// # ).unwrap();
    /// #
    /// # let unquantized = F32::new(3.0).unwrap();
    /// #
    /// // ... define `grid` and `unquantized` as above ...
    ///
    /// // Import the trait `QuantizationMethod` so that we can use its implementation for
    /// // `RateDistortionQuantization`.
    /// use constriction::quant::{QuantizationMethod, RateDistortionQuantization};
    ///
    /// let quantized = RateDistortionQuantization.quantize(
    ///     unquantized, &grid, |x, y| ((x - y) * (x - y)).get(), 1.0);
    /// assert_eq!(quantized.get(), 1.5);
    /// ```
    pub fn quantize(&self, unquantized: V, distortion: impl Fn(V, V) -> R, bit_penalty: R) -> V
    where
        V: Copy,
        R: Copy + PartialOrd + Zero + Bounded + Mul<Output = R>,
    {
        RateDistortionQuantization.quantize(unquantized, self, distortion, bit_penalty)
    }
}

impl<V, R> RatedGrid<V, R> for StaticRatedGrid<V, R>
where
    V: Copy,
    R: Copy,
{
    fn points_and_unnormalized_rates(&self) -> impl Iterator<Item = (V, R)> {
        self.points_and_rates().iter().copied()
    }
}

impl<V, R, C> FromPoints<V, C> for StaticRatedGrid<V, R>
where
    Self: Sized,
    V: Copy + Ord,
    C: Copy + Ord + AsPrimitive<R> + num_traits::Num,
    R: num_traits::real::Real + 'static,
{
    fn try_from_points<F>(
        points: impl IntoIterator<Item = F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>,
    {
        let mut map = BTreeMap::new();
        for point in points {
            map.entry(V::try_from(point)?)
                .and_modify(|count| *count = *count + C::one())
                .or_insert(C::one());
        }

        Ok(Self::from_points_and_counts(map))
    }

    fn try_from_points_and_counts<F>(
        points_and_counts: impl IntoIterator<Item = (F, C)>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>,
    {
        // This does not allocate if `points_and_counts> = Vec<(F, C)>` and
        // <V as TryFrom<F>>::Error = Infallible.
        let mut points_and_counts = points_and_counts
            .into_iter()
            .map(|(point, count)| Ok((V::try_from(point)?, count)))
            .collect::<Result<Vec<_>, _>>()?;

        assert!(
            !points_and_counts.is_empty(),
            "Cannot construct an empty rated grid."
        );

        // We sort by counts rather than by log-counts to avoid spurious ties due to rounding
        // errors. It's also probably faster (but requires an additional conversion below).
        points_and_counts.sort_unstable_by(|&(point1, count1), &(point2, count2)| {
            (count2, point1).cmp(&(count1, point2))
        });

        let mut total = C::zero();
        for &(_point, count) in &points_and_counts {
            total = total + count
        }
        let total = total.as_();

        // This conversion seems to directly reuse the existing allocation as long as `C` and `F`
        // have the same memory layout (e.g., if `C=u32` and `F=f32`, or if `C=u64` and `F=f64`).
        let mut entropy = R::zero();
        let grid = points_and_counts
            .into_iter()
            .map(|(point, count)| {
                let freq = count.as_() / total;
                let rate = -freq.log2();
                entropy = entropy + freq * rate;
                (point, rate)
            })
            .collect::<Vec<_>>();

        Ok(Self { grid, entropy })
    }
}

impl<'a, V, C, R, const CAP: usize> From<&'a EmpiricalDistribution<V, C, CAP>>
    for StaticRatedGrid<V, R>
where
    R: num_traits::real::Real + 'static,
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num + AsPrimitive<R> + Ord,
{
    fn from(distribution: &'a EmpiricalDistribution<V, C, CAP>) -> Self {
        StaticRatedGrid::from_points_and_counts(distribution.iter())
    }
}

/// A dynamic finite grid of (not necessarily one-dimensional) points, each weighted with a
/// "bit rate".
///
/// A `DynamicRatedGrid` supports all operations that a [`StaticRatedGrid`] supports. In addition, a
/// `DynamicRatedGrid` has methods [`insert`](Self::insert) and [`remove`](Self::remove) that allow
/// users to adjust the bit rates of each grid point after instantiation. If these operations are
/// not required, then a [`StaticRatedGrid`] should e used as it is more efficient in both memory
/// and runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicRatedGrid<V = F32, R = f32, C = u32> {
    /// Nonempty Vec of tuples `(grid_point, rate, count)`, sorted by decreasing count.
    grid: Vec<(V, R, C)>,
    total: C,
}

/// TODO: documentation with code examples
impl<V, R, C> DynamicRatedGrid<V, R, C>
where
    V: Copy,
    R: Copy + num_traits::real::Real + 'static,
    C: Copy + num_traits::AsPrimitive<R>,
{
    /// Returns the grid points and their associated bit rates, sorted by increasing rate.
    pub fn points_and_rates(
        &self,
    ) -> impl ExactSizeIterator<Item = (V, R)> + DoubleEndedIterator + '_ {
        let log_total = self.total.as_().log2();
        self.grid
            .iter()
            .map(move |&(point, unnormalized_rate, _count)| (point, unnormalized_rate + log_total))
    }

    /// Returns the grid points and their associated counts, sorted by decreasing count.
    pub fn points_and_counts(
        &self,
    ) -> impl ExactSizeIterator<Item = (V, C)> + DoubleEndedIterator + '_ {
        self.grid
            .iter()
            .map(|&(point, _unnormalized_rate, count)| (point, count))
    }

    /// Returns the entropy corresponding to the bit rates.
    ///
    /// The entropy could technically be calculated as follows,
    ///
    /// `entropy = - sum_i( 2^{-rate} * rate )`.
    ///
    /// But the current implementation instead pre-calculates it when calculating the rates in the
    /// various constructors as most of the operations to do this have to be performed there anyway.
    /// So calling `RatedGrid::entropy_base2` is cheap.
    pub fn entropy_base2(&self) -> R {
        let mut sum_count_unnormalized_rate = R::zero();
        for &(_value, unnormalized_rate, count) in &self.grid {
            let count = count.as_();
            sum_count_unnormalized_rate = sum_count_unnormalized_rate + count * unnormalized_rate
        }

        let total = self.total.as_();
        total.log2() + sum_count_unnormalized_rate / total
    }

    /// Quantizes a given value to a grid point by optimizing a rate/distortion-tradeoff.
    ///
    /// The quantization method minimizes the following objective:
    ///
    /// `quantized = arg_min[ distortion(unquantized, quantized) - bit_penalty * rate(quantized) ]`
    ///
    /// where `distortion` and `bit_penalty` are provided by the caller, `rate(quantized)` is the
    /// rate that the `RatedGrid` associates with the grid point `quantized`, and the minimization
    /// runs over all grid points `quantized`.
    ///
    /// The `distortion` function must be nonnegative for all arguments, but it does not need to be
    /// zero for `quantized == unquantized`, and it does not have to be unimodal.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{F32, quant::{StaticRatedGrid, FromPoints}};
    ///
    /// let points = [4.2, -0.3, 1.5, -0.3, 0.1, 1.5, 1.5];
    /// let grid = StaticRatedGrid::<F32, f32>::try_from_points::<_, u32>(
    ///     points.iter().copied()
    /// ).unwrap();
    ///
    /// let unquantized = F32::new(3.0).unwrap();
    /// let quantized = grid.quantize(unquantized, |x, y| ((x - y) * (x - y)).get(), 1.0);
    /// assert_eq!(quantized.get(), 1.5);
    /// ```
    ///
    /// This method can also be called in generic code through [`QuantizationMethod::quantize`].
    /// Thus, the above method call is equal to the following trait method call:
    ///
    /// ```
    /// # use constriction::{F32, quant::{StaticRatedGrid, FromPoints}};
    /// #
    /// # let points = [4.2, -0.3, 1.5, -0.3, 0.1, 1.5, 1.5];
    /// # let grid = StaticRatedGrid::<F32, f32>::try_from_points::<_, u32>(
    /// #     points.iter().copied()
    /// # ).unwrap();
    /// #
    /// # let unquantized = F32::new(3.0).unwrap();
    /// #
    /// // ... define `grid` and `unquantized` as above ...
    ///
    /// // Import the trait `QuantizationMethod` so that we can use its implementation for
    /// // `RateDistortionQuantization`.
    /// use constriction::quant::{QuantizationMethod, RateDistortionQuantization};
    ///
    /// let quantized = RateDistortionQuantization.quantize(
    ///     unquantized, &grid, |x, y| ((x - y) * (x - y)).get(), 1.0);
    /// assert_eq!(quantized.get(), 1.5);
    /// ```
    pub fn quantize(&self, unquantized: V, distortion: impl Fn(V, V) -> R, bit_penalty: R) -> V
    where
        V: Copy,
        R: Copy + PartialOrd + Zero + Bounded + Mul<Output = R>,
    {
        RateDistortionQuantization.quantize(unquantized, self, distortion, bit_penalty)
    }
}

impl<V, R, C> RatedGrid<V, R> for DynamicRatedGrid<V, R, C>
where
    V: Copy,
    R: Copy,
    C: Copy,
{
    fn points_and_unnormalized_rates(&self) -> impl Iterator<Item = (V, R)> {
        self.grid
            .iter()
            .map(move |&(point, unnormalized_rate, _count)| (point, unnormalized_rate))
    }
}

impl<V, R, C> FromPoints<V, C> for DynamicRatedGrid<V, R, C>
where
    Self: Sized,
    V: Copy + Ord,
    C: Copy + Ord + AsPrimitive<R> + num_traits::Num,
    R: num_traits::real::Real + 'static,
{
    fn try_from_points<F>(
        points: impl IntoIterator<Item = F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>,
    {
        let mut map = BTreeMap::new();
        for point in points {
            map.entry(V::try_from(point)?)
                .and_modify(|count| *count = *count + C::one())
                .or_insert(C::one());
        }

        Ok(Self::from_points_and_counts(map))
    }

    fn try_from_points_and_counts<F>(
        points_and_counts: impl IntoIterator<Item = (F, C)>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>,
    {
        // This does not allocate if `points_and_counts> = Vec<(F, C)>` and
        // <V as TryFrom<F>>::Error = Infallible.
        let mut grid = points_and_counts
            .into_iter()
            .map(|(point, count)| Ok((V::try_from(point)?, -count.as_().log2(), count)))
            .collect::<Result<Vec<_>, _>>()?;

        assert!(!grid.is_empty(), "Cannot construct an empty rated grid.");

        // We sort by counts rather than by log-counts to avoid spurious ties due to rounding
        // errors. It's also probably faster (but requires an additional conversion below).
        grid.sort_unstable_by(|&(point1, _r1, count1), &(point2, _r2, count2)| {
            (count2, point1).cmp(&(count1, point2))
        });

        let mut total = C::zero();
        for &(_, _, count) in &grid {
            total = total + count;
        }

        Ok(Self { grid, total })
    }
}

impl<'a, V, C, R, const CAP: usize> From<&'a EmpiricalDistribution<V, C, CAP>>
    for DynamicRatedGrid<V, R, C>
where
    R: num_traits::real::Real + 'static,
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num + AsPrimitive<R>,
{
    fn from(distribution: &'a EmpiricalDistribution<V, C, CAP>) -> Self {
        DynamicRatedGrid::from_points_and_counts(distribution.iter())
    }
}

impl<V, R, C> DynamicDistribution<V, C> for DynamicRatedGrid<V, R, C>
where
    V: Copy + Ord,
    R: Copy + num_traits::real::Real + 'static,
    C: Copy + Ord + num_traits::Num + AsPrimitive<R>,
{
    type InsertSuccess = C;
    type InsertError = NotFoundError;

    /// Inserts `count` points into the existing grid point at `position` and adjusts its rate
    /// accordingly.
    ///
    /// The grid point must already exist (with a nonzero number of points at that grid point).
    ///
    /// On success, returns `Ok(original_count)`, where `original_count > 0` is the number of
    /// points that were present at the grid point at `position` *before* the removal (thus, once
    /// the method returns, there are only `original_count - count` points left at that grid point).
    ///
    /// Returns `Err(NotFoundError)` if removal fails because there aren't enough points with the
    /// provided `value` (in this case, the `EmpiricalDistribution` remains unchanged).
    fn insert(&mut self, position: V, count: C) -> Result<C, NotFoundError>
    where
        V: PartialEq + PartialOrd,
        C: Ord + Add<Output = C>,
    {
        let (index, old_unnormalized_rate, old_count) = self
            .grid
            .iter_mut()
            .enumerate()
            .find_map(|(i, (value, unnormalized_rate, count))| {
                if *value == position {
                    Some((i, unnormalized_rate, count))
                } else {
                    None
                }
            })
            .ok_or(NotFoundError)?;

        let original_count = *old_count;
        let new_count = original_count + count;
        *old_count = new_count;
        *old_unnormalized_rate = -old_count.as_().log2();
        self.total = self.total + count;

        let target_index = self.grid[..index]
            .iter()
            .enumerate()
            .rev()
            .skip(1)
            .find(|(_i, &(value, _unnormalized_rate, count))| {
                (count, position) >= (new_count, value)
            })
            .map(|(i, _)| i + 1)
            .unwrap_or(0);

        self.grid[target_index..=index].rotate_right(1);

        Ok(original_count)
    }

    /// Removes `count` points from the grid point at `position` and adjusts its rate accordingly.
    ///
    /// `count` should be nonzero. If `count` is zero, then the grid remains unchanged but whether
    /// the method returns `Ok(..)` or `Err(..)` is unspecified.
    ///
    /// On success, returns `Ok(original_count)`, where `original_count >= count` is the number of
    /// points that were present at the grid point at `position` *before* the removal. Thus, once
    /// the method returns, there are only `original_count - count` points left at that grid point.
    /// If removal leads to zero resulting points at `position`, then this removes the grid point.
    ///
    /// Returns `Err(NotFoundError)` if removal fails because there aren't enough points with the
    /// provided `value` (in this case, the `EmpiricalDistribution` remains unchanged).
    fn remove(&mut self, position: V, count: C) -> Result<C, NotFoundError>
    where
        V: PartialEq + PartialOrd,
        C: Ord + Sub<Output = C>,
    {
        let (index, old_unnormalized_rate, old_count) = self
            .grid
            .iter_mut()
            .enumerate()
            .find_map(|(i, (value, unnormalized_rate, count))| {
                if *value == position {
                    Some((i, unnormalized_rate, count))
                } else {
                    None
                }
            })
            .ok_or(NotFoundError)?;

        let original_count = *old_count;
        match original_count.cmp(&count) {
            core::cmp::Ordering::Less => return Err(NotFoundError),
            core::cmp::Ordering::Equal => {
                self.grid.remove(index);
            }
            core::cmp::Ordering::Greater => {
                let new_count = *old_count - count;
                *old_count = new_count;
                *old_unnormalized_rate = -new_count.as_().log2();
                let index_after_target = self
                    .grid
                    .iter()
                    .enumerate()
                    .skip(index + 1)
                    .find(|(_i, &(value, _unnormalized_rate, count))| {
                        (count, position) <= (new_count, value)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(self.grid.len());

                self.grid[index..index_after_target].rotate_left(1);
            }
        }

        self.total = self.total - count;
        Ok(original_count)
    }

    fn remove_all(&mut self, position: V) -> C {
        let index = self
            .grid
            .iter()
            .position(|(value, _unnormalized_rate, _count)| *value == position);

        match index {
            Some(index) => self.grid.remove(index).2,
            None => C::zero(),
        }
    }
}

/// Error type returned by [`EmpiricalDistribution::remove`] if removal fails.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct NotFoundError;

impl Display for NotFoundError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Attempted to remove a value from an `EmpiricalDistribution` that does not exist in \
            the distribution."
        )
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NotFoundError {}

/// Quantizer that implements the [Variational Bayesian Quantization (VBQ)] method.
///
/// VBQ is a quantization method that takes into account (i) the "prior" distribution of
/// unquantized points (putting a higher density of grid points in regions of high prior density)
/// and (ii) the saliency of the value that we quantize, i.e., how much a given distortion of a
/// single value would hurt the overall quality of the quantization of a collection of values.
///
/// # Overview of the VBQ Method
///
/// For a given input `unquantized`, VBQ returns a point `quantized` by minimizing the following
/// objective function:
///
/// `loss(quantized) = distortion(quantized - unquantized) + bit_penalty * rate_estimate(quantized)`
///
/// Here, the function `distortion` and the scalar `bit_penalty` are provided by the caller, and
/// `rate_estimate` is an estimate of the information content that the result `quantized` will have
/// under the empirical distribution of the collection of all quantized points (see below). To use
/// VBQ as described in the [original paper] (Eq. 9), set `bit_penalty = 2.0 * lambda *
/// posterior_variance`, where `lambda > 0` is a parameter that trades off between rate and
/// distortion (`lambda` is the same for all points you want to quantize) and `posterior_variance`
/// is an estimate of the variance of the point you currently want to quantize under its Bayesian
/// posterior distribution (`posterior_variance` will typically be different for each point you
/// want to quantize, or else a different quantization method may be more suitable).
///
/// The `rate_estimate` in the above objective function is calculated based on the provided `prior`
/// distribution and on some theoretical considerations of how the VBQ algorithm works. You will
/// get better estimates (and therefore better  quantization results) if the `prior` approximates
/// the distribution of quantized points. Since you don't have access to the quantized points before
/// running VBQ, it is recommended to run VBQ on a given set of points several times in a row. In
/// the first run, set the `prior` to the empirical distribution of *unquantized* points (using,
/// e.g., an [`EmpiricalDistribution`]). In subsequent runs of VBQ, set the prior to the empirical
/// distribution of the quantized points you obtained in the previous run.
///
/// # Arguments
///
/// This type implements the trait [`QuantizationMethod`], where the arguments of its method
/// [`quantize`](QuantizationMethod::quantize) are interpreted as follows:
///
/// - `unquantized`: the value that you want to quantize.
/// - `grid`: a distribution that influences how VBQ distributes its grid points (see discussion
///   above). Typically an [`EmpiricalDistribution`] that estimates the distribution of final
///   quantized values, e.g., by taking the empirical distribution of either the unquantized points
///   or of quantized points from a previous run of VBQ on the same set of points.
/// - `distortion`: a function that assigns a penalty to any pair of unquantized and quantized
///   value. A common choice is a quadratic distortion, i.e., `|x, y| ((x - y) * (x - y)).get()`.
///   The distortion must satisfy the following:
///   - `distortion(unquantized, unquantized) == L::zero()`;
///   - `distortion(unquantized, quantized) > L::zero()` for all `x != unquantized`; and
///   - `distortion(unquantized, quantized)` must be unimodal in its second argument, i.e., it must
///     be monotonically nonincreasing in its second argument for `quantized <= unquantized`, and
///     monotonically nondecreasing in its second argument for `quantized >= unquantized`.
/// - `bit_penalty`: conversion rate from bit rates to penalties, see documentation of
///   [`quantize`](QuantizationMethod::quantize). Higher values of `bit_penalty` lead to a more
///   coarse quantization, i.e., higher distortion and lower bit entropy.
///
/// # References
///
/// VBQ was originally proposed and empirically evaluated for the compression of images and word
/// embeddings by [Yang et al., ICML 2020]. For an empirical evaluation of VBQ for the compression
/// of neural network weights, see [Tan and Bamler, Deploy & Monitor ML Workshop @ NeurIPS 2022].
///
/// [Variational Bayesian Quantization (VBQ)]: http://proceedings.mlr.press/v119/yang20a/yang20a.pdf
/// [Yang et al., ICML 2020]: http://proceedings.mlr.press/v119/yang20a/yang20a.pdf
/// [original paper]: http://proceedings.mlr.press/v119/yang20a/yang20a.pdf
/// [Tan and Bamler, Deploy & Monitor ML Workshop @ NeurIPS 2022]:
///   https://raw.githubusercontent.com/dmml-workshop/dmml-neurips-2022/main/accepted-papers/paper_21.pdf
#[derive(Debug, Clone, Copy, Default)]
pub struct Vbq;

impl<P, L> QuantizationMethod<P, P::Value, L> for Vbq
where
    P: UnnormalizedInverse,
    P::Count: PartialOrd + Bounded,
    L: Copy + PartialOrd + Zero + Bounded,
{
    fn quantize(
        &self,
        unquantized: P::Value,
        grid: &P,
        distortion: impl Fn(P::Value, P::Value) -> L,
        bit_penalty: L,
    ) -> P::Value {
        let total = grid.total();
        assert!(total != P::Count::zero());
        let two = P::Count::one() + P::Count::one();

        let conversion = if P::Count::one() / two > P::Count::zero() {
            // `P::Count` is a float type. No conversion necessary (will be compiled out).
            P::Count::one()
        } else {
            // `P::Count` is an integer type. Use fixed point arithmetic.
            ((P::Count::max_value() - P::Count::one()) / two + P::Count::one()) / total
        };
        assert!(conversion != P::Count::zero(), "prior must not contain more points than half the range of numbers representable by `P::Count`");

        let mut lower = P::Count::zero();
        let mut upper = total * conversion;
        let mut prev_mid_converted = total;

        let target = conversion * grid.cdf(unquantized);

        let mut current_rate = L::zero();
        let mut record_point = unquantized; // Will be overwritten at least once.
        let mut record_objective = L::max_value();

        loop {
            let mid = (lower + upper) / two; // Rounds down (which is probably what we want).
            if mid <= target {
                lower = mid;
            }
            if mid >= target {
                upper = mid;
            }

            let mid_converted = mid / conversion; // Rounds down (which is what we want).
            if mid_converted == prev_mid_converted {
                // Can't be reached in the first iteration of the loop because `mid_converted < total`.
                break;
            }
            prev_mid_converted = mid_converted;

            let candidate = grid
                .ppf(mid_converted)
                .expect("`mid_converted < prior.total()`");
            let current_objective = distortion(unquantized, candidate) + current_rate;
            if current_objective <= record_objective {
                record_point = candidate;
                record_objective = current_objective;
            }

            current_rate = current_rate + bit_penalty;

            match current_rate.partial_cmp(&record_objective) {
                Some(core::cmp::Ordering::Less) => {
                    // Continue with the next iteration since we might still be able to improve
                    // upon `record_objective`.
                }
                None | Some(_) => {
                    // We either won't be able to improve upon `record_objective` (because all
                    // subsequent candidate objectives will be lower bounded by `current_rate`), or
                    // we encountered NaN (which should never happen for a well defined prior, but
                    // if it does then it's probably best to break).
                    break;
                }
            }
        }

        record_point
    }
}

/// Explicit rate/distortion quantization.
///
/// See documentation of [`StaticRatedGrid::quantize`] and [`DynamicRatedGrid::quantize`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RateDistortionQuantization;

impl<G, V, R> QuantizationMethod<G, V, R> for RateDistortionQuantization
where
    G: RatedGrid<V, R>,
    V: Copy,
    R: Copy + PartialOrd + Zero + Bounded + Mul<Output = R>,
{
    fn quantize(
        &self,
        unquantized: V,
        grid: &G,
        distortion: impl Fn(V, V) -> R,
        bit_penalty: R,
    ) -> V {
        let mut grid = grid.points_and_unnormalized_rates();
        let (mut record_candidate, rate) = grid.next().expect("grid isn't empty");
        let first_distortion = distortion(unquantized, record_candidate);
        let mut record_loss = first_distortion + bit_penalty * rate;

        for (candidate, rate) in grid {
            let scaled_rate = bit_penalty * rate;
            if scaled_rate >= record_loss {
                // Rate is monotonically nondecreasing, and distortion is nonnegative. So once we're at
                // this point, there won't be any better candidate than the best we've found so far.
                break;
            }
            let candidate_distortion = distortion(unquantized, candidate);
            let loss = candidate_distortion + scaled_rate;
            if loss < record_loss {
                record_loss = loss;
                record_candidate = candidate
            }
        }

        record_candidate
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::F64;

    use super::*;

    use alloc::vec::Vec;
    use rand::{seq::SliceRandom, RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;

    #[test]
    fn dynamic_empirical_distribution() {
        #[cfg(not(miri))]
        let amt = 1000;

        #[cfg(miri)]
        let amt = 100;

        let mut rng = Xoshiro256StarStar::seed_from_u64(202312115);
        let mut points = (0..amt)
            .flat_map(|_| {
                let num_repeats = 1 + rng.next_u32() % 4;
                let value = rng.next_u32() as f32 / u32::MAX as f32;
                core::iter::repeat(value).take(num_repeats as usize)
            })
            .collect::<Vec<_>>();
        points.shuffle(&mut rng);
        assert!(points.len() > amt);
        assert!(points.len() < 5 * amt);

        let dist =
            EmpiricalDistribution::<F32, u32>::try_from_points(points.iter().copied()).unwrap();

        #[cfg(not(miri))]
        let num_moves = 100;

        #[cfg(miri)]
        let num_moves = 10;

        assert_eq!(dist.total() as usize, points.len());
        for _ in 0..num_moves {
            let index = rng.next_u32() as usize % amt;
            let x = points[index];
            let expected = points.iter().filter(|&&y| y < x).count() as u32;
            let x = F32::new(x).unwrap();
            assert_eq!(dist.cdf(x), expected);
            assert_eq!(dist.ppf(expected).unwrap(), x);
        }
    }

    #[test]
    fn vbq_empirical() {
        #[cfg(not(miri))]
        let amt = 1000;

        #[cfg(miri)]
        let amt = 100;

        let mut rng = Xoshiro256StarStar::seed_from_u64(202312116);
        let mut points = (0..amt)
            .flat_map(|_| {
                let num_repeats = 1 + rng.next_u32() % 4;
                let value = F32::new(rng.next_u32() as f32 / u32::MAX as f32).unwrap();
                core::iter::repeat(value).take(num_repeats as usize)
            })
            .collect::<Vec<_>>();
        points.shuffle(&mut rng);
        assert!(points.len() > amt);
        assert!(points.len() < 5 * amt);

        let prior = EmpiricalDistribution::<F32, u32>::from_points(&points);
        let initial_entropy = prior.entropy_base2::<f32>();
        let mut entropy_previous_coarseness = initial_entropy;

        #[cfg(not(miri))]
        let (num_repeats, betas) = (5, [1e-7, 1e-5, 0.001, 0.01, 0.1]);

        #[cfg(miri)]
        let (num_repeats, betas) = (2, [0.001, 0.1]);

        for beta in betas {
            let beta = F32::new(beta).unwrap();
            let mut prior = prior.clone();
            let mut shifted_points = points.clone();
            let mut previous_entropy = initial_entropy;

            for i in 0..num_repeats {
                for (point, shifted_point) in points.iter().zip(shifted_points.iter_mut()) {
                    let quant = prior.vbq(*point, |x, y| (x - y) * (x - y), beta);
                    prior.remove(*shifted_point, 1).unwrap();
                    prior.insert(quant, 1).unwrap();
                    *shifted_point = quant;
                }
                let entropy = prior.entropy_base2::<f32>();
                if i <= 1 {
                    assert!(entropy < previous_entropy)
                } else if i == 4 {
                    assert_eq!(entropy, previous_entropy)
                } else {
                    assert!(entropy <= previous_entropy);
                }
                previous_entropy = entropy
            }

            assert!(previous_entropy < entropy_previous_coarseness);
            entropy_previous_coarseness = previous_entropy;
        }
    }

    #[test]
    fn vbq_gaussian() {
        #[cfg(not(miri))]
        let amt = 1000;

        #[cfg(miri)]
        let amt = 100;

        let mut rng = Xoshiro256StarStar::seed_from_u64(20240117);
        let mut points = (0..amt)
            .flat_map(|_| {
                let num_repeats = 1 + rng.next_u32() % 4;
                let value = rng.next_u32() as f64 / (u32::MAX / 2) as f64 - 1.0;
                core::iter::repeat(value).take(num_repeats as usize)
            })
            .collect::<Vec<_>>();
        points.shuffle(&mut rng);
        assert!(points.len() > amt);
        assert!(points.len() < 5 * amt);

        let empirical_distribution =
            EmpiricalDistribution::<F64, u32>::try_from_points(points.iter().copied()).unwrap();
        let initial_entropy = empirical_distribution.entropy_base2::<f64>();
        core::mem::drop(empirical_distribution);

        let prior = probability::distribution::Gaussian::new(0.1, 1.3);

        #[cfg(not(miri))]
        let betas = [1e-7, 1e-5, 0.001, 0.01, 0.1, 1.0];

        #[cfg(miri)]
        let betas = [0.001, 0.1];

        let mut previous_entropy = initial_entropy;

        for beta in betas {
            let shifted_points = points
                .iter()
                .map(|&point| Vbq.quantize(point, &prior, |x, y| (x - y) * (x - y), beta));
            let empirical_distribution =
                EmpiricalDistribution::<F64, u32>::try_from_points(shifted_points).unwrap();
            let entropy = empirical_distribution.entropy_base2::<f64>();
            core::mem::drop(empirical_distribution);
            assert!(entropy < previous_entropy);
            assert!(entropy > 0.0);

            previous_entropy = entropy
        }
    }

    #[test]
    fn rate_distortion_quantization() {
        #[cfg(not(miri))]
        let amt = 1000;

        #[cfg(miri)]
        let amt = 100;

        let mut rng = Xoshiro256StarStar::seed_from_u64(20240213);
        let mut points = (0..amt)
            .flat_map(|_| {
                let num_repeats = 1 + rng.next_u32() % 4;
                let value = F32::new(rng.next_u32() as f32 / u32::MAX as f32).unwrap();
                core::iter::repeat(value).take(num_repeats as usize)
            })
            .collect::<Vec<_>>();
        points.shuffle(&mut rng);
        assert!(points.len() > amt);
        assert!(points.len() < 5 * amt);

        let prior = EmpiricalDistribution::<F32, u32>::from_points(&points);
        let initial_entropy = prior.entropy_base2::<f32>();
        let mut entropy_previous_coarseness = initial_entropy;

        #[cfg(not(miri))]
        let (num_repeats, betas) = (3, [1e-7, 1e-5, 0.001, 0.01, 0.1]);

        #[cfg(miri)]
        let (num_repeats, betas) = (2, [0.001, 0.1]);

        for beta in betas {
            let beta = F32::new(beta).unwrap();
            let prior = prior.clone();
            let mut previous_entropy = initial_entropy;

            // Run VBQ once to get an estimate of the grid.
            let mut quantized = points
                .iter()
                .map(|point| prior.vbq(*point, |x, y| (x - y) * (x - y), beta))
                .collect::<Vec<_>>();

            for i in 0..num_repeats {
                let grid = StaticRatedGrid::<_, f32>::from_points::<u32, _>(&quantized);
                let entropy = grid.entropy_base2();
                if i < 2 {
                    assert!(entropy < previous_entropy);
                }
                previous_entropy = entropy;

                // TODO: maybe use `NonNanFloat` for `R`
                for point in quantized.iter_mut() {
                    *point = grid.quantize(*point, |x, y| ((x - y) * (x - y)).get(), beta.get())
                }
            }

            let grid = StaticRatedGrid::<F32, f32>::from_points::<u32, _>(&quantized);
            let entropy = grid.entropy_base2();
            assert!(entropy == previous_entropy);
            previous_entropy = entropy;

            assert!(previous_entropy < entropy_previous_coarseness);
            entropy_previous_coarseness = previous_entropy;
        }
    }

    #[test]
    fn static_rated_grid() {
        let (points, grid, counts, _rng) = create_random_rated_grid::<StaticRatedGrid>();

        let points_and_rates = grid.points_and_rates();
        assert_eq!(points_and_rates.len(), counts.len());
        let (mut previous_point, mut previous_rate) = (F32::new(0.0).unwrap(), 0.0);
        let neg_log_normalizer = (points.len() as f32).log2();

        for &(point, rate) in points_and_rates {
            let count = counts[&point.get().to_bits()];
            let expected_rate = neg_log_normalizer - (count as f32).log2();
            assert!((rate - expected_rate).abs() < 1e-6);
            assert!(rate > previous_rate || (rate == previous_rate || point > previous_point));
            previous_point = point;
            previous_rate = rate;
        }
    }

    #[test]
    fn dynamic_rated_grid() {
        let (points, mut grid, mut counts, mut rng) =
            create_random_rated_grid::<DynamicRatedGrid>();

        let points_and_rates = grid.points_and_rates();
        let points_and_unnormalized_rates = grid.points_and_unnormalized_rates();
        assert_eq!(points_and_rates.len(), counts.len());
        let (mut previous_point, mut previous_rate) = (F32::new(0.0).unwrap(), 0.0);
        let neg_log_normalizer = (points.len() as f32).log2();

        let mut num_grid_points = 0;
        for ((point, rate), (point2, unnormalized_rate)) in
            points_and_rates.zip(points_and_unnormalized_rates)
        {
            let count = counts[&point.get().to_bits()];
            let expected_unnormalized_rate = -(count as f32).log2();
            let expected_rate = neg_log_normalizer + expected_unnormalized_rate;
            assert_eq!(point, point2);
            assert!((unnormalized_rate - expected_unnormalized_rate).abs() < 1e-6);
            assert!((rate - expected_rate).abs() < 1e-6);
            assert!(rate > previous_rate || (rate == previous_rate || point > previous_point));
            previous_point = point;
            previous_rate = rate;
            num_grid_points += 1;
        }

        assert_eq!(num_grid_points, counts.len());

        #[cfg(not(miri))]
        let num_moves = 100;

        #[cfg(miri)]
        let num_moves = 10;

        let mut total_moves = 0;

        for _ in 0..num_moves {
            let (from_index, from_point, from_count) = loop {
                let from_index = rng.next_u32() as usize % points.len();
                let from_point = points[from_index];
                if let Some(from_count) = counts.get_mut(&from_point.to_bits()) {
                    break (from_index, from_point, from_count);
                }
            };

            let move_count = rng.next_u32() % *from_count + 1;
            total_moves += move_count;
            assert_eq!(
                grid.remove(F32::new(from_point).unwrap(), move_count)
                    .unwrap(),
                *from_count
            );
            *from_count -= move_count;
            if *from_count == 0 {
                counts.remove(&from_point.to_bits()).unwrap();
            }

            let (to_point, to_count) = loop {
                let to_index = rng.next_u32() as usize % points.len();
                if to_index == from_index {
                    continue;
                }
                let to_point = points[to_index];
                if let Some(to_count) = counts.get_mut(&to_point.to_bits()) {
                    break (to_point, to_count);
                }
            };

            assert_eq!(
                grid.insert(F32::new(to_point).unwrap(), 2 * move_count)
                    .unwrap(),
                *to_count
            );
            *to_count += 2 * move_count;
        }

        let points_and_rates = grid.points_and_rates();
        let points_and_unnormalized_rates = grid.points_and_unnormalized_rates();
        assert_eq!(points_and_rates.len(), counts.len());
        let (mut previous_point, mut previous_rate) = (F32::new(0.0).unwrap(), 0.0);
        let neg_log_normalizer = ((points.len() + total_moves as usize) as f32).log2();

        let mut num_grid_points = 0;
        for ((point, rate), (point2, unnormalized_rate)) in
            points_and_rates.zip(points_and_unnormalized_rates)
        {
            let count = counts[&point.get().to_bits()];
            let expected_unnormalized_rate = -(count as f32).log2();
            let expected_rate = neg_log_normalizer + expected_unnormalized_rate;
            assert_eq!(point, point2);
            assert!((unnormalized_rate - expected_unnormalized_rate).abs() < 1e-6);
            assert!((rate - expected_rate).abs() < 1e-6);
            assert!(rate > previous_rate || (rate == previous_rate || point > previous_point));
            previous_point = point;
            previous_rate = rate;
            num_grid_points += 1;
        }

        assert_eq!(num_grid_points, counts.len());
    }

    fn create_random_rated_grid<G: FromPoints<F32, u32>>(
    ) -> (Vec<f32>, G, HashMap<u32, u32>, Xoshiro256StarStar) {
        #[cfg(not(miri))]
        let amt = 1000;

        #[cfg(miri)]
        let amt = 100;

        let mut rng = Xoshiro256StarStar::seed_from_u64(202402291);
        let mut points = (0..amt)
            .flat_map(|_| {
                let num_repeats = 1 + rng.next_u32() % 4;
                let value = rng.next_u32() as f32 / u32::MAX as f32;
                core::iter::repeat(value).take(num_repeats as usize)
            })
            .collect::<Vec<_>>();
        points.shuffle(&mut rng);
        assert!(points.len() > amt);
        assert!(points.len() < 5 * amt);

        let grid = G::try_from_points(points.iter().copied()).unwrap();

        let mut counts = HashMap::new();
        for &point in &points {
            // Using `.to_bits()` is a dirty hack to make `pos` hashable. It works
            // here because we don't have NaNs, and all zeros have the same sign.
            counts
                .entry(point.to_bits())
                .and_modify(|count| *count += 1u32)
                .or_insert(1);
        }

        (points, grid, counts, rng)
    }
}
