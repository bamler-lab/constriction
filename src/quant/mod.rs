#[cfg(not(feature = "benchmark-internals"))]
mod augmented_btree;

#[cfg(feature = "benchmark-internals")]
pub mod augmented_btree;

use core::{
    convert::TryFrom,
    fmt::{Debug, Display},
    hash::Hash,
    ops::{Add, Sub},
};

use num_traits::{AsPrimitive, Bounded, One, Zero};

use crate::{UnwrapInfallible, F32};

use self::augmented_btree::AugmentedBTree;

pub trait UnnormalizedDistribution {
    type Value: Copy;

    type Count: Copy + num_traits::Num;

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
    /// function [`vbq`] still works exactly as expected if `cdf` returns the right-sided CDF as
    /// long as [`UnnormalizedInverse::ppf`] behaves accordingly; and if `cdf` and `ppf` follow
    /// opposite conventions then `vbq` still terminates and quantizes to a grid point from the
    /// provided `prior`, but the quantization might be biased towards rounding up or down).
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
    /// `cum >= self.total()`), and so the the right-most one would just be the largest number that
    /// can be represented by `Self::Value` (e.g., `f32::INFINITY`), which is probably not what
    /// callers would expect.
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

pub struct EmpiricalDistribution<V = F32, C = u32>(AugmentedBTree<V, CountWrapper<C>, 64>);

impl<V, C> EmpiricalDistribution<V, C>
where
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num,
{
    pub fn new() -> Self {
        Self(AugmentedBTree::new())
    }

    pub fn try_add_points<'a, F>(
        &mut self,
        points: impl IntoIterator<Item = &'a F>,
    ) -> Result<(), <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>,
        F: Copy + 'a,
    {
        for &point in points {
            self.0.insert(V::try_from(point)?, CountWrapper(C::one()))
        }

        Ok(())
    }

    pub fn try_from_points<'a, F>(
        points: impl IntoIterator<Item = &'a F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        Self: Sized + Default,
        V: TryFrom<F>,
        F: Copy + 'a,
    {
        let mut this = Self::default();
        this.try_add_points(points)?;
        Ok(this)
    }

    pub fn add_points<'a>(&mut self, points: impl IntoIterator<Item = &'a V>)
    where
        Self: Sized,
        V: Copy + 'a,
    {
        self.try_add_points(points).unwrap_infallible();
    }

    pub fn from_points<'a>(points: impl IntoIterator<Item = &'a V>) -> Self
    where
        Self: Sized + Default,
        V: Copy + 'a,
    {
        Self::try_from_points(points).unwrap_infallible()
    }

    pub fn try_add_points_hashable<'a, F>(
        &mut self,
        points: impl IntoIterator<Item = &'a F>,
    ) -> Result<(), <V as TryFrom<F>>::Error>
    where
        Self: Sized,
        V: TryFrom<F> + Hash,
        F: Copy + 'a,
    {
        self.try_add_points(points)
    }

    pub fn try_from_points_hashable<'a, F>(
        points: impl IntoIterator<Item = &'a F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        Self: Sized + Default,
        V: TryFrom<F> + Hash,
        F: Copy + 'a,
    {
        let mut this = Self::default();
        this.try_add_points_hashable(points)?;
        Ok(this)
    }

    pub fn from_points_hashable<'a>(points: impl IntoIterator<Item = &'a V>) -> Self
    where
        Self: Sized + Default,
        V: Hash + Copy + 'a,
    {
        Self::try_from_points_hashable(points).unwrap_infallible()
    }

    // pub fn try_from_points_hashable<'a, F>(
    //     points: impl IntoIterator<Item = &'a F>,
    // ) -> Result<Self, <V as TryFrom<F>>::Error>
    // where
    //     Self: Sized,
    //     V: TryFrom<F> + Hash,
    //     F: Copy + 'a,
    // {
    //     let mut counts = HashMap::new();
    //     for &point in points {
    //         let point = V::try_from(point)?;
    //         counts
    //             .entry(point)
    //             .and_modify(|count| *count = *count + CountWrapper(C::one()))
    //             .or_insert(CountWrapper(C::one()));
    //     }

    //     let mut sorted = counts.into_iter().collect::<Vec<_>>();
    //     sorted.sort_unstable_by_key(|(v, _)| *v);
    //     let tree = unsafe { AugmentedBTree::from_sorted_unchecked(&sorted) };

    //     Ok(Self(tree))
    // }

    pub fn entropy_base2<F>(&self) -> F
    where
        F: num_traits::float::Float + 'static,
        C: num_traits::AsPrimitive<F>,
    {
        let mut last_accum = C::zero();
        let mut sum_count_log_count = F::zero();
        for (_, accum) in self.0.iter() {
            let count = (accum.0 - last_accum).as_();
            sum_count_log_count = sum_count_log_count + count * count.log2();
            last_accum = accum.0;
        }

        let total = self.0.total().0.as_();
        total.log2() - sum_count_log_count / total
    }

    pub fn insert(&mut self, value: V) {
        self.0.insert(value, CountWrapper(C::one()))
    }

    /// Removes a point at position `pos`.
    ///
    /// On success, returns `Some(remaining)` where `remaining` is the number of points that still
    /// remain at position `pos` after the removal. Returns `None` if removal fails because there
    /// was no point at position `pos`.
    pub fn remove(&mut self, value: V) -> Option<C> {
        self.0.remove(value, CountWrapper(C::one())).map(|c| c.0)
    }
}

impl<V, C> UnnormalizedDistribution for EmpiricalDistribution<V, C>
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

impl<V, C> UnnormalizedInverse for EmpiricalDistribution<V, C>
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

impl<V: Display, C: Display + Debug> Debug for EmpiricalDistribution<V, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("DynamicEmpiricalDistribution")
            .field(&self.0)
            .finish()
    }
}

impl<V: Copy, C: Copy> Clone for EmpiricalDistribution<V, C> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<V: Copy, C: Copy> Default for EmpiricalDistribution<V, C>
where
    V: Copy + Ord,
    C: Ord + Copy + num_traits::Num,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Quantizes a single value using [Variational Bayesian Quantization (VBQ)].
///
/// VBQ is a quantization method that takes into account (i) the "prior" distribution of
/// unquantized points (putting a higher density of grid points in regions of high prior density)
/// and (ii) the saliency of the value that we quantize, i.e., how much a given distortion of a
/// single value would hurt the overall quality of the quantization of a collection of values.
///
/// # Overview of the Method
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
/// - `unquantized`: the value that you want to quantize.
/// - `prior`: a distribution that influences how VBQ distributes its grid points (see discussion
///   above). Typically an [`EmpiricalDistribution`] that estimates the distribution of final
///   quantized values, e.g., by taking the empirical distribution of either the unquantized points
///   or of quantized points from a previous run of VBQ on the same set of points.
/// - `distortion`: a function that assigns a penalty to any given quantization error. A common
///   choice is a quadratic distortion, i.e., `|x| x * x`. The distortion must satisfy the
///   following:
///   - `distortion(P::Value::zero()) == L::zero()`;
///   - `distortion(x) > L::zero()` for all `x != P::Value::zero()`; and
///   - `distortion` must be unimodal, i.e., it must be monotonically nonincreasing for negative
///     arguments, and nondecreasing for positive arguments.
/// - `bit_penalty`: conversion rate from bit rates to penalties, see above objective function.
///   Higher values of `bit_penalty` lead to a more coarse quantization, i.e., higher distortion
///   and lower bit entropy.
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
pub fn vbq<P, L>(
    unquantized: P::Value,
    prior: &P,
    mut distortion: impl FnMut(P::Value) -> L,
    bit_penalty: L,
) -> P::Value
where
    P: UnnormalizedInverse,
    P::Count: Ord + Bounded,
    P::Value: Sub<Output = P::Value>,
    L: Copy + Ord + Zero + Bounded,
{
    let total = prior.total();
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

    let target = conversion * prior.cdf(unquantized);

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

        let candidate = prior
            .ppf(mid_converted)
            .expect("`mid_converted < prior.total()`");
        let deviation = candidate - unquantized;
        let current_objective = distortion(deviation) + current_rate;
        if current_objective <= record_objective {
            record_point = candidate;
            record_objective = current_objective;
        }

        current_rate = current_rate + bit_penalty;
        if current_rate >= record_objective {
            // We won't be able to improve upon `record_objective` because all subsequent
            // candidate objectives will be lower bounded by `current_rate`.
            break;
        }
    }

    record_point
}

#[cfg(test)]
mod tests {
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

        let dist = EmpiricalDistribution::<F32, u32>::try_from_points(&points).unwrap();

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
    fn vbq() {
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

        let prior = EmpiricalDistribution::<F32, u32>::try_from_points(&points).unwrap();
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
                    let quant = super::vbq(*point, &prior, |x| x * x, beta);
                    prior.remove(*shifted_point).unwrap();
                    prior.insert(quant);
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

    // TODO: add test for VBQ with a prior that implements `probability::distribution::inverse`,
    // and that is not an `EmpiricalDistribution`
}
