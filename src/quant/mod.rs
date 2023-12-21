pub mod vbq;

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

use crate::{UnwrapInfallible, F32};

use self::augmented_btree::AugmentedBTree;

pub trait EmpiricalDistribution<V, C>
where
    V: Ord,
    C: Ord + num_traits::Num,
{
    fn try_from_points<'a, F>(
        points: impl IntoIterator<Item = &'a F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        Self: Sized,
        V: TryFrom<F>,
        F: Copy + 'a;

    fn from_points<'a>(points: impl IntoIterator<Item = &'a V>) -> Self
    where
        Self: Sized,
        V: Copy + 'a,
    {
        Self::try_from_points(points).unwrap_infallible()
    }

    fn try_from_points_hashable<'a, F>(
        points: impl IntoIterator<Item = &'a F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        Self: Sized,
        V: TryFrom<F> + Hash,
        F: Copy + 'a,
    {
        Self::try_from_points(points)
    }

    fn from_points_hashable<'a>(points: impl IntoIterator<Item = &'a V>) -> Self
    where
        Self: Sized,
        V: Hash + Copy + 'a,
    {
        Self::try_from_points_hashable(points).unwrap_infallible()
    }

    fn total(&self) -> C;

    fn left_sided_cumulative(&self, x: V) -> C;

    fn entropy_base2<F>(&self) -> F
    where
        F: num_traits::float::Float + 'static,
        C: num_traits::AsPrimitive<F>;

    /// Returns the inverse of [`left_sided_cumulative`](Self::left_sided_cumulative)
    ///
    /// More precisely, the returned value is the right-sided inverse of the left-sided CDF, i.e.,
    /// it is the right-most position where the left-sided CDF is smaller than or equal to the
    /// argument `cum`. Returns `None` if `cum >= self.total()` (in this case, the left-sided
    /// CDF is smaller than or equal to `cum` everywhere, so there is no *single* right-most
    /// position that satisfies this criterion).
    ///
    /// The following two relations hold (where `self` is an `EmpiricalDistribution`):
    ///
    /// - `self.left_cumulative(self.inverse_cumulative(self.left_cumulative(pos)).unwrap())` is
    ///   equal to `self.left_cumulative(pos)` (assuming that `unwrap()` succeeds).
    /// - `self.inverse_cumulative(self.left_cumulative(self.inverse_cumulative(cum).unwrap())).unwrap())`
    ///   is equal to `self.inverse_cumulative(cum).unwrap()` (assuming that the inner `unwrap()`
    ///   succeeds—in which case the outer `unwrap()` is guaranteed to succeed).
    fn inverse_cumulative(&self, cum: C) -> Option<V>;
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

pub struct DynamicEmpiricalDistribution<V = F32, C = u32>(AugmentedBTree<V, CountWrapper<C>, 64>);

impl<V: Display, C: Display + Debug> Debug for DynamicEmpiricalDistribution<V, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("DynamicEmpiricalDistribution")
            .field(&self.0)
            .finish()
    }
}

impl<V: Copy, C: Copy> Clone for DynamicEmpiricalDistribution<V, C> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<V, C> EmpiricalDistribution<V, C> for DynamicEmpiricalDistribution<V, C>
where
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num,
{
    fn try_from_points<'a, F>(
        points: impl IntoIterator<Item = &'a F>,
    ) -> Result<Self, <V as TryFrom<F>>::Error>
    where
        V: TryFrom<F>,
        F: Copy + 'a,
    {
        let mut tree = AugmentedBTree::new();
        for &point in points {
            tree.insert(V::try_from(point)?, CountWrapper(C::one()))
        }

        Ok(Self(tree))
    }

    // fn try_from_points_hashable<'a, F>(
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

    fn total(&self) -> C {
        self.0.total().0
    }

    fn left_sided_cumulative(&self, x: V) -> C {
        self.0.left_cumulative(x).0
    }

    fn inverse_cumulative(&self, cum: C) -> Option<V> {
        self.0.quantile_function(CountWrapper(cum))
    }

    fn entropy_base2<F>(&self) -> F
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
}

impl<V, C> DynamicEmpiricalDistribution<V, C>
where
    V: Copy + Ord,
    C: Copy + Ord + num_traits::Num,
{
    pub fn insert(&mut self, value: V) {
        self.0.insert(value, CountWrapper(C::one()))
    }

    /// Removes a point at position `pos`.
    ///
    /// On success, returns `Some(remaining)` where `remaining` is the number of points that still
    /// remain at position `pos` after the removal. Returns `None` if removal fails because there
    /// was no point at position `pos`.
    #[allow(clippy::result_unit_err)]
    pub fn remove(&mut self, value: V) -> Option<C> {
        self.0.remove(value, CountWrapper(C::one())).map(|c| c.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::vec::Vec;
    use rand::{seq::SliceRandom, RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;

    #[test]
    fn dynamic_empirical_distribution() {
        let amt = 1000;
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

        let dist = DynamicEmpiricalDistribution::<F32, u32>::try_from_points(&points).unwrap();

        assert_eq!(dist.total() as usize, points.len());
        for _ in 0..100 {
            let index = rng.next_u32() as usize % amt;
            let x = points[index];
            let expected = points.iter().filter(|&&y| y < x).count() as u32;
            let x = F32::new(x).unwrap();
            assert_eq!(dist.left_sided_cumulative(x), expected);
            assert_eq!(dist.inverse_cumulative(expected).unwrap(), x);
        }
    }
}
