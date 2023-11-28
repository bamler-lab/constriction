mod augmented_btree;

use core::{fmt::Debug, hash::Hash, ops::Add};

use alloc::vec::Vec;
use grove::{example_data::Unit, locators::ByKey, splay::SplayTree, Keyed, SomeTree, ToSummary};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::hash_map::{
    Entry::{Occupied, Vacant},
    HashMap,
};

pub mod vbq;

#[derive(Debug, Clone, Copy)]
struct Entry<V, C> {
    value: V,
    count: C,
}

impl<V: PartialEq, C> PartialEq for Entry<V, C> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<V: PartialOrd, C> PartialOrd for Entry<V, C> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<V: Eq, C> Eq for Entry<V, C> {}

impl<V: Ord, C> Ord for Entry<V, C> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

trait EmpiricalDistribution<V, C>
where
    C: num_traits::Num + Copy,
    V: Hash + Ord + Copy,
{
    fn from_points<'a>(points: impl IntoIterator<Item = &'a V>) -> Self
    where
        V: 'a;

    fn total(&self) -> C;

    fn left_sided_cumulative(&mut self, x: V) -> C;
}

#[derive(Clone, Debug)]
pub struct StaticEmpiricalDistribution<V, C = usize> {
    total: C,
    sorted: Vec<Entry<V, C>>,
}

impl<V, C> EmpiricalDistribution<V, C> for StaticEmpiricalDistribution<V, C>
where
    C: num_traits::Num + Copy,
    V: Hash + Ord + Copy,
{
    fn from_points<'a>(points: impl IntoIterator<Item = &'a V>) -> Self
    where
        V: 'a,
    {
        let mut counts = HashMap::new();
        for &point in points {
            counts
                .entry(point)
                .and_modify(|count| *count = *count + C::one())
                .or_insert(C::one());
        }
        let mut sorted = counts
            .into_iter()
            .map(|(value, count)| Entry { value, count })
            .collect::<Vec<_>>();
        sorted.sort_unstable();
        let total = sorted
            .iter_mut()
            .fold(C::zero(), |total, Entry { value: _, count }| {
                let a = *count;
                *count = total;
                a + total //TODO: this might wrap
            });
        Self { total, sorted }
    }

    fn left_sided_cumulative(&mut self, x: V) -> C {
        let index = self.sorted.partition_point(|entry| entry.value < x);
        self.sorted
            .get(index)
            .map(|entry| entry.count)
            .unwrap_or(self.total)
    }

    #[inline(always)]
    fn total(&self) -> C {
        self.total
    }
}

pub struct DynamicEmpiricalDistribution<V, C: Add<Output = C> + Default + Copy> {
    total: C,
    sorted: SplayTree<(Entry<V, C>, SumSummary<C>, Unit)>,
}

impl<V, C: Add<Output = C> + Default + Copy + Debug> Debug for DynamicEmpiricalDistribution<V, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DynamicEmpiricalDistribution")
            .field("total", &self.total)
            .finish_non_exhaustive()
    }
}

#[derive(Default, Clone, Copy, Debug)]
struct SumSummary<C> {
    sum: C,
}

impl<C: Add<Output = C>> Add for SumSummary<C> {
    type Output = SumSummary<C>;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            sum: self.sum + rhs.sum,
        }
    }
}

impl<V, C: Add<Output = C> + Copy> ToSummary<SumSummary<C>> for Entry<V, C> {
    fn to_summary(&self) -> SumSummary<C> {
        SumSummary { sum: self.count }
    }
}

impl<V: Ord, C> Keyed<V> for Entry<V, C> {
    fn get_key(&self) -> &V {
        &self.value
    }
}

impl<V, C> EmpiricalDistribution<V, C> for DynamicEmpiricalDistribution<V, C>
where
    C: Add<Output = C> + Default + Copy + num_traits::Num,
    V: Hash + Ord + Copy,
{
    fn from_points<'a>(points: impl IntoIterator<Item = &'a V>) -> Self
    where
        V: 'a,
    {
        let mut counts = HashMap::new();
        for &point in points {
            counts
                .entry(point)
                .and_modify(|count| *count = *count + C::one())
                .or_insert(C::one());
        }
        let mut sorted = counts
            .into_iter()
            .map(|(value, count)| Entry { value, count })
            .collect::<Vec<_>>();
        sorted.sort_unstable();
        let mut sorted = sorted
            .into_iter()
            .collect::<SplayTree<(Entry<V, C>, SumSummary<C>, Unit)>>();
        let total = sorted.slice(..).summary().sum;

        Self { total, sorted }
    }

    fn left_sided_cumulative(&mut self, x: V) -> C {
        self.sorted.slice(ByKey(..&x)).summary().sum
    }

    #[inline(always)]
    fn total(&self) -> C {
        self.total
    }
}

impl<V, C> DynamicEmpiricalDistribution<V, C>
where
    C: Add<Output = C> + Default + Copy + num_traits::Num,
    V: Ord + Copy,
{
    pub fn replace(&mut self, old: V, new: V) -> Result<(C, C), ()> {
        let new_count_at_old = match self.sorted.slice(ByKey((&old,))).delete() {
            None => return Err(()),
            Some(Entry {
                value: _,
                mut count,
            }) => {
                count = count - C::one();
                if count != C::zero() {
                    self.sorted
                        .slice(ByKey((&old,)))
                        .insert(Entry { value: old, count });
                }
                count
            }
        };

        let mut count = self
            .sorted
            .slice(ByKey((&new,)))
            .delete()
            .map(|entry| entry.count)
            .unwrap_or_default();
        count = count + C::one();
        self.sorted
            .slice(ByKey((&new,)))
            .insert(Entry { value: new, count });
        Ok((new_count_at_old, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_empirical_distribution() {
        todo!()
    }

    #[test]
    fn dynamic_empirical_distribution() {
        todo!()
    }
}
