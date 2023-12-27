use core::ops::{Mul, Sub};

use super::EmpiricalDistribution;

/// # Notes
///
/// - `F` is the float type with which the bisection in quantile-space is performed.
pub fn vbq_quadratic_distortion<F, V, C>(
    prior: &impl EmpiricalDistribution<V, C>,
    unquantized: V,
    beta: V,
) -> V
where
    F: Copy + num_traits::float::FloatCore + num_traits::AsPrimitive<C> + 'static,
    V: Copy
        + Ord
        + Mul<Output = V>
        + Sub<Output = V>
        + num_traits::Zero
        + num_traits::Bounded
        + 'static,
    C: Copy + Ord + num_traits::Num + num_traits::AsPrimitive<F>,
{
    let half = F::one() / (F::one() + F::one());
    let mut lower = F::zero();
    let upper = prior.total();
    assert!(upper > C::zero());
    let mut last_mid_int = upper + C::one();
    let mut upper = upper.as_();

    let target = prior.left_sided_cumulative(unquantized).as_();

    let mut current_rate = V::zero();
    let mut record_point = unquantized; // Will be overwritten at least once.
    let mut record_objective = V::max_value();

    loop {
        let mid = half * (lower + upper);
        if mid <= target {
            lower = mid;
        }
        if mid >= target {
            upper = mid;
        }

        let mid_int = mid.as_();
        if mid_int == last_mid_int {
            break;
        }
        last_mid_int = mid_int;

        let candidate = prior
            .inverse_cumulative(mid_int)
            .expect("`mid_int < prior.total()`");
        let deviation = candidate - unquantized;
        let distortion = deviation * deviation;
        let current_objective = distortion + current_rate;
        if current_objective <= record_objective {
            record_point = candidate;
            record_objective = current_objective;
        }

        current_rate = current_rate + beta;
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
    use std::dbg;

    use super::*;

    use super::super::DynamicEmpiricalDistribution;
    use crate::F32;

    use alloc::vec::Vec;
    use rand::{seq::SliceRandom, RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;

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

        let prior = DynamicEmpiricalDistribution::<F32, u32>::try_from_points(&points).unwrap();
        let initial_entropy = prior.entropy_base2::<f32>();
        dbg!(initial_entropy);
        let mut entropy_previous_coarseness = initial_entropy;

        #[cfg(not(miri))]
        let (num_repeats, betas) = (5, [1e-7, 1e-5, 0.001, 0.01, 0.1]);

        #[cfg(miri)]
        let (num_repeats, betas) = (2, [0.001, 0.1]);

        for beta in betas {
            dbg!(beta);
            let beta = F32::new(beta).unwrap();
            let mut prior = prior.clone();
            let mut shifted_points = points.clone();
            let mut previous_entropy = initial_entropy;

            for i in 0..num_repeats {
                for (point, shifted_point) in points.iter().zip(shifted_points.iter_mut()) {
                    let quant = vbq_quadratic_distortion::<f32, _, _>(&prior, *point, beta);
                    prior.remove(*shifted_point).unwrap();
                    prior.insert(quant);
                    *shifted_point = quant;
                }
                let entropy = prior.entropy_base2::<f32>();
                dbg!(entropy);
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
}
