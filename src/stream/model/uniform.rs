use core::borrow::Borrow;

use num_traits::AsPrimitive;

use crate::{generic_static_asserts, wrapping_pow2, BitArray, NonZeroBitArray};

use super::{DecoderModel, EncoderModel, EntropyModel, IterableEntropyModel};

#[derive(Debug, Clone, Copy)]
pub struct UniformModel<Probability: BitArray, const PRECISION: usize> {
    probability_per_bin: Probability::NonZero,
    last_symbol: Probability,
}

impl<Probability: BitArray, const PRECISION: usize> UniformModel<Probability, PRECISION> {
    pub fn new(range: usize) -> Self
    where
        usize: AsPrimitive<Probability>,
        Probability: AsPrimitive<usize>,
    {
        generic_static_asserts!(
            (Probability: BitArray; const PRECISION: usize);
            PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
            USIZE_MUST_SUPPORT_PRECISION: PRECISION <= <usize as BitArray>::BITS;
            PRECISION_MUST_BE_NONZERO: PRECISION > 0;
        );

        assert!(range > 1); // We don't support degenerate probability distributions (i.e. range=1).
        let range = unsafe { range.into_nonzero_unchecked() }; // For performance hint.
        let last_symbol_usize = NonZeroBitArray::get(range) - 1;
        let last_symbol = last_symbol_usize.as_();
        assert!(
            last_symbol
                <= wrapping_pow2::<Probability>(PRECISION).wrapping_sub(&Probability::one())
                && last_symbol.as_() == last_symbol_usize
        );

        if PRECISION == Probability::BITS {
            let probability_per_bin = (wrapping_pow2::<usize>(PRECISION)
                .wrapping_sub(NonZeroBitArray::get(range))
                / NonZeroBitArray::get(range))
            .as_()
                + Probability::one();
            unsafe {
                Self {
                    probability_per_bin: probability_per_bin.into_nonzero_unchecked(),
                    last_symbol,
                }
            }
        } else {
            let probability_per_bin =
                (Probability::one() << PRECISION) / NonZeroBitArray::get(range).as_();
            let probability_per_bin = probability_per_bin
                .into_nonzero()
                .expect("range <= (1 << PRECISION)");
            Self {
                probability_per_bin,
                last_symbol,
            }
        }
    }
}

impl<Probability: BitArray, const PRECISION: usize> EntropyModel<PRECISION>
    for UniformModel<Probability, PRECISION>
{
    type Symbol = usize;
    type Probability = Probability;
}

impl<Probability: BitArray, const PRECISION: usize> EncoderModel<PRECISION>
    for UniformModel<Probability, PRECISION>
where
    usize: AsPrimitive<Probability>,
{
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Self::Symbol>,
    ) -> Option<(Self::Probability, <Self::Probability as BitArray>::NonZero)> {
        let symbol = symbol.borrow().as_();
        let left_cumulative = symbol.wrapping_mul(&self.probability_per_bin.get());

        #[allow(clippy::comparison_chain)]
        if symbol < self.last_symbol {
            // Most common case.
            Some((left_cumulative, self.probability_per_bin))
        } else if symbol == self.last_symbol {
            // Less common but possible case.
            let probability =
                wrapping_pow2::<Probability>(PRECISION).wrapping_sub(&left_cumulative);
            let probability = unsafe { probability.into_nonzero_unchecked() };
            Some((left_cumulative, probability))
        } else {
            // Least common case.
            None
        }
    }
}

impl<Probability: BitArray, const PRECISION: usize> DecoderModel<PRECISION>
    for UniformModel<Probability, PRECISION>
where
    Probability: AsPrimitive<usize>,
{
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (
        Self::Symbol,
        Self::Probability,
        <Self::Probability as BitArray>::NonZero,
    ) {
        let symbol_guess = quantile / self.probability_per_bin.get(); // Might be 1 too large for last symbol.
        let remainder = quantile % self.probability_per_bin.get();
        if symbol_guess < self.last_symbol {
            (
                symbol_guess.as_(),
                quantile - remainder,
                self.probability_per_bin,
            )
        } else {
            let left_cumulative = self.last_symbol * self.probability_per_bin.get();
            let prob = wrapping_pow2::<Probability>(PRECISION).wrapping_sub(&left_cumulative);
            let prob = unsafe {
                // SAFETY: prob can't be zero because we have a `quantile` that is contained in its interval.
                prob.into_nonzero_unchecked()
            };
            (self.last_symbol.as_(), left_cumulative, prob)
        }
    }
}

impl<'m, Probability: BitArray, const PRECISION: usize> IterableEntropyModel<'m, PRECISION>
    for UniformModel<Probability, PRECISION>
where
    Probability: AsPrimitive<usize>,
    usize: AsPrimitive<Probability>,
{
    fn symbol_table(
        &'m self,
    ) -> impl Iterator<
        Item = (
            Self::Symbol,
            Self::Probability,
            <Self::Probability as BitArray>::NonZero,
        ),
    > {
        // The following doesn't truncate on the conversion or overflow on the addition because it
        // inverts an operation that was performed in the constructor (which checked for both
        // potential sources of error).
        let last_symbol = self.last_symbol.as_();
        let range = last_symbol + 1;
        let probability_per_bin = self.probability_per_bin;

        (0..range).map(move |symbol| {
            let left_cumulative = symbol.as_() * probability_per_bin.get();
            let probability = if symbol != last_symbol {
                probability_per_bin
            } else {
                let probability =
                    wrapping_pow2::<Probability>(PRECISION).wrapping_sub(&left_cumulative);

                // SAFETY: the constructor ensures that `range < 2^PRECISION`, so every bin has a
                // nonzero probability mass.
                unsafe { probability.into_nonzero_unchecked() }
            };

            (symbol, left_cumulative, probability)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::tests::test_entropy_model;

    #[test]
    fn uniform() {
        for range in [2, 3, 4, 5, 6, 7, 8, 9, 62, 63, 64, 254, 255, 256] {
            test_entropy_model(&UniformModel::<u32, 24>::new(range), 0..range);
            test_entropy_model(&UniformModel::<u32, 32>::new(range), 0..range);
            test_entropy_model(&UniformModel::<u16, 12>::new(range), 0..range);
            test_entropy_model(&UniformModel::<u16, 16>::new(range), 0..range);
            if range < 255 {
                test_entropy_model(&UniformModel::<u8, 8>::new(range), 0..range);
            }
            if range <= 64 {
                test_entropy_model(&UniformModel::<u8, 6>::new(range), 0..range);
            }
        }
    }
}
