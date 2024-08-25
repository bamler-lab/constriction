use core::{borrow::Borrow, marker::PhantomData};

use alloc::{boxed::Box, vec::Vec};
use num_traits::{float::FloatCore, AsPrimitive};

use crate::{generic_static_asserts, wrapping_pow2, BitArray};

use super::{
    super::{DecoderModel, EntropyModel, IterableEntropyModel},
    accumulate_nonzero_probabilities,
    contiguous::ContiguousCategoricalEntropyModel,
    fast_quantized_cdf, iter_extended_cdf, perfectly_quantized_probabilities,
};

/// A tabularized [`DecoderModel`] that is optimized for fast decoding of i.i.d. symbols
///
/// TODO: documentation
#[derive(Debug, Clone, Copy)]
pub struct ContiguousLookupDecoderModel<
    Probability = u16,
    Cdf = Vec<Probability>,
    LookupTable = Box<[Probability]>,
    const PRECISION: usize = 12,
> where
    Probability: BitArray,
{
    /// Satisfies invariant:
    /// `lookup_table.as_ref().len() == 1 << PRECISION`
    lookup_table: LookupTable,

    /// Satisfies invariant:
    /// `left_sided_cumulative_and_symbol.as_ref().len()
    /// == *lookup_table.as_ref().iter().max() as usize + 2`
    cdf: Cdf,

    phantom: PhantomData<Probability>,
}

impl<Probability, const PRECISION: usize>
    ContiguousLookupDecoderModel<Probability, Vec<Probability>, Box<[Probability]>, PRECISION>
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
{
    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities_contiguous_fast<F>(
        probabilities: &[F],
        normalization: Option<F>,
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + AsPrimitive<Probability>,
        Probability: AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability> + AsPrimitive<F>,
    {
        generic_static_asserts!(
            (Probability: BitArray; const PRECISION: usize);
            PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
            PRECISION_MUST_BE_NONZERO: PRECISION > 0;
            USIZE_MUST_STRICTLY_SUPPORT_PRECISION: PRECISION < <usize as BitArray>::BITS;
        );

        let mut cdf =
            fast_quantized_cdf::<Probability, F, PRECISION>(probabilities, normalization)?;

        let mut extended_cdf = Vec::with_capacity(probabilities.len() + 1);
        extended_cdf.push(cdf.next().expect("cdf is not empty"));
        let mut lookup_table = Vec::with_capacity(1 << PRECISION);

        for (index, right_cumulative) in cdf.enumerate() {
            extended_cdf.push(right_cumulative);
            lookup_table.resize(right_cumulative.as_(), index.as_());
        }

        extended_cdf.push(wrapping_pow2(PRECISION));
        lookup_table.resize(1 << PRECISION, (probabilities.len() - 1).as_());

        Ok(Self {
            lookup_table: lookup_table.into_boxed_slice(),
            cdf: extended_cdf,
            phantom: PhantomData,
        })
    }

    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities_contiguous_perfect<F>(
        probabilities: &[F],
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        let slots = perfectly_quantized_probabilities::<_, _, PRECISION>(probabilities)?;
        Self::from_nonzero_fixed_point_probabilities_contiguous(
            slots.into_iter().map(|slot| slot.weight),
            false,
        )
    }

    /// Create a `LookupDecoderModel` over a contiguous range of symbols.
    ///
    /// TODO: example
    #[deprecated(
        since = "0.4.0",
        note = "Please use `from_symbols_and_floating_point_probabilities_fast` or \
        `from_symbols_and_floating_point_probabilities_perfect` instead. See documentation for \
        detailed upgrade instructions."
    )]
    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities_contiguous<F>(probabilities: &[F]) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        Self::from_floating_point_probabilities_contiguous_perfect(probabilities)
    }

    /// Create a `LookupDecoderModel` over a contiguous range of symbols using fixed point arighmetic.
    ///
    /// # Example
    ///
    /// See [`SmallContiguousLookupDecoderModel`].
    #[allow(clippy::result_unit_err)]
    pub fn from_nonzero_fixed_point_probabilities_contiguous<I>(
        probabilities: I,
        infer_last_probability: bool,
    ) -> Result<Self, ()>
    where
        I: IntoIterator,
        I::Item: Borrow<Probability>,
    {
        generic_static_asserts!(
            (Probability: BitArray; const PRECISION: usize);
            PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
            PRECISION_MUST_BE_NONZERO: PRECISION > 0;
            USIZE_MUST_STRICTLY_SUPPORT_PRECISION: PRECISION < <usize as BitArray>::BITS;
        );

        let mut lookup_table = Vec::with_capacity(1 << PRECISION);
        let probabilities = probabilities.into_iter();
        let mut cdf =
            Vec::with_capacity(probabilities.size_hint().0 + 1 + infer_last_probability as usize);
        accumulate_nonzero_probabilities::<_, _, _, _, _, PRECISION>(
            core::iter::repeat(()),
            probabilities,
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
            cdf,
            phantom: PhantomData,
        })
    }
}

impl<Probability, Cdf, LookupTable, const PRECISION: usize>
    ContiguousLookupDecoderModel<Probability, Cdf, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Cdf: AsRef<[Probability]>,
    LookupTable: AsRef<[Probability]>,
{
    /// Makes a very cheap shallow copy of the model that can be used much like a shared
    /// reference.
    ///
    /// The returned `LookupDecoderModel` implements `Copy`, which is a requirement for some
    /// methods, such as [`Decode::decode_iid_symbols`]. These methods could also accept a
    /// shared reference to a `NonContiguousCategoricalDecoderModel` (since all references
    /// to entropy models are also entropy models, and all shared references implement
    /// `Copy`), but passing a *view* instead may be slightly more efficient because it
    /// avoids one level of dereferencing.
    ///
    /// [`Decode::decode_iid_symbols`]: super::Decode::decode_iid_symbols
    pub fn as_view(
        &self,
    ) -> ContiguousLookupDecoderModel<Probability, &[Probability], &[Probability], PRECISION> {
        ContiguousLookupDecoderModel {
            lookup_table: self.lookup_table.as_ref(),
            cdf: self.cdf.as_ref(),
            phantom: PhantomData,
        }
    }

    /// Temporarily converts a lookup model into a non-lookup model (rarely useful).
    ///
    /// # See also
    ///
    /// - [`into_contiguous_categorical`](Self::into_contiguous_categorical)
    pub fn as_contiguous_categorical(
        &self,
    ) -> ContiguousCategoricalEntropyModel<Probability, &[Probability], PRECISION> {
        ContiguousCategoricalEntropyModel {
            cdf: self.cdf.as_ref(),
            phantom: PhantomData,
        }
    }

    /// Converts a lookup model into a non-lookup model.
    ///
    /// This drops the lookup table, so its only conceivable use case is to
    /// free memory while still holding on to the model.
    ///
    /// # See also
    ///
    /// - [`as_contiguous_categorical`](Self::as_contiguous_categorical)
    pub fn into_contiguous_categorical(
        self,
    ) -> ContiguousCategoricalEntropyModel<Probability, Cdf, PRECISION> {
        ContiguousCategoricalEntropyModel {
            cdf: self.cdf,
            phantom: PhantomData,
        }
    }
}

impl<Probability, Cdf, LookupTable, const PRECISION: usize> EntropyModel<PRECISION>
    for ContiguousLookupDecoderModel<Probability, Cdf, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
{
    type Symbol = usize;
    type Probability = Probability;
}

impl<Probability, Cdf, LookupTable, const PRECISION: usize> DecoderModel<PRECISION>
    for ContiguousLookupDecoderModel<Probability, Cdf, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
    Cdf: AsRef<[Probability]>,
    LookupTable: AsRef<[Probability]>,
{
    #[inline(always)]
    fn quantile_function(
        &self,
        quantile: Probability,
    ) -> (Self::Symbol, Probability, Probability::NonZero) {
        generic_static_asserts!(
            (Probability: BitArray; const PRECISION: usize);
            PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
            PRECISION_MUST_BE_NONZERO: PRECISION > 0;
        );

        if Probability::BITS != PRECISION {
            // It would be nice if we could avoid this but we currently don't statically enforce
            // `quantile` to fit into `PRECISION` bits.
            assert!(quantile < Probability::one() << PRECISION);
        }

        // SAFETY:
        // - `quantile_to_index` has length `1 << PRECISION` and we verified that
        //   `quantile` fits into `PRECISION` bits above.
        // - `left_sided_cumulative_and_symbol` has length
        //   `*quantile_to_index.as_ref().iter().max() as usize + 2`, so we can always
        //   access it at `index + 1` for `index` coming from `quantile_to_index`.
        let (left_sided_cumulative, symbol, next_cumulative) = unsafe {
            let index = *self.lookup_table.as_ref().get_unchecked(quantile.into());
            let index = index.into();
            let cdf = self.cdf.as_ref();
            (
                *cdf.get_unchecked(index),
                index,
                *cdf.get_unchecked(index + 1),
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

impl<'m, Probability, Cdf, const PRECISION: usize>
    From<&'m ContiguousCategoricalEntropyModel<Probability, Cdf, PRECISION>>
    for ContiguousLookupDecoderModel<Probability, Vec<Probability>, Box<[Probability]>, PRECISION>
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Cdf: AsRef<[Probability]>,
{
    fn from(model: &'m ContiguousCategoricalEntropyModel<Probability, Cdf, PRECISION>) -> Self {
        let cdf = model.cdf.as_ref().to_vec();
        let mut lookup_table = Vec::with_capacity(1 << PRECISION);
        for (symbol, &cumulative) in model.cdf.as_ref()[1..model.cdf.as_ref().len() - 1]
            .iter()
            .enumerate()
        {
            lookup_table.resize(cumulative.into(), symbol.as_());
        }
        lookup_table.resize(1 << PRECISION, (model.cdf.as_ref().len() - 2).as_());

        Self {
            lookup_table: lookup_table.into_boxed_slice(),
            cdf,
            phantom: PhantomData,
        }
    }
}

impl<'m, Probability, Cdf, LookupTable, const PRECISION: usize> IterableEntropyModel<'m, PRECISION>
    for ContiguousLookupDecoderModel<Probability, Cdf, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Cdf: AsRef<[Probability]>,
    LookupTable: AsRef<[Probability]>,
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
        iter_extended_cdf(
            self.cdf
                .as_ref()
                .iter()
                .enumerate()
                .map(|(symbol, &cumulative)| (cumulative, symbol)),
        )
    }
}

#[cfg(test)]
mod tests {
    use alloc::{ vec};

    use crate::stream::{model::EncoderModel, stack::DefaultAnsCoder, Decode};

    use super::*;

    #[test]
    fn lookup_contiguous() {
        let probabilities = vec![3u8, 18, 1, 42];
        let model =
            ContiguousCategoricalEntropyModel::<_, _, 6>::from_nonzero_fixed_point_probabilities(
                probabilities,
                false,
            )
            .unwrap();
        let lookup_decoder_model = model.to_lookup_decoder_model();

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
}
