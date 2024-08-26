use core::{borrow::Borrow, marker::PhantomData};

use alloc::{boxed::Box, vec::Vec};
use num_traits::{float::FloatCore, AsPrimitive};

use crate::{generic_static_asserts, wrapping_pow2, BitArray, NonZeroBitArray};

use super::{
    super::{DecoderModel, EntropyModel, IterableEntropyModel},
    accumulate_nonzero_probabilities, fast_quantized_cdf, iter_extended_cdf,
    non_contiguous::NonContiguousCategoricalDecoderModel,
    perfectly_quantized_probabilities,
};

/// A tabularized [`DecoderModel`] that is optimized for fast decoding of i.i.d. symbols
///
/// TODO: documentation
#[derive(Debug, Clone, Copy)]
pub struct NonContiguousLookupDecoderModel<
    Symbol,
    Probability = u16,
    Cdf = Vec<(Probability, Symbol)>,
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

    phantom: PhantomData<(Probability, Symbol)>,
}

impl<Symbol, Probability, const PRECISION: usize>
    NonContiguousLookupDecoderModel<
        Symbol,
        Probability,
        Vec<(Probability, Symbol)>,
        Box<[Probability]>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    Symbol: Clone,
    usize: AsPrimitive<Probability>,
{
    #[allow(clippy::result_unit_err)]
    pub fn from_symbols_and_floating_point_probabilities_fast<F>(
        symbols: impl IntoIterator<Item = Symbol>,
        probabilities: &[F],
        normalization: Option<F>,
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + AsPrimitive<Probability>,
        Probability: AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability> + AsPrimitive<F>,
    {
        let mut cdf =
            fast_quantized_cdf::<Probability, F, PRECISION>(probabilities, normalization)?;
        let mut left_cumulative = cdf.next().expect("cdf is not empty");
        let cdf = cdf.chain(core::iter::once(wrapping_pow2(PRECISION)));

        let symbol_table = symbols
            .into_iter()
            .zip(cdf)
            .map(|(symbol, right_cumulative)| {
                let probability = right_cumulative
                    .wrapping_sub(&left_cumulative)
                    .into_nonzero()
                    .expect("quantization is leaky");
                let old_left_cumulative = left_cumulative;
                left_cumulative = right_cumulative;
                (symbol, old_left_cumulative, probability)
            });

        Ok(Self::from_symbol_table(symbol_table))
    }

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

    /// Create a `LookupDecoderModel` over arbitrary symbols.
    ///
    /// TODO: example
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

    /// Create a `LookupDecoderModel` over arbitrary symbols.
    ///
    /// TODO: example
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
        generic_static_asserts!(
            (Probability: BitArray; const PRECISION: usize);
            PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
            PRECISION_MUST_BE_NONZERO: PRECISION > 0;
            USIZE_MUST_STRICTLY_SUPPORT_PRECISION: PRECISION < <usize as BitArray>::BITS;
        );

        let mut lookup_table = Vec::with_capacity(1 << PRECISION);
        let symbols = symbols.into_iter();
        let mut cdf =
            Vec::with_capacity(symbols.size_hint().0 + 1 + infer_last_probability as usize);
        let mut symbols = accumulate_nonzero_probabilities::<_, _, _, _, _, PRECISION>(
            symbols,
            probabilities.into_iter(),
            |symbol, _, probability| {
                let index = cdf.len().as_();
                cdf.push((lookup_table.len().as_(), symbol));
                lookup_table.resize(lookup_table.len() + probability.into(), index);
                Ok(())
            },
            infer_last_probability,
        )?;

        let last_symbol = cdf.last().expect("cdf is not empty").1.clone();
        cdf.push((wrapping_pow2(PRECISION), last_symbol));

        if symbols.next().is_some() {
            Err(())
        } else {
            Ok(Self {
                lookup_table: lookup_table.into_boxed_slice(),
                cdf,
                phantom: PhantomData,
            })
        }
    }

    pub fn from_iterable_entropy_model<'m, M>(model: &'m M) -> Self
    where
        M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
    {
        Self::from_symbol_table(model.symbol_table())
    }

    fn from_symbol_table(
        symbol_table: impl Iterator<Item = (Symbol, Probability, Probability::NonZero)>,
    ) -> Self {
        generic_static_asserts!(
            (Probability: BitArray; const PRECISION: usize);
            PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
            PRECISION_MUST_BE_NONZERO: PRECISION > 0;
            USIZE_MUST_STRICTLY_SUPPORT_PRECISION: PRECISION < <usize as BitArray>::BITS;
        );

        let mut lookup_table = Vec::with_capacity(1 << PRECISION);
        let mut cdf = Vec::with_capacity(symbol_table.size_hint().0 + 1);
        for (symbol, left_sided_cumulative, probability) in symbol_table {
            let index = cdf.len().as_();
            debug_assert_eq!(left_sided_cumulative, lookup_table.len().as_());
            cdf.push((lookup_table.len().as_(), symbol));
            lookup_table.resize(lookup_table.len() + probability.get().into(), index);
        }
        let last_symbol = cdf.last().expect("cdf is not empty").1.clone();
        cdf.push((wrapping_pow2(PRECISION), last_symbol));

        Self {
            lookup_table: lookup_table.into_boxed_slice(),
            cdf,
            phantom: PhantomData,
        }
    }
}

impl<Symbol, Probability, Cdf, LookupTable, const PRECISION: usize>
    NonContiguousLookupDecoderModel<Symbol, Probability, Cdf, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Cdf: AsRef<[(Probability, Symbol)]>,
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
    /// [`Decode::decode_iid_symbols`]: crate::stream::Decode::decode_iid_symbols
    pub fn as_view(
        &self,
    ) -> NonContiguousLookupDecoderModel<
        Symbol,
        Probability,
        &[(Probability, Symbol)],
        &[Probability],
        PRECISION,
    > {
        NonContiguousLookupDecoderModel {
            lookup_table: self.lookup_table.as_ref(),
            cdf: self.cdf.as_ref(),
            phantom: PhantomData,
        }
    }

    /// Temporarily converts a lookup model into a non-lookup model (rarely useful).
    ///
    /// # See also
    ///
    /// - [`into_non_contiguous_categorical`](Self::into_non_contiguous_categorical)
    pub fn as_non_contiguous_categorical(
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

    /// Converts a lookup model into a non-lookup model.
    ///
    /// This drops the lookup table, so its only conceivable use case is to
    /// free memory while still holding on to the model.
    ///
    /// # See also
    ///
    /// - [`as_non_contiguous_categorical`](Self::as_non_contiguous_categorical)
    pub fn into_non_contiguous_categorical(
        self,
    ) -> NonContiguousCategoricalDecoderModel<Symbol, Probability, Cdf, PRECISION> {
        NonContiguousCategoricalDecoderModel {
            cdf: self.cdf,
            phantom: PhantomData,
        }
    }
}

impl<Symbol, Probability, Cdf, LookupTable, const PRECISION: usize> EntropyModel<PRECISION>
    for NonContiguousLookupDecoderModel<Symbol, Probability, Cdf, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
{
    type Symbol = Symbol;
    type Probability = Probability;
}

impl<Symbol, Probability, Cdf, LookupTable, const PRECISION: usize> DecoderModel<PRECISION>
    for NonContiguousLookupDecoderModel<Symbol, Probability, Cdf, LookupTable, PRECISION>
where
    Probability: BitArray + Into<usize>,
    Cdf: AsRef<[(Probability, Symbol)]>,
    LookupTable: AsRef<[Probability]>,
    Symbol: Clone,
{
    #[inline(always)]
    fn quantile_function(
        &self,
        quantile: Probability,
    ) -> (Symbol, Probability, Probability::NonZero) {
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
        let ((left_sided_cumulative, symbol), next_cumulative) = unsafe {
            let index = *self.lookup_table.as_ref().get_unchecked(quantile.into());
            let index = index.into();
            let cdf = self.cdf.as_ref();
            (
                cdf.get_unchecked(index).clone(),
                cdf.get_unchecked(index + 1).0,
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

impl<'m, Symbol, Probability, M, const PRECISION: usize> From<&'m M>
    for NonContiguousLookupDecoderModel<
        Symbol,
        Probability,
        Vec<(Probability, Symbol)>,
        Box<[Probability]>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    Symbol: Clone,
    usize: AsPrimitive<Probability>,
    M: IterableEntropyModel<'m, PRECISION, Symbol = Symbol, Probability = Probability> + ?Sized,
{
    #[inline(always)]
    fn from(model: &'m M) -> Self {
        Self::from_iterable_entropy_model(model)
    }
}

impl<'m, Symbol, Probability, Cdf, LookupTable, const PRECISION: usize>
    IterableEntropyModel<'m, PRECISION>
    for NonContiguousLookupDecoderModel<Symbol, Probability, Cdf, LookupTable, PRECISION>
where
    Symbol: Clone + 'm,
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Cdf: AsRef<[(Probability, Symbol)]>,
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
        iter_extended_cdf(self.cdf.as_ref().iter().cloned())
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use crate::stream::{
        model::{EncoderModel, NonContiguousCategoricalEncoderModel},
        stack::DefaultAnsCoder,
        Decode,
    };

    use super::*;

    #[test]
    fn lookup_noncontiguous() {
        let symbols = "axcy";
        let probabilities = [3u8, 18, 1, 42];
        let encoder_model = NonContiguousCategoricalEncoderModel::<_, u8, 6>::from_symbols_and_nonzero_fixed_point_probabilities(
            symbols.chars(),probabilities.iter(),false
        )
        .unwrap();
        let decoder_model = NonContiguousCategoricalDecoderModel::<_, _,_, 6>::from_symbols_and_nonzero_fixed_point_probabilities(
            symbols.chars(),probabilities.iter(),false
        )
        .unwrap();
        let lookup_decoder_model =
            NonContiguousLookupDecoderModel::from_iterable_entropy_model(&decoder_model);

        // Verify that `decode(encode(x)) == x` and that `lookup_decode(encode(x)) == x`.
        for symbol in symbols.chars() {
            let (left_cumulative, probability) = encoder_model
                .left_cumulative_and_probability(symbol)
                .unwrap();
            for quantile in left_cumulative..left_cumulative + probability.get() {
                assert_eq!(
                    decoder_model.quantile_function(quantile),
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
            let (symbol, left_cumulative, probability) = decoder_model.quantile_function(quantile);
            assert_eq!(
                lookup_decoder_model.quantile_function(quantile),
                (symbol, left_cumulative, probability)
            );
            assert_eq!(
                encoder_model
                    .left_cumulative_and_probability(symbol)
                    .unwrap(),
                (left_cumulative, probability)
            );
        }

        // Test encoding and decoding a few symbols.
        let symbols = "axcxcyaac";
        let mut ans = DefaultAnsCoder::new();
        ans.encode_iid_symbols_reverse(symbols.chars(), &encoder_model)
            .unwrap();
        assert!(!ans.is_empty());
        let decoded = ans
            .decode_iid_symbols(9, &decoder_model)
            .collect::<Result<String, _>>()
            .unwrap();
        assert_eq!(decoded, symbols);
        assert!(ans.is_empty());
    }
}
