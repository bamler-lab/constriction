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
/// over a arbitrary (i.e., not necessarily contiguous) alphabet of symbols
///
/// The default type parameters correspond to the "small" [preset], i.e., they allow
/// decoding with a [`SmallAnsCoder`] or a [`SmallRangeDecoder`] (as well as with a
/// [`DefaultAnsCoder`] or a [`DefaultRangeDecoder`], since you can always use a "bigger"
/// coder on a "smaller" model). Increasing the const generic `PRECISION` by much beyond its
/// default value is not recommended because the size of the lookup table grows
/// exponentially in `PRECISION`, thus increasing both memory consumption and runtime (due
/// to reduced cache locality).
///
/// # See also
///
/// - [`ContiguousLookupDecoderModel`]
///
/// # Example
///
/// ## Full typical usage example
///
/// ```
/// use constriction::stream::{
///     model::{
///         EncoderModel, NonContiguousLookupDecoderModel, IterableEntropyModel,
///         SmallNonContiguousCategoricalEncoderModel,
///     },
///     queue::{SmallRangeDecoder, SmallRangeEncoder},
///     Decode, Encode,
/// };
///
/// // Let's first encode some message. We use a `SmallContiguousCategoricalEntropyModel`
/// // for encoding, so that we can decode with a lookup model later.
/// let message = "Mississippi";
/// let symbols = ['M', 'i', 's', 'p'];
/// let floating_point_probabilities = [1.0f32 / 11., 4. / 11., 4. / 11., 2. / 11.];
/// let encoder_model = SmallNonContiguousCategoricalEncoderModel
///     ::from_symbols_and_floating_point_probabilities_perfect(
///         symbols.iter().cloned(),
///         &floating_point_probabilities,
///     )
///     .unwrap();
/// let mut encoder = SmallRangeEncoder::new();
/// encoder.encode_iid_symbols(message.chars(), &encoder_model);
///
/// let compressed = encoder.get_compressed();
/// let fixed_point_probabilities = symbols
///     .iter()
///     .map(|symbol| encoder_model.left_cumulative_and_probability(symbol).unwrap().1.get())
///     .collect::<Vec<_>>();
///
/// // ... write `compressed` and `fixed_point_probabilities` to a file and read them back ...
///
/// let lookup_decoder_model =
///     NonContiguousLookupDecoderModel::<_, u16, _, _>::from_symbols_and_nonzero_fixed_point_probabilities(
///         symbols.iter().cloned(),
///         &fixed_point_probabilities,
///         false,
///     )
///     .unwrap();
/// let mut decoder = SmallRangeDecoder::from_compressed(compressed).unwrap();
///
/// let reconstructed = decoder
///     .decode_iid_symbols(11, &lookup_decoder_model)
///     .collect::<Result<String, _>>()
///     .unwrap();
///
/// assert_eq!(&reconstructed, message);
/// ```
///
/// ## Compatibility with "default" entropy coders
///
/// The above example uses coders with the "small" [preset] to demonstrate typical usage of
/// lookup decoder models. However, lookup models are also compatible with coders with the
/// "default" preset (you can always use a "smaller" model with a "larger" coder; so you
/// could, e.g., encode part of a message with a model that uses the "default" preset and
/// another part of the message with a model that uses the "small" preset so it can be
/// decoded with a lookup model).
///
/// ```
/// // Same imports, `message`, `symbols` and `floating_point_probabilities` as above ...
/// # use constriction::stream::{
/// #     model::{
/// #         EncoderModel, NonContiguousLookupDecoderModel, IterableEntropyModel,
/// #         SmallNonContiguousCategoricalEncoderModel,
/// #     },
/// #     queue::{DefaultRangeDecoder, DefaultRangeEncoder},
/// #     Decode, Encode,
/// # };
/// #
/// # let message = "Mississippi";
/// # let symbols = ['M', 'i', 's', 'p'];
/// # let floating_point_probabilities = [1.0f32 / 11., 4. / 11., 4. / 11., 2. / 11.];
///
/// let encoder_model = SmallNonContiguousCategoricalEncoderModel
///     ::from_symbols_and_floating_point_probabilities_perfect(
///         symbols.iter().cloned(),
///         &floating_point_probabilities,
///     )
///     .unwrap(); // We're using a "small" encoder model again ...
/// let mut encoder = DefaultRangeEncoder::new(); // ... but now with a "default" coder.
/// encoder.encode_iid_symbols(message.chars(), &encoder_model);
///
/// // ... obtain `compressed` and `fixed_point_probabilities` as in the example above ...
/// # let compressed = encoder.get_compressed();
/// # let fixed_point_probabilities = symbols
/// #     .iter()
/// #     .map(|symbol| encoder_model.left_cumulative_and_probability(symbol).unwrap().1.get())
/// #     .collect::<Vec<_>>();
///
/// // Then decode with the same lookup model as before, but now with a "default" decoder:
/// let lookup_decoder_model =
///     NonContiguousLookupDecoderModel::<_, u16, _, _>::from_symbols_and_nonzero_fixed_point_probabilities(
///         symbols.iter().cloned(),
///         &fixed_point_probabilities,
///         false,
///     )
///     .unwrap();
/// let mut decoder = DefaultRangeDecoder::from_compressed(compressed).unwrap();
///
/// let reconstructed = decoder
///     .decode_iid_symbols(11, &lookup_decoder_model)
///     .collect::<Result<String, _>>()
///     .unwrap();
///
/// assert_eq!(&reconstructed, message);
/// ```
///
/// You can also use an [`AnsCoder`] instead of a range coder of course.
///
/// [`AnsCoder`]: crate::stream::stack::AnsCoder
/// [`SmallAnsCoder`]: crate::stream::stack::SmallAnsCoder
/// [`SmallRangeDecoder`]: crate::stream::queue::SmallRangeDecoder
/// [`DefaultAnsCoder`]: crate::stream::stack::DefaultAnsCoder
/// [`DefaultRangeDecoder`]: crate::stream::queue::DefaultRangeDecoder
/// [`ContiguousLookupDecoderModel`]:
///     crate::stream::model::ContiguousLookupDecoderModel
/// [preset]: crate::stream#presets
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
    /// Constructs a lookup table for decoding whose PMF approximates given `probabilities`.
    ///
    /// Similar to [`from_symbols_and_floating_point_probabilities_fast`], but the resulting
    /// (fixed-point precision) model typically approximates the provided floating point
    /// `probabilities` slightly better.
    ///
    /// See [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect`]
    /// for a detailed comparison between `..._fast` and `..._perfect` constructors of
    /// categorical entropy models. However, note that, different to the case with
    /// non-lookup models, using this `..._perfect` variant of the constructor may be
    /// justified in more cases because
    /// - lookup decoder models use a smaller `PRECISION` by default, so the differences in
    ///   bit rate between the `..._fast` and the `..._perfect` constructor are more
    ///   pronounced; and
    /// - lookup decoder models should only be used if you expect to use them to decode
    ///   *lots of* symbols anyway, so an increased upfront cost for construction is less of
    ///   an issue.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{
    ///     model::NonContiguousLookupDecoderModel,
    ///     stack::SmallAnsCoder,
    ///     Decode, Code,
    /// };
    ///
    /// let probabilities = [0.3f32, 0.1, 0.4, 0.2];
    /// let symbols = ['a', 'b', 'x', 'y'];
    /// let decoder_model = NonContiguousLookupDecoderModel::<_, u16, _, _>
    ///     ::from_symbols_and_floating_point_probabilities_perfect(
    ///         symbols.iter().copied(),
    ///         &probabilities
    ///     ).unwrap();
    ///
    /// let compressed = [0x956Eu16, 0x0155]; // (imagine this was read from a file)
    /// let expected = ['b', 'a', 'a', 'x', 'x', 'y', 'x', 'x', 'a'];
    /// let mut coder = SmallAnsCoder::from_compressed_slice(&compressed).unwrap();
    ///
    /// let reconstructed = coder
    ///     .decode_iid_symbols(9, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
    /// assert!(coder.is_empty());
    /// assert_eq!(reconstructed, expected);
    /// ```
    ///
    /// [`from_symbols_and_floating_point_probabilities_fast`]:
    ///     Self::from_symbols_and_floating_point_probabilities_fast
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect`]:
    ///     crate::stream::model::ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect
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

    /// Faster but less accurate variant of
    /// [`from_symbols_and_floating_point_probabilities_perfect`]
    ///
    /// Semantics are analogous to
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast`],
    /// except that this method constructs a *lookup table*, i.e., a model that takes
    /// considerably more runtime to build but, once built, is optimized for very fast
    /// decoding of lots i.i.d. symbols.
    ///
    /// # See also
    ///
    /// - [`from_symbols_and_floating_point_probabilities_perfect`], which can be slower but
    ///   typically approximates the provided `probabilities` better, which may be a good
    ///   trade-off in the kind of situations that lookup decoder models are intended for.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{
    ///     model::NonContiguousLookupDecoderModel,
    ///     stack::SmallAnsCoder,
    ///     Decode, Code,
    /// };
    ///
    /// let probabilities = [0.3f32, 0.1, 0.4, 0.2];
    /// let symbols = ['a', 'b', 'x', 'y'];
    /// let decoder_model = NonContiguousLookupDecoderModel::<_, u16, _, _>
    ///     ::from_symbols_and_floating_point_probabilities_fast(
    ///         symbols.iter().copied(),
    ///         &probabilities,
    ///         None,
    ///     ).unwrap();
    ///
    /// let compressed = [0xF592u16, 0x0133]; // (imagine this was read from a file)
    /// let expected = ['b', 'a', 'a', 'x', 'x', 'y', 'x', 'x', 'a'];
    /// let mut coder = SmallAnsCoder::from_compressed_slice(&compressed).unwrap();
    ///
    /// let reconstructed = coder
    ///     .decode_iid_symbols(9, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
    /// assert!(coder.is_empty());
    /// assert_eq!(reconstructed, expected);
    /// ```
    ///
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast`]:
    ///     crate::stream::model::ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast
    /// [`from_symbols_and_floating_point_probabilities_perfect`]:
    ///     Self::from_symbols_and_floating_point_probabilities_perfect
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

    /// Deprecated constructor.
    ///
    /// This constructor has been deprecated in constriction version 0.4.0, and it will be
    /// removed in constriction version 0.5.0.
    ///
    /// # Upgrade Instructions
    ///
    /// Call either of the following two constructors instead:
    /// - [`from_symbols_and_floating_point_probabilities_perfect`] (referred to as
    ///   `..._perfect` in the following); or
    /// - [`from_symbols_and_floating_point_probabilities_fast`] (referred to as `..._fast`
    ///   in the following).
    ///
    /// Both constructors approximate the given (floating-point) probability distribution in
    /// fixed point arithmetic, and both construct a valid (exactly invertible) model that
    /// is guaranteed to assign a nonzero probability to all symbols. The `..._perfect`
    /// constructor finds the best possible approximation of the provided fixed-point
    /// probability distribution (thus leading to the lowest bit rates), while the
    /// `..._fast` constructor is faster but may find a *slightly* imperfect approximation.
    /// Note that, since lookup models use a smaller fixed-point `PRECISION` than other
    /// models (e.g., [`NonContiguousCategoricalDecoderModel`]), the difference in bit rate
    /// between the two models is more pronounced.
    ///
    /// Note that the `..._fast` constructor breaks binary compatibility with `constriction`
    /// version <= 0.3.5. If you need to be able to exchange binary compressed data with a
    /// program that uses a lookup decoder model or a categorical entropy model from
    /// `constriction` version <= 0.3.5, then call
    /// [`from_symbols_and_floating_point_probabilities_perfect`].
    ///
    /// # Compatibility Table
    ///
    /// (Lookup decoder models can only be used for decoding; in the following table,
    /// "encoding" refers to [`NonContiguousCategoricalEncoderModel`])
    ///
    /// | constructor used for encoding → <br> ↓ constructor used for decoding ↓ | [legacy](crate::stream::model::NonContiguousCategoricalEncoderModel::from_symbols_and_floating_point_probabilities) |  [`..._perfect`](crate::stream::model::NonContiguousCategoricalEncoderModel::from_symbols_and_floating_point_probabilities_perfect) | [`..._fast`](crate::stream::model::NonContiguousCategoricalEncoderModel::from_symbols_and_floating_point_probabilities_fast) |
    /// | --------------------: | --------------- | --------------- | --------------- |
    /// | **legacy (this one)** | ✅ compatible   | ✅ compatible   | ❌ incompatible |
    /// | **[`..._perfect`]**   | ✅ compatible   | ✅ compatible   | ❌ incompatible |
    /// | **[`..._fast`]**      | ❌ incompatible | ❌ incompatible | ✅ compatible   |
    ///
    /// [`from_symbols_and_floating_point_probabilities_perfect`]:
    ///     Self::from_symbols_and_floating_point_probabilities_perfect
    /// [`..._perfect`]: Self::from_symbols_and_floating_point_probabilities_perfect
    /// [`from_symbols_and_floating_point_probabilities_fast`]:
    ///     Self::from_symbols_and_floating_point_probabilities_fast
    /// [`..._fast`]: Self::from_symbols_and_floating_point_probabilities_fast
    /// [`NonContiguousCategoricalEncoderModel`]:
    ///     crate::stream::model::NonContiguousCategoricalEncoderModel
    /// [`NonContiguousCategoricalDecoderModel`]:
    ///     crate::stream::model::NonContiguousCategoricalDecoderModel
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

    /// Create a `NonContiguousLookupDecoderModel` from pre-calculated fixed-point
    /// probabilities.
    ///
    /// # Example
    ///
    /// See [type level documentation](Self).
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
    /// This drops the lookup table, so its only conceivable use case is to free memory
    /// while still holding on to (a slower variant of) the model.
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
