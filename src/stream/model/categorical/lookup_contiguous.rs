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

/// Type alias for a [`ContiguousLookupDecoderModel`] with sane [presets].
///
/// See documentation of [`ContiguousLookupDecoderModel`] for a detailed code example.
///
/// Note that, in contrast to most other models (and entropy coders), there is no type alias
/// for the "default" [preset] because using lookup tables with these presets is strongly
/// discouraged (the lookup tables would be enormous).
///
/// [preset]: crate::stream#presets
/// [presets]: crate::stream#presets
pub type SmallContiguousLookupDecoderModel<Cdf = Vec<u16>, LookupTable = Box<[u16]>> =
    ContiguousLookupDecoderModel<u16, Cdf, LookupTable, 12>;

/// A tabularized [`DecoderModel`] that is optimized for fast decoding of i.i.d. symbols
/// over a contiguous alphabet of symbols (i.e., `{0, 1, ..., n-1}`)
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
/// - [`NonContiguousLookupDecoderModel`]
///
/// # Example
///
/// ## Full typical usage example
///
/// ```
/// use constriction::stream::{
///     model::{
///         IterableEntropyModel, SmallContiguousCategoricalEntropyModel,
///         SmallContiguousLookupDecoderModel,
///     },
///     queue::{SmallRangeDecoder, SmallRangeEncoder},
///     Decode, Encode,
/// };
///
/// // Let's first encode some message. We use a `SmallContiguousCategoricalEntropyModel`
/// // for encoding, so that we can decode with a lookup model later.
/// let message = [2, 1, 3, 0, 0, 2, 0, 2, 1, 0, 2];
/// let floating_point_probabilities = [0.4f32, 0.2, 0.1, 0.3];
/// let encoder_model =
///     SmallContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(
///         &floating_point_probabilities,
///     )
///     .unwrap();
/// let mut encoder = SmallRangeEncoder::new();
/// encoder.encode_iid_symbols(message, &encoder_model);
///
/// // Note: we could construct a matching `decoder` and `lookup_decoder_model` as follows:
/// //   let mut decoder = encoder.into_decoder().unwrap();
/// //   let lookup_decoder_model = encoder_model.to_lookup_decoder_model();
/// # { // (actually run this in the doc test to be sure)
/// #    let mut decoder = encoder.clone().into_decoder().unwrap();
/// #    let lookup_decoder_model = encoder_model.to_lookup_decoder_model();
/// # }
/// // But in a more realistic compression setup, we'd want to serialize the compressed bit string
/// // (and possibly the model) to a file and read it back. So let's simulate this here:
///
/// let compressed = encoder.get_compressed();
/// let fixed_point_probabilities = encoder_model
///     .symbol_table()
///     .map(|(_symbol, _cdf, probability)| probability.get())
///     .collect::<Vec<_>>();
///
/// // ... write `compressed` and `fixed_point_probabilities` to a file and read them back ...
///
/// let lookup_decoder_model =
///     SmallContiguousLookupDecoderModel::from_nonzero_fixed_point_probabilities(
///         &fixed_point_probabilities,
///         false,
///     )
///     .unwrap();
/// let mut decoder = SmallRangeDecoder::from_compressed(compressed).unwrap();
///
/// let reconstructed = decoder
///     .decode_iid_symbols(11, &lookup_decoder_model)
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
///
/// assert_eq!(&reconstructed[..], message);
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
/// // Same imports, `message`, and `floating_point_probabilities` as in the example above ...
/// # use constriction::stream::{
/// #     model::{
/// #         IterableEntropyModel, SmallContiguousCategoricalEntropyModel,
/// #         SmallContiguousLookupDecoderModel,
/// #     },
/// #     queue::{DefaultRangeDecoder, DefaultRangeEncoder},
/// #     Decode, Encode,
/// # };
/// #
/// # let message = [2, 1, 3, 0, 0, 2, 0, 2, 1, 0, 2];
/// # let floating_point_probabilities = [0.4f32, 0.2, 0.1, 0.3];
///
/// let encoder_model =
///     SmallContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(
///         &floating_point_probabilities,
///     )
///     .unwrap(); // We're using a "small" encoder model again ...
/// let mut encoder = DefaultRangeEncoder::new(); // ... but now with a "default" coder.
/// encoder.encode_iid_symbols(message, &encoder_model);
///
/// // ... obtain `compressed` and `fixed_point_probabilities` as in the example above ...
/// # let compressed = encoder.get_compressed();
/// # let fixed_point_probabilities = encoder_model
/// #     .symbol_table()
/// #     .map(|(_symbol, _cdf, probability)| probability.get())
/// #     .collect::<Vec<_>>();
///
/// // Then decode with the same lookup model as before, but now with a "default" decoder:
/// let lookup_decoder_model =
///     SmallContiguousLookupDecoderModel::from_nonzero_fixed_point_probabilities(
///         &fixed_point_probabilities,
///         false,
///     )
///     .unwrap();
/// let mut decoder = DefaultRangeDecoder::from_compressed(compressed).unwrap();
///
/// let reconstructed = decoder
///     .decode_iid_symbols(11, &lookup_decoder_model)
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
///
/// assert_eq!(&reconstructed[..], message);
/// ```
///
/// You can also use an [`AnsCoder`] instead of a range coder of course.
///
/// [`AnsCoder`]: crate::stream::stack::AnsCoder
/// [`SmallAnsCoder`]: crate::stream::stack::SmallAnsCoder
/// [`SmallRangeDecoder`]: crate::stream::queue::SmallRangeDecoder
/// [`DefaultAnsCoder`]: crate::stream::stack::DefaultAnsCoder
/// [`DefaultRangeDecoder`]: crate::stream::queue::DefaultRangeDecoder
/// [`NonContiguousLookupDecoderModel`]:
///     crate::stream::model::NonContiguousLookupDecoderModel
/// [preset]: crate::stream#presets
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
    /// Constructs a lookup table for decoding whose PMF approximates given `probabilities`.
    ///
    /// Similar to [`from_floating_point_probabilities_fast`], but the resulting
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
    ///     model::SmallContiguousLookupDecoderModel,
    ///     stack::SmallAnsCoder,
    ///     Decode, Code,
    /// };
    ///
    /// let probabilities = [0.3f32, 0.1, 0.4, 0.2];
    /// let decoder_model = SmallContiguousLookupDecoderModel
    ///     ::from_floating_point_probabilities_perfect(&probabilities).unwrap();
    ///
    /// let compressed = [0x956Eu16, 0x0155]; // (imagine this was read from a file)
    /// let expected = [1, 0, 0, 2, 2, 3, 2, 2, 0];
    /// let mut coder = SmallAnsCoder::from_compressed_slice(&compressed).unwrap();
    ///
    /// let reconstructed = coder
    ///     .decode_iid_symbols(9, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
    /// assert!(coder.is_empty());
    /// assert_eq!(reconstructed, expected);
    /// ```
    ///
    /// [`from_floating_point_probabilities_fast`]:
    ///     Self::from_floating_point_probabilities_fast
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect`]:
    ///     crate::stream::model::ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect
    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities_perfect<F>(probabilities: &[F]) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        let slots = perfectly_quantized_probabilities::<_, _, PRECISION>(probabilities)?;
        Self::from_nonzero_fixed_point_probabilities(
            slots.into_iter().map(|slot| slot.weight),
            false,
        )
    }

    /// Faster but less accurate variant of [`from_floating_point_probabilities_perfect`]
    ///
    /// Semantics are analogous to
    /// [`ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast`],
    /// except that this method constructs a *lookup table*, i.e., a model that takes
    /// considerably more runtime to build but, once built, is optimized for very fast
    /// decoding of lots i.i.d. symbols.
    ///
    /// # See also
    ///
    /// - [`from_floating_point_probabilities_perfect`], which can be slower but
    ///   typically approximates the provided `probabilities` better, which may be a good
    ///   trade-off in the kind of situations that lookup decoder models are intended for.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{
    ///     model::SmallContiguousLookupDecoderModel,
    ///     stack::SmallAnsCoder,
    ///     Decode, Code,
    /// };
    ///
    /// let probabilities = [0.3f32, 0.1, 0.4, 0.2];
    /// let decoder_model = SmallContiguousLookupDecoderModel
    ///     ::from_floating_point_probabilities_fast(&probabilities, None).unwrap();
    ///
    /// let compressed = [0xF592u16, 0x0133]; // (imagine this was read from a file)
    /// let expected = [1, 0, 0, 2, 2, 3, 2, 2, 0];
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
    /// [`from_floating_point_probabilities_perfect`]:
    ///     Self::from_floating_point_probabilities_perfect
    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities_fast<F>(
        probabilities: &[F],
        normalization: Option<F>,
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + AsPrimitive<Probability>,
        Probability: AsPrimitive<usize>,
        usize: AsPrimitive<F>,
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

    /// Deprecated constructor.
    ///
    /// This constructor has been deprecated in constriction version 0.4.0, and it will be
    /// removed in constriction version 0.5.0.
    ///
    /// # Upgrade Instructions
    ///
    /// Call either of the following two constructors instead:
    /// - [`from_floating_point_probabilities_perfect`] (referred to as `..._perfect` in the
    ///   following); or
    /// - [`from_floating_point_probabilities_fast`] (referred to as `..._fast` in the
    ///   following).
    ///
    /// Both constructors approximate the given (floating-point) probability distribution in
    /// fixed point arithmetic, and both construct a valid (exactly invertible) model that
    /// is guaranteed to assign a nonzero probability to all symbols. The `..._perfect`
    /// constructor finds the best possible approximation of the provided fixed-point
    /// probability distribution (thus leading to the lowest bit rates), while the
    /// `..._fast` constructor is faster but may find a *slightly* imperfect approximation.
    /// Note that, since lookup models use a smaller fixed-point `PRECISION` than other
    /// models (e.g., [`ContiguousCategoricalEntropyModel`]), the difference in bit rate
    /// between the two models is more pronounced.
    ///
    /// Note that the `..._fast` constructor breaks binary compatibility with `constriction`
    /// version <= 0.3.5. If you need to be able to exchange binary compressed data with a
    /// program that uses a lookup decoder model or a categorical entropy model from
    /// `constriction` version <= 0.3.5, then call
    /// [`from_floating_point_probabilities_perfect`].
    ///
    /// # Compatibility Table
    ///
    /// (Lookup decoder models can only be used for decoding; in the following table,
    /// "encoding" refers to [`ContiguousCategoricalEntropyModel`])
    ///
    /// | constructor used for encoding → <br> ↓ constructor used for decoding ↓ | [legacy](ContiguousCategoricalEntropyModel::from_floating_point_probabilities) |  [`..._perfect`](ContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect) | [`..._fast`](ContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast) |
    /// | --------------------: | --------------- | --------------- | --------------- |
    /// | **legacy (this one)** | ✅ compatible   | ✅ compatible   | ❌ incompatible |
    /// | **[`..._perfect`]**   | ✅ compatible   | ✅ compatible   | ❌ incompatible |
    /// | **[`..._fast`]**      | ❌ incompatible | ❌ incompatible | ✅ compatible   |
    ///
    /// [`from_floating_point_probabilities_perfect`]:
    ///     Self::from_floating_point_probabilities_perfect
    /// [`..._perfect`]: Self::from_floating_point_probabilities_perfect
    /// [`from_floating_point_probabilities_fast`]:
    ///     Self::from_floating_point_probabilities_fast
    /// [`..._fast`]: Self::from_floating_point_probabilities_fast
    #[deprecated(
        since = "0.4.0",
        note = "Please use `from_floating_point_probabilities_perfect` or \
        `from_floating_point_probabilities_fast` instead. See documentation for \
        detailed upgrade instructions."
    )]
    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities<F>(probabilities: &[F]) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        Self::from_floating_point_probabilities_perfect(probabilities)
    }

    /// Create a `ContiguousLookupDecoderModel` from pre-calculated fixed-point
    /// probabilities.
    ///
    /// # Example
    ///
    /// See [type level documentation](Self#example).
    #[allow(clippy::result_unit_err)]
    pub fn from_nonzero_fixed_point_probabilities<I>(
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
    /// The returned `ContiguousLookupDecoderModel` implements `Copy`, which is a
    /// requirement for some methods, such as [`Decode::decode_iid_symbols`]. These methods
    /// could also accept a shared reference to a `NonContiguousCategoricalDecoderModel`
    /// (since all references to entropy models are also entropy models, and all shared
    /// references implement `Copy`), but passing a *view* instead may be slightly more
    /// efficient because it avoids one level of dereferencing.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::stream::{
    ///     model::{
    ///         IterableEntropyModel, SmallContiguousCategoricalEntropyModel,
    ///         SmallContiguousLookupDecoderModel
    ///     },
    ///     queue::SmallRangeDecoder,
    ///     Decode, Encode,
    /// };
    ///
    /// let expected = [2, 1, 3, 0, 0, 2, 0, 2, 1, 0, 2];
    /// let probabilities = [0.4f32, 0.2, 0.1, 0.3];
    /// let decoder_model = SmallContiguousLookupDecoderModel
    ///     ::from_floating_point_probabilities_perfect(&probabilities).unwrap();
    ///
    /// let compressed = [0xA78Cu16, 0xA856]; // (imagine this was read from a file)
    /// let mut decoder = SmallRangeDecoder::from_compressed(&compressed).unwrap();
    ///
    /// // We can decode symbols by passing a shared reference to `decoder_model`:
    /// let reconstructed = decoder
    ///     .decode_iid_symbols(11, &decoder_model)
    ///     .collect::<Result<Vec<_>, _>>()
    ///     .unwrap();
    /// assert_eq!(&reconstructed[..], &expected);
    ///
    /// // However, `decode_iid_symbols` internally calls `decode_symbol` multiple times.
    /// // If we encode lots of symbols then it's slightly cheaper to first get a view of
    /// // the decoder model, which we can then pass by value:
    /// let mut decoder = SmallRangeDecoder::from_compressed(&compressed).unwrap(); // (same as before)
    /// let decoder_model_view = decoder_model.as_view();
    /// let reconstructed2 = decoder
    ///     .decode_iid_symbols(11, decoder_model_view)
    ///     .collect::<Result<Vec<_>, _>>()
    ///     .unwrap();
    /// assert_eq!(&reconstructed2[..], &expected);
    ///
    /// // `decoder_model_view` can be used again here if necessary (since it implements `Copy`).
    /// ```
    ///
    /// [`Decode::decode_iid_symbols`]: crate::stream::Decode::decode_iid_symbols
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
    /// This drops the lookup table, so its only conceivable use case is to free memory
    /// while still holding on to (a slower variant of) the model.
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
    use alloc::vec;

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
