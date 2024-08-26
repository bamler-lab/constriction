use core::{borrow::Borrow, marker::PhantomData};

use alloc::{boxed::Box, vec::Vec};
use num_traits::{float::FloatCore, AsPrimitive};

use crate::{
    stream::model::{DecoderModel, EncoderModel, EntropyModel, IterableEntropyModel},
    wrapping_pow2, BitArray,
};

use super::{
    accumulate_nonzero_probabilities, fast_quantized_cdf, iter_extended_cdf,
    lookup_contiguous::ContiguousLookupDecoderModel, perfectly_quantized_probabilities,
};

/// Type alias for a typical [`ContiguousCategoricalEntropyModel`].
///
/// See:
/// - [`ContiguousCategoricalEntropyModel`]
/// - [discussion of presets](crate::stream#presets)
pub type DefaultContiguousCategoricalEntropyModel<Cdf = Vec<u32>> =
    ContiguousCategoricalEntropyModel<u32, Cdf, 24>;

/// Type alias for a [`ContiguousCategoricalEntropyModel`] optimized for compatibility with
/// lookup decoder models.
///
/// See:
/// - [`ContiguousCategoricalEntropyModel`]
/// - [discussion of presets](crate::stream#presets)
pub type SmallContiguousCategoricalEntropyModel<Cdf = Vec<u16>> =
    ContiguousCategoricalEntropyModel<u16, Cdf, 12>;

/// An entropy model for a categorical probability distribution over a contiguous range of
/// integers starting at zero.
///
/// You will usually want to use this type through one of its type aliases,
/// [`DefaultContiguousCategoricalEntropyModel`] or
/// [`SmallContiguousCategoricalEntropyModel`], see [discussion of presets](crate::stream#presets).
///
/// This entropy model implements both [`EncoderModel`] and [`DecoderModel`], which means
/// that it can be used for both encoding and decoding with any of the stream coders
/// provided by the `constriction` crate.
///
/// # Example
///
/// ```
/// use constriction::{
///     stream::{stack::DefaultAnsCoder, model::DefaultContiguousCategoricalEntropyModel, Decode},
///     UnwrapInfallible,
/// };
///
/// // Create a `ContiguousCategoricalEntropyModel` that approximates floating point probabilities.
/// let probabilities = [0.3, 0.0, 0.4, 0.1, 0.2]; // Note that `probabilities[1] == 0.0`.
/// let model = DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_fast(
///     &probabilities, None
/// ).unwrap();
/// assert_eq!(model.support_size(), 5); // `model` supports the symbols `0..5usize`.
///
/// // Use `model` for entropy coding.
/// let message = vec![2, 0, 3, 1, 2, 4, 3, 2, 0];
/// let mut ans_coder = DefaultAnsCoder::new();
///
/// // We could pass `model` by reference but passing `model.as_view()` is slightly more efficient.
/// ans_coder.encode_iid_symbols_reverse(message.iter().cloned(), model.as_view()).unwrap();
/// // Note that `message` contains the symbol `1`, and that `probabilities[1] == 0.0`. However, we
/// // can still encode the symbol because the `ContiguousCategoricalEntropyModel` is "leaky", i.e.,
/// // it assigns a nonzero probability to all symbols in the range `0..model.support_size()`.
///
/// // Decode the encoded message and verify correctness.
/// let decoded = ans_coder
///     .decode_iid_symbols(9, model.as_view())
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap_infallible();
/// assert_eq!(decoded, message);
/// assert!(ans_coder.is_empty());
///
/// // The `model` assigns zero probability to any symbols that are not in the support
/// // `0..model.support_size()`, so trying to encode a message that contains such a symbol fails.
/// assert!(ans_coder.encode_iid_symbols_reverse(&[2, 0, 5, 1], model.as_view()).is_err())
/// // ERROR: symbol `5` is not in the support of `model`.
/// ```
///
/// # When Should I Use This Type of Entropy Model?
///
/// Use a `ContiguousCategoricalEntropyModel` for probabilistic models that can *only* be
/// represented as an explicit probability table, and not by some more compact analytic
/// expression. If you have a probability model that can be expressed by some analytical
/// expression (e.g., a [`Binomial`](probability::distribution::Binomial) distribution),
/// then use [`LeakyQuantizer`] instead (unless you want to encode lots of symbols with the
/// same entropy model, in which case the explicitly tabulated representation of a
/// categorical entropy model could improve runtime performance).
///
/// Further, a `ContiguousCategoricalEntropyModel` can only represent probability
/// distribution whose support (i.e., the set of symbols to which the model assigns a
/// non-zero probability) is a contiguous range of integers starting at zero. If the support
/// of your probability distribution has a more complicated structure (or if the `Symbol`
/// type is not an integer type), then you can use a
/// [`NonContiguousCategoricalEncoderModel`] or a [`NonContiguousCategoricalDecoderModel`],
/// which are strictly more general than a `ContiguousCategoricalEntropyModel` but which
/// have a larger memory footprint and slightly worse runtime performance.
///
/// If you want to *decode* lots of symbols with the same entropy model, and if reducing the
/// `PRECISION` to a moderate value is acceptable to you, then you may want to consider
/// using a [`LookupDecoderModel`] instead for even better runtime performance (at the cost
/// of a larger memory footprint and worse compression efficiency due to lower `PRECISION`).
///
/// # Computational Efficiency
///
/// For a probability distribution with a support of `N` symbols, a
/// `ContiguousCategoricalEntropyModel` has the following asymptotic costs:
///
/// - creation:
///   - runtime cost: `Θ(N)` when creating from fixed point probabilities, `Θ(N log(N))`
///     when creating from floating point probabilities;
///   - memory footprint: `Θ(N)`;
///   - both are cheaper by a constant factor than for a
///     [`NonContiguousCategoricalEncoderModel`] or a
///     [`NonContiguousCategoricalDecoderModel`].
/// - encoding a symbol (calling [`EncoderModel::left_cumulative_and_probability`]):
///   - runtime cost: `Θ(1)` (cheaper than for [`NonContiguousCategoricalEncoderModel`]
///     since it compiles to a simiple array lookup rather than a `HashMap` lookup)
///   - memory footprint: no heap allocations, constant stack space.
/// - decoding a symbol (calling [`DecoderModel::quantile_function`]):
///   - runtime cost: `Θ(log(N))` (both expected and worst-case; probably slightly cheaper
///     than for [`NonContiguousCategoricalDecoderModel`] due to better memory locality)
///   - memory footprint: no heap allocations, constant stack space.
///
/// [`EntropyModel`]: trait.EntropyModel.html
/// [`Encode`]: crate::Encode
/// [`Decode`]: crate::Decode
/// [`HashMap`]: std::hash::HashMap
#[derive(Debug, Clone, Copy)]
pub struct ContiguousCategoricalEntropyModel<Probability, Cdf, const PRECISION: usize> {
    /// Invariants:
    /// - `cdf.len() >= 2` (actually, we currently even guarantee `cdf.len() >= 3` but
    ///   this may be relaxed in the future)
    /// - `cdf[0] == 0`
    /// - `cdf` is monotonically increasing except that it may wrap around only at
    ///   the very last entry (this happens iff `PRECISION == Probability::BITS`).
    ///   Thus, all probabilities within range are guaranteed to be nonzero.
    pub(super) cdf: Cdf,

    pub(super) phantom: PhantomData<Probability>,
}

impl<Probability: BitArray, const PRECISION: usize>
    ContiguousCategoricalEntropyModel<Probability, Vec<Probability>, PRECISION>
{
    /// Constructs a leaky distribution whose PMF approximates given probabilities.
    ///
    /// The returned distribution will be defined for symbols of type `usize` from the range
    /// `0..probabilities.len()`. Every symbol will have a strictly nonzero probability,
    /// even if its corresponding entry in the provided argument `probabilities` is smaller
    /// than the smallest nonzero probability that can be represented with `PRECISION` bits
    /// (including if the provided probability is exactly zero). This guarantee ensures that
    /// every symbol in the range `0..probabilities.len()` can be encoded with the resulting
    /// model.
    ///
    /// # Arguments
    ///
    /// - `probabilities`: a slice of floating point values (`F` is typically `f64` or
    ///   `f32`). All entries must be nonnegative and at least one entry has to be nonzero.
    ///   The entries do not necessarily need to add up to one (see argument
    ///   `normalization`).
    /// - `normalization`: optional sum of `probabilities`, which will be used to normalize
    ///   the probability distribution. Will be calculated internally if not provided. Only
    ///   provide this argument if you know its value *exactly*; it must be obtained by
    ///   summing up the elements of `probability` left to right (otherwise, you'll get
    ///   different rounding errors, which may lead to overflowing probabilities in edge
    ///   cases, e.g., if the last element has a very small probability). If in doubt, do
    ///   not provide.
    ///
    /// # Runtime Complexity
    ///
    /// O(`probabilities.len()`). This is in contrast to
    /// [`from_floating_point_probabilities_perfect`], which may be considerably slower.
    ///
    /// # Error Handling
    ///
    /// Returns an error if the normalization (regardless of whether it is provided or
    /// calculated) is not a finite positive value. Also returns an error if `probability`
    /// is of length zero or one (degenerate probability distributions are not supported by
    /// `constriction`) or if `probabilities` contains more than `2^PRECISION` elements (in
    /// which case we could not assign a nonzero fixed-point probability to every symbol).
    ///
    /// # See also
    ///
    /// - [`from_floating_point_probabilities_perfect`]
    ///
    /// [`from_floating_point_probabilities_perfect`]:
    ///     Self::from_floating_point_probabilities_perfect
    #[allow(clippy::result_unit_err)]
    pub fn from_floating_point_probabilities_fast<F>(
        probabilities: &[F],
        normalization: Option<F>,
    ) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + AsPrimitive<Probability>,
        Probability: BitArray + AsPrimitive<usize>,
        usize: AsPrimitive<Probability> + AsPrimitive<F>,
    {
        let cdf = fast_quantized_cdf::<_, _, PRECISION>(probabilities, normalization)?;
        Self::from_fixed_point_cdf(cdf)
    }

    /// Slower variant of [`from_floating_point_probabilities_fast`].
    ///
    /// Constructs a leaky distribution whose PMF approximates given probabilities as well
    /// as possible within a `PRECISION`-bit fixed-point representation. The returned
    /// distribution will be defined for symbols of type `usize` from the range
    /// `0..probabilities.len()`.
    ///
    /// # Comparison to `from_floating_point_probabilities_fast`
    ///
    /// This method explicitly minimizes the Kullback-Leibler divergence from the resulting
    /// fixed-point precision probabilities to the floating-point argument `probabilities`.
    /// This method may find a slightly better quantization than
    /// [`from_floating_point_probabilities_fast`], thus leading to a very slightly lower
    /// expected bit rate. However, this method can have a *significantly* longer runtime
    /// than `from_floating_point_probabilities_fast`.
    ///
    /// For most applications, [`from_floating_point_probabilities_fast`] is the better
    /// choice because the marginal reduction in bit rate due to
    /// `from_floating_point_probabilities_perfect` is rarely worth its significantly longer
    /// runtime. This advice applies in particular to autoregressive compression techniques,
    /// i.e., methods that use a different probability distribution for every single encoded
    /// symbol.
    ///
    /// However, the following edge cases may justify using
    /// `from_floating_point_probabilities_perfect` despite its poor runtime behavior:
    ///
    /// - you're constructing an entropy model that will be used to encode a very large
    ///   number of symbols; or
    /// - you're constructing an entropy model on the encoder side whose fixed-point
    ///   representation will be stored as a part of the compressed data (which will then be
    ///   read in every time the data gets decoded); or
    /// - you need backward compatibility with constriction <= version 0.3.5.
    ///
    /// # Details
    ///
    ///
    ///
    /// The argument `probabilities` is a slice of floating point values (`F` is typically
    /// `f64` or `f32`). All entries must be nonnegative, and at least one entry has to be
    /// nonzero. The entries do not necessarily need to add up to one (the resulting
    /// distribution will automatically get normalized, and an overall scaling of all
    /// entries of `probabilities` does not affect the result, up to effects due to rounding
    /// errors).
    ///
    /// The probability mass function of the returned distribution will approximate the
    /// provided probabilities as well as possible, subject to the following constraints:
    /// - probabilities are represented in fixed point arithmetic, where the const generic
    ///   parameter `PRECISION` controls the number of bits of precision. This typically
    ///   introduces rounding errors;
    /// - despite the possibility of rounding errors, the returned probability distribution
    ///   will be exactly normalized; and
    /// - each symbol in the support `0..probabilities.len()` gets assigned a strictly
    ///   nonzero probability, even if the provided probability for the symbol is zero or
    ///   below the threshold that can be resolved in fixed point arithmetic with
    ///   `PRECISION` bits. We refer to this property as the resulting distribution being
    ///   "leaky". The leakiness guarantees that all symbols within the support can be
    ///   encoded when this distribution is used as an entropy model.
    ///
    /// More precisely, the resulting probability distribution minimizes the cross entropy
    /// from the provided (floating point) to the resulting (fixed point) probabilities
    /// subject to the above three constraints.
    ///
    /// # Error Handling
    ///
    /// Returns an error if the provided probability distribution cannot be normalized,
    /// either because `probabilities` is of length zero, or because one of its entries is
    /// negative with a nonzero magnitude, or because the sum of its elements is zero,
    /// infinite, or NaN.
    ///
    /// Also returns an error if the probability distribution is degenerate, i.e., if
    /// `probabilities` has only a single element, because degenerate probability
    /// distributions currently cannot be represented.
    ///
    /// Also returns an error if `probabilities` contains more than `2^PRECISION` elements
    /// (in which case we could not assign a nonzero fixed-point probability to every
    /// symbol).
    ///
    /// # See also
    ///
    /// - [`from_floating_point_probabilities_fast`]
    ///
    /// [`from_floating_point_probabilities_fast`]:
    ///     Self::from_floating_point_probabilities_fast
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

    /// Deprecated constructor.
    ///
    /// This constructor has been deprecated in constriction version 0.4.0, and it will be
    /// removed in constriction version 0.5.0.
    ///
    /// # Upgrade Instructions
    ///
    /// Most *new* use cases should call [`from_floating_point_probabilities_fast`] instead.
    /// Using the `..._fast` constructor may lead to very slightly larger bit rates, but it
    /// runs considerably faster.
    ///
    /// However, note that [`from_floating_point_probabilities_fast`] breaks binary
    /// compatibility with `constriction` version <= 0.3.5. If you need to be able to
    /// exchange binary compressed data with a program that uses a categorical entropy model
    /// from `constriction` version <= 0.3.5, then call
    /// [`from_floating_point_probabilities_perfect`] instead. Another reason for using the
    /// `..._perfect` constructor could be if compression performance is much more important
    /// to you than runtime performance.
    ///
    /// # Compatibility Table
    ///
    /// This deprecated constructor currently delegates to
    /// [`from_floating_point_probabilities_perfect`] (referred to as `..._perfect` in the
    /// table below).
    ///
    /// | Constructor used for compression → <br> ↓ Constructor used for decompression ↓ | This one |  `..._perfect` | `..._fast` |
    /// | ----------------: | ------------ | ------------ | ------------ |
    /// | **This one**      | compatible   | compatible   | incompatible |
    /// | **`..._perfect`** | compatible   | compatible   | incompatible |
    /// | **`..._fast`**    | incompatible | incompatible | compatible   |
    ///
    /// [`from_floating_point_probabilities_perfect`]:
    ///     Self::from_floating_point_probabilities_perfect
    /// [`from_floating_point_probabilities_fast`]:
    ///     Self::from_floating_point_probabilities_fast
    #[deprecated(
        since = "0.4.0",
        note = "Please use `from_floating_point_probabilities_fast` or \
        `from_floating_point_probabilities_perfect` instead. See documentation for detailed \
        upgrade instructions."
    )]
    #[allow(clippy::result_unit_err)]
    #[inline(always)]
    pub fn from_floating_point_probabilities<F>(probabilities: &[F]) -> Result<Self, ()>
    where
        F: FloatCore + core::iter::Sum<F> + Into<f64>,
        Probability: Into<f64> + AsPrimitive<usize>,
        f64: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
    {
        Self::from_floating_point_probabilities_perfect(probabilities)
    }

    /// Constructs a distribution with a PMF given in fixed point arithmetic.
    ///
    /// This is a low level method that allows, e.g,. reconstructing a probability
    /// distribution previously exported with [`symbol_table`]. The more common way to
    /// construct a `LeakyCategorical` distribution is via
    /// [`from_floating_point_probabilities`].
    ///
    /// The items of `probabilities` have to be nonzero and smaller than `1 << PRECISION`,
    /// where `PRECISION` is a const generic parameter on the
    /// `ContiguousCategoricalEntropyModel`.
    ///
    /// If `infer_last_probability` is `false` then the items yielded by `probabilities`
    /// have to (logically) sum up to `1 << PRECISION`. If `infer_last_probability` is
    /// `true` then they must sum up to a value strictly smaller than `1 << PRECISION`, and
    /// the method will add an additional symbol at the end that takes the remaining
    /// probability mass.
    ///
    /// # Examples
    ///
    /// If `infer_last_probability` is `false`, the provided probabilities have to sum up to
    /// `1 << PRECISION`:
    ///
    /// ```
    /// use constriction::stream::model::{
    ///     DefaultContiguousCategoricalEntropyModel, IterableEntropyModel
    /// };
    ///
    /// let probabilities = vec![1u32 << 21, 1 << 22, 1 << 22, 1 << 22, 1 << 21];
    /// // `probabilities` sums up to `1 << PRECISION` as required:
    /// assert_eq!(probabilities.iter().sum::<u32>(), 1 << 24);
    ///
    /// let model = DefaultContiguousCategoricalEntropyModel
    ///     ::from_nonzero_fixed_point_probabilities(&probabilities, false).unwrap();
    /// let symbol_table = model.floating_point_symbol_table::<f64>().collect::<Vec<_>>();
    /// assert_eq!(
    ///     symbol_table,
    ///     vec![
    ///         (0, 0.0, 0.125),
    ///         (1, 0.125, 0.25),
    ///         (2, 0.375, 0.25),
    ///         (3, 0.625, 0.25),
    ///         (4, 0.875, 0.125),
    ///     ]
    /// );
    /// ```
    ///
    /// If `PRECISION` is set to the maximum value supported by the type `Probability`, then
    /// the provided probabilities still have to *logically* sum up to `1 << PRECISION`
    /// (i.e., the summation has to wrap around exactly once):
    ///
    /// ```
    /// use constriction::stream::model::{
    ///     ContiguousCategoricalEntropyModel, IterableEntropyModel
    /// };
    ///
    /// let probabilities = vec![1u32 << 29, 1 << 30, 1 << 30, 1 << 30, 1 << 29];
    /// // `probabilities` sums up to `1 << 32` (logically), i.e., it wraps around once.
    /// assert_eq!(probabilities.iter().fold(0u32, |accum, &x| accum.wrapping_add(x)), 0);
    ///
    /// let model = ContiguousCategoricalEntropyModel::<u32, Vec<u32>, 32>
    ///     ::from_nonzero_fixed_point_probabilities(&probabilities, false).unwrap();
    /// let symbol_table = model.floating_point_symbol_table::<f64>().collect::<Vec<_>>();
    /// assert_eq!(
    ///     symbol_table,
    ///     vec![
    ///         (0, 0.0, 0.125),
    ///         (1, 0.125, 0.25),
    ///         (2, 0.375, 0.25),
    ///         (3, 0.625, 0.25),
    ///         (4, 0.875, 0.125)
    ///     ]
    /// );
    /// ```
    ///
    /// Wrapping around twice fails:
    ///
    /// ```
    /// use constriction::stream::model::ContiguousCategoricalEntropyModel;
    /// let probabilities = vec![1u32 << 30, 1 << 31, 1 << 31, 1 << 31, 1 << 30];
    /// // `probabilities` sums up to `1 << 33` (logically), i.e., it would wrap around twice.
    /// assert!(
    ///     ContiguousCategoricalEntropyModel::<u32, Vec<u32>, 32>
    ///         ::from_nonzero_fixed_point_probabilities(&probabilities, false).is_err()
    /// );
    /// ```
    ///
    /// So does providing probabilities that just don't sum up to `1 << FREQUENCY`:
    ///
    /// ```
    /// use constriction::stream::model::ContiguousCategoricalEntropyModel;
    /// let probabilities = vec![1u32 << 21, 5 << 8, 1 << 22, 1 << 21];
    /// // `probabilities` sums up to `1 << 33` (logically), i.e., it would wrap around twice.
    /// assert!(
    ///     ContiguousCategoricalEntropyModel::<u32, Vec<u32>, 32>
    ///         ::from_nonzero_fixed_point_probabilities(&probabilities, false).is_err()
    /// );
    /// ```
    ///
    /// [`symbol_table`]: IterableEntropyModel::symbol_table
    /// [`fixed_point_probabilities`]: #method.fixed_point_probabilities
    /// [`from_floating_point_probabilities`]: #method.from_floating_point_probabilities
    #[allow(clippy::result_unit_err)]
    pub fn from_nonzero_fixed_point_probabilities<I>(
        probabilities: I,
        infer_last_probability: bool,
    ) -> Result<Self, ()>
    where
        I: IntoIterator,
        I::Item: Borrow<Probability>,
    {
        let probabilities = probabilities.into_iter();
        let mut cdf =
            Vec::with_capacity(probabilities.size_hint().0 + 1 + infer_last_probability as usize);
        accumulate_nonzero_probabilities::<_, _, _, _, _, PRECISION>(
            core::iter::repeat(()),
            probabilities,
            |(), left_sided_cumulative, _| {
                cdf.push(left_sided_cumulative);
                Ok(())
            },
            infer_last_probability,
        )?;
        cdf.push(wrapping_pow2(PRECISION));

        Ok(Self {
            cdf,
            phantom: PhantomData,
        })
    }

    fn from_fixed_point_cdf<I>(cdf: I) -> Result<Self, ()>
    where
        I: ExactSizeIterator<Item = Probability>,
    {
        let extended_cdf = cdf
            .chain(core::iter::once(wrapping_pow2(PRECISION)))
            .collect();

        Ok(Self {
            cdf: extended_cdf,
            phantom: PhantomData,
        })
    }
}

impl<Probability, Cdf, const PRECISION: usize>
    ContiguousCategoricalEntropyModel<Probability, Cdf, PRECISION>
where
    Probability: BitArray,
    Cdf: AsRef<[Probability]>,
{
    /// Returns the number of symbols supported by the model.
    ///
    /// The distribution is defined on the contiguous range of symbols from zero
    /// (inclusively) to `support_size()` (exclusively). All symbols within this range are
    /// guaranteed to have a nonzero probability, while all symbols outside of this range
    /// have a zero probability.
    #[inline(always)]
    pub fn support_size(&self) -> usize {
        self.cdf.as_ref().len() - 1
    }

    /// Makes a very cheap shallow copy of the model that can be used much like a shared
    /// reference.
    ///
    /// The returned `ContiguousCategoricalEntropyModel` implements `Copy`, which is a
    /// requirement for some methods, such as [`Encode::encode_iid_symbols`] or
    /// [`Decode::decode_iid_symbols`]. These methods could also accept a shared reference
    /// to a `ContiguousCategoricalEntropyModel` (since all references to entropy models are
    /// also entropy models, and all shared references implement `Copy`), but passing a
    /// *view* instead may be slightly more efficient because it avoids one level of
    /// dereferencing.
    ///
    /// [`Encode::encode_iid_symbols`]: super::Encode::encode_iid_symbols
    /// [`Decode::decode_iid_symbols`]: super::Decode::decode_iid_symbols
    #[inline]
    pub fn as_view(
        &self,
    ) -> ContiguousCategoricalEntropyModel<Probability, &[Probability], PRECISION> {
        ContiguousCategoricalEntropyModel {
            cdf: self.cdf.as_ref(),
            phantom: PhantomData,
        }
    }

    /// Creates a [`LookupDecoderModel`] for efficient decoding of i.i.d. data
    ///
    /// While a `ContiguousCategoricalEntropyModel` can already be used for decoding (since
    /// it implements [`DecoderModel`]), you may prefer converting it to a
    /// `LookupDecoderModel` first for improved efficiency. Logically, the two will be
    /// equivalent.
    ///
    /// # Warning
    ///
    /// You should only call this method if both of the following conditions are satisfied:
    ///
    /// - `PRECISION` is relatively small (typically `PRECISION == 12`, as in the "Small"
    ///   [preset]) because the memory footprint of a `LookupDecoderModel` grows
    ///   exponentially in `PRECISION`; and
    /// - you're about to decode a relatively large number of symbols with the resulting
    ///   model; the conversion to a `LookupDecoderModel` bears a significant runtime and
    ///   memory overhead, so if you're going to use the resulting model only for a single
    ///   or a handful of symbols then you'll end up paying more than you gain.
    ///
    /// [preset]: crate::stream#presets
    #[inline(always)]
    pub fn to_lookup_decoder_model(
        &self,
    ) -> ContiguousLookupDecoderModel<Probability, Vec<Probability>, Box<[Probability]>, PRECISION>
    where
        Probability: Into<usize>,
        usize: AsPrimitive<Probability>,
    {
        self.into()
    }
}

impl<Probability, Cdf, const PRECISION: usize> EntropyModel<PRECISION>
    for ContiguousCategoricalEntropyModel<Probability, Cdf, PRECISION>
where
    Probability: BitArray,
{
    type Symbol = usize;
    type Probability = Probability;
}

impl<'m, Probability, Cdf, const PRECISION: usize> IterableEntropyModel<'m, PRECISION>
    for ContiguousCategoricalEntropyModel<Probability, Cdf, PRECISION>
where
    Probability: BitArray,
    Cdf: AsRef<[Probability]>,
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
        iter_extended_cdf(
            self.cdf
                .as_ref()
                .iter()
                .enumerate()
                .map(|(symbol, &cumulative)| (cumulative, symbol)),
        )
    }
}

impl<Probability, Cdf, const PRECISION: usize> DecoderModel<PRECISION>
    for ContiguousCategoricalEntropyModel<Probability, Cdf, PRECISION>
where
    Probability: BitArray,
    Cdf: AsRef<[Probability]>,
{
    #[inline(always)]
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (usize, Probability, Probability::NonZero) {
        let cdf = self.cdf.as_ref();
        // SAFETY: `cdf` is not empty.
        let monotonic_part_of_cdf = unsafe { cdf.get_unchecked(..cdf.len() - 1) };
        let Err(next_symbol) = monotonic_part_of_cdf.binary_search_by(|&x| {
            if x <= quantile {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Greater
            }
        }) else {
            // SAFETY: our search criterion never returns `Equal`, so the search cannot succeed.
            unsafe { core::hint::unreachable_unchecked() }
        };

        let symbol = next_symbol - 1;

        // SAFETY:
        // - `next_symbol < cdf.len()` because we searched only within `monotonic_part_of_cdf`, which
        //   is one element shorter than `cdf`. Thus `cdf.get_unchecked(next_symbol)` is sound.
        // - `next_symbol > 0` because `cdf[0] == 0` and our search goes right on equality; thus,
        //   `next_symbol - 1` does not wrap around, and so `next_symbol - 1` is also within bounds.
        let (right_cumulative, left_cumulative) =
            unsafe { (*cdf.get_unchecked(next_symbol), *cdf.get_unchecked(symbol)) };

        // SAFETY: our constructors don't allow zero probabilities.
        let probability = unsafe {
            right_cumulative
                .wrapping_sub(&left_cumulative)
                .into_nonzero_unchecked()
        };

        (symbol, left_cumulative, probability)
    }
}

impl<Probability, Cdf, const PRECISION: usize> EncoderModel<PRECISION>
    for ContiguousCategoricalEntropyModel<Probability, Cdf, PRECISION>
where
    Probability: BitArray,
    Cdf: AsRef<[Probability]>,
{
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<usize>,
    ) -> Option<(Probability, Probability::NonZero)> {
        let index = *symbol.borrow();
        if index >= self.support_size() {
            return None;
        }
        let cdf = self.cdf.as_ref();

        // SAFETY: we verified that  index is within bounds (we compare `index >= len - 1`
        // here and not `index + 1 >= len` because the latter could overflow/wrap but `len`
        // is guaranteed to be nonzero; once the check passes, we know that `index + 1`
        // doesn't wrap because `cdf.len()` can't be `usize::max_value()` since that would
        // mean that there's no space left even for the call stack).
        let (left_cumulative, right_cumulative) =
            unsafe { (*cdf.get_unchecked(index), *cdf.get_unchecked(index + 1)) };

        // SAFETY: The constructors ensure that all probabilities within bounds are nonzero.
        let probability = unsafe {
            right_cumulative
                .wrapping_sub(&left_cumulative)
                .into_nonzero_unchecked()
        };

        Some((left_cumulative, probability))
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::tests::{test_entropy_model, verify_iterable_entropy_model};
    use super::*;

    /// Test that `optimal_weights` reproduces the same distribution when fed with an
    /// already quantized model.
    #[test]
    fn trivial_optimal_weights() {
        let hist = [
            56319u32, 134860032, 47755520, 60775168, 75699200, 92529920, 111023616, 130420736,
            150257408, 169970176, 188869632, 424260864, 229548800, 236082432, 238252287, 234666240,
            1, 1, 227725568, 216746240, 202127104, 185095936, 166533632, 146508800, 126643712,
            107187968, 88985600, 72576000, 57896448, 45617664, 34893056, 26408448, 19666688,
            14218240, 10050048, 7164928, 13892864,
        ];
        assert_eq!(hist.iter().map(|&x| x as u64).sum::<u64>(), 1 << 32);

        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();
        let categorical =
            ContiguousCategoricalEntropyModel::<u32, _, 32>::from_floating_point_probabilities_perfect(
                &probabilities,
            )
            .unwrap();
        let weights: Vec<_> = categorical
            .symbol_table()
            .map(|(_, _, probability)| probability.get())
            .collect();

        assert_eq!(&weights[..], &hist[..]);
    }

    #[test]
    fn nontrivial_optimal_weights() {
        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        assert_ne!(hist.iter().map(|&x| x as u64).sum::<u64>(), 1 << 32);

        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();

        let fast =
            ContiguousCategoricalEntropyModel::<u32, _, 32>::from_floating_point_probabilities_fast(
                &probabilities,
                None
            )
            .unwrap();
        let kl_fast = verify_iterable_entropy_model(&fast, &hist, 1e-6);

        let perfect =
            ContiguousCategoricalEntropyModel::<u32, _, 32>::from_floating_point_probabilities_perfect(
                &probabilities,
            )
            .unwrap();
        let kl_perfect = verify_iterable_entropy_model(&perfect, &hist, 1e-6);

        assert!(kl_perfect < kl_fast);
    }

    /// Regression test for convergence of `optimize_leaky_categorical`.
    #[test]
    fn perfect_converges() {
        // Two example probability distributions that lead to an infinite loop in constriction 0.2.6
        // (see <https://github.com/bamler-lab/constriction/issues/20>).
        let example1 = [0.15, 0.69, 0.15];
        let example2 = [
            1.34673042e-04,
            6.52306480e-04,
            3.14999325e-03,
            1.49921896e-02,
            6.67127371e-02,
            2.26679876e-01,
            3.75356406e-01,
            2.26679876e-01,
            6.67127594e-02,
            1.49922138e-02,
            3.14990873e-03,
            6.52299321e-04,
            1.34715927e-04,
        ];

        let categorical1 =
            DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(
                &example1,
            )
            .unwrap();
        let prob0 = categorical1.left_cumulative_and_probability(0).unwrap().1;
        let prob2 = categorical1.left_cumulative_and_probability(2).unwrap().1;
        assert!((-1..=1).contains(&(prob0.get() as i64 - prob2.get() as i64)));
        verify_iterable_entropy_model(&categorical1, &example1, 1e-10);

        let categorical2 =
            DefaultContiguousCategoricalEntropyModel::from_floating_point_probabilities_perfect(
                &example2,
            )
            .unwrap();
        verify_iterable_entropy_model(&categorical2, &example2, 1e-10);
    }

    #[test]
    fn contiguous_categorical() {
        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        let probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();

        let fast =
            ContiguousCategoricalEntropyModel::<u32, _, 32>::from_floating_point_probabilities_fast(
                &probabilities,
                None
            )
            .unwrap();
        test_entropy_model(&fast, 0..probabilities.len());
        let kl_fast = verify_iterable_entropy_model(&fast, &hist, 1e-8);

        let perfect =
            ContiguousCategoricalEntropyModel::<u32, _, 32>::from_floating_point_probabilities_perfect(
                &probabilities,
            )
            .unwrap();
        test_entropy_model(&perfect, 0..probabilities.len());
        let kl_perfect = verify_iterable_entropy_model(&perfect, &hist, 1e-8);

        assert!(kl_perfect < kl_fast);
    }
}
