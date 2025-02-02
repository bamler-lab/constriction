use core::{borrow::Borrow, marker::PhantomData, ops::RangeInclusive};

use num_traits::{float::FloatCore, AsPrimitive, PrimInt, WrappingAdd, WrappingSub};

use crate::{generic_static_asserts, wrapping_pow2, BitArray};

use super::{
    DecoderModel, Distribution, EncoderModel, EntropyModel, Inverse, IterableEntropyModel,
};

/// Quantizes probability distributions and represents them in fixed-point precision.
///
/// You will usually want to use this type through one of its type aliases,
/// [`DefaultLeakyQuantizer`] or [`SmallLeakyQuantizer`], see [discussion of
/// presets](crate::stream#presets).
///
/// # Examples
///
/// ## Quantizing Continuous Distributions
///
/// ```
/// use constriction::{
///     stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Encode, Decode},
///     UnwrapInfallible,
/// };
///
/// // Create a quantizer that supports integer symbols from -5 to 20 (inclusively),
/// // using the "default" preset for `Probability` and `PRECISION`.
/// let quantizer = DefaultLeakyQuantizer::new(-5..=20);
///
/// // Quantize a normal distribution with mean 8.3 and standard deviation 4.1.
/// let continuous_distribution1 = probability::distribution::Gaussian::new(8.3, 4.1);
/// let entropy_model1 = quantizer.quantize(continuous_distribution1);
///
/// // You can reuse the same quantizer for more than one distribution, and the distributions don't
/// // even have to be of the same type (e.g., one can be a `Gaussian` and another a `Laplace`).
/// let continuous_distribution2 = probability::distribution::Laplace::new(-1.4, 2.7);
/// let entropy_model2 = quantizer.quantize(continuous_distribution2);
///
/// // Use the entropy models with an entropy coder.
/// let mut ans_coder = DefaultAnsCoder::new();
/// ans_coder.encode_symbol(4, entropy_model1).unwrap();
/// ans_coder.encode_symbol(-3, entropy_model2).unwrap();
///
/// // Decode symbols (in reverse order, since the `AnsCoder` is a stack) and verify correctness.
/// assert_eq!(ans_coder.decode_symbol(entropy_model2).unwrap_infallible(), -3);
/// assert_eq!(ans_coder.decode_symbol(entropy_model1).unwrap_infallible(), 4);
/// assert!(ans_coder.is_empty());
/// ```
///
/// ## Quantizing a Discrete Distribution (That Has an Analytic Expression)
///
/// If you pass a discrete probability distribution to the method [`quantize`] then it no
/// longer needs to perform any quantization in the data space, but it will still perform
/// steps 2 and 3 in the list below, i.e., it will still convert to a "leaky" fixed-point
/// approximation that can be used by any of `constrictions`'s stream codes. In the
/// following example, we'll quantize a [`Binomial`](probability::distribution::Binomial)
/// distribution (as discussed [below](#dont-quantize-categorical-distributions-though), you
/// should *not* quantize a [`Categorical`](probability::distribution::Categorical)
/// distribution since there are more efficient specialized types for this use case).
///
/// ```
/// use constriction::stream::{
///     model::DefaultLeakyQuantizer, queue::DefaultRangeEncoder, Encode, Decode
/// };
///
/// let distribution = probability::distribution::Binomial::new(1000, 0.1); // arguments: `n, p`
/// let quantizer = DefaultLeakyQuantizer::new(0..=1000); // natural support is `0..=n`
/// let entropy_model = quantizer.quantize(distribution);
///
/// // Let's use a Range Coder this time, just for fun (we could as well use an ANS Coder again).
/// let mut range_encoder = DefaultRangeEncoder::new();
///
/// // Encode a "typical" symbol from the distribution (i.e., one with non-negligible probability).
/// range_encoder.encode_symbol(107, entropy_model).unwrap();
///
/// // Due to the "leakiness" of the quantizer, the following still works despite the fact that
/// // the symbol `1000` has a ridiculously low probability under the binomial distribution.
/// range_encoder.encode_symbol(1000, entropy_model).unwrap();
///
/// // Decode symbols (in forward order, since range coding operates as a queue) and verify.
/// let mut range_decoder = range_encoder.into_decoder().unwrap();
/// assert_eq!(range_decoder.decode_symbol(entropy_model).unwrap(), 107);
/// assert_eq!(range_decoder.decode_symbol(entropy_model).unwrap(), 1000);
/// assert!(range_decoder.maybe_exhausted());
/// ```
///
/// # Detailed Description
///
/// A `LeakyQuantizer` is a builder of [`LeakilyQuantizedDistribution`]s. It takes an
/// arbitrary probability distribution that implements the [`Distribution`] trait from the
/// crate [`probability`] and turns it into a [`LeakilyQuantizedDistribution`] by performing
/// the following three steps:
///
/// 1. **quantization**: lossless entropy coding can only be performed over *discrete* data.
///    Any continuous (real-valued) data has to be approximated by some discrete set of
///    points. If you provide a continuous distributions (i.e., a probability density
///    function) to this builder, then it will quantize the data space by rounding values to
///    the nearest integer. This step is optional, see
///    [below](#continuous-vs-discrete-probability-distributions).
/// 2. **approximation with fixed-point arithmetic**: an entropy model that is used for
///    compressing and decompressing has to be *exactly* invertible, so that its
///    [`EncoderModel`] implementation is compatible with its [`DecoderModel`]
///    implementation. The `LeakilyQuantizedDistribution`s that are built by this builder
///    represent probabilities and quantiles in fixed-point arithmetic with `PRECISION`
///    bits. This allows them to avoid rounding errors when inverting the model, so that
///    they can implement both `EncoderModel` and `DecoderModel` in such a way that one is
///    the *exact* inverse of the other.
/// 3. **introducing leakiness**: naively approximating a probability distribution with
///    fixed point arithmetic could lead to problems: it could round some very small
///    probabilities to zero. This would have the undesirable effect that the corresponding
///    symbol then could no longer be encoded. This builder ensures that the
///    `LeakilyQuantizedDistribution`s that it creates assign a nonzero probability to all
///    symbols within a user-defined range, so that these symbols can always be encoded,
///    even if their probabilities under the *original* probability distribution are very
///    low (or even zero).
///
/// # Continuous vs. Discrete Probability Distributions
///
/// The method [`quantize`] accepts both continuous probability distributions (i.e.,
/// probability density functions, such as [`Gaussian`]) and discrete distributions that are
/// defined only on (some) integers (i.e., probability mass functions, such as
/// [`Binomial`]). The resulting [`LeakilyQuantizedDistribution`] will always be a discrete
/// probability distribution. If the original probability distribution is continuous, then
/// the quantizer implicitly creates bins of size one by rounding to the nearest integer
/// (i.e., the bins range from `i - 0.5` to `i + 0.5` for each integer `i`). If the original
/// probability distribution is discrete then no rounding in the symbol space occurs, but
/// the quantizer still performs steps 2 and 3 above, i.e., it still rounds probabilities
/// and quantiles to fixed-point arithmetic in a way that ensures that all probabilities
/// within a user-defined range are nonzero.
///
/// ## Don't Quantize *Categorical* Distributions, Though.
///
/// Although you can use a `LeakyQuantizer` for *discrete* probability distributions, you
/// should *not* use it for probability distributions of the type
/// [`probability::distribution::Categorical`]. While this will technically work, it will
/// lead to poor computational performance (and also to *slightly* suboptimal compression
/// efficiency). If you're dealing with categorical distributions, use one of the dedicated
/// types [`ContiguousCategoricalEntropyModel`], [`NonContiguousCategoricalEncoderModel`],
/// [`NonContiguousCategoricalDecoderModel`], or [`ContiguousLookupDecoderModel`] or
/// [`NonContiguousLookupDecoderModel`] instead.
///
/// By contrast, *do* use a `LeakyQuantizer` if the underlying probability [`Distribution`]
/// can be described by some analytic function (e.g., the function `f(x) ∝ e^{-(x-\mu)^2/2}`
/// describing the bell curve of a Gaussian distribution, or the function `f_n(k) = (n
/// choose k) p^k (1-p)^{n-k}` describing the probability mass function of a binomial
/// distribution). For such parameterized distributions, both the cumulative distribution
/// function and its inverse can often be expressed as, or at least approximated by, some
/// analytic expression that can be evaluated in constant time, independent of the number of
/// possible symbols.
///
/// # Computational Efficiency
///
/// Two things should be noted about computational efficiency:
///
/// - **quantization is lazy:** both the constructor of a `LeakyQuantizer` and the method
///   [`quantize`] perform only a small constant amount of work, independent of the
///   `PRECISION` and the number of symbols on which the resulting entropy model will be
///   defined. The actual quantization is done once the resulting
///   [`LeakilyQuantizedDistribution`] is used for encoding and/or decoding, and it is only
///   done for the involved symbols.
/// - **quantization for decoding is more expensive than for encoding:** using a
///   `LeakilyQuantizedDistribution` as an [`EncoderModel`] only requires evaluating the
///   cumulative distribution function (CDF) of the underlying continuous probability
///   distribution a constant number of times (twice, to be precise). By contrast, using it
///   as a [`DecoderModel`] requires numerical inversion of the cumulative distribution
///   function. This numerical inversion starts by calling [`Inverse::inverse`] from the
///   crate [`probability`] on the underlying continuous probability distribution. But the
///   result of this method call then has to be refined by repeatedly probing the CDF in
///   order to deal with inevitable rounding errors in the implementation of
///   `Inverse::inverse`. The number of required iterations will depend on how accurate the
///   implementation of `Inverse::inverse` is.
///
/// The laziness means that it is relatively cheap to use a different
/// `LeakilyQuantizedDistribution` for each symbol of the message, which is a common
/// thing to do in machine-learning based compression methods. By contrast, if you want to
/// use the *same* entropy model for many symbols then a `LeakilyQuantizedDistribution` can
/// become unnecessarily expensive, especially for decoding, because you might end up
/// calculating the inverse CDF in the same region over and over again. If this is the case,
/// consider tabularizing the `LeakilyQuantizedDistribution` that you obtain from the method
/// [`quantize`] by calling [`to_generic_encoder_model`] or [`to_generic_decoder_model`] on
/// it (or, if you use a low `PRECISION`, you may even consider calling
/// [`to_generic_lookup_decoder_model`]). You'll have to bring the trait
/// [`IterableEntropyModel`] into scope to call these conversion methods (`use
/// constriction::stream::model::IterableEntropyModel`).
///
/// # Requirements for Correctness
///
/// The original distribution that you pass to the method [`quantize`] can only be an
/// approximation of a true (normalized) probability distribution because it represents
/// probabilities with finite (floating point) precision. Despite the possibility of
/// rounding errors in the underlying (floating point) distribution, a `LeakyQuantizer` is
/// guaranteed to generate a valid entropy model with exactly compatible implementations of
/// [`EncoderModel`] and [`DecoderModel`] as long as both of the following requirements are
/// met:
///
/// - The cumulative distribution function (CDF) [`Distribution::distribution`] is defined
///   on all mid points between integers that lie within the range that is provided as
///   argument `support` to the `new` method; it is monotonically nondecreasing, and its
///   values do not exceed the closed interval `[0.0, 1.0]`. It is OK if the CDF does not
///   cover the entire interval from `0.0` to `1.0` (e.g., due to rounding errors or
///   clipping); any remaining probability mass on the tails is added to the probability
///   of the symbols at the respective ends of the `support`.
/// - The quantile function or inverse CDF [`Inverse::inverse`] evaluates to a finite
///   non-NaN value everywhere on the open interval `(0.0, 1.0)`, and it is monotonically
///   nondecreasing on this interval. It does not have to be defined at the boundaries `0.0`
///   or `1.0` (more precisely, it only has to be defined on the closed interval
///   `[epsilon, 1.0 - epsilon]` where `epsilon := 2.0^{-(PRECISION+1)}` and `^` denotes
///   mathematical exponentiation). Further, the implementation of `Inverse::inverse` does
///   not actually have to be the inverse of `Distribution::distribution` because it is only
///   used as an initial hint where to start a search for the true inverse. It is OK if
///   `Inverse::inverse` is just some approximation of the true inverse CDF. Any deviations
///   between `Inverse::inverse` and the true inverse CDF will negatively impact runtime
///   performance but will otherwise have no observable effect.
///
/// [`quantize`]: Self::quantize
/// [`Gaussian`]: probability::distribution::Gaussian
/// [`Binomial`]: probability::distribution::Binomial
/// [`to_generic_encoder_model`]: IterableEntropyModel::to_generic_encoder_model
/// [`to_generic_decoder_model`]: IterableEntropyModel::to_generic_decoder_model
/// [`to_generic_lookup_decoder_model`]: IterableEntropyModel::to_generic_lookup_decoder_model
/// [`IterableEntropyModel`]: IterableEntropyModel
/// [`ContiguousCategoricalEntropyModel`]: crate::stream::model::ContiguousCategoricalEntropyModel
/// [`NonContiguousCategoricalEncoderModel`]: crate::stream::model::NonContiguousCategoricalEncoderModel
/// [`NonContiguousCategoricalDecoderModel`]: crate::stream::model::NonContiguousCategoricalDecoderModel
/// [`ContiguousLookupDecoderModel`]: crate::stream::model::ContiguousLookupDecoderModel
/// [`NonContiguousLookupDecoderModel`]: crate::stream::model::NonContiguousLookupDecoderModel
#[derive(Debug, Clone, Copy)]
pub struct LeakyQuantizer<F, Symbol, Probability, const PRECISION: usize> {
    min_symbol_inclusive: Symbol,
    max_symbol_inclusive: Symbol,
    free_weight: F,
    phantom: PhantomData<Probability>,
}

/// Type alias for a typical [`LeakyQuantizer`].
///
/// See:
/// - [`LeakyQuantizer`]
/// - [discussion of presets](crate::stream#presets)
pub type DefaultLeakyQuantizer<F, Symbol> = LeakyQuantizer<F, Symbol, u32, 24>;

/// Type alias for a [`LeakyQuantizer`] optimized for compatibility with lookup decoder
/// models.
///
/// See:
/// - [`LeakyQuantizer`]
/// - [discussion of presets](crate::stream#presets)
pub type SmallLeakyQuantizer<F, Symbol> = LeakyQuantizer<F, Symbol, u16, 12>;

impl<F, Symbol, Probability, const PRECISION: usize>
    LeakyQuantizer<F, Symbol, Probability, PRECISION>
where
    Probability: BitArray + Into<F>,
    Symbol: PrimInt + AsPrimitive<Probability> + WrappingSub + WrappingAdd,
    F: FloatCore,
{
    /// Constructs a `LeakyQuantizer` with a finite support.
    ///
    /// The `support` is an inclusive range (which can be expressed with the `..=` notation,
    /// as in `-100..=100`). All [`LeakilyQuantizedDistribution`]s generated by this
    /// `LeakyQuantizer` are then guaranteed to assign a nonzero probability to all symbols
    /// within the `support`, and a zero probability to all symbols outside of the
    /// `support`. Having a known support is often a useful property of entropy models
    /// because it ensures that all symbols within the `support` can indeed be encoded, even
    /// if their probability under the underlying probability distribution is extremely
    /// small.
    ///
    /// This method takes `support` as a `RangeInclusive` because we want to support, e.g.,
    /// probability distributions over the `Symbol` type `u8` with full support `0..=255`.
    ///
    /// # Panics
    ///
    /// Panics if either of the following conditions is met:
    ///
    /// - `support` is empty; or
    /// - `support` contains only a single value (we do not support degenerate probability
    ///   distributions that put all probability mass on a single symbol); or
    /// - `support` is larger than `1 << PRECISION` (because in this case, assigning any
    ///   representable nonzero probability to all elements of `support` would exceed our
    ///   probability budge).
    ///
    /// [`quantize`]: #method.quantize
    pub fn new(support: RangeInclusive<Symbol>) -> Self {
        generic_static_asserts!(
            (Probability: BitArray; const PRECISION: usize);
            PROBABILITY_MUST_SUPPORT_PRECISION: PRECISION <= Probability::BITS;
            PRECISION_MUST_BE_NONZERO: PRECISION > 0;
        );

        // We don't support degenerate probability distributions (i.e., distributions that
        // place all probability mass on a single symbol).
        assert!(support.end() > support.start());

        let support_size_minus_one = support.end().wrapping_sub(support.start()).as_();
        let max_probability = Probability::max_value() >> (Probability::BITS - PRECISION);
        let free_weight = max_probability
            .checked_sub(&support_size_minus_one)
            .expect("The support is too large to assign a nonzero probability to each element.")
            .into();

        LeakyQuantizer {
            min_symbol_inclusive: *support.start(),
            max_symbol_inclusive: *support.end(),
            free_weight,
            phantom: PhantomData,
        }
    }

    /// Quantizes the given probability distribution and returns an [`EntropyModel`].
    ///
    /// See [struct documentation](Self) for details and code examples.
    ///
    /// Note that this method takes `self` only by reference, i.e., you can reuse
    /// the same `Quantizer` to quantize arbitrarily many distributions.
    #[inline]
    pub fn quantize<D: Distribution>(
        self,
        distribution: D,
    ) -> LeakilyQuantizedDistribution<F, Symbol, Probability, D, PRECISION> {
        LeakilyQuantizedDistribution {
            inner: distribution,
            quantizer: self,
        }
    }

    /// Returns the exact range of symbols that have nonzero probability.
    ///
    /// The returned inclusive range is the same as the one that was passed to the
    /// constructor [`new`](Self::new). All entropy models created by the method
    /// [`quantize`](Self::quantize) will assign a nonzero probability to all elements in
    /// the `support`, and they will assign a zero probability to all elements outside of
    /// the `support`. The support contains at least two and at most `1 << PRECISION`
    /// elements.
    #[inline]
    pub fn support(&self) -> RangeInclusive<Symbol> {
        self.min_symbol_inclusive..=self.max_symbol_inclusive
    }
}

/// An [`EntropyModel`] that approximates a parameterized probability [`Distribution`].
///
/// A `LeakilyQuantizedDistribution` can be created with a [`LeakyQuantizer`]. It can be
/// used for encoding and decoding with any of the stream codes provided by the
/// `constriction` crate (it can only be used for decoding if the underlying
/// [`Distribution`] implements the the trait [`Inverse`] from the [`probability`] crate).
///
/// # When Should I Use This Type of Entropy Model?
///
/// Use a `LeakilyQuantizedDistribution` when you have a probabilistic model that is defined
/// through some analytic expression (e.g., a mathematical formula for the probability
/// density function of a continuous probability distribution, or a mathematical formula for
/// the probability mass functions of some discrete probability distribution). Examples of
/// probabilistic models that lend themselves to being quantized are continuous
/// distributions such as [`Gaussian`], [`Laplace`], or [`Exponential`], as well as discrete
/// distributions with some analytic expression, such as [`Binomial`].
///
/// Do *not* use a `LeakilyQuantizedDistribution` if your probabilistic model can only be
/// presented as an explicit probability table. While you could, in principle, apply a
/// [`LeakyQuantizer`] to such a [`Categorical`] distribution, you will get better
/// computational performance (and also *slightly* better compression effectiveness) if you
/// instead use one of the dedicated types [`ContiguousCategoricalEntropyModel`],
/// [`NonContiguousCategoricalEncoderModel`], [`NonContiguousCategoricalDecoderModel`],
/// [`ContiguousLookupDecoderModel`], [`NonContiguousLookupDecoderModel`], or
/// [`LazyContiguousCategoricalEntropyModel`].
///
/// # Examples
///
/// See [examples for `LeakyQuantizer`](LeakyQuantizer#examples).
///
/// # Computational Efficiency
///
/// See [discussion for `LeakyQuantizer`](LeakyQuantizer#computational-efficiency).
///
/// [`Gaussian`]: probability::distribution::Gaussian
/// [`Laplace`]: probability::distribution::Laplace
/// [`Exponential`]: probability::distribution::Exponential
/// [`Binomial`]: probability::distribution::Binomial
/// [`Categorical`]: probability::distribution::Categorical
/// [`ContiguousCategoricalEntropyModel`]:
///     crate::stream::model::ContiguousCategoricalEntropyModel
/// [`NonContiguousCategoricalEncoderModel`]:
///     crate::stream::model::NonContiguousCategoricalEncoderModel
/// [`NonContiguousCategoricalDecoderModel`]:
///     crate::stream::model::NonContiguousCategoricalDecoderModel
/// [`ContiguousLookupDecoderModel`]: crate::stream::model::ContiguousLookupDecoderModel
/// [`LazyContiguousCategoricalEntropyModel`]:
///     crate::stream::model::LazyContiguousCategoricalEntropyModel
/// [`NonContiguousLookupDecoderModel`]:
///     crate::stream::model::NonContiguousLookupDecoderModel
#[derive(Debug, Clone, Copy)]
pub struct LeakilyQuantizedDistribution<F, Symbol, Probability, D, const PRECISION: usize> {
    inner: D,
    quantizer: LeakyQuantizer<F, Symbol, Probability, PRECISION>,
}

impl<F, Symbol, Probability, D, const PRECISION: usize>
    LeakilyQuantizedDistribution<F, Symbol, Probability, D, PRECISION>
where
    Probability: BitArray + Into<F>,
    Symbol: PrimInt + AsPrimitive<Probability> + WrappingSub + WrappingAdd,
    F: FloatCore,
{
    /// Returns the quantizer that was used to create this entropy model.
    ///
    /// You may want to reuse this quantizer to quantize further probability distributions.
    #[inline]
    pub fn quantizer(self) -> LeakyQuantizer<F, Symbol, Probability, PRECISION> {
        self.quantizer
    }

    /// Returns a reference to the underlying (floating-point) probability [`Distribution`].
    ///
    /// Returns the floating-point probability distribution which this
    /// `LeakilyQuantizedDistribution` approximates in fixed-point arithmetic.
    ///
    /// # See also
    ///
    /// - [`inner_mut`](Self::inner_mut)
    /// - [`into_inner`](Self::into_inner)
    ///
    /// [`Distribution`]: probability::distribution::Distribution
    #[inline]
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Returns a mutable reference to the underlying (floating-point) probability
    /// [`Distribution`].
    ///
    /// You can use this method to mutate parameters of the underlying [`Distribution`]
    /// after it was already quantized. This is safe and cheap since quantization is done
    /// lazily anyway. Note that you can't mutate the [`support`](Self::support) since it is a
    /// property of the [`LeakyQuantizer`], not of the `Distribution`. If you want to modify
    /// the `support` then you have to create a new `LeakyQuantizer` with a different support.
    ///
    /// # See also
    ///
    /// - [`inner`](Self::inner)
    /// - [`into_inner`](Self::into_inner)
    ///
    /// [`Distribution`]: probability::distribution::Distribution
    #[inline]
    pub fn inner_mut(&mut self) -> &mut D {
        &mut self.inner
    }

    /// Consumes the entropy model and returns the underlying (floating-point) probability
    /// [`Distribution`].
    ///
    /// Returns the floating-point probability distribution which this
    /// `LeakilyQuantizedDistribution` approximates in fixed-point arithmetic.
    ///
    /// # See also
    ///
    /// - [`inner`](Self::inner)
    /// - [`inner_mut`](Self::inner_mut)
    ///
    /// [`Distribution`]: probability::distribution::Distribution
    #[inline]
    pub fn into_inner(self) -> D {
        self.inner
    }

    /// Returns the exact range of symbols that have nonzero probability.
    ///
    /// See [`LeakyQuantizer::support`].
    #[inline]
    pub fn support(&self) -> RangeInclusive<Symbol> {
        self.quantizer.support()
    }
}

#[inline(always)]
fn slack<Probability, Symbol>(symbol: Symbol, min_symbol_inclusive: Symbol) -> Probability
where
    Probability: BitArray,
    Symbol: AsPrimitive<Probability> + WrappingSub,
{
    // This whole `mask` business is only relevant if `Symbol` is a signed type smaller than
    // `Probability`, which should be very uncommon. In all other cases, this whole stuff
    // will be optimized away.
    let mask = wrapping_pow2::<Probability>(8 * core::mem::size_of::<Symbol>())
        .wrapping_sub(&Probability::one());
    symbol.wrapping_sub(&min_symbol_inclusive).as_() & mask
}

impl<F, Symbol, Probability, D, const PRECISION: usize> EntropyModel<PRECISION>
    for LeakilyQuantizedDistribution<F, Symbol, Probability, D, PRECISION>
where
    Probability: BitArray,
{
    type Probability = Probability;
    type Symbol = Symbol;
}

impl<Symbol, Probability, D, const PRECISION: usize> EncoderModel<PRECISION>
    for LeakilyQuantizedDistribution<f64, Symbol, Probability, D, PRECISION>
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + Into<f64> + WrappingSub,
    Probability: BitArray + Into<f64>,
    D: Distribution,
    D::Value: AsPrimitive<Symbol>,
{
    /// Performs (one direction of) the quantization.
    ///
    /// # Panics
    ///
    /// Panics if it detects some invalidity in the underlying probability distribution.
    /// This means that there is a bug in the implementation of [`Distribution`] for the
    /// distribution `D`: the cumulative distribution function is either not monotonically
    /// nondecreasing, returns NaN, or its values exceed the interval `[0.0, 1.0]` at some
    /// point.
    ///
    /// More precisely, this method panics if the quantization procedure leads to a zero
    /// probability despite the added leakiness (and despite the fact that the constructor
    /// checks that `min_symbol_inclusive < max_symbol_inclusive`, i.e., that there are at
    /// least two symbols with nonzero probability and therefore the probability of a single
    /// symbol should not be able to overflow).
    ///
    /// See [requirements for correctness](LeakyQuantizer#requirements-for-correctness).
    ///
    /// [`Distribution`]: probability::distribution::Distribution
    fn left_cumulative_and_probability(
        &self,
        symbol: impl Borrow<Symbol>,
    ) -> Option<(Probability, Probability::NonZero)> {
        let min_symbol_inclusive = self.quantizer.min_symbol_inclusive;
        let max_symbol_inclusive = self.quantizer.max_symbol_inclusive;
        let free_weight = self.quantizer.free_weight;

        if symbol.borrow() < &min_symbol_inclusive || symbol.borrow() > &max_symbol_inclusive {
            return None;
        };
        let slack = slack(*symbol.borrow(), min_symbol_inclusive);

        // Round both cumulatives *independently* to fixed point precision.
        let left_sided_cumulative = if symbol.borrow() == &min_symbol_inclusive {
            // Corner case: make sure that the probabilities add up to one. The generic
            // calculation in the `else` branch may lead to a lower total probability
            // because we're cutting off the left tail of the distribution.
            Probability::zero()
        } else {
            let non_leaky: Probability =
                (free_weight * self.inner.distribution((*symbol.borrow()).into() - 0.5)).as_();
            non_leaky + slack
        };

        let right_sided_cumulative = if symbol.borrow() == &max_symbol_inclusive {
            // Corner case: make sure that the probabilities add up to one. The generic
            // calculation in the `else` branch may lead to a lower total probability
            // because we're cutting off the right tail of the distribution and we're
            // rounding down.
            wrapping_pow2(PRECISION)
        } else {
            let non_leaky: Probability =
                (free_weight * self.inner.distribution((*symbol.borrow()).into() + 0.5)).as_();
            non_leaky + slack + Probability::one()
        };

        let probability = right_sided_cumulative
            .wrapping_sub(&left_sided_cumulative)
            .into_nonzero()
            .expect("Invalid underlying continuous probability distribution.");

        Some((left_sided_cumulative, probability))
    }
}

impl<Symbol, Probability, D, const PRECISION: usize> DecoderModel<PRECISION>
    for LeakilyQuantizedDistribution<f64, Symbol, Probability, D, PRECISION>
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + Into<f64> + WrappingSub + WrappingAdd,
    Probability: BitArray + Into<f64>,
    D: Inverse,
    D::Value: AsPrimitive<Symbol>,
{
    fn quantile_function(
        &self,
        quantile: Probability,
    ) -> (Self::Symbol, Probability, Probability::NonZero) {
        let max_probability = Probability::max_value() >> (Probability::BITS - PRECISION);
        // This check should usually compile away in inlined and verifiably correct usages
        // of this method.
        assert!(quantile <= max_probability);

        let inverse_denominator = 1.0 / (max_probability.into() + 1.0);

        let min_symbol_inclusive = self.quantizer.min_symbol_inclusive;
        let max_symbol_inclusive = self.quantizer.max_symbol_inclusive;
        let free_weight = self.quantizer.free_weight;

        // Make an initial guess for the inverse of the leaky CDF.
        let mut symbol: Self::Symbol = self
            .inner
            .inverse((quantile.into() + 0.5) * inverse_denominator)
            .as_();

        let mut left_sided_cumulative = if symbol <= min_symbol_inclusive {
            // Corner case: we're in the left cut off tail of the distribution.
            symbol = min_symbol_inclusive;
            Probability::zero()
        } else {
            if symbol > max_symbol_inclusive {
                // Corner case: we're in the right cut off tail of the distribution.
                symbol = max_symbol_inclusive;
            }

            let non_leaky: Probability =
                (free_weight * self.inner.distribution(symbol.into() - 0.5)).as_();
            non_leaky + slack(symbol, min_symbol_inclusive)
        };

        // SAFETY: We have to ensure that all paths lead to a state where
        // `right_sided_cumulative != left_sided_cumulative`.
        let mut step = Self::Symbol::one(); // `step` will always be a power of 2.
        let right_sided_cumulative = if left_sided_cumulative > quantile {
            // Our initial guess for `symbol` was too high. Reduce it until we're good.
            symbol = symbol - step;
            let mut found_lower_bound = false;

            loop {
                let old_left_sided_cumulative = left_sided_cumulative;

                if symbol == min_symbol_inclusive {
                    left_sided_cumulative = Probability::zero();
                    if step <= Symbol::one() {
                        // This can only be reached from a downward search, so `old_left_sided_cumulative`
                        // is the right sided cumulative since the step size is one.
                        // SAFETY: `old_left_sided_cumulative > quantile >= 0 = left_sided_cumulative`
                        break old_left_sided_cumulative;
                    }
                } else {
                    let non_leaky: Probability =
                        (free_weight * self.inner.distribution(symbol.into() - 0.5)).as_();
                    left_sided_cumulative = non_leaky + slack(symbol, min_symbol_inclusive);
                }

                if left_sided_cumulative <= quantile {
                    found_lower_bound = true;
                    // We found a lower bound, so we're either done or we have to do a binary
                    // search now.
                    if step <= Symbol::one() {
                        let right_sided_cumulative = if symbol == max_symbol_inclusive {
                            wrapping_pow2(PRECISION)
                        } else {
                            let non_leaky: Probability =
                                (free_weight * self.inner.distribution(symbol.into() + 0.5)).as_();
                            (non_leaky + slack(symbol, min_symbol_inclusive))
                                .wrapping_add(&Probability::one())
                        };
                        // SAFETY: `old_left_sided_cumulative > quantile >= left_sided_cumulative`
                        break right_sided_cumulative;
                    } else {
                        step = step >> 1;
                        // The following addition can't overflow because we're in the binary search phase.
                        symbol = symbol + step;
                    }
                } else if found_lower_bound {
                    // We're in the binary search phase, so all following guesses will be within bounds.
                    if step > Symbol::one() {
                        step = step >> 1
                    }
                    symbol = symbol - step;
                } else {
                    // We're still in the downward search phase with exponentially increasing step size.
                    if step << 1 != Symbol::zero() {
                        step = step << 1;
                    }

                    // Find a smaller `symbol` that is still `>= min_symbol_inclusive`.
                    symbol = loop {
                        let new_symbol = symbol.wrapping_sub(&step);
                        if new_symbol >= min_symbol_inclusive && new_symbol <= symbol {
                            break new_symbol;
                        }
                        // The following cannot set `step` to zero because this would mean that
                        // `step == 1` and thus either the above `if` branch would have been
                        // chosen, or `symbol == min_symbol_inclusive` (which would imply
                        // `left_sided_cumulative <= quantile`), or `symbol` would be the
                        // lowest representable symbol (which would also require
                        // `symbol == min_symbol_inclusive`).
                        step = step >> 1;
                    };
                }
            }
        } else {
            // Our initial guess for `symbol` was either exactly right or too low.
            // Check validity of the right sided cumulative. If it isn't valid,
            // keep increasing `symbol` until it is.
            let mut found_upper_bound = false;

            loop {
                let right_sided_cumulative = if symbol == max_symbol_inclusive {
                    let right_sided_cumulative = wrapping_pow2(PRECISION);
                    if step <= Symbol::one() {
                        let non_leaky: Probability =
                            (free_weight * self.inner.distribution(symbol.into() - 0.5)).as_();
                        left_sided_cumulative = non_leaky + slack(symbol, min_symbol_inclusive);

                        // SAFETY: we have to manually check here.
                        if right_sided_cumulative == left_sided_cumulative {
                            panic!("Invalid underlying probability distribution.");
                        }

                        break right_sided_cumulative;
                    } else {
                        right_sided_cumulative
                    }
                } else {
                    let non_leaky: Probability =
                        (free_weight * self.inner.distribution(symbol.into() + 0.5)).as_();
                    (non_leaky + slack(symbol, min_symbol_inclusive))
                        .wrapping_add(&Probability::one())
                };

                if right_sided_cumulative > quantile
                    || right_sided_cumulative == Probability::zero()
                {
                    found_upper_bound = true;
                    // We found an upper bound, so we're either done or we have to do a binary
                    // search now.
                    if step <= Symbol::one() {
                        left_sided_cumulative = if symbol == min_symbol_inclusive {
                            Probability::zero()
                        } else {
                            let non_leaky: Probability =
                                (free_weight * self.inner.distribution(symbol.into() - 0.5)).as_();
                            non_leaky + slack(symbol, min_symbol_inclusive)
                        };

                        if left_sided_cumulative <= quantile || symbol == min_symbol_inclusive {
                            // SAFETY: we have `left_sided_cumulative <= quantile < right_sided_sided_cumulative`
                            break right_sided_cumulative;
                        }
                    } else {
                        step = step >> 1;
                    }
                    // The following subtraction can't overflow because we're in the binary search phase.
                    symbol = symbol - step;
                } else if found_upper_bound {
                    // We're in the binary search phase, so all following guesses will be within bounds.
                    if step > Symbol::one() {
                        step = step >> 1
                    }
                    symbol = symbol + step;
                } else {
                    // We're still in the upward search phase with exponentially increasing step size.
                    if step << 1 != Symbol::zero() {
                        step = step << 1;
                    }

                    symbol = loop {
                        let new_symbol = symbol.wrapping_add(&step);
                        if new_symbol <= max_symbol_inclusive && new_symbol >= symbol {
                            break new_symbol;
                        }
                        // The following cannot set `step` to zero because this would mean that
                        // `step == 1` and thus either the above `if` branch would have been
                        // chosen, or `symbol == max_symbol_inclusive` (which would imply
                        // `right_sided_cumulative > quantile || right_sided_cumulative == 0`),
                        // or `symbol` would be the largest representable symbol (which would
                        // also require `symbol == max_symbol_inclusive`).
                        step = step >> 1;
                    };
                }
            }
        };

        let probability = unsafe {
            // SAFETY: see above "SAFETY" comments on all paths that lead here.
            right_sided_cumulative
                .wrapping_sub(&left_sided_cumulative)
                .into_nonzero_unchecked()
        };
        (symbol, left_sided_cumulative, probability)
    }
}

impl<'m, Symbol, Probability, D, const PRECISION: usize> IterableEntropyModel<'m, PRECISION>
    for LeakilyQuantizedDistribution<f64, Symbol, Probability, D, PRECISION>
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + AsPrimitive<usize> + Into<f64> + WrappingSub,
    Probability: BitArray + Into<f64>,
    D: Distribution + 'm,
    D::Value: AsPrimitive<Symbol>,
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
        LeakilyQuantizedDistributionIter {
            model: self,
            symbol: Some(self.quantizer.min_symbol_inclusive),
            left_sided_cumulative: Probability::zero(),
        }
    }
}

/// Iterator over the [`symbol_table`] of a [`LeakilyQuantizedDistribution`].
///
/// [`symbol_table`]: IterableEntropyModel::symbol_table
#[derive(Debug)]
struct LeakilyQuantizedDistributionIter<Symbol, Probability, M, const PRECISION: usize> {
    model: M,
    symbol: Option<Symbol>,
    left_sided_cumulative: Probability,
}

impl<Symbol, Probability, D, const PRECISION: usize> Iterator
    for LeakilyQuantizedDistributionIter<
        Symbol,
        Probability,
        &LeakilyQuantizedDistribution<f64, Symbol, Probability, D, PRECISION>,
        PRECISION,
    >
where
    f64: AsPrimitive<Probability>,
    Symbol: PrimInt + AsPrimitive<Probability> + AsPrimitive<usize> + Into<f64> + WrappingSub,
    Probability: BitArray + Into<f64>,
    D: Distribution,
    D::Value: AsPrimitive<Symbol>,
{
    type Item = (Symbol, Probability, Probability::NonZero);

    fn next(&mut self) -> Option<Self::Item> {
        let symbol = self.symbol?;

        let right_sided_cumulative = if symbol == self.model.quantizer.max_symbol_inclusive {
            self.symbol = None;
            wrapping_pow2(PRECISION)
        } else {
            let next_symbol = symbol + Symbol::one();
            self.symbol = Some(next_symbol);
            let non_leaky: Probability = (self.model.quantizer.free_weight
                * self.model.inner.distribution((symbol).into() - 0.5))
            .as_();
            non_leaky + slack(next_symbol, self.model.quantizer.min_symbol_inclusive)
        };

        let probability = unsafe {
            // SAFETY: probabilities of
            right_sided_cumulative
                .wrapping_sub(&self.left_sided_cumulative)
                .into_nonzero_unchecked()
        };

        let left_sided_cumulative = self.left_sided_cumulative;
        self.left_sided_cumulative = right_sided_cumulative;

        Some((symbol, left_sided_cumulative, probability))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if let Some(symbol) = self.symbol {
            let len = slack::<usize, _>(symbol, self.model.quantizer.max_symbol_inclusive)
                .saturating_add(1);
            (len, None)
        } else {
            (0, Some(0))
        }
    }
}

#[cfg(test)]
mod tests {
    use probability::prelude::*;

    use super::*;

    #[test]
    fn split_almost_delta_distribution() {
        fn inner(distribution: impl Distribution<Value = f64>) {
            let quantizer = DefaultLeakyQuantizer::new(-10..=10);
            let model = quantizer.quantize(distribution);
            let (left_cdf, left_prob) = model.left_cumulative_and_probability(2).unwrap();
            let (right_cdf, right_prob) = model.left_cumulative_and_probability(3).unwrap();

            assert_eq!(
                left_prob.get(),
                right_prob.get() - 1,
                "Peak not split evenly."
            );
            assert_eq!(
                (1u32 << 24) - left_prob.get() - right_prob.get(),
                19,
                "Peak has wrong probability mass."
            );
            assert_eq!(left_cdf + left_prob.get(), right_cdf);
            // More thorough generic consistency checks of the CDF are done in `test_quantized_*()`.
        }

        inner(Gaussian::new(2.5, 1e-40));
        inner(Cauchy::new(2.5, 1e-40));
        inner(Laplace::new(2.5, 1e-40));
    }

    #[test]
    fn leakily_quantized_normal() {
        #[cfg(not(miri))]
        let (support, std_devs, means) = (
            -127..=127,
            [1e-40, 0.0001, 0.1, 3.5, 123.45, 1234.56],
            [
                -300.6, -127.5, -100.2, -4.5, 0.0, 50.3, 127.5, 180.2, 2000.0,
            ],
        );

        // We use different settings when testing on miri so that the test time stays reasonable.
        #[cfg(miri)]
        let (support, std_devs, means) = (
            -20..=20,
            [1e-40, 0.0001, 3.5, 1234.56],
            [-300.6, -20.5, -5.2, 8.5, 20.5, 2000.0],
        );

        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(support.clone());
        for &std_dev in &std_devs {
            for &mean in &means {
                let distribution = Gaussian::new(mean, std_dev);
                super::super::tests::test_entropy_model(
                    &quantizer.quantize(distribution),
                    *support.start()..*support.end() + 1,
                );
            }
        }
    }

    #[test]
    fn leakily_quantized_cauchy() {
        #[cfg(not(miri))]
        let (support, gammas, means) = (
            -127..=127,
            [1e-40, 0.0001, 0.1, 3.5, 123.45, 1234.56],
            [
                -300.6, -127.5, -100.2, -4.5, 0.0, 50.3, 127.5, 180.2, 2000.0,
            ],
        );

        // We use different settings when testing on miri so that the test time stays reasonable.
        #[cfg(miri)]
        let (support, gammas, means) = (
            -20..=20,
            [1e-40, 0.0001, 3.5, 1234.56],
            [-300.6, -20.5, -5.2, 8.5, 20.5, 2000.0],
        );
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(support.clone());
        for &gamma in &gammas {
            for &mean in &means {
                let distribution = Cauchy::new(mean, gamma);
                super::super::tests::test_entropy_model(
                    &quantizer.quantize(distribution),
                    *support.start()..*support.end() + 1,
                );
            }
        }
    }

    #[test]
    fn leakily_quantized_laplace() {
        #[cfg(not(miri))]
        let (support, bs, means) = (
            -127..=127,
            [1e-40, 0.0001, 0.1, 3.5, 123.45, 1234.56],
            [
                -300.6, -127.5, -100.2, -4.5, 0.0, 50.3, 127.5, 180.2, 2000.0,
            ],
        );

        // We use different settings when testing on miri so that the test time stays reasonable.
        #[cfg(miri)]
        let (support, bs, means) = (
            -20..=20,
            [1e-40, 0.0001, 3.5, 1234.56],
            [-300.6, -20.5, -5.2, 8.5, 20.5, 2000.0],
        );
        let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(support.clone());
        for &b in &bs {
            for &mean in &means {
                let distribution = Laplace::new(mean, b);
                super::super::tests::test_entropy_model(
                    &quantizer.quantize(distribution),
                    *support.start()..*support.end() + 1,
                );
            }
        }
    }

    #[test]
    fn leakily_quantized_binomial() {
        #[cfg(not(miri))]
        let (ns, ps) = (
            [1, 2, 10, 100, 1000, 10_000],
            [1e-30, 1e-20, 1e-10, 0.1, 0.4, 0.9],
        );

        // We use different settings when testing on miri so that the test time stays reasonable.
        #[cfg(miri)]
        let (ns, ps) = ([1, 2, 100], [1e-30, 0.1, 0.4]);

        for &n in &ns {
            for &p in &ps {
                if n < 1000 || p >= 0.1 {
                    // In the excluded situations, `<Binomial as Inverse>::inverse` currently doesn't terminate.
                    // TODO: file issue to `probability` repo.
                    let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=n as u32);
                    let distribution = Binomial::new(n, p);
                    super::super::tests::test_entropy_model(
                        &quantizer.quantize(distribution),
                        0..(n as u32 + 1),
                    );
                }
            }
        }
    }
}
