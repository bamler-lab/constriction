#[cfg(feature = "pybindings")]
pub mod pybindings;

pub mod distributions;

use std::ops::Deref;

use distributions::DiscreteDistribution;
use num::{
    cast::AsPrimitive,
    traits::{WrappingAdd, WrappingSub},
    CheckedDiv, One, PrimInt, Unsigned, Zero,
};

pub unsafe trait CompressedWord:
    PrimInt + Unsigned + WrappingSub + WrappingAdd + 'static
{
    /// Must be exactly twice as large as `Self`.
    type State: PrimInt + From<Self> + AsPrimitive<Self>;

    fn bits() -> usize {
        8 * std::mem::size_of::<Self>()
    }

    fn min_state() -> Self::State {
        Self::State::from(Self::max_value()) + Self::State::one()
    }

    fn split_state(state: Self::State) -> (Self, Self) {
        let high = (state >> Self::bits()).as_();
        let low = state.as_();
        (low, high)
    }

    fn compose_state(low: Self, high: Self) -> Self::State {
        (Self::State::from(high) << Self::bits()) | Self::State::from(low)
    }
}

unsafe impl CompressedWord for u8 {
    type State = u16;
}

unsafe impl CompressedWord for u16 {
    type State = u32;
}

unsafe impl CompressedWord for u32 {
    type State = u64;
}

unsafe impl CompressedWord for u64 {
    type State = u128;
}

pub struct Coder<W: CompressedWord> {
    buf: Vec<W>,
    state: W::State,
}

#[non_exhaustive]
#[derive(Debug)]
pub enum CoderError {
    InvalidCompressedData,
    InvalidSymbol,
    DistributionsYieldedTooManyItems,
    DistributionsYieldedTooFewItems,
    InvalidDistribution,
}

type Result<T> = std::result::Result<T, CoderError>;

impl<W: CompressedWord> Coder<W> {
    /// Creates an Empty ANS coder.
    ///
    /// This is usually the starting point if you want to *compress` data.
    ///
    /// # Example
    ///
    /// ```
    /// let mut coder = ans::Coder::<u32>::new();
    ///
    /// // ... push some symbols on the coder ...
    ///
    /// // Finally, get the compressed data.
    /// let compressed = coder.into_compressed();
    /// ```
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            state: W::State::zero(),
        }
    }

    /// Creates ANS coder with some initial.
    ///
    /// This is usually the starting point if you want to *decompress` data
    /// previously obtained from [`into_compressed`](#method.into_compressed).
    /// However, it can also be used to append more symbols to an existing
    /// compressed stream of data.
    pub fn with_compressed_data(mut compressed: Vec<W>) -> Self {
        let (low, high) = match compressed.len() {
            0 => (W::zero(), W::zero()),
            1 => (compressed.pop().unwrap(), W::zero()),
            _ => {
                let high = compressed.pop().unwrap();
                let low = compressed.pop().unwrap();
                (low, high)
            }
        };

        Self {
            buf: compressed,
            state: W::compose_state(low, high),
        }
    }

    /// Encodes a single symbol and appends it to the compressed data.
    ///
    /// This is a low level method. You usually probably want to call a batch method
    /// like [`push_symbols`](#method.push_symbols) or
    /// [`push_iid_symbols`](#method.push_iid_symbols) instead. See examples there.
    ///
    /// This method is called `push_symbol` rather than `encode_symbol` to highlight
    /// the fact that the `Coder` is a stack: the last symbol `pushed` onto the
    /// stack will be the first symbol that [`pop_symbol`](#method.pop_symbol) will
    /// retrieve.
    ///
    /// Returns [`Err(InvalidSymbol)`](enum.CoderError.html#variant.InvalidSymbol)
    /// if `symbol` has zero probability under the entropy model `distribution`.
    /// This error can usually be avoided by using a "leaky" distribution, i.e., a
    /// distribution that assign a nonzero probability to all symbols within a
    /// finite domain. Leaky distributions can be constructed with, e.g., a
    /// [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_continuous_probabilities`](
    /// distributions/struct.Categorical.html#method.from_continuous_probabilities).
    pub fn push_symbol<S>(
        &mut self,
        symbol: S,
        distribution: impl DiscreteDistribution<S, W>,
    ) -> Result<()> {
        let (left_sided_cumulative, probability) = distribution
            .left_cumulative_and_probability(symbol)
            .map_err(|()| CoderError::InvalidSymbol)?;
        if self.state >= W::compose_state(W::zero(), probability) {
            let (low, high) = W::split_state(self.state);
            self.buf.push(low);
            self.state = W::State::from(high);
        }
        let prefix = self
            .state
            .checked_div(&W::State::from(probability))
            .ok_or(CoderError::InvalidSymbol)?;
        let suffix =
            W::State::from(left_sided_cumulative) + self.state % W::State::from(probability);
        self.state = (prefix << W::bits()) | suffix;

        Ok(())
    }

    /// Decodes a single symbol and pops it off the compressed data.
    ///
    /// This is a low level method. You usually probably want to call a batch method
    /// like [`pop_symbols`](#method.pop_symbols) or
    /// [`pop_iid_symbols`](#method.pop_iid_symbols) instead.
    ///
    /// This method is called `pop_symbol` rather than `decode_symbol` to highlight
    /// the fact that the `Coder` is a stack: `pop_symbol` will return the *last*
    /// symbol that was previously encoded via [`push_symbol`](#method.push_symbol).
    ///
    /// Note that this method cannot fail. It will still produce symbols even if the
    /// coder is empty, but such symbols will not recover any previously encoded
    /// data and will generally have low entropy. Still, being able to pop off an
    /// arbitrary number of symbols can sometimes be useful in edge cases of, e.g.,
    /// the bits-back algorithm.
    pub fn pop_symbol<S>(&mut self, distribution: impl DiscreteDistribution<S, W>) -> S {
        let (low, high) = W::split_state(self.state);
        let (symbol, left_sided_cumulative, probability) = distribution.quantile_function(low);
        self.state = W::State::from(probability) * W::State::from(high)
            + W::State::from(low - left_sided_cumulative);

        if self.state < W::min_state() {
            if let Some(word) = self.buf.pop() {
                self.state = (self.state << W::bits()) | W::State::from(word);
            }
        }

        symbol
    }

    /// Encodes multiple symbols with individual entropy models in reverse order.
    ///
    /// The symbols are encoded in reverse order so that calling [`pop_symbols`]
    /// retrieves them in forward order (since the ANS coder is a stack).
    ///
    /// Returns [`Err(InvalidSymbol)`](enum.CoderError.html#variant.InvalidSymbol)
    /// if one of the symbols has zero probability under its entropy model.
    /// In this case, a part of the symbols may have already been pushed on the
    /// coder. This error can usually be avoided by using "leaky" distributions,
    /// i.e., distributions that assign a nonzero probability to all symbols within
    /// a finite domain. Leaky distributions can be constructed with, e.g., a
    /// [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_continuous_probabilities`](
    /// distributions/struct.Categorical.html#method.from_continuous_probabilities).
    ///
    /// # Example
    ///
    /// ```
    /// use statrs::distribution::Normal;
    ///
    /// let mut coder = ans::Coder::<u32>::new();
    /// let quantizer = ans::distributions::LeakyQuantizer::new(-20..=20);
    /// let symbols = vec![5, -1, -3, 4];
    /// let means_and_stds = vec![(3.2, 1.9), (5.7, 8.2), (-1.4, 2.1), (2.6, 5.3)];
    ///
    /// // Create entropy models lazily to reduce memory overhead.
    /// let entropy_models = means_and_stds
    ///     .iter()
    ///     .map(|&(mean, std)| quantizer.quantize(Normal::new(mean, std).unwrap()));
    ///
    /// // Encode the sequence of symbols.
    /// coder.push_symbols(symbols.iter().cloned().zip(entropy_models.clone()));
    ///
    /// // Decode the encoded data.
    /// let reconstructed = coder.pop_symbols(entropy_models).collect::<Vec<_>>();
    ///
    /// assert_eq!(symbols, reconstructed);
    /// assert!(coder.is_empty());
    /// ```
    ///
    /// [`pop_symbols`]: #method.pop_symbols
    pub fn push_symbols<S, D: DiscreteDistribution<S, W>>(
        &mut self,
        symbols_and_distributions: impl Iterator<Item = (S, D)> + DoubleEndedIterator,
    ) -> Result<()> {
        for (symbol, distribution) in symbols_and_distributions.rev() {
            self.push_symbol(symbol, distribution)?;
        }

        Ok(())
    }

    /// Decodes a sequence of symbols.
    ///
    /// This method is the inverse of [`push_symbols`]. See example there.
    ///
    /// # TODO
    ///
    /// Once specialization is stable, return an `impl ExactSizeIterator` if
    /// `distribution` implements `ExactSizeIterator` (same for
    /// [`pop_iid_symbols`](#method.pop_iid_symbols)).
    ///
    /// [`push_symbols`]: #method.push_symbols
    pub fn pop_symbols<'s, S, D: DiscreteDistribution<S, W>>(
        &'s mut self,
        distributions: impl Iterator<Item = D> + 's,
    ) -> impl Iterator<Item = S> + 's {
        distributions.map(move |distribution| self.pop_symbol(distribution))
    }

    /// Encodes a sequence of symbols using a fixed entropy model.
    ///
    /// This is a convenience wrapper around [`push_symbols`] and the inverse of
    /// [`pop_iid_symbols`].
    ///
    /// Returns [`Err(InvalidSymbol)`](enum.CoderError.html#variant.InvalidSymbol)
    /// if one of the symbols has zero probability under its entropy model.
    /// In this case, a part of the symbols may have already been pushed on the
    /// coder. This error can usually be avoided by using "leaky" distributions,
    /// i.e., distributions that assign a nonzero probability to all symbols within
    /// a finite domain. Leaky distributions can be constructed with, e.g., a
    /// [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_continuous_probabilities`](
    /// distributions/struct.Categorical.html#method.from_continuous_probabilities).
    ///
    /// # Example
    ///
    /// ```
    /// let mut coder = ans::Coder::<u32>::new();
    /// let symbols = vec![5, -1, -3, 4];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution =
    ///     ans::distributions::Categorical::from_continuous_probabilities(&probabilities, -3);
    ///
    /// coder.push_iid_symbols(symbols.iter().cloned(), &distribution).unwrap();
    /// let reconstructed = coder.pop_iid_symbols(4, &distribution).collect::<Vec<_>>();
    ///
    /// assert_eq!(symbols, reconstructed);
    /// assert!(coder.is_empty());
    /// ```
    ///
    /// [`push_symbols`]: #method.push_symbols
    /// [`pop_iid_symbols`]: #method.pop_iid_symbols
    pub fn push_iid_symbols<S>(
        &mut self,
        symbols: impl Iterator<Item = S> + DoubleEndedIterator,
        distribution: &impl DiscreteDistribution<S, W>,
    ) -> Result<()> {
        self.push_symbols(symbols.map(|symbol| (symbol, distribution)))
    }

    /// Decode a sequence of symbols using a fixed entropy model.
    ///
    /// This is a convenience wrapper around [`pop_symbols`], and the inverse of
    /// [`push_iid_symbols`]. See example in the latter.
    ///
    /// [`pop_symbols`]: #method.pop_symbols
    /// [`push_iid_symbols`]: #method.push_iid_symbols
    pub fn pop_iid_symbols<'c, S: 'c>(
        &'c mut self,
        amt: usize,
        distribution: &'c impl DiscreteDistribution<S, W>,
    ) -> impl Iterator<Item = S> + 'c {
        self.pop_symbols((0..amt).map(move |_| distribution))
    }

    /// Discards all compressed data and resets the coder to the same state as
    /// [`Coder::new`](#method.new).
    pub fn clear(&mut self) {
        self.buf.clear();
        self.state = W::State::zero();
    }

    /// Check if no data for decoding is left.
    ///
    /// This method returns `true` if no data is left for decoding. This means that
    /// the coder is in the same state as it would be after being constructed with
    /// [`Coder::new`](#method.new) or after calling [`clear`](#method.clear).
    ///
    /// Note that you can still pop symbols off an empty coder, but this is only
    /// useful in rare edge cases, see documentation of
    /// [`pop_symbol`](#method.pop_symbol).
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty() && self.state == W::State::zero()
    }

    /// Returns a view into the full compressed data currently on the stack.
    ///
    /// Due to the way how an `Coder` works internally, the compressed data is
    /// split up into a `bulk` and a `head` part. This method returns these two
    /// parts as a tuple `(bulk, head)`.
    ///
    /// Here, `bulk: &[W]` has variable size (and can be empty). It changes only
    /// infrequently (when `head` overflows or underflows). By contrast, the `head`
    /// part has a fixed size: `head: [W: 2]` always contains two words of
    /// compressed data.
    ///
    /// # See also
    ///
    /// The full compressed data is comprised of the concatenation of `bulk` and
    /// `head`. The methods [`to_compressed`](#method.as_compressed) and
    /// [`into_compressed`](#method.into_compressed) both perform this concatenation
    /// for you (truncating any trailing zero words, which is compatible with what
    /// [`with_compressed_data`](method.with_compressed_data) expects). Thus, these
    /// methods are usually more convenient as they return the complete compressed
    /// data in a single slice or `Vec` of words. However, they need mutable access
    /// to, or ownership of, the coder, respectively.
    pub fn as_compressed(&self) -> (&[W], [W; 2]) {
        // Return the head as an array rather than a tuple of compressed words to make
        // it more obvious how it should logically be appended to the bulk.
        let (low, high) = W::split_state(self.state);
        (&self.buf, [low, high])
    }

    /// Assembles the current compressed data into a single slice.
    ///
    /// This method is similar to [`as_compressed`](#method.as_compressed) with the
    /// difference that it concatenates the `bulk` and `head` before returning them.
    /// The concatenation truncates any trailing zero words, which is compatible with
    /// what [`with_compressed_data`](method.with_compressed_data) expects.
    ///
    /// The returned `CoderGuard` dereferences to `&[W]`, thus providing read-only
    /// access to the compressed data. If you need to modify the compressed data,
    /// consider calling [`into_compressed`](#method.into_compressed) instead.
    ///
    /// # Example
    ///
    /// ```
    /// let mut coder = ans::Coder::<u32>::new();
    ///
    /// // Push some data on the coder.
    /// let symbols = vec![5, -1, -3, 4];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution =
    ///     ans::distributions::Categorical::from_continuous_probabilities(&probabilities, -3);
    /// coder.push_iid_symbols(symbols.iter().cloned(), &distribution).unwrap();
    ///
    /// // Inspect the compressed data.
    /// let compressed = coder.to_compressed();
    /// dbg!(&*compressed);
    ///
    /// // Need to drop `compressed` before the `coder` can be mutated again.
    /// // (Alternatively, we could have enclosed the above inspection in a block.)
    /// std::mem::drop(compressed);
    ///
    /// // Now `coder` can be used again.
    /// let reconstructed = coder.pop_iid_symbols(4, &distribution).collect::<Vec<_>>();
    /// assert_eq!(reconstructed, symbols);
    /// ```
    pub fn to_compressed(&mut self) -> CoderGuard<W> {
        CoderGuard::new(self)
    }

    /// Consumes the coder and returns the compressed data.
    ///
    /// The returned data can be used to recreate a coder with the same state
    /// (e.g., for decoding) by passing it to
    /// [`with_compressed_data`](#with_compressed_data).
    ///
    /// If you don't want to consume the coder, consider calling
    /// [`to_compressed`](#method.to_compressed) or
    /// [`as_compressed`](#method.as_compressed) instead.
    ///
    /// # Example
    ///
    /// ```
    /// let mut coder = ans::Coder::<u32>::new();
    ///
    /// // Push some data on the coder.
    /// let symbols = vec![5, -1, -3, 4];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution =
    ///     ans::distributions::Categorical::from_continuous_probabilities(&probabilities, -3);
    /// coder.push_iid_symbols(symbols.iter().cloned(), &distribution).unwrap();
    ///
    /// // Get the compressed data, consuming the coder.
    /// let compressed = coder.into_compressed();
    ///
    /// // ... write `compressed` to a file and then read it back later ...
    ///
    /// // Create a new coder with the same state and use it for decompression.
    /// let mut coder = ans::Coder::with_compressed_data(compressed);
    /// let reconstructed = coder.pop_iid_symbols(4, &distribution).collect::<Vec<_>>();
    /// assert_eq!(reconstructed, symbols);
    /// assert!(coder.is_empty())
    /// ```
    pub fn into_compressed(mut self) -> Vec<W> {
        self.flush_state();
        self.buf
    }

    /// Concatenates `state` to `buf` and returns the number of words appended.
    ///
    /// This puts the coder into a temporary inconsistent state, which is why this
    /// method is private.
    fn flush_state(&mut self) -> usize {
        let (low, high) = W::split_state(self.state);
        if high == W::zero() {
            debug_assert!(self.buf.is_empty());
            if low == W::zero() {
                0
            } else {
                self.buf.push(low);
                1
            }
        } else {
            self.buf.push(low);
            self.buf.push(high);
            2
        }
    }

    /// Returns the number of compressed words `W` on the stack.
    ///
    /// This method returns the length of the slice or `Vec` that would be returned
    /// by [`to_compressed`](#method.to_compressed) or
    /// [`into_compressed`](#method.into_compressed), respectively, without mutating
    /// the coder. See also [`num_bits`](#method.num_bits).
    pub fn num_words(&self) -> usize {
        let (low, high) = W::split_state(self.state);
        if high == W::zero() {
            debug_assert!(self.buf.is_empty());
            if low == W::zero() {
                0
            } else {
                1
            }
        } else {
            self.buf.len() + 2
        }
    }

    /// Returns the number of compressed bits on the stack.
    ///
    /// The returned value is a multiple of the bitlength of the compressed word
    /// type `W`. See also [`num_words`](#method.num_words).
    pub fn num_bits(&self) -> usize {
        W::bits() * self.num_words()
    }
}

/// Provides temporary read-only access to the compressed data wrapped in an
/// [`Coder`].
///
/// Dereferences to `&[W]`. See [`Coder::to_compressed`] for an example.
///
/// [`Coder`]: struct.Coder.html
/// [`Coder::to_compressed`]: struct.Coder.html#method.to_compressed
pub struct CoderGuard<'a, W: CompressedWord> {
    inner: &'a mut Coder<W>,
    num_appended: usize,
}

impl<'a, W: CompressedWord> CoderGuard<'a, W> {
    fn new(coder: &'a mut Coder<W>) -> Self {
        let num_appended = coder.flush_state();
        Self {
            inner: coder,
            num_appended,
        }
    }
}

impl<'a, W: CompressedWord> Drop for CoderGuard<'a, W> {
    fn drop(&mut self) {
        // No need to reset `coder.state`, it cannot have changed.
        for _ in 0..self.num_appended {
            self.inner.buf.pop();
        }
    }
}

impl<'a, W: CompressedWord> Deref for CoderGuard<'a, W> {
    type Target = [W];

    fn deref(&self) -> &Self::Target {
        &self.inner.buf
    }
}

#[cfg(test)]
mod tests {
    use super::distributions::{Categorical, DiscreteDistribution, LeakyQuantizer};
    use super::*;

    use rand_xoshiro::rand_core::{RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;
    use statrs::distribution::{InverseCDF, Normal};

    #[test]
    fn compress_none() {
        let coder1 = Coder::<u32>::new();
        assert!(coder1.is_empty());
        let compressed = coder1.into_compressed();
        assert!(compressed.is_empty());

        let coder2 = Coder::<u32>::with_compressed_data(compressed);
        assert!(coder2.is_empty());
    }

    #[test]
    fn compress_one() {
        let mut coder = Coder::<u32>::new();
        let quantizer = LeakyQuantizer::new(-127..=127);
        let distribution = quantizer.quantize(Normal::new(3.2, 5.1).unwrap());

        coder.push_symbol(2, &distribution).unwrap();

        // Test if import/export of compressed data works.
        let compressed = coder.into_compressed();
        assert_eq!(compressed.len(), 1);
        let mut coder = Coder::with_compressed_data(compressed);

        assert_eq!(coder.pop_symbol(&distribution), 2);

        assert!(coder.is_empty());
    }

    #[test]
    fn compress_many() {
        const AMT: usize = 1000;
        let mut symbols_gaussian = Vec::with_capacity(AMT);
        let mut means = Vec::with_capacity(AMT);
        let mut stds = Vec::with_capacity(AMT);

        let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
        for _ in 0..AMT {
            let mean = (200.0 / u32::MAX as f64) * rng.next_u32() as f64 - 100.0;
            let std_dev = (10.0 / u32::MAX as f64) * rng.next_u32() as f64 + 0.001;
            let quantile = (rng.next_u32() as f64 + 0.5) / (1u64 << 32) as f64;
            let dist = Normal::new(mean, std_dev).unwrap();
            let symbol = std::cmp::max(
                -127,
                std::cmp::min(127, (dist.inverse_cdf(quantile) + 0.5) as i32),
            );

            symbols_gaussian.push(symbol);
            means.push(mean);
            stds.push(std_dev);
        }

        let hist = [
            1u32, 186545, 237403, 295700, 361445, 433686, 509456, 586943, 663946, 737772, 1657269,
            896675, 922197, 930672, 916665, 0, 0, 0, 0, 0, 723031, 650522, 572300, 494702, 418703,
            347600, 1, 283500, 226158, 178194, 136301, 103158, 76823, 55540, 39258, 27988, 54269,
        ];
        let categorical_probabilities = hist.iter().map(|&x| x as f64).collect::<Vec<_>>();
        let categorical =
            Categorical::from_continuous_probabilities(&categorical_probabilities, -127);
        let mut symbols_categorical = Vec::with_capacity(AMT);
        for _ in 0..AMT {
            let quantile = rng.next_u32();
            let symbol = categorical.quantile_function(quantile).0;
            symbols_categorical.push(symbol);
        }

        let mut coder = Coder::new();

        coder
            .push_iid_symbols(symbols_categorical.iter().cloned(), &categorical)
            .unwrap();
        dbg!(coder.num_bits(), AMT as f64 * categorical.entropy::<f64>());

        let quantizer = LeakyQuantizer::new(-127..=127);
        coder
            .push_symbols(symbols_gaussian.iter().zip(&means).zip(&stds).map(
                |((&symbol, &mean), &std)| {
                    (symbol, quantizer.quantize(Normal::new(mean, std).unwrap()))
                },
            ))
            .unwrap();
        dbg!(coder.num_bits());

        // Test if import/export of compressed data works.
        let compressed = coder.into_compressed();
        let mut coder = Coder::with_compressed_data(compressed);

        let reconstructed_gaussian = coder
            .pop_symbols(
                means
                    .iter()
                    .zip(&stds)
                    .map(|(&mean, &std)| quantizer.quantize(Normal::new(mean, std).unwrap())),
            )
            .collect::<Vec<_>>();
        let reconstructed_categorical =
            coder.pop_iid_symbols(AMT, &categorical).collect::<Vec<_>>();

        assert!(coder.is_empty());

        assert_eq!(symbols_gaussian, reconstructed_gaussian);
        assert_eq!(symbols_categorical, reconstructed_categorical);
    }
}
