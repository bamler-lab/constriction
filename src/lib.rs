//! Low-level entropy coding utilities using [range Asymmetric Numeral Systems (rANS)]
//!
//! The main entry point to this crate is the type [`Coder`].
//!
//! # TODO
//!
//! Move some documentation from [`Coder`] here.
//!
//! [range Asymmetric Numeral Systems (rANS)]:
//! https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#Range_variants_(rANS)_and_streaming

#![warn(missing_docs, rust_2018_idioms, missing_debug_implementations)]

#[cfg(feature = "pybindings")]
pub mod pybindings;

/// Probability distributions that can be used as entropy models for compression.
///
///
///
/// # Example
///
/// See documentation of [`Coder`] for an example how to use these distributions for
/// data compression or decompression.
///
/// [`Coder`]: crate::Coder
pub mod distributions;

use std::{borrow::Borrow, error::Error, fmt::Debug};

use distributions::DiscreteDistribution;
use num::{
    cast::AsPrimitive,
    traits::{WrappingAdd, WrappingSub},
    CheckedDiv, PrimInt, Unsigned, Zero,
};

/// A trait for the smallest unit of compressed data in a [`Coder`]
///
/// [`Coder`]: struct.Coder.html
pub unsafe trait CompressedWord:
    PrimInt + Unsigned + WrappingSub + WrappingAdd + 'static
{
    /// The type that holds the current head of the [`Coder`].
    ///
    /// Must be twice as large as `Self`, so that [`split_state`] can split it into two
    /// words.
    ///
    /// [`Coder`]: struct.Coder.html
    /// [`split_state`]: #method.split_state
    type State: PrimInt + From<Self> + AsPrimitive<Self>;

    /// Returns the number of compressed bits in a `CompressedWord`.
    ///
    /// Defaults to `8 * std::mem::size_of::<Self>()`, which is suitable for all
    /// primitive unsigned integers.
    ///
    /// This should really be a `const fn`, except that these aren't allowed on trait
    /// methods yet.
    fn bits() -> usize {
        8 * std::mem::size_of::<Self>()
    }

    /// Splits `state` into two `CompressedWord`s and returns `(low, high)`.
    ///
    /// Here, `low` holds the least significant bits and `high` the most significant
    /// bits of `state`. Inverse of [`compose_state`](#method.compose_state).
    ///
    /// # Example
    ///
    /// ```
    /// use ans::CompressedWord;
    ///
    /// let state: u64 = 0x0123_4567_89ab_cdef;
    /// let (low, high) = u32::split_state(state);
    /// assert_eq!(low, 0x89ab_cdef);
    /// assert_eq!(high, 0x0123_4567);
    ///
    /// let reconstructed = u32::compose_state(low, high);
    /// assert_eq!(reconstructed, state);
    /// ```
    fn split_state(state: Self::State) -> (Self, Self) {
        let high = (state >> Self::bits()).as_();
        let low = state.as_();
        (low, high)
    }

    /// Composes a `State` from two compressed words.
    ///
    /// Here, `low` holds the least significant bits and `high` the most significant
    /// bits of the returned `State`. See [`split_state`] for an example.
    ///
    /// [`split_state`]: #method.split_state
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

/// Error type for [`ans::Coder`]
///
/// [`ans::Coder`]: struct.Coder.html
#[non_exhaustive]
#[derive(Debug)]
pub enum CoderError {
    /// Tried to encode a symbol with zero probability under the used entropy model.
    ///
    /// This error can usually be avoided by using a "leaky" distribution, i.e., a
    /// distribution that assigns a nonzero probability to all symbols within a
    /// finite domain. Leaky distributions can be constructed with, e.g., a
    /// [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_floating_point_probabilities`](
    /// distributions/struct.Categorical.html#method.from_floating_point_probabilities).
    ImpossibleSymbol,

    /// The iterator provided to [`Coder::try_push_symbols`] or
    /// [`Coder::try_pop_symbols`] yielded `Err(_)`.
    ///
    /// The variant wraps the original error, which can also be retrieved via
    /// [`source`](#method.source).
    ///
    /// [`Coder::try_push_symbols`]: struct.Coder.html#method.try_push_symbols
    /// [`Coder::try_pop_symbols`]: struct.Coder.html#method.try_pop_symbols
    IterationError(Box<dyn Error + 'static>),
}

impl Error for CoderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::IterationError(ref source) => Some(&**source),
            _ => None,
        }
    }
}

impl std::fmt::Display for CoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Error during entropy coding.")
    }
}

type Result<T> = std::result::Result<T, CoderError>;

/// Entropy coder for both encoding and decoding on a stack
///
/// This is is a very general entropy coder that provides both encoding and
/// decoding. The coder uses an entropy coding algorithm called [range Asymmetric
/// Numeral Systems (rANS)], which means that it operates as a stack, i.e., a "last
/// in first out" data structure: encoding "pushes symbols on" the stack and
/// decoding "pops symbols off" the stack in reverse order. In contrast to
/// [`SeekableDecoder`], decoding with a `Coder` *consumes* the compressed data for
/// the decoded symbols. This means that encoding and decoding can be interleaved
/// arbitrarily, thus growing and shrinking the stack along the way.
///
/// # Example
///
/// The following shows a basic usage example. For more examples, see
/// [`push_symbols`] or [`push_iid_symbols`].
///
/// ```
/// use statrs::distribution::Normal;
///
/// let mut coder = ans::Coder::<u32>::new();
/// let quantizer = ans::distributions::LeakyQuantizer::new(-100..=100);
/// let entropy_model = quantizer.quantize(Normal::new(0.0, 10.0).unwrap());
///
/// let symbols = vec![-10, 4, 0, 3];
/// coder.push_iid_symbols(symbols.iter(), &entropy_model);
/// println!("Encoded into {} bits: {:?}", coder.num_bits(), &*coder.get_compressed());
///
/// // Above, `push_iid_symbols` encoded the symbols in reverse order (see documentation).
/// // So popping them off now will yield the same symbols in non-reversed order.
/// let reconstructed = coder.pop_iid_symbols(4, &entropy_model).collect::<Vec<_>>();
/// assert_eq!(reconstructed, symbols);
/// ```
///
/// # Compressed Word Type `W`
///
/// The `Coder` is generic over a type `W`, which is a [`CompressedWord`]. It is
/// typically a primitive unsigned integer type. **If you're unsure, use `W = u32`
/// as a default choice** unless your entropy models are lookup tables that operate
/// on the entire `W` space (in which case a smaller type like `u16` or even `u8`
/// may be more appropriate to reduce the memory footprint).
///
/// In more detail, the choice of the compressed word type`W` has several
/// consequences:
///
/// - It influences the accuracy with which entropy models used for encoding or
///   decoding can represent probabilities. This accuracy in turn has an influence
///   on:
///   - the runtime and memory efficiency of the entropy models; larger `W` types
///     may lead to more expensive entropy models, but it depends on the type of
///     entropy model.
///   - how well the models can represent the true distribution of the data; small
///     `W` types therefore lead to some *relative* overhead over the theoretically
///     optimal bitrate for entropy coding (i.e., an overhead that is roughly
///     proportional to the number of encoded symbols).
/// - Besides, the choice of `W` influences the *absolute* (i.e., constant) overhead
///   in bitrate over the theoretically possible lowest bitrate for entropy coding.
///   This constant overhead is between one and two words of type `W`. Thus, larger
///   `W` types such as `u32` lead to a slightly larger constant overhead, but the
///   constant overhead is typically negligible unless one is compressing a very
///   short message.
/// - Finally, the choice of `W` may have a minor influence on the run-time
///   efficiency of the coder itself. On the one hand, the coder operates most of
///   the time on a state of twice the size as `W`, suggesting higher cost for large
///   `W` types. On the other hand, very small `W` types will require the coder to
///   flush its state more often. In practice, however, the actual entropy coding
///   part is unlikely to be the computational bottleneck unless one uses entropy
///   models that are heavily optimized for speed (such as lookup tables).
///
/// # Entropy Models and Streaming Entropy Coding
///
/// Entropy coding is an approach to lossless compression that employs a
/// probabilistic model over the encoded data. This so called *entropy model* allows
/// an entropy coding algorithm to assign short codewords to data it will likely
/// see, at the cost of mapping unlikely data to longer codewords. The module
/// [`distributions`] provides tools to construct entropy models that you can use
/// with a `Coder`.
///
/// This `Coder` (as well as [`SeekableDecoder`]) models data as a sequence of
/// "symbols" of arbitrary (possibly even heterogeneous) types. Each symbol is
/// associated with its own entropy model (see [below](#encoding-correlated-data)
/// for ways to model correlations between symbols). However, in contrast to, e.g.,
/// [Huffman  Coding], rANS coding provides streaming entropy coding where the
/// compressed file size is amortized over the entire sequence of symbols. This
/// provides better (lower) bitrates, in particular for entropy models where the
/// [entropy] of each individual symbol is small, possibly below 1 bit. Huffman
/// coding encodes each symbol independently into a bitstring of integer length, and
/// it concatenates these individual bitstrings to the full compressed message. This
/// leads to an overhead since the contribution of each symbol to the length of the
/// compressed message is the symbol's [information content] *rounded up* to the
/// nearest integer number of bits. By contrast, streaming entropy coding with rANS,
/// as implemented in this `Coder`, amortizes over the (typically fractional)
/// information content of all symbols so that the effective contribution of each
/// symbol to the length of the compressed message is closer to the actual
/// information content of the symbol (*not* rounded up to the nearest integer).
///
/// # Consistency Between Encoding and Decoding
///
/// As described above, each symbol can be encoded and decoded with its own entropy
/// model. If your goal is to reconstruct the originally encoded symbols during
/// decoding, then you must employ the same sequence of entropy models (in reversed
/// order) during encoding and decoding. However, this is not required in general.
/// It is perfectly legal to push symbols on the `Coder` using some entropy models,
/// and then pop off symbols using different entropy models. The popped off symbols
/// will then in general be different from the original symbols, but will be
/// generated in a deterministic way. If there is no deterministic relation between
/// the entropy models used for pushing and popping, and if there is still
/// compressed  data left at the end (i.e., if [`is_empty`] returns false), then the
/// popped off symbols are approximately distributed as independent samples from the
/// entropy models provided when popping them off the coder. Such random samples,
/// which consume parts of the compressed data, are useful in the bits-back
/// algorithm.
///
/// # Encoding Correlated Data
///
/// While the `Coder` expects an individual entropy model for each symbol as
/// described above, the caller can still encode correlations between symbols. The
/// most straight-forward way to encode correlations is by choosing the entropy
/// model of each symbol conditionally on other symbols, as described in the example
/// below. Another way to encode correlations is via hierarchical probabilistic
/// models and the bits-back algorithm, which can naturally be implemented on top of
/// a stack-based entropy coder such as the rANS algorithm used by this `Coder`.
///
/// As an example for correlations via conditional entropy models, consider a
/// message that consists of a sequence of three symbols `[s1, s2, s3]`. The full
/// entropy model for this message is a probability distribution `p(s1, s2, s3)`.
/// Such a distribution can, at least in principle, always be factorized as
/// `p(s1, s2, s3) = p1(s1) * p2(s2 | s1) * p3(s3 | s1, s2)`. Here, `p1`, `p2`, and
/// `p3` are entropy models for the individual symbols `s1`, `s2`, and `s3`,
/// respectively. In this notation, the bar "`|`" separates the symbol on the left,
/// which is the one that the given entropy model describes, from the symbols on the
/// right, which one must already know if one  wants to construct the entropy model.
/// During encoding, we know the entire message `[s1, s2, s3]`, and so we can
/// construct all three entropy models `p1`, `p2`, and `p3`. We then use these
/// entropy models to encode ("push on the stack") the symbols `s1`, `s2`, and `s3`
/// *in reverse order* (the method [`push_symbols`] automatically reverses the order
/// of the provided symbols). When decoding the compressed message, we do not know
/// the symbols `s1`, `s2`, and `s3` upfront, so we initially cannot construct `p2`
/// or `p3`. But the entropy model `p1` of the first symbol does not depend on any
/// other symbols, so we can use it to decode ("pop off") the first symbol `s1`.
/// Using this decoded symbol, we can construct the entropy model `p2` and use it to
/// decode ("pop off") the second symbol `s2`. Finally, we use both decoded symbols
/// `s1` and `s2` to construct the entropy model `p3` and we decode `s3`.
///
/// [range Asymmetric Numeral Systems (rANS)]:
/// https://en.wikipedia.org/wiki/Asymmetric_numeral_systems#Range_variants_(rANS)_and_streaming
/// [`push_symbols`]: #method.push_symbols
/// [`push_iid_symbols`]: #method.push_iid_symbols
/// [`distributions`]: distributions/index.html
/// [Huffman Coding]: https://en.wikipedia.org/wiki/Huffman_coding
/// [entropy]: https://en.wikipedia.org/wiki/Entropy_(information_theory)
/// [information content]: https://en.wikipedia.org/wiki/Information_content
/// [`push_symbols`]: #method.push_symbols
/// [`is_empty`]: #method.is_empty
pub struct Coder<W: CompressedWord> {
    buf: Vec<W>,
    state: W::State,
}

impl<W: CompressedWord> Debug for Coder<W>
where
    W: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter_compressed()).finish()
    }
}

impl<W: CompressedWord> Coder<W> {
    /// Creates an empty ANS entropy coder.
    ///
    /// This is usually the starting point if you want to *compress* data.
    ///
    /// # Example
    ///
    /// ```
    /// let mut coder = ans::Coder::<u32>::new();
    ///
    /// // ... push some symbols onto the coder ...
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

    /// Creates ANS coder with some initial compressed data.
    ///
    /// This is usually the starting point if you want to *decompress* data
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
    /// This is a low level method. You probably usually want to call a batch method
    /// like [`push_symbols`](#method.push_symbols) or
    /// [`push_iid_symbols`](#method.push_iid_symbols) instead. See examples there.
    ///
    /// The bound `impl Borrow<S>` on argument `symbol` essentially means that you
    /// can provide the symbol either by value or by reference.
    ///
    /// This method is called `push_symbol` rather than `encode_symbol` to stress
    /// the fact that the `Coder` is a stack: the last symbol *pushed onto* the
    /// stack will be the first symbol that [`pop_symbol`](#method.pop_symbol) will
    /// *pop off* the stack.
    ///
    /// Returns [`Err(ImpossibleSymbol)`] if `symbol` has zero probability under the
    /// entropy model `distribution`. This error can usually be avoided by using a
    /// "leaky" distribution, i.e., a distribution that assigns a nonzero
    /// probability to all symbols within a finite domain. Leaky distributions can
    /// be constructed with, e.g., a
    /// [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_floating_point_probabilities`](
    /// distributions/struct.Categorical.html#method.from_floating_point_probabilities).
    ///
    /// [`Err(ImpossibleSymbol)`]: enum.CoderError.html#variant.ImpossibleSymbol
    pub fn push_symbol<S>(
        &mut self,
        symbol: impl Borrow<S>,
        distribution: impl DiscreteDistribution<W, Symbol = S>,
    ) -> Result<()> {
        let (left_sided_cumulative, probability) = distribution
            .left_cumulative_and_probability(symbol)
            .map_err(|()| CoderError::ImpossibleSymbol)?;
        if self.state >= W::compose_state(W::zero(), probability) {
            let (low, high) = W::split_state(self.state);
            self.buf.push(low);
            self.state = W::State::from(high);
        }
        let prefix = self
            .state
            .checked_div(&W::State::from(probability))
            .ok_or(CoderError::ImpossibleSymbol)?;
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
    /// This method is called `pop_symbol` rather than `decode_symbol` to stress the
    /// fact that the `Coder` is a stack: `pop_symbol` will return the *last* symbol
    /// that was previously encoded via [`push_symbol`](#method.push_symbol).
    ///
    /// Note that this method cannot fail. It will still produce symbols in a
    /// deterministic way even if the coder is empty, but such symbols will not
    /// recover any previously encoded data and will generally have low entropy.
    /// Still, being able to pop off an arbitrary number of symbols can sometimes be
    /// useful in edge cases of, e.g., the bits-back algorithm.
    pub fn pop_symbol<S>(&mut self, distribution: impl DiscreteDistribution<W, Symbol = S>) -> S {
        let (low, high) = W::split_state(self.state);
        let (symbol, left_sided_cumulative, probability) = distribution.quantile_function(low);
        self.state = W::State::from(probability) * W::State::from(high)
            + W::State::from(low - left_sided_cumulative);

        if self.state <= W::max_value().into() {
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
    /// The provided iterator `symbols_and_distributions` must yield pairs of
    /// symbols and their entropy models. The bound `S: Borrow<D::Symbol>`
    /// essentially means that the iterator may yield symbols either by value or by
    /// reference.
    ///
    /// Returns [`Err(ImpossibleSymbol)`] if one of the symbols has zero probability
    /// under its entropy model.  In this case, some of the symbols may have already
    /// been pushed onto the coder. This error can usually be avoided by using
    /// "leaky" distributions, i.e., distributions that assign a nonzero probability
    /// to all symbols within a finite domain. Leaky distributions can be
    /// constructed with, e.g., a  
    /// [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_floating_point_probabilities`](
    /// distributions/struct.Categorical.html#method.from_floating_point_probabilities).
    ///
    /// If the the iteration over symbols and their distributions itself can fail
    /// (e.g., because of an invalid parameterization of a distribution), consider
    /// using [`try_push_symbols`] instead.
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
    /// coder.push_symbols(symbols.iter().zip(entropy_models.clone()));
    ///
    /// // Decode the encoded data.
    /// let reconstructed = coder.pop_symbols(entropy_models).collect::<Vec<_>>();
    ///
    /// assert_eq!(symbols, reconstructed);
    /// assert!(coder.is_empty());
    /// ```
    ///
    /// [`pop_symbols`]: #method.pop_symbols
    /// [`Err(ImpossibleSymbol)`]: enum.CoderError.html#variant.ImpossibleSymbol
    /// [`try_push_symbols`]: #method.try_push_symbols
    pub fn push_symbols<D, S, I>(&mut self, symbols_and_distributions: I) -> Result<()>
    where
        D: DiscreteDistribution<W>,
        S: Borrow<D::Symbol>,
        I: Iterator<Item = (S, D)> + DoubleEndedIterator,
    {
        for (symbol, distribution) in symbols_and_distributions.rev() {
            self.push_symbol(symbol, distribution)?;
        }

        Ok(())
    }

    /// Decodes a sequence of symbols.
    ///
    /// This method is the inverse of [`push_symbols`]. See example there.
    ///
    /// If the iterator `distributions` can fail, consider using [`try_pop_symbols`]
    /// instead.
    ///
    /// # TODO
    ///
    /// Once specialization is stable, return an `impl ExactSizeIterator` if
    /// `distribution` implements `ExactSizeIterator` (same for
    /// [`pop_iid_symbols`](#method.pop_iid_symbols) and for [`try_pop_symbols`]).
    ///
    /// [`push_symbols`]: #method.push_symbols
    /// [`try_pop_symbols`]: #method.try_pop_symbols
    pub fn pop_symbols<'s, S, D: DiscreteDistribution<W, Symbol = S>>(
        &'s mut self,
        distributions: impl Iterator<Item = D> + 's,
    ) -> impl Iterator<Item = S> + 's {
        distributions.map(move |distribution| self.pop_symbol(distribution))
    }

    /// Like [`push_symbols`] but for fallible input.
    ///
    /// Returns:
    /// - `Ok(())` if no error occurred (same as in [`push_symbols`]);
    /// - [`Err(ImpossibleSymbol)`] if one tries to push a symbol that has probability
    ///   zero under its entropy model (same as in [`push_symbols`]);
    /// - [`Err(IterationError(source))`] if `symbols_and_distribution` yields
    ///   `Err(source)` (this is the only difference to [`push_symbols`]).
    ///
    /// In the event of an error, iteration over `symbols_and_distributions`
    /// terminates.
    ///
    /// # Example
    ///
    /// ```
    /// use statrs::{distribution::Normal, StatsError};
    ///
    /// let mut coder = ans::Coder::<u32>::new();
    /// let quantizer = ans::distributions::LeakyQuantizer::new(-10..=10);
    /// let symbols = vec![1, 2, 3];
    /// let means_and_stds = [
    ///     (1.1, 1.0),
    ///     (2.2, -2.0), // <-- Negative standard deviation: this will fail.
    ///     (3.3, 3.0),
    /// ];
    ///
    /// let result = coder.try_push_symbols(symbols.iter().zip(&means_and_stds).map(
    ///     |(&symbol, &(mean, std))| {
    ///         Normal::new(mean, std)
    ///             .map(|distribution| (symbol, quantizer.quantize(distribution)))
    ///     },
    /// ));
    ///
    /// assert!(result.is_err()); // <-- Verify that it did indeed fail.
    /// dbg!(result);
    /// ```
    ///
    /// [`push_symbols`]: #method.push_symbols
    /// [`Err(ImpossibleSymbol)`]: enum.CoderError.html#variant.ImpossibleSymbol
    /// [`Err(IterationError(source))`]: enum.CoderError.html#variant.IterationError
    pub fn try_push_symbols<E, D, S, I>(&mut self, symbols_and_distributions: I) -> Result<()>
    where
        E: Error + 'static,
        D: DiscreteDistribution<W>,
        S: Borrow<D::Symbol>,
        I: Iterator<Item = std::result::Result<(S, D), E>> + DoubleEndedIterator,
    {
        for symbol_and_distribution in symbols_and_distributions.rev() {
            let (symbol, distribution) =
                symbol_and_distribution.map_err(|err| CoderError::IterationError(Box::new(err)))?;
            self.push_symbol(symbol, distribution)?;
        }

        Ok(())
    }

    /// Like [`pop_symbols`] but for fallible input.
    ///
    /// Forwards any errors that occur in the iteration of `distributions` and pops
    /// symbols only for items where `distributions` yields `Ok(_)`. Forwarded
    /// errors `e` will be wrapped as [`CoderError::IterationError(e)`].
    ///
    /// Note that the returned iterator does *not* terminate after yielding its
    /// first error. But you will very likely want to terminate iteration yourself
    /// after the first error: even if `distributions` yields any valid
    /// distributions after the first error, the coder will likely be in an
    /// unexpected (yet technically valid) state since it skipped decoding of the
    /// symbol with the erroneous distribution. A simple way to terminate after the
    /// first error is by `.collect`ing the iterator returned by this method into a
    /// `Result<Vec<_>, _>`, see example below.
    ///
    /// # Example
    ///
    /// ```
    /// use statrs::{distribution::Normal, StatsError};
    ///
    /// let mut coder = ans::Coder::<u32>::new();
    /// let quantizer = ans::distributions::LeakyQuantizer::new(-10..=10);
    /// let symbols = vec![1, 2, 3];
    /// let mut means_and_stds = [(1.1, 1.0), (2.2, 2.0), (3.3, 3.0)];
    ///
    /// coder.try_push_symbols(symbols.iter().zip(&means_and_stds).map(
    ///     |(&symbol, &(mean, std))| {
    ///         Normal::new(mean, std)
    ///             .map(|distribution| (symbol, quantizer.quantize(distribution)))
    ///     },
    /// )).unwrap(); // <-- We don't expect any errors so far.
    ///
    /// means_and_stds[1].1 = -2.0; // <-- Negative standard deviation: this will fail.
    ///
    /// let decoded = coder
    ///     .try_pop_symbols(means_and_stds.iter().map(|&(mean, std)| {
    ///         Normal::new(mean, std).map(|distribution| quantizer.quantize(distribution))
    ///     }))
    ///     .collect::<Result<Vec<_>, _>>(); // <-- Short-circuit the fallible iterator.
    ///
    /// assert!(decoded.is_err()); // <-- Verify that it did indeed fail.
    /// dbg!(decoded);
    /// assert!(!coder.is_empty()); // <-- Therefore, there's still some data left on the coder.
    ///
    /// means_and_stds[1].1 = 2.0; // <-- Fixed it. We should be able to decode the remaining data now.
    ///
    /// let remaining_decoded = coder
    ///     .try_pop_symbols(means_and_stds[1..].iter().map(|&(mean, std)| {
    ///         Normal::new(mean, std).map(|distribution| quantizer.quantize(distribution))
    ///     }))
    ///     .collect::<Result<Vec<_>, _>>();
    ///
    /// assert_eq!(remaining_decoded.unwrap(), &symbols[1..]); // <-- Verify remaining data.
    /// assert!(coder.is_empty());
    /// ```
    ///
    /// [`pop_symbols`]: #method.pop_symbols
    /// [`CoderError::IterationError(e)`]: enum.CoderError.html#variant.IterationError
    pub fn try_pop_symbols<'s, E, D, I>(
        &'s mut self,
        distributions: I,
    ) -> impl Iterator<Item = Result<D::Symbol>> + 's
    where
        E: Error + 'static,
        D: DiscreteDistribution<W>,
        I: Iterator<Item = std::result::Result<D, E>> + 's,
    {
        distributions.map(move |distribution| {
            let distribution =
                distribution.map_err(|err| CoderError::IterationError(Box::new(err)))?;
            Ok(self.pop_symbol(distribution))
        })
    }

    /// Encodes a sequence of symbols using a fixed entropy model.
    ///
    /// This is a convenience wrapper around [`push_symbols`], and the inverse of
    /// [`pop_iid_symbols`].
    ///
    /// The bound `S: Borrow<D::Symbol>` essentially means that the provided
    /// iterator `symbols` may yield symbols either by value or by reference.
    ///
    /// Returns [`Err(ImpossibleSymbol)`] if one of the symbols has zero probability
    /// under its entropy model. In this case, some of the symbols may have already
    /// been pushed on the coder. This error can usually be avoided by using "leaky"
    /// distributions, i.e., distributions that assign a nonzero probability to all
    /// symbols within a finite domain. Leaky distributions can be constructed with,
    /// e.g., a [`LeakyQuantizer`](distributions/struct.LeakyQuantizer.html) or with
    /// [`Categorical::from_floating_point_probabilities`](
    /// distributions/struct.Categorical.html#method.from_floating_point_probabilities).
    ///
    /// # Example
    ///
    /// ```
    /// let mut coder = ans::Coder::<u32>::new();
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution =
    ///     ans::distributions::Categorical::from_floating_point_probabilities(&probabilities).unwrap();
    ///
    /// coder.push_iid_symbols(symbols.iter(), &distribution).unwrap();
    /// let reconstructed = coder.pop_iid_symbols(4, &distribution).collect::<Vec<_>>();
    ///
    /// assert_eq!(symbols, reconstructed);
    /// assert!(coder.is_empty());
    /// ```
    ///
    /// [`push_symbols`]: #method.push_symbols
    /// [`pop_iid_symbols`]: #method.pop_iid_symbols
    /// [`Err(ImpossibleSymbol)`]: enum.CoderError.html#variant.ImpossibleSymbol
    pub fn push_iid_symbols<D, S, I>(&mut self, symbols: I, distribution: &D) -> Result<()>
    where
        D: DiscreteDistribution<W>,
        S: Borrow<D::Symbol>,
        I: Iterator<Item = S> + DoubleEndedIterator,
    {
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
        distribution: &'c impl DiscreteDistribution<W, Symbol = S>,
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
    /// This is a low level method that provides a view into the current compressed
    /// data at zero cost, but in a somewhat inconvenient representation. In most
    /// cases you will likely want to call one of [`get_compressed`],
    /// [`into_compressed`], or [`iter_compressed`] instead, as these methods
    /// provide the the compressed data in more convenient forms (which is also the
    /// form expected by the constructor [`with_compressed_data`]).
    ///
    /// The return value of this method is a tuple `(bulk, head)`, where
    /// `bulk: &[W]` has variable size (and can be empty) and `head: [W; 2]` has a
    /// fixed size. When encoding or decoding data, `head` typically changes with
    /// each encoded or decoded symbol while `bulk` changes only infrequently
    /// (whenever `head` overflows or underflows).
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`with_compressed_data`]: #method.with_compressed_data
    pub fn as_compressed_raw(&self) -> (&[W], [W; 2]) {
        // Return the head as an array rather than a tuple of compressed words to make
        // it more obvious how it should logically be appended to the bulk.
        let (low, high) = W::split_state(self.state);
        (&self.buf, [low, high])
    }

    /// Assembles the current compressed data into a single slice.
    ///
    /// This method is similar to [`as_compressed_raw`] with the difference that it
    /// concatenates the `bulk` and `head` before returning them. The concatenation
    /// truncates any trailing zero words, which is compatible with the constructor
    /// [`with_compressed_data`].
    ///
    /// This method requires a `&mut self` receiver. If you only have a shared
    /// reference to a `Coder`, consider calling [`as_compressed_raw`] or
    /// [`iter_compressed`] instead.
    ///
    /// The returned `CoderGuard` dereferences to `&[W]`, thus providing read-only
    /// access to the compressed data. If you need ownership of the compressed data,
    /// consider calling [`into_compressed`] instead.
    ///
    /// # Example
    ///
    /// ```
    /// let mut coder = ans::Coder::<u32>::new();
    ///
    /// // Push some data on the coder.
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution =
    ///     ans::distributions::Categorical::from_floating_point_probabilities(&probabilities)
    ///         .unwrap();
    /// coder.push_iid_symbols(symbols.iter(), &distribution).unwrap();
    ///
    /// // Inspect the compressed data.
    /// dbg!(coder.get_compressed());
    ///
    /// // We can still use the coder afterwards.
    /// let reconstructed = coder.pop_iid_symbols(4, &distribution).collect::<Vec<_>>();
    /// assert_eq!(reconstructed, symbols);
    /// ```
    ///
    /// [`as_compressed_raw`]: #method.as_compressed_raw
    /// [`with_compressed_data`]: #method.with_compressed_data
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn get_compressed(&mut self) -> CoderGuard<'_, W> {
        CoderGuard::new(self)
    }

    /// Iterates over the compressed data currently on the stack.
    ///
    /// In contrast to [`get_compressed`] or [`into_compressed`], this method does
    /// not require mutable access or even ownership of the `Coder`.
    ///
    /// # Example
    ///
    /// ```
    /// use statrs::distribution::Normal;
    ///
    /// // Create a coder and encode some stuff.
    /// let mut coder = ans::Coder::<u32>::new();
    /// let quantizer = ans::distributions::LeakyQuantizer::new(-100..=100);
    /// let distribution = quantizer.quantize(Normal::new(0.0, 10.0).unwrap());
    /// coder.push_iid_symbols(-100..=100, &distribution);
    ///
    /// // Iterate over compressed data, collect it into to a vector, and compare to more direct method.
    /// let compressed_iter = coder.iter_compressed();
    /// let compressed_collected = compressed_iter.collect::<Vec<_>>();
    /// assert!(!compressed_collected.is_empty());
    /// assert_eq!(compressed_collected, &*coder.get_compressed());
    ///
    /// // We can also iterate in reverse direction, which is useful for streaming decoding.
    /// let compressed_iter_reverse = coder.iter_compressed().rev();
    /// let compressed_collected_reverse = compressed_iter_reverse.collect::<Vec<_>>();
    /// let mut compressed_direct = coder.into_compressed();
    /// assert!(!compressed_collected_reverse.is_empty());
    /// assert_ne!(compressed_collected_reverse, compressed_direct);
    /// compressed_direct.reverse();
    /// assert_eq!(compressed_collected_reverse, compressed_direct);
    /// ```
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    pub fn iter_compressed(
        &self,
    ) -> impl Iterator<Item = W> + ExactSizeIterator + DoubleEndedIterator + '_ {
        CompressedIter::new(self)
    }

    /// Consumes the coder and returns the compressed data.
    ///
    /// The returned data can be used to recreate a coder with the same state
    /// (e.g., for decoding) by passing it to
    /// [`with_compressed_data`](#method.with_compressed_data).
    ///
    /// If you don't want to consume the coder, consider calling
    /// [`get_compressed`](#method.get_compressed),
    /// [`as_compressed_raw`](#method.as_compressed_raw), or
    /// [`iter_compressed`](#method.iter_compressed) instead.
    ///
    /// # Example
    ///
    /// ```
    /// let mut coder = ans::Coder::<u32>::new();
    ///
    /// // Push some data on the coder.
    /// let symbols = vec![8, 2, 0, 7];
    /// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
    /// let distribution =
    ///     ans::distributions::Categorical::from_floating_point_probabilities(&probabilities)
    ///         .unwrap();
    /// coder.push_iid_symbols(symbols.iter(), &distribution).unwrap();
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
        Self::state_len(self.state, |w| self.buf.push(w));
        self.buf
    }

    /// Returns the number of compressed words on the stack.
    ///
    /// This includes a constant overhead of between one and two words unless the
    /// coder is completely empty.
    ///
    /// This method returns the length of the slice, the `Vec<W>`, or the iterator
    /// that would be returned by [`get_compressed`], [`into_compressed`], or
    /// [`iter_compressed`], respectively, when called at this time.
    ///
    /// See also [`num_bits`].
    ///
    /// [`get_compressed`]: #method.get_compressed
    /// [`into_compressed`]: #method.into_compressed
    /// [`iter_compressed`]: #method.iter_compressed
    /// [`num_bits`]: #method.num_bits
    pub fn num_words(&self) -> usize {
        self.buf.len() + Self::state_len(self.state, |_| ())
    }

    /// Returns the size of the current stack of compressed data in bits.
    ///
    /// This includes some constant overhead unless the coder is completely empty
    /// (see [`num_words`](#method.num_words)).
    ///
    /// The returned value is a multiple of the bitlength of the compressed word
    /// type `W`.
    pub fn num_bits(&self) -> usize {
        W::bits() * self.num_words()
    }

    fn state_len(state: W::State, mut append_word: impl FnMut(W)) -> usize {
        let (low, high) = W::split_state(state);
        if high == W::zero() {
            if low == W::zero() {
                0
            } else {
                append_word(low);
                1
            }
        } else {
            append_word(low);
            append_word(high);
            2
        }
    }
}

/// Provides temporary read-only access to the compressed data wrapped in a
/// [`Coder`].
///
/// Dereferences to `&[W]`. See [`Coder::get_compressed`] for an example.
///
/// [`Coder`]: struct.Coder.html
/// [`Coder::get_compressed`]: struct.Coder.html#method.get_compressed
pub struct CoderGuard<'a, W: CompressedWord> {
    inner: &'a mut Coder<W>,
}

impl<W: CompressedWord> Debug for CoderGuard<'_, W>
where
    W: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&**self, f)
    }
}

impl<'a, W: CompressedWord> CoderGuard<'a, W> {
    fn new(coder: &'a mut Coder<W>) -> Self {
        // Append state. Will be undone in `<Self as Drop>::drop`.
        Coder::<W>::state_len(coder.state, |w| coder.buf.push(w));
        Self { inner: coder }
    }
}

impl<'a, W: CompressedWord> Drop for CoderGuard<'a, W> {
    fn drop(&mut self) {
        // Revert what we did in `Self::new`.
        Coder::<W>::state_len(self.inner.state, |_| {
            self.inner.buf.pop();
        });
    }
}

impl<'a, W: CompressedWord> std::ops::Deref for CoderGuard<'a, W> {
    type Target = [W];

    fn deref(&self) -> &Self::Target {
        &self.inner.buf
    }
}

struct CompressedIter<'a, W> {
    buf: &'a [W],
    head: [W; 2],
    index_front: usize,
    index_back: usize,
}

impl<'a, W: CompressedWord> CompressedIter<'a, W> {
    fn new(coder: &'a Coder<W>) -> Self {
        let (buf, head) = coder.as_compressed_raw();

        // This cannot overflow because `state_len` is at most 2 and even if `W` is `u8`
        // then `buf.len() + 2` cannot overflow the entire address space because there
        // are definitely 2 bytes worth of some other data floating around somewhere
        // (e.g., to hold the `buf` pointer itself as well as `head`, `index_front`, and
        // `index_back`).
        let len = buf.len() + Coder::<W>::state_len(coder.state, |_| ());

        Self {
            buf,
            head,
            index_front: 0,
            index_back: len,
        }
    }
}

impl<W: CompressedWord> Iterator for CompressedIter<'_, W> {
    type Item = W;

    fn next(&mut self) -> Option<Self::Item> {
        let index_front = self.index_front;
        if index_front >= self.index_back {
            None
        } else {
            self.index_front += 1;
            let result = *self.buf.get(index_front).unwrap_or_else(|| {
                // SAFETY:
                // - `index_front >= self.buf.len()` because the above `get` failed.
                // - `index_front - self.buf.len() < 2` because we checked above that
                //   `index_front < self.index_back` and we maintain the invariant
                //   `self.index_back <= self.buf.len() + 2`.
                unsafe { self.head.get_unchecked(index_front - self.buf.len()) }
            });
            Some(result)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.index_back - self.index_front;
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.index_back - self.index_front
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.index_front = self.index_front.saturating_add(n);
        self.next()
    }
}

impl<W: CompressedWord> ExactSizeIterator for CompressedIter<'_, W> {}

impl<W: CompressedWord> DoubleEndedIterator for CompressedIter<'_, W> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index_front >= self.index_back {
            None
        } else {
            // We can subtract one because `self.index_back > self.index_front >= 0`.
            self.index_back -= 1;
            let result = *self.buf.get(self.index_back).unwrap_or_else(|| {
                // SAFETY:
                // - `self.index_back >= self.buf.len()` because the above `get` failed.
                // - `self.index_back - self.buf.len() < 2` because we maintain the
                //   invariant `self.index_back <= self.buf.len() + 2`, and we just
                //   decreased `self.index_back`.
                unsafe { self.head.get_unchecked(self.index_back - self.buf.len()) }
            });
            Some(result)
        }
    }

    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.index_back = self.index_back.saturating_sub(n);
        self.next_back()
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
            Categorical::from_floating_point_probabilities(&categorical_probabilities).unwrap();
        let mut symbols_categorical = Vec::with_capacity(AMT);
        for _ in 0..AMT {
            let quantile = rng.next_u32();
            let symbol = categorical.quantile_function(quantile).0;
            symbols_categorical.push(symbol);
        }

        let mut coder = Coder::new();

        coder
            .push_iid_symbols(symbols_categorical.iter(), &categorical)
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
