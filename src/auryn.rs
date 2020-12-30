use std::{borrow::Borrow, error::Error};

use num::cast::AsPrimitive;

use crate::{
    distributions::DiscreteDistribution, stack::Stack, BitArray, Code, Decode, Encode,
    EncodingError, TryCodingError,
};

/// # Origin of the Name "Auryn"
///
/// AURYN is a medallion in Michael Ende's novel "The Neverending Story". It is
/// described as two serpents that bite each other's tails. The name therefore keeps
/// with constriction's snake theme while at the same time serving as a metaphor for
/// the two buffers of compressed data, where encoding and decoding transfers data
/// from one buffer to the other (just like two serpents that "eat up" each other).
///
/// In the book, the two serpents represent the two realms of reality and fantasy.
/// If worn by a person from the realm of reality, AURYN grants the bearer all
/// whishes in the realm of fantasy; but with every whish granted in the realm of
/// fantasy, AURYN takes away some of its bearer's memories from the realm of
/// reality. Similarly, the `Auryn` data structure allows decoding binary data with
/// arbitrary entropy models, i.e., even with entropy models that are unrelated to
/// the origin of the binary data. This may be used in bits-back like algorithms to
/// "make up" ("fantasize") a sequence of symbols; each fantasized symbol takes away
/// a fixed number of bits from the original ("real") binary data.
#[derive(Debug)]
pub struct Auryn<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    /// The supply of bits.
    supply: Stack<CompressedWord, State>,

    /// Remaining information not used up by decoded symbols.
    waste: Stack<CompressedWord, State>,
}

/// Type alias for an [`Auryn`] with sane parameters for typical use cases.
///
/// This type alias sets the generic type arguments `CompressedWord` and `State` to
/// sane values for many typical use cases.
pub type DefaultAuryn = Auryn<u32, u64>;

impl<CompressedWord, State> From<Stack<CompressedWord, State>> for Auryn<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn from(stack: Stack<CompressedWord, State>) -> Self {
        Auryn::with_supply(stack)
    }
}

impl<CompressedWord, State> Auryn<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    pub fn with_supply(supply: Stack<CompressedWord, State>) -> Self {
        Self {
            supply,
            waste: Default::default(),
        }
    }

    pub fn with_supply_and_waste(
        supply: Stack<CompressedWord, State>,
        waste: Stack<CompressedWord, State>,
    ) -> Self {
        Self { supply, waste }
    }

    pub fn with_compressed_data(compressed: Vec<CompressedWord>) -> Self {
        Self::with_supply(Stack::with_compressed_data(compressed))
    }

    pub fn supply(&self) -> &Stack<CompressedWord, State> {
        &self.supply
    }

    pub fn supply_mut(&mut self) -> &mut Stack<CompressedWord, State> {
        &mut self.supply
    }

    pub fn waste(&self) -> &Stack<CompressedWord, State> {
        &self.waste
    }

    pub fn waste_mut(&mut self) -> &mut Stack<CompressedWord, State> {
        &mut self.waste
    }

    pub fn into_supply_and_waste(
        self,
    ) -> (Stack<CompressedWord, State>, Stack<CompressedWord, State>) {
        (self.supply, self.waste)
    }

    pub fn encode_symbols_reverse<D, S, I>(
        &mut self,
        symbols_and_distributions: I,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
        S: Borrow<D::Symbol>,
        I: IntoIterator<Item = (S, D)>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_symbols(symbols_and_distributions.into_iter().rev())
    }

    pub fn try_encode_symbols_reverse<E, D, S, I>(
        &mut self,
        symbols_and_distributions: I,
    ) -> Result<(), TryCodingError<EncodingError, E>>
    where
        E: Error + 'static,
        D: DiscreteDistribution,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
        S: Borrow<D::Symbol>,
        I: IntoIterator<Item = std::result::Result<(S, D), E>>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.try_encode_symbols(symbols_and_distributions.into_iter().rev())
    }

    pub fn encode_iid_symbols_reverse<D, S, I>(
        &mut self,
        symbols: I,
        distribution: &D,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = S>,
        S: Borrow<D::Symbol>,
        I::IntoIter: DoubleEndedIterator,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_iid_symbols(symbols.into_iter().rev(), distribution)
    }
}

impl<CompressedWord, State> Code for Auryn<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type CompressedWord = CompressedWord;

    type State = (State, State);

    fn state(&self) -> Self::State {
        (self.supply.state(), self.waste.state())
    }
}

impl<CompressedWord, State> Decode for Auryn<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type DecodingError = std::convert::Infallible;

    fn decode_symbol<D>(&mut self, distribution: D) -> Result<D::Symbol, Self::DecodingError>
    where
        D: DiscreteDistribution,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        let quantile = self.supply.chop_quantile_off_state::<D>();
        self.supply.refill_state_if_possible();

        let (symbol, left_sided_cumulative, probability) = distribution.quantile_function(quantile);
        let remainder = quantile - left_sided_cumulative;

        if self.waste.state()
            > (State::max_value() - remainder.into().into()) / probability.into().into()
        {
            self.waste.flush_state::<D>();
            // At this point, the invariant on `self.waste.state` (see its doc comment) is
            // temporarily violated, but it will be restored below.
        }
        self.waste
            .encode_remainder_onto_state(remainder, probability);

        Ok(symbol)
    }

    fn maybe_finished(&self) -> bool {
        self.supply.maybe_finished()
    }
}

impl<CompressedWord, State> Encode for Auryn<CompressedWord, State>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn encode_symbol<S, D>(
        &mut self,
        symbol: impl Borrow<S>,
        distribution: D,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution<Symbol = S>,
        D::Probability: Into<Self::CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
    {
        let (left_sided_cumulative, probability) = distribution
            .left_cumulative_and_probability(symbol)
            .map_err(|()| EncodingError::ImpossibleSymbol)?;

        let remainder = self.waste.decode_remainder_off_state(probability)?;
        // We have to refill here if the decoder flushed here; the decoder flushed iff
        // waste.state would have been >= 1<<(S-P); in this case, waste.state is now
        // < 1<<(S-W) and >= 1<<(S-P-W).
        self.waste.refill_state_if_possible();

        if (self.supply.state() >> (State::BITS - D::PRECISION)) != State::zero() {
            self.supply.flush_state::<D>();
        }
        self.supply
            .append_quantile_to_state::<D>(left_sided_cumulative + remainder);

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::distributions::{Categorical, DiscreteDistribution, LeakyQuantizer};

    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };
    use statrs::distribution::{InverseCDF, Normal};

    #[test]
    fn compress_none() {
        let auryn1 = DefaultAuryn::with_compressed_data(Vec::new());
        assert!(auryn1.maybe_finished());
        let (supply, mut waste) = auryn1.into_supply_and_waste();
        assert!(supply.is_empty());
        assert!(waste.is_empty());

        let auryn2 = DefaultAuryn::with_supply_and_waste(supply, waste);
        assert!(auryn2.maybe_finished());
    }
    #[test]
    fn restore_none() {
        generic_restore_many::<u32, u64, u32, 24>(3, 0);
    }

    #[test]
    fn restore_one() {
        generic_restore_many::<u32, u64, u32, 24>(3, 1);
    }

    #[test]
    fn restore_two() {
        generic_restore_many::<u32, u64, u32, 24>(3, 2);
    }

    #[test]
    fn restore_ten() {
        generic_restore_many::<u32, u64, u32, 24>(20, 10);
    }

    #[test]
    fn restore_twenty() {
        generic_restore_many::<u32, u64, u32, 24>(18, 20);
    }

    #[test]
    fn restore_many_u32_u64_32() {
        generic_restore_many::<u32, u64, u32, 32>(1024, 1000);
    }

    #[test]
    fn restore_many_u32_u64_24() {
        generic_restore_many::<u32, u64, u32, 24>(1024, 1000);
    }

    #[test]
    fn restore_many_u32_u64_16() {
        generic_restore_many::<u32, u64, u16, 16>(1024, 1000);
    }

    #[test]
    fn restore_many_u16_u64_16() {
        generic_restore_many::<u16, u64, u16, 16>(1024, 1000);
    }

    #[test]
    fn restore_many_u32_u64_8() {
        generic_restore_many::<u32, u64, u8, 8>(1024, 1000);
    }

    #[test]
    fn restore_many_u16_u64_8() {
        generic_restore_many::<u16, u64, u8, 8>(1024, 1000);
    }

    #[test]
    fn restore_many_u8_u64_8() {
        generic_restore_many::<u8, u64, u8, 8>(1024, 1000);
    }

    #[test]
    fn restore_many_u16_u32_16() {
        generic_restore_many::<u16, u32, u16, 16>(1024, 1000);
    }

    #[test]
    fn restore_many_u16_u32_8() {
        generic_restore_many::<u16, u32, u8, 8>(1024, 1000);
    }

    #[test]
    fn restore_many_u8_u32_8() {
        generic_restore_many::<u8, u32, u8, 8>(1024, 1000);
    }

    fn generic_restore_many<CompressedWord, State, Probability, const PRECISION: usize>(
        amt_compressed_words: usize,
        amt_symbols: usize,
    ) where
        State: BitArray + AsPrimitive<CompressedWord>,
        CompressedWord: BitArray + Into<State> + AsPrimitive<Probability>,
        Probability: BitArray + Into<CompressedWord> + AsPrimitive<usize> + Into<f64>,
        u64: AsPrimitive<CompressedWord>,
        u32: AsPrimitive<Probability>,
        usize: AsPrimitive<Probability>,
        f64: AsPrimitive<Probability>,
        i32: AsPrimitive<Probability>,
    {
        let mut rng = Xoshiro256StarStar::seed_from_u64(
            (amt_compressed_words as u64).rotate_left(32) ^ amt_symbols as u64,
        );
        let mut compressed = (0..amt_compressed_words)
            .map(|_| rng.next_u64().as_())
            .collect::<Vec<_>>();

        // Set highest bit so that invariant of a `Stack` is satisfied.
        compressed
            .last_mut()
            .map(|w| *w = *w | (CompressedWord::one() << (CompressedWord::BITS - 1)));

        let distributions = (0..amt_symbols)
            .map(|_| {
                let mean = (200.0 / u32::MAX as f64) * rng.next_u32() as f64 - 100.0;
                let std_dev = (10.0 / u32::MAX as f64) * rng.next_u32() as f64 + 0.001;
                Normal::new(mean, std_dev)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-127..=127);

        let mut auryn = Auryn::<CompressedWord, State>::with_compressed_data(compressed.clone());
        assert!(auryn.waste().is_empty());

        let symbols = auryn
            .decode_symbols(
                distributions
                    .iter()
                    .map(|&distribution| quantizer.quantize(distribution)),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(!auryn.maybe_finished());
        if amt_symbols != 0 {
            assert!(!auryn.waste().is_empty());
        }

        auryn
            .encode_symbols_reverse(
                symbols
                    .iter()
                    .zip(distributions)
                    .map(|(&symbol, distribution)| (symbol, quantizer.quantize(distribution))),
            )
            .unwrap();

        let (supply, waste) = auryn.into_supply_and_waste();
        assert!(waste.is_empty());
        assert_eq!(supply.into_compressed(), compressed);
    }
}
