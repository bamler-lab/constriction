use std::{
    borrow::Borrow,
    error::Error,
    ops::{Deref, DerefMut},
};

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
#[derive(Debug, Clone)]
pub struct Auryn<CompressedWord, State, const PRECISION: usize>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    /// The supply of bits.
    ///
    /// Satisfies the normal invariant of a `Stack`.
    supply: Stack<CompressedWord, State>,

    /// Remaining information not used up by decoded symbols.
    ///
    /// Satisfies different invariants than a usual `Stack`:
    /// - `waste.state() >= State::one() << (State::BITS - PRECISION - CompressedWord::BITS)`
    ///   unless `waste.buf().is_empty()`; and
    /// - `waste.state() < State::one() << (State::BITS - PRECISION)`
    waste: Stack<CompressedWord, State>,
}

/// Type alias for an [`Auryn`] with sane parameters for typical use cases.
///
/// This type alias sets the generic type arguments `CompressedWord` and `State` to
/// sane values for many typical use cases.
pub type DefaultAuryn = Auryn<u32, u64, 24>;

impl<CompressedWord, State, const PRECISION: usize> Auryn<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    pub fn from_stack_for_decoding(
        stack: Stack<CompressedWord, State>,
    ) -> Result<Self, Stack<CompressedWord, State>> {
        assert!(CompressedWord::BITS > 0);
        assert!(State::BITS >= 2 * CompressedWord::BITS);
        assert!(State::BITS % CompressedWord::BITS == 0);
        assert!(PRECISION <= CompressedWord::BITS);
        assert!(PRECISION > 0);

        if stack.state() < State::one() << (State::BITS - CompressedWord::BITS) {
            // Not enough data to initialize `supply`.
            return Err(stack);
        }

        let mut waste_state = State::one();
        let mut word_iter = stack.buf().iter().rev();
        while waste_state < State::one() << (State::BITS - CompressedWord::BITS - PRECISION) {
            if let Some(&word) = word_iter.next() {
                waste_state = (waste_state << CompressedWord::BITS) | word.into();
            } else {
                // Not enough data to initialize `waste`.
                return Err(stack);
            }
        }

        let remaining_words = word_iter.len();
        let (mut buf, state) = stack.into_buf_and_state();
        buf.resize(remaining_words, CompressedWord::zero());

        let supply = Stack::with_buf_and_state(buf, state)
            .expect("Original `Stack` was valid and we only shrank its `buf`.");
        let waste = Stack::with_state_and_empty_buf(waste_state);

        Ok(Self { supply, waste })

        // let (mut buf, mut waste_state) = stack.into_buf_and_state();
        // let mut supply_state = State::one();

        // let needs_transfer = waste_state >= State::one() << (State::BITS - PRECISION);

        // let threshold =
        //     State::one() << (State::BITS - CompressedWord::BITS * (1 + needs_transfer as usize));
        // while supply_state < threshold {
        //     if let Some(word) = buf.pop() {
        //         supply_state = (supply_state << CompressedWord::BITS) | word.into();
        //     } else {
        //         break;
        //     }
        // }

        // if needs_transfer {
        //     // Transfer one word from `waste_state` to `supply_state`.
        //     supply_state = (supply_state << CompressedWord::BITS) | waste_state.as_().into();
        //     waste_state = waste_state >> CompressedWord::BITS;
        // }

        // let supply = Stack::with_buf_and_state(buf, supply_state)
        //     .expect("We enforced validity of `supply` manually above.");
        // let waste = Stack::with_state_and_empty_buf(waste_state);

        // Self { supply, waste }
    }

    pub fn from_stack_for_encoding(
        stack: Stack<CompressedWord, State>,
    ) -> Result<Self, Stack<CompressedWord, State>> {
        assert!(CompressedWord::BITS > 0);
        assert!(State::BITS >= 2 * CompressedWord::BITS);
        assert!(State::BITS % CompressedWord::BITS == 0);
        assert!(PRECISION <= CompressedWord::BITS);
        assert!(PRECISION > 0);

        let (buf, state) = stack.into_buf_and_state();

        let waste = Stack::from_compressed(buf)
            .and_then(|waste| {
                if waste.state() < State::one() << (State::BITS - PRECISION - CompressedWord::BITS)
                {
                    // Not enough data available to initialize `waste`.
                    Err(waste.into_compressed())
                } else {
                    Ok(waste)
                }
            })
            .map_err(|buf| {
                Stack::with_buf_and_state(buf, state)
                    .expect("We're reconstructing the original stack, which was valid.")
            })?;

        let supply = Stack::with_state_and_empty_buf(state);

        Ok(Self::with_supply_and_waste(supply, waste))
    }

    pub fn finish_decoding(
        self,
    ) -> Result<(Vec<CompressedWord>, Stack<CompressedWord, State>), Self> {
        if self.supply.state() < State::one() << (State::BITS - CompressedWord::BITS) {
            // Ran out of binary data to decode. Resulting `Stack` might have invalid state
            // (possibly only after concatenation to `prefix`).
            return Err(self);
        }

        if self.waste.state() < State::one() << (State::BITS - PRECISION - CompressedWord::BITS) {
            // Produced too little waste. Concatenation of `waste.state` to existing data
            // wouldn't be reversible. (This cannot happen if `from_stack_for_decoding` was
            // called on a stack with at least one `State` worth of `buf`, since it tries
            // to initialize the `Auryn` with enough `waste` to start with).
            return Err(self);
        }

        let (prefix, supply_state) = self.supply.into_buf_and_state();
        let suffix = Stack::with_buf_and_state(self.waste.into_compressed(), supply_state)
            .expect("We explicitly verified validity of `supply_state` above.");

        Ok((prefix, suffix))
    }

    pub fn finish_decoding_and_concatenate(self) -> Result<Stack<CompressedWord, State>, Self> {
        let (mut prefix, suffix) = self.finish_decoding()?;
        let (suffix_buf, state) = suffix.into_buf_and_state();
        prefix.extend_from_slice(&suffix_buf);
        let stack = Stack::with_buf_and_state(prefix, state)
            .expect("`finish_decoding` would have failed if `state` was too small.");
        Ok(stack)
    }

    pub fn finish_encoding(
        self,
    ) -> Result<(Vec<CompressedWord>, Stack<CompressedWord, State>), Self> {
        if self.supply.state() < State::one() << (State::BITS - CompressedWord::BITS) {
            // Encoding ended with fewer bits than it started.
            return Err(self);
        }

        if self.waste.state() < State::one() << (State::BITS - CompressedWord::BITS - PRECISION) {
            // Encoding consumed more waste than available.
            return Err(self);
        }

        let waste_state_bits =
            (State::BITS - 1).wrapping_sub(self.waste.state().leading_zeros() as usize);
        if waste_state_bits % CompressedWord::BITS != 0 || waste_state_bits == usize::max_value() {
            // Waste's state (without leading 1 bit) must fit into an integer number of words.
            return Err(self);
        }

        let (mut buf, state) = self.supply.into_buf_and_state();
        let (prefix, mut waste_state) = self.waste.into_buf_and_state();

        while waste_state != State::one() {
            buf.push(waste_state.as_());
            waste_state = waste_state >> CompressedWord::BITS;
        }

        let suffix = Stack::with_buf_and_state(buf, state)
            .expect("We explicitly verified that `self.supply.state()` was large enough.");

        Ok((prefix, suffix))
    }

    pub fn finish_encoding_and_concatenate(self) -> Result<Stack<CompressedWord, State>, Self> {
        let (mut prefix, suffix) = self.finish_encoding()?;
        let (suffix_buf, state) = suffix.into_buf_and_state();
        prefix.extend_from_slice(&suffix_buf);
        let stack = Stack::with_buf_and_state(prefix, state)
            .expect("`finish_encoding` would have failed if `state` was too small.");
        Ok(stack)
    }

    pub fn with_supply_and_waste(
        supply: Stack<CompressedWord, State>,
        mut waste: Stack<CompressedWord, State>,
    ) -> Self {
        assert!(PRECISION <= CompressedWord::BITS);
        assert!(PRECISION > 0);

        // `waste` has to satisfy slightly different invariants than a usual `Stack`.
        // If they're violated then flushing one word is guaranteed to restore them.
        if waste.state() >= State::one() << (State::BITS - PRECISION) {
            waste.flush_state();
            // Now, `waste` satisfies both required invariants:
            // - waste.state() >= State::one() << (State::BITS - PRECISION - CompressedWord::BITS)
            // - waste.state() < State::one() << (State::BITS - CompressedWord::BITS)
            //                 <= State::one() << (State::BITS - PRECISION)
        }

        Self { supply, waste }
    }

    /// Converts the `Auryn` into a new `Auryn` for coding with a different precision.
    ///
    /// Here, "precision" refers to the number of bits with which probabilities are
    /// represented in entropy models passed to the `encode_XXX` and `decode_XXX`
    /// methods.
    ///
    /// The generic argument `NEW_PRECISION` can usually be omitted because the compiler
    /// can infer its value from the first time the new `Auryn` is used for encoding or
    /// decoding. The recommended usage pattern is to store the returned `Auryn` in a
    /// variable that shadows the old `Auryn` (since the old one gets consumed anyway),
    /// i.e., `let mut auryn = auryn.change_precision()`. See example below.
    ///
    /// # Example
    ///
    /// ```
    /// use constriction::{distributions::LeakyQuantizer, Decode, stack::DefaultStack};
    ///
    /// // Construct two entropy models with 24 and 20 bits of precision, respectively.
    /// let continuous_distribution = statrs::distribution::Normal::new(0.0, 10.0).unwrap();
    /// let quantizer24 = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let distribution24 = quantizer24.quantize(continuous_distribution);
    /// let quantizer20 = LeakyQuantizer::<_, _, u32, 20>::new(-100..=100);
    /// let distribution20 = quantizer20.quantize(continuous_distribution);
    ///
    /// // Construct an `Auryn` and decode data with the 24 bit precision entropy model.
    /// let data = vec![0x0123_4567u32, 0x89ab_cdef];
    /// let mut auryn = DefaultStack::from_binary(data).into_auryn_for_decoding().unwrap();
    /// let _symbol_a = auryn.decode_symbol(distribution24);
    ///
    /// // Change the `Auryn`'s precision and decode data with the 20 bit precision entropy model.
    /// let mut auryn = auryn.change_precision(); // <-- Compiler can infer `NEW_PRECISION`.
    /// let _symbol_b = auryn.decode_symbol(distribution20);
    /// ```
    pub fn change_precision<const NEW_PRECISION: usize>(
        mut self,
    ) -> Auryn<CompressedWord, State, NEW_PRECISION> {
        assert!(NEW_PRECISION <= CompressedWord::BITS);

        if NEW_PRECISION > PRECISION {
            if self.waste.state() >= State::one() << (State::BITS - NEW_PRECISION) {
                self.waste.flush_state()
            }
        } else if NEW_PRECISION < PRECISION {
            if self.waste.state()
                < State::one() << (State::BITS - NEW_PRECISION - CompressedWord::BITS)
            {
                self.waste.refill_state_unchecked()
            }
        }

        Auryn {
            supply: self.supply,
            waste: self.waste,
        }
    }

    pub fn supply(&self) -> &Stack<CompressedWord, State> {
        &self.supply
    }

    pub fn supply_mut(&mut self) -> &mut Stack<CompressedWord, State> {
        &mut self.supply
    }

    /// TODO: document that this may violate constraints (but that's OK because
    /// the caller only gets a shared reference, and all methods that rely on the
    /// invariant of `Stack` require mutable access).
    ///
    ///  If you have mutable access to the `Auryn`
    /// then calling `auryn.waste_mut().get_compressed()` may be more efficient. If you
    /// have ownership of the `Auryn` and you won't need it
    ///
    /// # Example
    ///
    /// This method is mainly useful to read out `waste`s state, e.g., by calling
    /// `auryn.waste().iter_compressed()`. However, if you have mutable access to or
    /// even ownership of the `Auryn`, then it may be better to call
    /// [`waste_mut`] or [`into_supply_and_waste`], respectively, followed by
    /// [`Stack::get_compressed`] as in the example below:
    ///
    /// ```
    /// use constriction::{distributions::LeakyQuantizer, stack::DefaultStack, Decode};
    ///
    /// let data = vec![0x0123_4567u32, 0x89ab_cdef];
    /// let mut auryn = DefaultStack::from_binary(data).into_auryn_for_encoding::<24>().unwrap();
    ///
    /// // Calling `auryn.waste()` only needs shared access to `auryn`.
    /// dbg!(auryn.waste()); // `Debug` implementation calls `auryn.waste().iter_compressed()`.
    ///
    /// // Since we have mutable access to `auryn`, the following is also possible and
    /// // might be slightly more efficient in expectation:
    /// dbg!(auryn.waste_mut().get_compressed()); // Prints the same compressed words as above.
    ///
    /// // If we no longer want to use `auryn` then we can also deconstruct it into its constituents:
    /// let (_supply, mut waste) = auryn.into_supply_and_waste();
    /// dbg!(waste.get_compressed()); // Prints the same compressed words as above.
    /// ```
    ///
    /// [`waste_mut`]: #method.waste_mut
    /// [`into_supply_and_waste`]: #method.into_supply_and_waste
    /// [`Stack::get_compressed`]: Stack::get_compressed
    pub fn waste(&self) -> &Stack<CompressedWord, State> {
        &self.waste
    }

    pub fn waste_mut<'a>(
        &'a mut self,
    ) -> impl DerefMut<Target = Stack<CompressedWord, State>> + Drop + 'a {
        WasteGuard::<'a, _, _, PRECISION>::new(&mut self.waste)
    }

    pub fn into_supply_and_waste(
        mut self,
    ) -> (Stack<CompressedWord, State>, Stack<CompressedWord, State>) {
        // `self.waste` satisfies slightly different invariants than a usual `Stack`.
        // We therefore first restore the usual `Stack` invariant.
        self.waste.refill_state_if_possible();

        (self.supply, self.waste)
    }

    pub fn encode_symbols_reverse<S, D, I>(
        &mut self,
        symbols_and_distributions: I,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = (S, D)>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_symbols(symbols_and_distributions.into_iter().rev())
    }

    pub fn try_encode_symbols_reverse<S, D, E, I>(
        &mut self,
        symbols_and_distributions: I,
    ) -> Result<(), TryCodingError<EncodingError, E>>
    where
        S: Borrow<D::Symbol>,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
        E: Error + 'static,
        I: IntoIterator<Item = std::result::Result<(S, D), E>>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.try_encode_symbols(symbols_and_distributions.into_iter().rev())
    }

    pub fn encode_iid_symbols_reverse<S, D, I>(
        &mut self,
        symbols: I,
        distribution: &D,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
        I: IntoIterator<Item = S>,
        I::IntoIter: DoubleEndedIterator,
    {
        self.encode_iid_symbols(symbols.into_iter().rev(), distribution)
    }
}

impl<CompressedWord, State, const PRECISION: usize> Code for Auryn<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type CompressedWord = CompressedWord;

    type State = (State, State);

    fn state(&self) -> Self::State {
        (self.supply.state(), self.waste.state())
    }

    fn maybe_empty(&self) -> bool {
        // `self.supply.state()` must always be above threshold if we still want to call
        // `finish_decoding`, so we only check if `supply.buf` is empty here.
        self.supply.buf().is_empty()
    }
}

impl<CompressedWord, State, const PRECISION: usize> Decode<PRECISION>
    for Auryn<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type DecodingError = std::convert::Infallible;

    fn decode_symbol<D>(&mut self, distribution: D) -> Result<D::Symbol, Self::DecodingError>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        let quantile = self.supply.chop_quantile_off_state::<D, PRECISION>();
        self.supply.refill_state_if_possible();

        let (symbol, left_sided_cumulative, probability) = distribution.quantile_function(quantile);
        let remainder = quantile - left_sided_cumulative;
        dbg!(probability, remainder);

        self.waste
            .encode_remainder_onto_state::<D, PRECISION>(remainder, probability);

        if self.waste.state() >= State::one() << (State::BITS - PRECISION) {
            // The invariant on `self.waste.state` (see its doc comment) is violated and must
            // be restored:
            self.waste.flush_state();
        }

        Ok(symbol)
    }
}

impl<CompressedWord, State, const PRECISION: usize> Encode<PRECISION>
    for Auryn<CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        distribution: D,
    ) -> Result<(), EncodingError>
    where
        D: DiscreteDistribution<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        CompressedWord: AsPrimitive<D::Probability>,
    {
        let (left_sided_cumulative, probability) = distribution
            .left_cumulative_and_probability(symbol)
            .map_err(|()| EncodingError::ImpossibleSymbol)?;

        if self.waste.state()
            < probability.into().into() << (State::BITS - CompressedWord::BITS - PRECISION)
        {
            self.waste.refill_state_unchecked();
            // At this point, the invariant on `self.waste` (see its doc comment) is
            // temporarily violated (but will be restored below). This is how `decode_symbol`
            // can detect that it has to flush `waste.state`.
        }

        let remainder = self
            .waste
            .decode_remainder_off_state::<D, PRECISION>(probability)?;

        if (self.supply.state() >> (State::BITS - PRECISION)) != State::zero() {
            self.supply.flush_state();
        }
        self.supply
            .append_quantile_to_state::<D, PRECISION>(left_sided_cumulative + remainder);

        Ok(())
    }
}

struct WasteGuard<'a, CompressedWord, State, const PRECISION: usize>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    waste: &'a mut Stack<CompressedWord, State>,
}

impl<'a, CompressedWord, State, const PRECISION: usize>
    WasteGuard<'a, CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn new(waste: &'a mut Stack<CompressedWord, State>) -> Self {
        // `Auryn::waste` satisfies slightly different invariants than a usual `Stack`.
        // We therefore restore the usual `Stack` invariant here. This is reversed
        // when the `WasteGuard` gets dropped.
        waste.refill_state_if_possible();

        Self { waste }
    }
}

impl<'a, CompressedWord, State, const PRECISION: usize> Deref
    for WasteGuard<'a, CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    type Target = Stack<CompressedWord, State>;

    fn deref(&self) -> &Self::Target {
        self.waste
    }
}

impl<'a, CompressedWord, State, const PRECISION: usize> DerefMut
    for WasteGuard<'a, CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.waste
    }
}

impl<'a, CompressedWord, State, const PRECISION: usize> Drop
    for WasteGuard<'a, CompressedWord, State, PRECISION>
where
    CompressedWord: BitArray + Into<State>,
    State: BitArray + AsPrimitive<CompressedWord>,
{
    fn drop(&mut self) {
        // Reverse the mutation done in `CoderGuard::new` to restore `Auryn`'s special
        // invariants for `waste`.
        if self.waste.state() >= State::one() << (State::BITS - PRECISION) {
            self.waste.flush_state();
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::distributions::LeakyQuantizer;

    use rand_xoshiro::{
        rand_core::{RngCore, SeedableRng},
        Xoshiro256StarStar,
    };
    use statrs::distribution::Normal;

    #[test]
    fn restore_none() {
        generic_restore_many::<u32, u64, u32, 24>(3, 0);
    }

    #[test]
    fn restore_one() {
        generic_restore_many::<u32, u64, u32, 24>(5, 1);
    }

    #[test]
    fn restore_two() {
        generic_restore_many::<u32, u64, u32, 24>(5, 2);
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
        let compressed = (0..amt_compressed_words)
            .map(|_| rng.next_u64().as_())
            .collect::<Vec<_>>();

        let distributions = (0..amt_symbols)
            .map(|_| {
                let mean = (100.0 / u32::MAX as f64) * rng.next_u32() as f64 - 100.0;
                let std_dev = (10.0 / u32::MAX as f64) * rng.next_u32() as f64 + 0.001;
                Normal::new(mean, std_dev)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(-100..=100);

        let mut auryn = Stack::from_binary(compressed.clone())
            .into_auryn_for_decoding()
            .unwrap();

        let symbols = auryn
            .decode_symbols(
                distributions
                    .iter()
                    .map(|&distribution| quantizer.quantize(distribution)),
            )
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(!auryn.maybe_empty());

        // Test exporting to stack and re-importing into Auryn:
        let stack = auryn.finish_decoding_and_concatenate().unwrap();
        let mut auryn = stack.into_auryn_for_encoding().unwrap();

        auryn
            .encode_symbols_reverse(
                symbols
                    .iter()
                    .zip(distributions)
                    .map(|(&symbol, distribution)| (symbol, quantizer.quantize(distribution))),
            )
            .unwrap();

        let stack = auryn.finish_encoding_and_concatenate().unwrap();
        assert_eq!(stack.into_binary().unwrap(), compressed);
    }
}
