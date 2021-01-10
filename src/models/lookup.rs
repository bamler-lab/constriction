use std::{
    collections::{
        hash_map::Entry::{Occupied, Vacant},
        HashMap,
    },
    convert::{TryFrom, TryInto},
    hash::Hash,
    marker::PhantomData,
    mem::MaybeUninit,
};

use num::cast::AsPrimitive;

use crate::BitArray;

use super::{DecoderModel, EncoderModel, EntropyModel};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderLookupTable<Symbol, Probability, const PRECISION: usize>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    symbol_to_left_cumulative_and_probability: HashMap<Symbol, (Probability, Probability)>,
}

impl<Symbol, Probability, const PRECISION: usize> EncoderLookupTable<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    pub fn new(
        symbols_and_probabilities: impl IntoIterator<Item = (Symbol, Probability)>,
    ) -> Result<Self, ()> {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);

        let (symbol_to_left_cumulative_and_probability, remainder) =
            accumulate::<_, _, _, PRECISION>(symbols_and_probabilities, 0)?;

        if remainder != Probability::zero() {
            return Err(());
        }

        Ok(Self {
            symbol_to_left_cumulative_and_probability,
        })
    }

    pub fn with_inferred_last_probability(
        symbols: impl IntoIterator<Item = Symbol>,
        probabilities: impl IntoIterator<Item = Probability>,
    ) -> Result<Self, ()> {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);

        let mut symbols = symbols.into_iter();
        let mut error = false;
        let error_reporting_zip = probabilities.into_iter().scan((), |(), probability| {
            if let Some(symbol) = symbols.next() {
                Some((symbol, probability))
            } else {
                // Fewer symbols than probabilities were provided; terminate and report error.
                error = true;
                None
            }
        });

        let (mut symbol_to_left_cumulative_and_probability, remainder) =
            accumulate::<_, _, _, PRECISION>(error_reporting_zip, 1)?;

        if !error {
            if let (Some(symbol), None) = (symbols.next(), symbols.next()) {
                match symbol_to_left_cumulative_and_probability.entry(symbol) {
                    Occupied(_) => return Err(()),
                    Vacant(slot) => {
                        slot.insert((Probability::one() << PRECISION, remainder));
                    }
                }

                return Ok(Self {
                    symbol_to_left_cumulative_and_probability,
                });
            }
        }

        Err(())
    }

    pub fn num_symbols(&self) -> usize {
        self.symbol_to_left_cumulative_and_probability.len()
    }

    pub fn encoder_table(
        &self,
    ) -> impl ExactSizeIterator<Item = (&Symbol, Probability, Probability)> {
        self.symbol_to_left_cumulative_and_probability.iter().map(
            |(symbol, &(left_sided_cumulative, probability))| {
                (symbol, left_sided_cumulative, probability)
            },
        )
    }

    pub fn to_decoder_model(
        &self,
    ) -> DecoderLookupTable<
        Symbol,
        Probability,
        Box<[Probability]>,
        Box<[(Probability, Symbol)]>,
        PRECISION,
    >
    where
        Symbol: Clone,
        usize: AsPrimitive<Probability>,
        Probability: Into<usize>,
    {
        self.into()
    }
}

fn accumulate<Symbol, Probability, I, const PRECISION: usize>(
    symbols_and_probabilities: I,
    extra_size_hint: usize,
) -> Result<(HashMap<Symbol, (Probability, Probability)>, Probability), ()>
where
    Probability: BitArray,
    I: IntoIterator<Item = (Symbol, Probability)>,
    Symbol: Hash + Eq,
{
    let mut left_sided_cumulative = Probability::zero();
    let mut laps = 0;

    let symbols_and_probabilities = symbols_and_probabilities.into_iter();
    let mut collection =
        HashMap::with_capacity(symbols_and_probabilities.size_hint().0 + extra_size_hint);

    for (symbol, probability) in symbols_and_probabilities {
        let old_left_sided_cumulative = left_sided_cumulative;
        left_sided_cumulative = old_left_sided_cumulative.wrapping_add(&probability);
        laps += (left_sided_cumulative <= old_left_sided_cumulative
            && probability != Probability::zero()) as usize;

        match collection.entry(symbol) {
            Occupied(_) => return Err(()),
            Vacant(slot) => {
                slot.insert((old_left_sided_cumulative, probability));
            }
        }
    }

    if (left_sided_cumulative < (Probability::one() << PRECISION) && laps == 0)
        || left_sided_cumulative == Probability::one() << PRECISION
            && laps == (PRECISION == Probability::BITS) as usize
    {
        let remainder = (Probability::one() << PRECISION) - left_sided_cumulative;
        Ok((collection, remainder))
    } else {
        Err(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderLookupTable<Symbol, Probability, Table1, Table2, const PRECISION: usize>
where
    Probability: BitArray,
{
    /// Satisfies invariant:
    /// `quantile_to_index.len() == 1 << PRECISION`
    quantile_to_index: Table1,

    /// Satisfies invariant:
    /// `left_sided_cumulative_and_symbol.len() == quantile_to_index.into_iter().max() as usize + 1`
    left_sided_cumulative_and_symbol: Table2,

    phantom: PhantomData<*mut (Symbol, Probability)>,
}

impl<Symbol, Probability, const PRECISION: usize>
    DecoderLookupTable<
        Symbol,
        Probability,
        Box<[Probability]>,
        Box<[(Probability, Symbol)]>,
        PRECISION,
    >
where
    Probability: BitArray,
{
    pub fn new(
        symbols_and_probabilities: impl IntoIterator<Item = (Symbol, Probability)>,
    ) -> Result<Self, ()>
    where
        usize: AsPrimitive<Probability>,
        Probability: Into<usize>,
        Symbol: Clone,
    {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);
        assert!(1usize << PRECISION != 0);

        let mut quantile_to_index = Vec::with_capacity(1 << PRECISION);
        let symbols_and_probabilities = symbols_and_probabilities.into_iter();
        let mut left_sided_cumulative_and_symbol =
            Vec::with_capacity(symbols_and_probabilities.size_hint().0 + 1);

        for (symbol, probability) in symbols_and_probabilities {
            if probability != Probability::zero() {
                let index = quantile_to_index.len().as_();
                left_sided_cumulative_and_symbol.push((quantile_to_index.len().as_(), symbol));
                quantile_to_index.resize(quantile_to_index.len() + probability.into(), index);
            }
        }

        if quantile_to_index.len() != 1 << PRECISION {
            Err(())
        } else {
            let total_probability_wrapping = if PRECISION == Probability::BITS {
                Probability::zero()
            } else {
                Probability::one() << PRECISION
            };

            // Get a dummy symbol for the last slot. It will never be read.
            let (_, dummy_symbol) = left_sided_cumulative_and_symbol.last().expect(
                "We already pushed at least one entry because quantile_to_index.len() != 1 << PRECISION != 0");
            let dummy_symbol = dummy_symbol.clone();
            left_sided_cumulative_and_symbol.push((total_probability_wrapping, dummy_symbol));

            Ok(Self {
                quantile_to_index: quantile_to_index.into_boxed_slice(),
                left_sided_cumulative_and_symbol: left_sided_cumulative_and_symbol
                    .into_boxed_slice(),
                phantom: PhantomData,
            })
        }
    }

    pub fn with_inferred_last_probability(
        symbols: impl IntoIterator<Item = Symbol>,
        probabilities: impl IntoIterator<Item = Probability>,
    ) -> Result<Self, ()>
    where
        usize: AsPrimitive<Probability>,
        Probability: Into<usize>,
        Symbol: Clone,
    {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);
        assert!(1usize << PRECISION != 0);

        let mut symbols = symbols.into_iter();
        let mut quantile_to_index = Vec::with_capacity(1 << PRECISION);
        let mut left_sided_cumulative_and_symbol = Vec::with_capacity(symbols.size_hint().0 + 1);

        for probability in probabilities.into_iter() {
            if let Some(symbol) = symbols.next() {
                if probability != Probability::zero() {
                    let index = left_sided_cumulative_and_symbol.len().as_();
                    left_sided_cumulative_and_symbol.push((quantile_to_index.len().as_(), symbol));
                    quantile_to_index.resize(quantile_to_index.len() + probability.into(), index);
                }
            } else {
                // Fewer symbols than probabilities were provided.
                return Err(());
            }
        }

        let remainder = (1usize << PRECISION).checked_sub(quantile_to_index.len());
        if let (Some(symbol), None, Some(remainder)) = (symbols.next(), symbols.next(), remainder) {
            if remainder != 0 {
                let index = left_sided_cumulative_and_symbol.len().as_();
                left_sided_cumulative_and_symbol
                    .push((quantile_to_index.len().as_(), symbol.clone()));
                quantile_to_index.resize(1 << PRECISION, index);
            }

            let total_probability_wrapping = if PRECISION == Probability::BITS {
                Probability::zero()
            } else {
                Probability::one() << PRECISION
            };

            // Reuse the last symbol for the additional closing entry. This will never be read.
            left_sided_cumulative_and_symbol.push((total_probability_wrapping, symbol));

            Ok(Self {
                quantile_to_index: quantile_to_index.into_boxed_slice(),
                left_sided_cumulative_and_symbol: left_sided_cumulative_and_symbol
                    .into_boxed_slice(),
                phantom: PhantomData,
            })
        } else {
            Err(())
        }
    }
}

impl<Symbol, Probability, Table1, Table2, const PRECISION: usize>
    DecoderLookupTable<Symbol, Probability, Table1, Table2, PRECISION>
where
    Probability: BitArray,
    Table1: AsRef<[Probability]>,
    Table2: AsRef<[(Probability, Symbol)]>,
{
    pub fn encoder_table(
        &self,
    ) -> impl ExactSizeIterator<Item = (&Symbol, Probability, Probability)> {
        let mut iter = self.left_sided_cumulative_and_symbol.as_ref().iter();
        let mut old_entry = iter.next().expect("Table has at least 2 entries.");

        iter.map(move |new_entry| {
            let left_sided_cumulative = old_entry.0;
            let probability = new_entry.0.wrapping_sub(&left_sided_cumulative);
            let symbol = &old_entry.1;
            old_entry = new_entry;
            (symbol, left_sided_cumulative, probability)
        })
    }

    pub fn num_symbols(&self) -> usize {
        self.left_sided_cumulative_and_symbol.as_ref().len() - 1
    }

    pub fn to_encoder_model(&self) -> Result<EncoderLookupTable<Symbol, Probability, PRECISION>, ()>
    where
        Symbol: Clone + Hash + Eq,
        Probability: BitArray + Into<usize>,
        Table1: AsRef<[Probability]>,
        Table2: AsRef<[(Probability, Symbol)]>,
    {
        self.try_into()
    }
}

impl<Symbol, Probability, const PRECISION: usize> EntropyModel<PRECISION>
    for EncoderLookupTable<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    type Symbol = Symbol;
    type Probability = Probability;
}

impl<Symbol, Probability, const PRECISION: usize> EncoderModel<PRECISION>
    for EncoderLookupTable<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    fn left_cumulative_and_probability(
        &self,
        symbol: impl std::borrow::Borrow<Self::Symbol>,
    ) -> Result<(Self::Probability, Self::Probability), ()> {
        self.symbol_to_left_cumulative_and_probability
            .get(symbol.borrow())
            .ok_or(())
            .map(Clone::clone)
    }
}

impl<Symbol, Probability, Table1, Table2, const PRECISION: usize> EntropyModel<PRECISION>
    for DecoderLookupTable<Symbol, Probability, Table1, Table2, PRECISION>
where
    Probability: BitArray + Into<usize>,
    Table1: AsRef<[Probability]>,
    Table2: AsRef<[(Probability, Symbol)]>,
{
    type Symbol = Symbol;
    type Probability = Probability;
}

impl<Symbol, Probability, Table1, Table2, const PRECISION: usize> DecoderModel<PRECISION>
    for DecoderLookupTable<Symbol, Probability, Table1, Table2, PRECISION>
where
    Probability: BitArray + Into<usize>,
    Table1: AsRef<[Probability]>,
    Table2: AsRef<[(Probability, Symbol)]>,
    Symbol: Clone,
{
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Self::Symbol, Self::Probability, Self::Probability) {
        if Probability::BITS != PRECISION {
            // It would be nice if we could avoid this we we currently don't statically enforce
            // `quantile` fit into `PRECISION` bits.
            assert!(quantile < Probability::one() << PRECISION);
        }

        unsafe {
            let index = *self
                .quantile_to_index
                .as_ref()
                .get_unchecked(quantile.into());
            let (left_sided_cumulative, symbol) = self
                .left_sided_cumulative_and_symbol
                .as_ref()
                .get_unchecked(index.into())
                .clone();
            let (next_cumulative, _) = *self
                .left_sided_cumulative_and_symbol
                .as_ref()
                .get_unchecked(index.into() + 1);

            (
                symbol,
                left_sided_cumulative,
                next_cumulative.wrapping_sub(&left_sided_cumulative),
            )
        }
    }
}

impl<Symbol, Probability, const PRECISION: usize>
    From<&EncoderLookupTable<Symbol, Probability, PRECISION>>
    for DecoderLookupTable<
        Symbol,
        Probability,
        Box<[Probability]>,
        Box<[(Probability, Symbol)]>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    Symbol: Clone + Hash + Eq,
    usize: AsPrimitive<Probability>,
{
    fn from(encoder_model: &EncoderLookupTable<Symbol, Probability, PRECISION>) -> Self {
        assert!(1usize << PRECISION != 0);

        let mut left_sided_cumulative_and_symbol = Vec::with_capacity(
            encoder_model
                .symbol_to_left_cumulative_and_probability
                .len()
                + 1,
        );

        for (symbol, &(left_sided_cumulative, _)) in encoder_model
            .symbol_to_left_cumulative_and_probability
            .iter()
        {
            left_sided_cumulative_and_symbol.push((left_sided_cumulative, symbol.clone()));
        }

        left_sided_cumulative_and_symbol.sort_by_key(|&(cumulative, _)| cumulative);

        let total_probability_wrapping = if PRECISION == Probability::BITS {
            Probability::zero()
        } else {
            Probability::one() << PRECISION
        };

        // Get a dummy symbol for the last slot. It will never be read.
        let (_, dummy_symbol) = left_sided_cumulative_and_symbol.last().expect(
                        "We already pushed at least one entry because quantile_to_index.len() != 1 << PRECISION != 0");
        let dummy_symbol = dummy_symbol.clone();
        left_sided_cumulative_and_symbol.push((total_probability_wrapping, dummy_symbol));

        let mut quantile_to_index: Vec<MaybeUninit<Probability>> =
            vec![MaybeUninit::uninit(); 1 << PRECISION];

        let mut iter = left_sided_cumulative_and_symbol.iter();
        let mut old_entry = iter.next().expect("Table has at least 2 entries.");
        let mut index = Probability::zero();

        for new_entry in iter {
            let left_sided_cumulative = old_entry.0;
            let probability = new_entry.0.wrapping_sub(&left_sided_cumulative);
            old_entry = new_entry;

            let slice = unsafe {
                // SAFETY:
                // - `left_sided_cumulative + probability <= 1 << PRECISION == quantile_to_index.len()`
                //   because `encoder_model` is an `EncoderLookupTable`.
                // - `left_sided_cumulative + probability` fits into `usize` because
                //   `1usize << PRECISION != 0`.
                quantile_to_index.get_unchecked_mut(
                    left_sided_cumulative.into()..left_sided_cumulative.into() + probability.into(),
                )
            };

            slice.fill(MaybeUninit::new(index));
            index = index.wrapping_add(&Probability::one());
        }

        let quantile_to_index = unsafe {
            // SAFETY: `encoder_model` is a valid `EncoderModel`, so it must map each quantile
            // to exactly one symbol.
            std::mem::transmute::<_, Vec<Probability>>(quantile_to_index)
        };

        Self {
            quantile_to_index: quantile_to_index.into_boxed_slice(),
            left_sided_cumulative_and_symbol: left_sided_cumulative_and_symbol.into_boxed_slice(),
            phantom: PhantomData,
        }
    }
}

impl<Symbol, Probability, Table1, Table2, const PRECISION: usize>
    TryFrom<&DecoderLookupTable<Symbol, Probability, Table1, Table2, PRECISION>>
    for EncoderLookupTable<Symbol, Probability, PRECISION>
where
    Probability: BitArray,
    Symbol: Clone + Hash + Eq,
    Table1: AsRef<[Probability]>,
    Table2: AsRef<[(Probability, Symbol)]>,
{
    type Error = ();

    fn try_from(
        decoder_model: &DecoderLookupTable<Symbol, Probability, Table1, Table2, PRECISION>,
    ) -> Result<Self, ()> {
        let symbol_to_left_cumulative_and_probability = decoder_model
            .encoder_table()
            .map(|(symbol, left_sided_cumulative, probability)| {
                (symbol.clone(), (left_sided_cumulative, probability))
            })
            .collect::<HashMap<_, _>>();

        if symbol_to_left_cumulative_and_probability.len()
            != decoder_model
                .left_sided_cumulative_and_symbol
                .as_ref()
                .len()
                - 1
        {
            Err(())
        } else {
            Ok(Self {
                symbol_to_left_cumulative_and_probability,
            })
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        models::{DecoderModel, EncoderModel},
        stack::DefaultStack,
        Decode,
    };

    use super::EncoderLookupTable;

    #[test]
    fn minimal() {
        let probabilities = vec![3u8, 18, 1, 42];
        let encoder_model =
            EncoderLookupTable::<_, _, 6>::new(probabilities.into_iter().enumerate()).unwrap();
        let decoder_model = encoder_model.to_decoder_model();

        for symbol in 0..4 {
            let (left_cumulative, probability) = encoder_model
                .left_cumulative_and_probability(symbol)
                .unwrap();
            for quantile in left_cumulative..left_cumulative + probability {
                assert_eq!(
                    decoder_model.quantile_function(quantile),
                    (symbol, left_cumulative, probability)
                );
            }
        }

        let symbols = vec![0, 3, 2, 3, 1, 3, 2, 0, 3];
        let mut stack = DefaultStack::new();
        stack
            .encode_iid_symbols_reverse(&symbols, &encoder_model)
            .unwrap();
        let decoded = stack
            .decode_iid_symbols(9, &decoder_model)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(decoded, symbols);
    }
}
