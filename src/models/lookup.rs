use std::{
    borrow::Borrow,
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
pub struct EncoderHashLookupTable<Symbol, Probability, const PRECISION: usize>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    symbol_to_left_cumulative_and_probability: HashMap<Symbol, (Probability, Probability)>,
}

impl<Symbol, Probability, const PRECISION: usize>
    EncoderHashLookupTable<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    pub fn new(
        symbols_and_probabilities: impl IntoIterator<Item = (Symbol, Probability)>,
    ) -> Result<Self, ()> {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);

        let symbols_and_probabilities = symbols_and_probabilities.into_iter();
        let mut symbol_to_left_cumulative_and_probability =
            HashMap::with_capacity(symbols_and_probabilities.size_hint().0);

        let remainder = accumulate::<_, _, _, _, PRECISION>(
            symbols_and_probabilities,
            |symbol, left_sided_cumulative, probability| {
                match symbol_to_left_cumulative_and_probability.entry(symbol) {
                    Occupied(_) => Err(()),
                    Vacant(slot) => {
                        slot.insert((left_sided_cumulative, probability));
                        Ok(())
                    }
                }
            },
        )?;

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
        let cap = symbols.size_hint().0;
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

        let mut symbol_to_left_cumulative_and_probability = HashMap::with_capacity(cap);

        let remainder = accumulate::<_, _, _, _, PRECISION>(
            error_reporting_zip,
            |symbol, left_sided_cumulative, probability| {
                match symbol_to_left_cumulative_and_probability.entry(symbol) {
                    Occupied(_) => Err(()),
                    Vacant(slot) => {
                        slot.insert((left_sided_cumulative, probability));
                        Ok(())
                    }
                }
            },
        )?;

        if !error {
            if let (Some(symbol), None) = (symbols.next(), symbols.next()) {
                match symbol_to_left_cumulative_and_probability.entry(symbol) {
                    Occupied(_) => return Err(()),
                    Vacant(slot) => {
                        slot.insert((Probability::wrapping_pow2::<PRECISION>(), remainder));
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
        ExplicitSymbolTable<Box<[(Probability, Symbol)]>>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderArrayLookupTable<Probability, Table, const PRECISION: usize>
where
    Probability: BitArray,
{
    symbol_to_left_cumulative_and_probability: Table,

    phantom: PhantomData<*mut (usize, Probability)>,
}

impl<Probability, const PRECISION: usize>
    EncoderArrayLookupTable<Probability, Box<[(Probability, Probability)]>, PRECISION>
where
    Probability: BitArray,
{
    pub fn new<'a>(probabilities: impl IntoIterator<Item = &'a Probability>) -> Result<Self, ()> {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);

        let probabilities = probabilities.into_iter();
        let mut symbol_to_left_cumulative_and_probability =
            Vec::with_capacity(probabilities.size_hint().0);

        let remainder = accumulate::<_, _, _, _, PRECISION>(
            probabilities.map(|probability| ((), *probability.borrow())),
            |(), left_sided_cumulative, probability| {
                symbol_to_left_cumulative_and_probability
                    .push((left_sided_cumulative, probability));
                Ok(())
            },
        )?;

        if remainder != Probability::zero() {
            return Err(());
        }

        Ok(Self {
            symbol_to_left_cumulative_and_probability: symbol_to_left_cumulative_and_probability
                .into_boxed_slice(),
            phantom: PhantomData,
        })
    }

    pub fn with_inferred_last_probability<I>(probabilities: I) -> Result<Self, ()>
    where
        I: IntoIterator,
        I::Item: AsRef<Probability>,
    {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);

        let probabilities = probabilities.into_iter();
        let mut symbol_to_left_cumulative_and_probability =
            Vec::with_capacity(probabilities.size_hint().0);

        let remainder = accumulate::<_, _, _, _, PRECISION>(
            probabilities.map(|probability| ((), *probability.as_ref())),
            |(), left_sided_cumulative, probability| {
                symbol_to_left_cumulative_and_probability
                    .push((left_sided_cumulative, probability));
                Ok(())
            },
        )?;

        symbol_to_left_cumulative_and_probability
            .push((Probability::wrapping_pow2::<PRECISION>(), remainder));

        Ok(Self {
            symbol_to_left_cumulative_and_probability: symbol_to_left_cumulative_and_probability
                .into_boxed_slice(),
            phantom: PhantomData,
        })
    }
}

impl<Probability, Table, const PRECISION: usize>
    EncoderArrayLookupTable<Probability, Table, PRECISION>
where
    Probability: BitArray,
    Table: AsRef<[(Probability, Probability)]>,
{
    pub fn num_symbols(&self) -> usize {
        self.symbol_to_left_cumulative_and_probability
            .as_ref()
            .len()
    }

    pub fn encoder_table(
        &self,
    ) -> impl ExactSizeIterator<Item = (usize, Probability, Probability)> + '_ {
        self.symbol_to_left_cumulative_and_probability
            .as_ref()
            .iter()
            .enumerate()
            .map(|(symbol, &(left_sided_cumulative, probability))| {
                (symbol, left_sided_cumulative, probability)
            })
    }

    pub fn to_decoder_model(
        &self,
    ) -> DecoderLookupTable<
        usize,
        Probability,
        Box<[Probability]>,
        ImplicitSymbolTable<Box<[(Probability, ())]>>,
        PRECISION,
    >
    where
        Probability: Into<usize>,
        usize: AsPrimitive<Probability>,
    {
        self.into()
    }
}

fn accumulate<Symbol, Probability, I, F, const PRECISION: usize>(
    symbols_and_probabilities: I,
    mut operation: F,
) -> Result<Probability, ()>
where
    Probability: BitArray,
    I: Iterator<Item = (Symbol, Probability)>,
    F: FnMut(Symbol, Probability, Probability) -> Result<(), ()>,
{
    let mut left_sided_cumulative = Probability::zero();
    let mut laps = 0;

    for (symbol, probability) in symbols_and_probabilities {
        let old_left_sided_cumulative = left_sided_cumulative;
        left_sided_cumulative = old_left_sided_cumulative.wrapping_add(&probability);
        laps += (left_sided_cumulative <= old_left_sided_cumulative
            && probability != Probability::zero()) as usize;
        operation(symbol, old_left_sided_cumulative, probability)?;
    }

    let total = Probability::wrapping_pow2::<PRECISION>();
    if (left_sided_cumulative < total && laps == 0)
        || left_sided_cumulative == total && laps == (PRECISION == Probability::BITS) as usize
    {
        let remainder = total.wrapping_sub(&left_sided_cumulative);
        Ok(remainder)
    } else {
        Err(())
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ExplicitSymbolTable<Table>(Table);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ImplicitSymbolTable<Table>(Table);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderLookupTable<Symbol, Probability, Table1, Table2, const PRECISION: usize>
where
    Probability: BitArray,
{
    /// Satisfies invariant:
    /// `quantile_to_index.as_ref().len() == 1 << PRECISION`
    quantile_to_index: Table1,

    /// Satisfies invariant:
    /// `left_sided_cumulative_and_symbol.as_ref().len()
    /// == *quantile_to_index.as_ref().iter().max() as usize + 2`
    left_sided_cumulative_and_symbol: Table2,

    phantom: PhantomData<*mut (Symbol, Probability)>,
}

impl<Symbol, Probability, const PRECISION: usize>
    DecoderLookupTable<
        Symbol,
        Probability,
        Box<[Probability]>,
        ExplicitSymbolTable<Box<[(Probability, Symbol)]>>,
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
            // Get a dummy symbol for the last slot. It will never be read.
            let (_, dummy_symbol) = left_sided_cumulative_and_symbol.last().expect(
                "We already pushed at least one entry because quantile_to_index.len() != 1 << PRECISION != 0");
            let dummy_symbol = dummy_symbol.clone();
            left_sided_cumulative_and_symbol
                .push((Probability::wrapping_pow2::<PRECISION>(), dummy_symbol));

            Ok(Self {
                quantile_to_index: quantile_to_index.into_boxed_slice(),
                left_sided_cumulative_and_symbol: ExplicitSymbolTable(
                    left_sided_cumulative_and_symbol.into_boxed_slice(),
                ),
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

            // Reuse the last symbol for the additional closing entry. This will never be read.
            left_sided_cumulative_and_symbol
                .push((Probability::wrapping_pow2::<PRECISION>(), symbol));

            Ok(Self {
                quantile_to_index: quantile_to_index.into_boxed_slice(),
                left_sided_cumulative_and_symbol: ExplicitSymbolTable(
                    left_sided_cumulative_and_symbol.into_boxed_slice(),
                ),
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
    Table2: SymbolTable<Symbol, Probability>,
{
    pub fn num_symbols(&self) -> usize {
        self.left_sided_cumulative_and_symbol.num_symbols()
    }
}

impl<Symbol, Probability, Table1, Table2, const PRECISION: usize>
    DecoderLookupTable<Symbol, Probability, Table1, ExplicitSymbolTable<Table2>, PRECISION>
where
    Probability: BitArray,
    Table1: AsRef<[Probability]>,
    Table2: AsRef<[(Probability, Symbol)]>,
{
    pub fn encoder_table(
        &self,
    ) -> impl ExactSizeIterator<Item = (&Symbol, Probability, Probability)> {
        let mut iter = self.left_sided_cumulative_and_symbol.0.as_ref().iter();
        let mut old_entry = iter.next().expect("Table has at least 2 entries.");

        iter.map(move |new_entry| {
            let left_sided_cumulative = old_entry.0;
            let probability = new_entry.0.wrapping_sub(&left_sided_cumulative);
            let symbol = &old_entry.1;
            old_entry = new_entry;
            (symbol, left_sided_cumulative, probability)
        })
    }

    pub fn to_encoder_model(
        &self,
    ) -> Result<EncoderHashLookupTable<Symbol, Probability, PRECISION>, ()>
    where
        Symbol: Clone + Hash + Eq,
    {
        self.try_into()
    }
}

impl<Probability, Table1, Table2, const PRECISION: usize>
    DecoderLookupTable<usize, Probability, Table1, ImplicitSymbolTable<Table2>, PRECISION>
where
    Probability: BitArray,
    Table1: AsRef<[Probability]>,
    Table2: AsRef<[(Probability, ())]>,
{
    pub fn encoder_table(
        &self,
    ) -> impl ExactSizeIterator<Item = (usize, Probability, Probability)> + '_ {
        let mut iter = self.left_sided_cumulative_and_symbol.0.as_ref().iter();
        let mut old_cumulative = iter.next().expect("Table has at least 2 entries.").0;

        iter.enumerate().map(move |(symbol, (new_cumulative, ()))| {
            let left_sided_cumulative = old_cumulative;
            let probability = new_cumulative.wrapping_sub(&left_sided_cumulative);
            old_cumulative = *new_cumulative;
            (symbol, left_sided_cumulative, probability)
        })
    }

    pub fn to_encoder_model(
        &self,
    ) -> Result<
        EncoderArrayLookupTable<Probability, Box<[(Probability, Probability)]>, PRECISION>,
        (),
    > {
        self.try_into()
    }
}

impl<Symbol, Probability, const PRECISION: usize> EntropyModel<PRECISION>
    for EncoderHashLookupTable<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    type Symbol = Symbol;
    type Probability = Probability;
}

impl<Symbol, Probability, const PRECISION: usize> EncoderModel<PRECISION>
    for EncoderHashLookupTable<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    #[inline(always)]
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

impl<Probability, Table, const PRECISION: usize> EntropyModel<PRECISION>
    for EncoderArrayLookupTable<Probability, Table, PRECISION>
where
    Probability: BitArray,
{
    type Symbol = usize;
    type Probability = Probability;
}

impl<Probability, Table, const PRECISION: usize> EncoderModel<PRECISION>
    for EncoderArrayLookupTable<Probability, Table, PRECISION>
where
    Probability: BitArray,
    Table: AsRef<[(Probability, Probability)]>,
{
    #[inline(always)]
    fn left_cumulative_and_probability(
        &self,
        symbol: impl std::borrow::Borrow<Self::Symbol>,
    ) -> Result<(Self::Probability, Self::Probability), ()> {
        self.symbol_to_left_cumulative_and_probability
            .as_ref()
            .get(*symbol.borrow())
            .ok_or(())
            .map(Clone::clone)
    }
}

impl<Symbol, Probability, Table1, Table2, const PRECISION: usize> EntropyModel<PRECISION>
    for DecoderLookupTable<Symbol, Probability, Table1, Table2, PRECISION>
where
    Probability: BitArray,
{
    type Symbol = Symbol;
    type Probability = Probability;
}

pub trait SymbolTable<Symbol, Probability> {
    unsafe fn get_unchecked(&self, index: usize) -> (Probability, Symbol);

    fn num_symbols(&self) -> usize;
}

impl<Symbol, Probability, Table> SymbolTable<Symbol, Probability> for ExplicitSymbolTable<Table>
where
    Symbol: Clone,
    Probability: Clone,
    Table: AsRef<[(Probability, Symbol)]>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> (Probability, Symbol) {
        self.0.as_ref().get_unchecked(index).clone()
    }

    fn num_symbols(&self) -> usize {
        self.0.as_ref().len() - 1
    }
}

impl<Probability, Table> SymbolTable<usize, Probability> for ImplicitSymbolTable<Table>
where
    Probability: Clone,
    Table: AsRef<[(Probability, ())]>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> (Probability, usize) {
        (self.0.as_ref().get_unchecked(index).0.clone(), index)
    }

    fn num_symbols(&self) -> usize {
        self.0.as_ref().len() - 1
    }
}

impl<Symbol, Probability, Table1, Table2, const PRECISION: usize> DecoderModel<PRECISION>
    for DecoderLookupTable<Symbol, Probability, Table1, Table2, PRECISION>
where
    Probability: BitArray + Into<usize>,
    Table1: AsRef<[Probability]>,
    Table2: SymbolTable<Symbol, Probability>,
    Symbol: Clone,
{
    #[inline(always)]
    fn quantile_function(
        &self,
        quantile: Self::Probability,
    ) -> (Self::Symbol, Self::Probability, Self::Probability) {
        if Probability::BITS != PRECISION {
            // It would be nice if we could avoid this but we currently don't statically enforce
            // `quantile` to fit into `PRECISION` bits.
            assert!(PRECISION == Probability::BITS || quantile < Probability::one() << PRECISION);
        }

        let ((left_sided_cumulative, symbol), next_cumulative) = unsafe {
            // SAFETY:
            // - `quantile_to_index` has length `1 << PRECISION` and we verified that
            //   `quantile` fits into `PRECISION` bits above.
            // - `left_sided_cumulative_and_symbol` has length
            //   `*quantile_to_index.as_ref().iter().max() as usize + 2`, so we can always
            //   access it at `index + 1` for `index` coming from `quantile_to_index`.
            let index = *self
                .quantile_to_index
                .as_ref()
                .get_unchecked(quantile.into());
            let index = index.into();

            (
                self.left_sided_cumulative_and_symbol.get_unchecked(index),
                self.left_sided_cumulative_and_symbol
                    .get_unchecked(index + 1)
                    .0,
            )
        };

        (
            symbol,
            left_sided_cumulative,
            next_cumulative.wrapping_sub(&left_sided_cumulative),
        )
    }
}

impl<Symbol, Probability, const PRECISION: usize>
    From<&EncoderHashLookupTable<Symbol, Probability, PRECISION>>
    for DecoderLookupTable<
        Symbol,
        Probability,
        Box<[Probability]>,
        ExplicitSymbolTable<Box<[(Probability, Symbol)]>>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    Symbol: Clone + Hash + Eq,
    usize: AsPrimitive<Probability>,
{
    fn from(encoder_model: &EncoderHashLookupTable<Symbol, Probability, PRECISION>) -> Self {
        assert!(1usize << PRECISION != 0);

        let mut left_sided_cumulative_and_symbol = invert_encoder_table::<_, _, _, PRECISION>(
            encoder_model
                .symbol_to_left_cumulative_and_probability
                .iter()
                .map(|(symbol, entry)| (symbol.clone(), entry)),
        );

        let len = left_sided_cumulative_and_symbol.len();
        left_sided_cumulative_and_symbol[..len - 1].sort_by_key(|&(cumulative, _)| cumulative);

        let quantile_to_index = fill_decoder_table::<_, _, _, PRECISION>(
            left_sided_cumulative_and_symbol
                .iter()
                .map(|(cumulative, symbol)| (*cumulative, symbol.clone())),
        );

        Self {
            left_sided_cumulative_and_symbol: ExplicitSymbolTable(left_sided_cumulative_and_symbol),
            quantile_to_index,
            phantom: PhantomData,
        }
    }
}

impl<Table, Probability, const PRECISION: usize>
    From<&EncoderArrayLookupTable<Probability, Table, PRECISION>>
    for DecoderLookupTable<
        usize,
        Probability,
        Box<[Probability]>,
        ImplicitSymbolTable<Box<[(Probability, ())]>>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    Table: AsRef<[(Probability, Probability)]>,
    usize: AsPrimitive<Probability>,
{
    fn from(encoder_model: &EncoderArrayLookupTable<Probability, Table, PRECISION>) -> Self {
        assert!(1usize << PRECISION != 0);

        let left_sided_cumulative_and_symbol = invert_encoder_table::<_, _, _, PRECISION>(
            encoder_model
                .symbol_to_left_cumulative_and_probability
                .as_ref()
                .iter()
                .map(|entry| ((), entry)),
        );

        // No need to sort, `EncoderArrayLookupTable` always has non-decreasing cdf.

        let quantile_to_index = fill_decoder_table::<_, _, _, PRECISION>(
            left_sided_cumulative_and_symbol.iter().cloned(),
        );

        Self {
            left_sided_cumulative_and_symbol: ImplicitSymbolTable(left_sided_cumulative_and_symbol),
            quantile_to_index,
            phantom: PhantomData,
        }
    }
}

fn invert_encoder_table<'a, Probability, Symbol, I, const PRECISION: usize>(
    symbol_to_left_cumulative_and_probability: I,
) -> Box<[(Probability, Symbol)]>
where
    Probability: BitArray + 'a,
    Symbol: Clone,
    I: ExactSizeIterator<Item = (Symbol, &'a (Probability, Probability))>,
{
    let mut left_sided_cumulative_and_symbol =
        Vec::with_capacity(symbol_to_left_cumulative_and_probability.len() + 1);

    left_sided_cumulative_and_symbol.extend(
        symbol_to_left_cumulative_and_probability
            .map(|(symbol, &(left_sided_cumulative, _))| (left_sided_cumulative, symbol)),
    );

    // Get a dummy symbol for the last slot. It will never be read.
    let (_, dummy_symbol) = left_sided_cumulative_and_symbol.last().expect(
            "We already pushed at least one entry because quantile_to_index.len() != 1 << PRECISION != 0");
    let dummy_symbol = dummy_symbol.clone();
    left_sided_cumulative_and_symbol
        .push((Probability::wrapping_pow2::<PRECISION>(), dummy_symbol));

    left_sided_cumulative_and_symbol.into_boxed_slice()
}

fn fill_decoder_table<Probability, Symbol, I, const PRECISION: usize>(
    mut left_sided_cumulative_and_symbol: I,
) -> Box<[Probability]>
where
    Probability: BitArray + Into<usize>,
    I: Iterator<Item = (Probability, Symbol)>,
{
    let mut quantile_to_index: Vec<MaybeUninit<Probability>> =
        vec![MaybeUninit::uninit(); 1 << PRECISION];

    let mut old_entry = left_sided_cumulative_and_symbol
        .next()
        .expect("Table has at least 2 entries.");
    let mut index = Probability::zero();

    for new_entry in left_sided_cumulative_and_symbol {
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

    unsafe {
        // SAFETY: `encoder_model` is a valid `EncoderModel`, so it must map each quantile
        // to exactly one symbol.
        std::mem::transmute::<_, Vec<Probability>>(quantile_to_index).into_boxed_slice()
    }
}

impl<Symbol, Probability, Table1, Table2, const PRECISION: usize>
    TryFrom<
        &DecoderLookupTable<Symbol, Probability, Table1, ExplicitSymbolTable<Table2>, PRECISION>,
    > for EncoderHashLookupTable<Symbol, Probability, PRECISION>
where
    Probability: BitArray,
    Symbol: Clone + Hash + Eq,
    Table1: AsRef<[Probability]>,
    Table2: AsRef<[(Probability, Symbol)]>,
{
    type Error = ();

    fn try_from(
        decoder_model: &DecoderLookupTable<
            Symbol,
            Probability,
            Table1,
            ExplicitSymbolTable<Table2>,
            PRECISION,
        >,
    ) -> Result<Self, ()> {
        let symbol_to_left_cumulative_and_probability = decoder_model
            .encoder_table()
            .map(|(symbol, left_sided_cumulative, probability)| {
                (symbol.clone(), (left_sided_cumulative, probability))
            })
            .collect::<HashMap<_, _>>();

        if symbol_to_left_cumulative_and_probability.len()
            != decoder_model.left_sided_cumulative_and_symbol.num_symbols()
        {
            Err(())
        } else {
            Ok(Self {
                symbol_to_left_cumulative_and_probability,
            })
        }
    }
}

impl<Probability, Table1, Table2, const PRECISION: usize>
    TryFrom<&DecoderLookupTable<usize, Probability, Table1, ImplicitSymbolTable<Table2>, PRECISION>>
    for EncoderArrayLookupTable<Probability, Box<[(Probability, Probability)]>, PRECISION>
where
    Probability: BitArray,
    Table1: AsRef<[Probability]>,
    Table2: AsRef<[(Probability, ())]>,
{
    type Error = ();

    fn try_from(
        decoder_model: &DecoderLookupTable<
            usize,
            Probability,
            Table1,
            ImplicitSymbolTable<Table2>,
            PRECISION,
        >,
    ) -> Result<Self, ()> {
        let symbol_to_left_cumulative_and_probability = decoder_model
            .encoder_table()
            .map(|(_symbol, left_sided_cumulative, probability)| {
                (left_sided_cumulative, probability)
            })
            .collect::<Vec<_>>();

        if symbol_to_left_cumulative_and_probability.len()
            != decoder_model.left_sided_cumulative_and_symbol.num_symbols()
        {
            Err(())
        } else {
            Ok(Self {
                symbol_to_left_cumulative_and_probability:
                    symbol_to_left_cumulative_and_probability.into_boxed_slice(),
                phantom: PhantomData,
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

    use super::*;

    #[test]
    fn minimal_hash() {
        let symbols = "axcy";
        let probabilities = vec![3u8, 18, 1, 42];
        let encoder_model =
            EncoderHashLookupTable::<_, _, 6>::new(symbols.chars().zip(probabilities.into_iter()))
                .unwrap();
        let decoder_model = encoder_model.to_decoder_model();
        assert_eq!(encoder_model, decoder_model.to_encoder_model().unwrap());

        // Verify that `decoder_model(encoder_model(x)) = x`.
        for symbol in symbols.chars() {
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

        // Verify that `encoder_model(decoder_model(x)) = x`.
        for quantile in 0..1 << 6 {
            let (symbol, left_cumulative, probability) = decoder_model.quantile_function(quantile);
            assert_eq!(
                encoder_model
                    .left_cumulative_and_probability(symbol)
                    .unwrap(),
                (left_cumulative, probability)
            );
        }

        // Test encoding and decoding a few symbols.
        let symbols = "axcxcyaac";
        let mut stack = DefaultStack::new();
        stack
            .encode_iid_symbols_reverse(symbols.chars(), &encoder_model)
            .unwrap();
        assert!(!stack.is_empty());
        let decoded = stack
            .decode_iid_symbols(9, &decoder_model)
            .collect::<Result<String, _>>()
            .unwrap();
        assert_eq!(decoded, symbols);
        assert!(stack.is_empty());
    }

    #[test]
    fn minimal_array() {
        let probabilities = vec![3u8, 18, 1, 42];
        let encoder_model = EncoderArrayLookupTable::<_, _, 6>::new(&probabilities).unwrap();
        let decoder_model = encoder_model.to_decoder_model();
        assert_eq!(encoder_model, decoder_model.to_encoder_model().unwrap());

        // Verify that `decoder_model(encoder_model(x)) = x`.
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

        // Verify that `encoder_model(decoder_model(x)) = x`.
        for quantile in 0..1 << 6 {
            let (symbol, left_cumulative, probability) = decoder_model.quantile_function(quantile);
            assert_eq!(
                encoder_model
                    .left_cumulative_and_probability(symbol)
                    .unwrap(),
                (left_cumulative, probability)
            );
        }

        // Test encoding and decoding a few symbols.
        let symbols = vec![0, 3, 2, 3, 1, 3, 2, 0, 3];
        let mut stack = DefaultStack::new();
        stack
            .encode_iid_symbols_reverse(&symbols, &encoder_model)
            .unwrap();
        assert!(!stack.is_empty());
        let decoded = stack
            .decode_iid_symbols(9, &decoder_model)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(decoded, symbols);
        assert!(stack.is_empty());
    }
}
