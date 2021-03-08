#[cfg(feature = "std")]
use std::collections::{
    hash_map::Entry::{Occupied, Vacant},
    HashMap,
};

#[cfg(not(feature = "std"))]
use hashbrown::hash_map::{
    Entry::{Occupied, Vacant},
    HashMap,
};

use alloc::{boxed::Box, vec::Vec};

use core::{
    borrow::Borrow,
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

/// Type alias for an [`EncoderHashLookupTable`] with sane settings.
///
/// This hash lookup table can be used with a [`SmallAnsCoder`] or a [`SmallRangeEncoder`]
/// (as well as with a [`DefaultAnsCoder`] or a [`DefaultRangeEncoder`]).
///
/// # Example
///
/// ```
/// use constriction::stream::{
///     models::lookup::DefaultEncoderHashLookupTable,
///     stack::{SmallAnsCoder, DefaultAnsCoder},
///     queue::{SmallRangeEncoder, DefaultRangeEncoder},
///     Encode,
/// };
///
/// let allowed_symbols = "Misp";
/// let probabilities = [373, 1489, 1489, 745];
/// let encoder_model = DefaultEncoderHashLookupTable::from_symbols_and_probabilities(
///     allowed_symbols.chars().zip(probabilities.iter().cloned())
/// )
/// .unwrap();
///
/// let original = "Mississippi";
///
/// let mut small_ans_coder = SmallAnsCoder::new();
/// small_ans_coder.encode_iid_symbols_reverse(original.chars(), &encoder_model).unwrap();
/// assert_eq!(small_ans_coder.into_compressed(), [20510u16, 3988]);
///
/// let mut default_ans_coder = DefaultAnsCoder::new();
/// default_ans_coder.encode_iid_symbols_reverse(original.chars(), &encoder_model).unwrap();
/// assert_eq!(default_ans_coder.into_compressed(), [261378078u32]);
///
/// let mut small_range_encoder = SmallRangeEncoder::new();
/// small_range_encoder.encode_iid_symbols(original.chars(), &encoder_model).unwrap();
/// assert_eq!(small_range_encoder.into_compressed(), [1984u16, 56408]);
///
/// let mut default_range_encoder = DefaultRangeEncoder::new();
/// default_range_encoder.encode_iid_symbols(original.chars(), &encoder_model).unwrap();
/// assert_eq!(default_range_encoder.into_compressed(), [130092885u32]);
/// ```
///
/// See example in [`DefaultDecoderGenericLookupTable`] for decoding the above compressed
/// bit strings.
///
/// # See also
///
/// - [`DefaultEncoderArrayLookupTable`]
/// - [`DefaultDecoderGenericLookupTable`]
///
/// [`SmallAnsCoder`]: super::super::stack::SmallAnsCoder
/// [`SmallRangeEncoder`]: super::super::queue::SmallRangeEncoder
/// [`DefaultAnsCoder`]: super::super::stack::DefaultAnsCoder
/// [`DefaultRangeEncoder`]: super::super::queue::DefaultRangeEncoder
pub type DefaultEncoderHashLookupTable<Symbol> = EncoderHashLookupTable<Symbol, u16, 12>;

impl<Symbol, Probability, const PRECISION: usize>
    EncoderHashLookupTable<Symbol, Probability, PRECISION>
where
    Symbol: Hash + Eq,
    Probability: BitArray,
{
    pub fn from_symbols_and_probabilities(
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

    pub fn from_symbols_and_partial_probabilities(
        symbols: impl IntoIterator<Item = Symbol>,
        partial_probabilities: impl IntoIterator<Item = Probability>,
    ) -> Result<Self, ()> {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);

        let mut symbols = symbols.into_iter();
        let cap = symbols.size_hint().0;
        let mut error = false;
        let error_reporting_zip = partial_probabilities
            .into_iter()
            .scan((), |(), probability| {
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
        GenericSymbolTable<Box<[(Probability, Symbol)]>>,
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

    phantom: PhantomData<(usize, Probability)>,
}

/// Type alias for an [`EncoderArrayLookupTable`] with sane settings.
///
/// This array lookup table can be used with a [`SmallAnsCoder`] or a [`SmallRangeEncoder`]
/// (as well as with a [`DefaultAnsCoder`] or a [`DefaultRangeEncoder`]).
///
/// # Example
///
/// ```
/// use constriction::stream::{
///     models::lookup::DefaultEncoderArrayLookupTable,
///     stack::{SmallAnsCoder, DefaultAnsCoder},
///     queue::{SmallRangeEncoder, DefaultRangeEncoder},
///     Encode,
/// };
///
/// let probabilities = [1489, 745, 1489, 373];
/// let encoder_model = DefaultEncoderArrayLookupTable::from_probabilities(
///     probabilities.iter().cloned()
/// )
/// .unwrap();
///
/// let original = [2, 1, 3, 0, 0, 2, 0, 2, 1, 0, 2];
///
/// let mut small_ans_coder = SmallAnsCoder::new();
/// small_ans_coder.encode_iid_symbols_reverse(&original, &encoder_model).unwrap();
/// assert_eq!(small_ans_coder.into_compressed(), [55942u16, 10569]);
///
/// let mut default_ans_coder = DefaultAnsCoder::new();
/// default_ans_coder.encode_iid_symbols_reverse(&original, &encoder_model).unwrap();
/// assert_eq!(default_ans_coder.into_compressed(), [692705926u32]);
///
/// let mut small_range_encoder = SmallRangeEncoder::new();
/// small_range_encoder.encode_iid_symbols(&original, &encoder_model).unwrap();
/// assert_eq!(small_range_encoder.into_compressed(), [48376u16, 16073]);
///
/// let mut default_range_encoder = DefaultRangeEncoder::new();
/// default_range_encoder.encode_iid_symbols(&original, &encoder_model).unwrap();
/// assert_eq!(default_range_encoder.into_compressed(), [3170399034u32]);
/// ```
///
/// See example in [`DefaultDecoderIndexLookupTable`] for decoding the above compressed bit
/// strings.
///
/// # See also
///
/// - [`DefaultEncoderHashLookupTable`]
/// - [`DefaultDecoderIndexLookupTable`]
///
/// [`SmallAnsCoder`]: super::super::stack::SmallAnsCoder
/// [`SmallRangeEncoder`]: super::super::queue::SmallRangeEncoder
/// [`DefaultAnsCoder`]: super::super::stack::DefaultAnsCoder
/// [`DefaultRangeEncoder`]: super::super::queue::DefaultRangeEncoder
pub type DefaultEncoderArrayLookupTable<Symbol> =
    EncoderArrayLookupTable<Symbol, Box<[(u16, u16)]>, 12>;

impl<Probability, const PRECISION: usize>
    EncoderArrayLookupTable<Probability, Box<[(Probability, Probability)]>, PRECISION>
where
    Probability: BitArray,
{
    pub fn from_probabilities(
        probabilities: impl IntoIterator<Item = Probability>,
    ) -> Result<Self, ()> {
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

    pub fn from_partial_probabilities<I>(partial_probabilities: I) -> Result<Self, ()>
    where
        I: IntoIterator,
        I::Item: AsRef<Probability>,
    {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);

        let probabilities = partial_probabilities.into_iter();
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
        IndexSymbolTable<Box<[(Probability, ())]>>,
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
pub struct GenericSymbolTable<Table>(Table);

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct IndexSymbolTable<Table>(Table);

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

    phantom: PhantomData<(Probability, Symbol)>,
}

/// Type alias for a [`DecoderLookupTable`] over symbols `{0, 1, ..., n-1}` with sane settings.
///
/// This array lookup table can be used with a [`SmallAnsCoder`] or a [`SmallRangeDecoder`]
/// (as well as with a [`DefaultAnsCoder`] or a [`DefaultRangeDecoder`]).
///
/// # Example
///
/// Let's decode the compressed bit strings we generated in the example for
/// [`DefaultEncoderArrayLookupTable`].
///
/// ```
/// use constriction::stream::{
///     models::lookup::DefaultDecoderIndexLookupTable,
///     stack::{SmallAnsCoder, DefaultAnsCoder},
///     queue::{SmallRangeDecoder, DefaultRangeDecoder},
///     Decode, Code,
/// };
///
/// let probabilities = [1489, 745, 1489, 373];
/// let decoder_model = DefaultDecoderIndexLookupTable::from_probabilities(
///     probabilities.iter().cloned()
/// )
/// .unwrap();
///
/// let expected = [2, 1, 3, 0, 0, 2, 0, 2, 1, 0, 2];
///
/// let mut small_ans_coder = SmallAnsCoder::from_compressed(vec![55942, 10569]).unwrap();
/// let reconstructed = small_ans_coder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
/// assert!(small_ans_coder.is_empty());
/// assert_eq!(reconstructed, expected);
///
/// let mut default_ans_decoder = DefaultAnsCoder::from_compressed(vec![692705926]).unwrap();
/// let reconstructed = default_ans_decoder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
/// assert!(default_ans_decoder.is_empty());
/// assert_eq!(reconstructed, expected);
///
/// let mut small_range_decoder = SmallRangeDecoder::from_compressed(vec![48376, 16073]);
/// let reconstructed = small_range_decoder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
/// assert!(small_range_decoder.maybe_empty());
/// assert_eq!(reconstructed, expected);
///
/// let mut default_range_decoder = DefaultRangeDecoder::from_compressed(vec![3170399034]);
/// let reconstructed = default_range_decoder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<Vec<_>, _>>().unwrap();
/// assert!(default_range_decoder.maybe_empty());
/// assert_eq!(reconstructed, expected);
/// ```
///
/// # See also
///
/// - [`DefaultDecoderGenericLookupTable`]
/// - [`DefaultEncoderArrayLookupTable`]
///
/// [`SmallAnsCoder`]: super::super::stack::SmallAnsCoder
/// [`SmallRangeDecoder`]: super::super::queue::SmallRangeDecoder
/// [`DefaultAnsCoder`]: super::super::stack::DefaultAnsCoder
/// [`DefaultRangeDecoder`]: super::super::queue::DefaultRangeDecoder
pub type DefaultDecoderIndexLookupTable<Symbol> =
    DecoderLookupTable<Symbol, u16, Box<[u16]>, IndexSymbolTable<Box<[(u16, ())]>>, 12>;

/// Type alias for a [`DecoderLookupTable`] over arbitrary symbols with sane settings.
///
/// This array lookup table can be used with a [`SmallAnsCoder`] or a [`SmallRangeDecoder`]
/// (as well as with a [`DefaultAnsCoder`] or a [`DefaultRangeDecoder`]).
///
/// # Example
///
/// Let's decode the compressed bit strings we generated in the example for
/// [`DefaultEncoderHashLookupTable`].
///
/// ```
/// use constriction::stream::{
///     models::lookup::DefaultDecoderGenericLookupTable,
///     stack::{SmallAnsCoder, DefaultAnsCoder},
///     queue::{SmallRangeDecoder, DefaultRangeDecoder},
///     Decode, Code,
/// };
///
/// let allowed_symbols = "Misp";
/// let probabilities = [373, 1489, 1489, 745];
/// let decoder_model = DefaultDecoderGenericLookupTable::from_symbols_and_probabilities(
///     allowed_symbols.chars().zip(probabilities.iter().cloned())
/// )
/// .unwrap();
///
/// let expected = "Mississippi";
///
/// let mut small_ans_coder = SmallAnsCoder::from_compressed(vec![20510, 3988]).unwrap();
/// let reconstructed = small_ans_coder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<String, _>>().unwrap();
/// assert!(small_ans_coder.is_empty());
/// assert_eq!(reconstructed, expected);
///
/// let mut default_ans_decoder = DefaultAnsCoder::from_compressed(vec![261378078]).unwrap();
/// let reconstructed = default_ans_decoder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<String, _>>().unwrap();
/// assert!(default_ans_decoder.is_empty());
/// assert_eq!(reconstructed, expected);
///
/// let mut small_range_decoder = SmallRangeDecoder::from_compressed(vec![1984, 56408]);
/// let reconstructed = small_range_decoder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<String, _>>().unwrap();
/// assert!(small_range_decoder.maybe_empty());
/// assert_eq!(reconstructed, expected);
///
/// let mut default_range_decoder = DefaultRangeDecoder::from_compressed(vec![130092885]);
/// let reconstructed = default_range_decoder
///     .decode_iid_symbols(11, &decoder_model).collect::<Result<String, _>>().unwrap();
/// assert!(default_range_decoder.maybe_empty());
/// assert_eq!(reconstructed, expected);
/// ```
///
/// # See also
///
/// - [`DefaultDecoderIndexLookupTable`]
/// - [`DefaultEncoderArrayLookupTable`]
///
/// [`SmallAnsCoder`]: super::super::stack::SmallAnsCoder
/// [`SmallRangeDecoder`]: super::super::queue::SmallRangeDecoder
/// [`DefaultAnsCoder`]: super::super::stack::DefaultAnsCoder
/// [`DefaultRangeDecoder`]: super::super::queue::DefaultRangeDecoder
pub type DefaultDecoderGenericLookupTable<Symbol> =
    DecoderLookupTable<Symbol, u16, Box<[u16]>, GenericSymbolTable<Box<[(u16, Symbol)]>>, 12>;

impl<Symbol, Probability, const PRECISION: usize>
    DecoderLookupTable<
        Symbol,
        Probability,
        Box<[Probability]>,
        GenericSymbolTable<Box<[(Probability, Symbol)]>>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Symbol: Copy,
{
    pub fn from_symbols_and_probabilities(
        symbols_and_probabilities: impl IntoIterator<Item = (Symbol, Probability)>,
    ) -> Result<Self, ()>
where {
        Self::generic_from_symbols_and_probabilities(symbols_and_probabilities)
    }

    pub fn from_symbols_and_partial_probabilities(
        symbols: impl IntoIterator<Item = Symbol>,
        partial_probabilities: impl IntoIterator<Item = Probability>,
    ) -> Result<Self, ()>
    where
        usize: AsPrimitive<Probability>,
        Probability: Into<usize>,
        Symbol: Clone,
    {
        Self::generic_from_symbols_and_partial_probabilities(symbols, partial_probabilities, false)
    }
}

impl<Probability, const PRECISION: usize>
    DecoderLookupTable<
        usize,
        Probability,
        Box<[Probability]>,
        IndexSymbolTable<Box<[(Probability, ())]>>,
        PRECISION,
    >
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
{
    pub fn from_probabilities(
        probabilities: impl IntoIterator<Item = Probability>,
    ) -> Result<Self, ()>
where {
        Self::generic_from_symbols_and_probabilities(
            probabilities
                .into_iter()
                .map(|probability| ((), probability)),
        )
    }

    pub fn from_partial_probabilities(
        partial_probabilities: impl IntoIterator<Item = Probability>,
    ) -> Result<Self, ()>
    where
        usize: AsPrimitive<Probability>,
        Probability: Into<usize>,
    {
        Self::generic_from_symbols_and_partial_probabilities(
            core::iter::repeat(()),
            partial_probabilities,
            true,
        )
    }
}

impl<Symbol, Probability, Table, const PRECISION: usize>
    DecoderLookupTable<Symbol, Probability, Box<[Probability]>, Table, PRECISION>
where
    Probability: BitArray + Into<usize>,
    usize: AsPrimitive<Probability>,
    Symbol: Copy,
    Table: SymbolTable<
        Symbol,
        Probability,
        Inner = Box<
            [(
                Probability,
                <Table as SymbolTable<Symbol, Probability>>::SymbolRepresentation,
            )],
        >,
    >,
{
    fn generic_from_symbols_and_probabilities(
        symbols_and_probabilities: impl IntoIterator<Item = (Table::SymbolRepresentation, Probability)>,
    ) -> Result<Self, ()> {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);
        assert!(1usize << PRECISION != 0);

        let mut quantile_to_index = Vec::with_capacity(1 << PRECISION);
        let symbols_and_probabilities = symbols_and_probabilities.into_iter();
        let mut left_sided_cumulative_and_symbol =
            Vec::with_capacity(symbols_and_probabilities.size_hint().0 + 1);

        for (symbol, probability) in symbols_and_probabilities {
            if probability != Probability::zero() {
                let index = left_sided_cumulative_and_symbol.len().as_();
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
                left_sided_cumulative_and_symbol: Table::new(
                    left_sided_cumulative_and_symbol.into_boxed_slice(),
                ),
                phantom: PhantomData,
            })
        }
    }

    fn generic_from_symbols_and_partial_probabilities(
        symbols: impl IntoIterator<Item = Table::SymbolRepresentation>,
        partial_probabilities: impl IntoIterator<Item = Probability>,
        allow_excess_symbols: bool,
    ) -> Result<Self, ()> {
        assert!(PRECISION > 0);
        assert!(PRECISION <= Probability::BITS);
        assert!(1usize << PRECISION != 0);

        let mut symbols = symbols.into_iter();
        let mut quantile_to_index = Vec::with_capacity(1 << PRECISION);
        let mut left_sided_cumulative_and_symbol = Vec::with_capacity(symbols.size_hint().0 + 1);

        for probability in partial_probabilities.into_iter() {
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
        if let (Some(symbol), excess_symbol, Some(remainder)) =
            (symbols.next(), symbols.next(), remainder)
        {
            if !allow_excess_symbols && excess_symbol.is_some() {
                return Err(());
            }

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
                left_sided_cumulative_and_symbol: Table::new(
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
    DecoderLookupTable<Symbol, Probability, Table1, GenericSymbolTable<Table2>, PRECISION>
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
    DecoderLookupTable<usize, Probability, Table1, IndexSymbolTable<Table2>, PRECISION>
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
        symbol: impl core::borrow::Borrow<Self::Symbol>,
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
        symbol: impl core::borrow::Borrow<Self::Symbol>,
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
    type SymbolRepresentation: Clone;
    type Inner: AsRef<[(Probability, Self::SymbolRepresentation)]>;

    fn new(inner: Self::Inner) -> Self;

    unsafe fn get_unchecked(&self, index: usize) -> (Probability, Symbol);

    fn num_symbols(&self) -> usize;
}

impl<Symbol, Probability, Table> SymbolTable<Symbol, Probability> for GenericSymbolTable<Table>
where
    Symbol: Clone,
    Probability: Clone,
    Table: AsRef<[(Probability, Symbol)]>,
{
    type SymbolRepresentation = Symbol;
    type Inner = Table;

    fn new(inner: Self::Inner) -> Self {
        Self(inner)
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> (Probability, Symbol) {
        self.0.as_ref().get_unchecked(index).clone()
    }

    fn num_symbols(&self) -> usize {
        self.0.as_ref().len() - 1
    }
}

impl<Probability, Table> SymbolTable<usize, Probability> for IndexSymbolTable<Table>
where
    Probability: Clone,
    Table: AsRef<[(Probability, ())]>,
{
    type SymbolRepresentation = ();
    type Inner = Table;

    fn new(inner: Self::Inner) -> Self {
        Self(inner)
    }

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
        GenericSymbolTable<Box<[(Probability, Symbol)]>>,
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
            left_sided_cumulative_and_symbol: GenericSymbolTable(left_sided_cumulative_and_symbol),
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
        IndexSymbolTable<Box<[(Probability, ())]>>,
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
            left_sided_cumulative_and_symbol: IndexSymbolTable(left_sided_cumulative_and_symbol),
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
    let mut quantile_to_index = Vec::with_capacity(1 << PRECISION);
    quantile_to_index.resize(1 << PRECISION, MaybeUninit::uninit());

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
        core::mem::transmute::<_, Vec<Probability>>(quantile_to_index).into_boxed_slice()
    }
}

impl<Symbol, Probability, Table1, Table2, const PRECISION: usize>
    TryFrom<&DecoderLookupTable<Symbol, Probability, Table1, GenericSymbolTable<Table2>, PRECISION>>
    for EncoderHashLookupTable<Symbol, Probability, PRECISION>
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
            GenericSymbolTable<Table2>,
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
    TryFrom<&DecoderLookupTable<usize, Probability, Table1, IndexSymbolTable<Table2>, PRECISION>>
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
            IndexSymbolTable<Table2>,
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
    extern crate std;
    use std::{string::String, vec};

    use super::super::super::{
        models::{DecoderModel, EncoderModel},
        stack::DefaultAnsCoder,
        Decode,
    };

    use super::*;

    #[test]
    fn minimal_hash() {
        let symbols = "axcy";
        let probabilities = vec![3u8, 18, 1, 42];
        let encoder_model = EncoderHashLookupTable::<_, _, 6>::from_symbols_and_probabilities(
            symbols.chars().zip(probabilities.into_iter()),
        )
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

    #[test]
    fn minimal_array() {
        let probabilities = vec![3u8, 18, 1, 42];
        let encoder_model =
            EncoderArrayLookupTable::<_, _, 6>::from_probabilities(probabilities.iter().cloned())
                .unwrap();
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
        let mut ans = DefaultAnsCoder::new();
        ans.encode_iid_symbols_reverse(&symbols, &encoder_model)
            .unwrap();
        assert!(!ans.is_empty());
        let decoded = ans
            .decode_iid_symbols(9, &decoder_model)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(decoded, symbols);
        assert!(ans.is_empty());
    }
}
