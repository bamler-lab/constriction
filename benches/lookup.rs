use std::any::type_name;

use constriction::{
    stream::{
        models::lookup::EncoderHashLookupTable, queue::RangeEncoder, stack::AnsCoder, Code, Decode,
        Encode,
    },
    BitArray, Pos, Seek,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num::cast::AsPrimitive;
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

criterion_group!(
    benches,
    round_trip_u32_u64_u16_12,
    round_trip_u32_u64_u16_16,
    round_trip_u16_u32_u8_8,
    round_trip_u16_u32_u16_8,
    round_trip_u16_u32_u16_12
);
criterion_main!(benches);

fn round_trip_u32_u64_u16_12(c: &mut Criterion) {
    round_trip::<u32, u64, u16, 12>(c);
}

fn round_trip_u32_u64_u16_16(c: &mut Criterion) {
    round_trip::<u32, u64, u16, 16>(c);
}

fn round_trip_u16_u32_u8_8(c: &mut Criterion) {
    round_trip::<u16, u32, u8, 8>(c);
}

fn round_trip_u16_u32_u16_8(c: &mut Criterion) {
    round_trip::<u16, u32, u16, 8>(c);
}

fn round_trip_u16_u32_u16_12(c: &mut Criterion) {
    round_trip::<u16, u32, u16, 12>(c);
}

fn round_trip<Word, State, Probability, const PRECISION: usize>(c: &mut Criterion)
where
    Probability: BitArray,
    u32: AsPrimitive<Probability>,
    Word: BitArray + AsPrimitive<Probability> + From<Probability>,
    State: BitArray + AsPrimitive<Word> + AsPrimitive<usize> + From<Word>,
    u64: From<Probability>,
    usize: From<Probability> + AsPrimitive<Probability>,
{
    ans_round_trip::<Word, State, Probability, PRECISION>(c);
    range_round_trip::<Word, State, Probability, PRECISION>(c);
}

fn ans_round_trip<Word, State, Probability, const PRECISION: usize>(c: &mut Criterion)
where
    Probability: BitArray,
    u32: AsPrimitive<Probability>,
    Word: BitArray + AsPrimitive<Probability> + From<Probability>,
    State: BitArray + AsPrimitive<Word> + AsPrimitive<usize> + From<Word>,
    u64: From<Probability>,
    usize: From<Probability> + AsPrimitive<Probability>,
{
    let (symbols, probabilities) = make_symbols_and_probabilities(PRECISION, 100);
    let encoder_model =
        EncoderHashLookupTable::<u16, Probability, PRECISION>::from_symbols_and_probabilities(
            symbols.iter().cloned().zip(probabilities),
        )
        .unwrap();

    let data = make_data(&symbols, 10_000);
    let mut encoder = AnsCoder::<Word, State>::new();

    let label_suffix = format!(
        "{}_{}_{}_{}",
        type_name::<Word>(),
        type_name::<State>(),
        type_name::<Probability>(),
        PRECISION
    );
    c.bench_function(&format!("ans_encoding_{}", label_suffix), |b| {
        b.iter(|| {
            encoder.clear();
            encoder
                .encode_iid_symbols_reverse(black_box(&data), &encoder_model)
                .unwrap();

            // Access `encoder.state()` and `encoder.bulk()` at an unpredictable position.
            let index = AsPrimitive::<usize>::as_(encoder.state()) % encoder.bulk().len();
            black_box(encoder.bulk()[index]);
        })
    });

    let decoder_model = encoder_model.to_decoder_model();

    let mut backward_decoder = encoder.into_seekable_decoder();
    let reset_snapshot = backward_decoder.pos();

    c.bench_function(&format!("ans_backward_decoding_{}", label_suffix), |b| {
        b.iter(|| {
            backward_decoder.seek(black_box(reset_snapshot)).unwrap();
            let mut checksum = 1234u16;
            for symbol in backward_decoder.decode_iid_symbols(data.len(), &decoder_model) {
                checksum ^= symbol.unwrap();
            }
            black_box(checksum);
        })
    });

    backward_decoder.seek(reset_snapshot).unwrap();
    let decoded = backward_decoder
        .decode_iid_symbols(data.len(), &encoder_model.to_decoder_model())
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    assert_eq!(decoded, data);
    assert!(backward_decoder.is_empty());

    backward_decoder.seek(reset_snapshot).unwrap();
    let mut forward_decoder = backward_decoder.into_reversed();
    let reset_snapshot = forward_decoder.pos();

    c.bench_function(&format!("ans_forward_decoding_{}", label_suffix), |b| {
        b.iter(|| {
            forward_decoder.seek(black_box(reset_snapshot)).unwrap();
            let mut checksum = 1234u16;
            for symbol in forward_decoder.decode_iid_symbols(data.len(), &decoder_model) {
                checksum ^= symbol.unwrap();
            }
            black_box(checksum);
        })
    });

    forward_decoder.seek(reset_snapshot).unwrap();
    let decoded = forward_decoder
        .decode_iid_symbols(data.len(), &encoder_model.to_decoder_model())
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    assert_eq!(decoded, data);
    assert!(forward_decoder.is_empty());
}

fn range_round_trip<Word, State, Probability, const PRECISION: usize>(c: &mut Criterion)
where
    Probability: BitArray,
    u32: AsPrimitive<Probability>,
    Word: BitArray + AsPrimitive<Probability> + From<Probability>,
    State: BitArray + AsPrimitive<Word> + AsPrimitive<usize> + From<Word>,
    u64: From<Probability>,
    usize: From<Probability> + AsPrimitive<Probability>,
{
    let (symbols, probabilities) = make_symbols_and_probabilities(PRECISION, 100);
    let encoder_model =
        EncoderHashLookupTable::<u16, Probability, PRECISION>::from_symbols_and_probabilities(
            symbols.iter().cloned().zip(probabilities),
        )
        .unwrap();

    let data = make_data(&symbols, 10_000);
    let mut encoder = RangeEncoder::<Word, State>::new();
    let reset_snapshot = encoder.pos();

    let label_suffix = format!(
        "{}_{}_{}_{}",
        type_name::<Word>(),
        type_name::<State>(),
        type_name::<Probability>(),
        PRECISION
    );
    c.bench_function(&format!("range_encoding_{}", label_suffix), |b| {
        b.iter(|| {
            encoder.clear();
            encoder
                .encode_iid_symbols(black_box(&data), &encoder_model)
                .unwrap();

            // Access `encoder.state()` and `encoder.bulk()` at an unpredictable position.
            let index = AsPrimitive::<usize>::as_(encoder.state().lower()) % encoder.bulk().len();
            black_box(encoder.bulk()[index]);
        })
    });

    let decoder_model = encoder_model.to_decoder_model();

    let mut decoder = encoder.into_decoder().unwrap();

    c.bench_function(&format!("range_decoding_{}", label_suffix), |b| {
        b.iter(|| {
            decoder.seek(black_box(reset_snapshot)).unwrap();
            let mut checksum = 1234u16;
            for symbol in decoder.decode_iid_symbols(data.len(), &decoder_model) {
                checksum ^= symbol.unwrap();
            }
            black_box(checksum);
        })
    });

    decoder.seek(reset_snapshot).unwrap();
    let decoded = decoder
        .decode_iid_symbols(data.len(), &encoder_model.to_decoder_model())
        .map(Result::unwrap)
        .collect::<Vec<_>>();
    assert_eq!(decoded, data);
    assert!(decoder.decoder_maybe_exhausted::<PRECISION>());
}

fn make_symbols_and_probabilities<Probability>(
    precision: usize,
    num_symbols: u32,
) -> (Vec<u16>, Vec<Probability>)
where
    Probability: BitArray + Into<u64>,
    u32: AsPrimitive<Probability>,
{
    assert!(precision <= Probability::BITS);
    assert!(num_symbols as u64 <= 1u64 << precision);

    let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
    let mut total_probability = 0;
    let bound_minus_one: u32 = ((2 * (1u64 << precision)) as f64 / num_symbols as f64 - 1.5) as u32;
    assert!(bound_minus_one > 0);

    // Try to get roughly the right total total amount, then fix up later.
    let (symbols, mut probabilities): (Vec<_>, Vec<_>) = (0..num_symbols)
        .map(|_| {
            let probability = ((rng.next_u32() % bound_minus_one) + 1).as_();
            total_probability += probability.into();
            let symbol = rng.next_u32() as u16;
            (symbol, probability)
        })
        .unzip();

    if total_probability < 1u64 << precision {
        let remaining = (1u64 << precision) - total_probability;
        for _ in 0..remaining {
            let index = rng.next_u32() % probabilities.len() as u32;
            let x = &mut probabilities[index as usize];
            *x = *x + Probability::one();
        }
    } else {
        let mut exceeding = total_probability - (1u64 << precision);
        while exceeding != 0 {
            let index = rng.next_u32() % probabilities.len() as u32;
            let x = &mut probabilities[index as usize];
            if *x > Probability::one() {
                *x = *x - Probability::one();
                exceeding -= 1;
            }
        }
    }

    (symbols, probabilities)
}

fn make_data<Symbol: Copy>(symbols: &[Symbol], amt: usize) -> Vec<Symbol> {
    let mut rng = Xoshiro256StarStar::seed_from_u64(5678 ^ amt as u64);
    (0..amt)
        .map(|_| symbols[(rng.next_u32() % symbols.len() as u32) as usize])
        .collect::<Vec<_>>()
}
