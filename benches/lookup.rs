use std::any::type_name;

use constriction::{
    stream::{ans::AnsCoder, models::lookup::EncoderHashLookupTable, Code, Decode, Pos, Seek},
    BitArray,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num::cast::AsPrimitive;
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;

criterion_group!(
    benches,
    round_trip_u16_u32_u8_8,
    round_trip_u16_u32_u16_8,
    round_trip_u16_u32_u16_12
);
criterion_main!(benches);

fn round_trip_u16_u32_u8_8(c: &mut Criterion) {
    round_trip_u16_u32_ux_y::<u8, 8>(c)
}

fn round_trip_u16_u32_u16_8(c: &mut Criterion) {
    round_trip_u16_u32_ux_y::<u16, 8>(c)
}

fn round_trip_u16_u32_u16_12(c: &mut Criterion) {
    round_trip_u16_u32_ux_y::<u16, 12>(c)
}

fn round_trip_u16_u32_ux_y<Probability, const PRECISION: usize>(c: &mut Criterion)
where
    Probability: BitArray,
    u32: AsPrimitive<Probability>,
    u16: AsPrimitive<Probability> + From<Probability>,
    u64: From<Probability>,
    usize: From<Probability> + AsPrimitive<Probability>,
{
    let (symbols, probabilities) = make_symbols_and_probabilities(PRECISION, 100);
    let encoder_model = EncoderHashLookupTable::<u16, Probability, PRECISION>::new(
        symbols.iter().cloned().zip(probabilities),
    )
    .unwrap();

    let data = make_data(&symbols, 10_000);
    let mut encoder = AnsCoder::<u16, u32>::new();

    let label_suffix = format!("u16_u32_{}_{}", type_name::<Probability>(), PRECISION);
    c.bench_function(&format!("encoding_{}", label_suffix), |b| {
        b.iter(|| {
            encoder.clear();
            encoder
                .encode_iid_symbols_reverse(black_box(&data), &encoder_model)
                .unwrap();

            // Access `ans.state()` and `ans.buf()` at an unpredictable position.
            let index = encoder.state() as usize % encoder.buf().len();
            black_box(encoder.buf()[index]);
        })
    });

    let decoder_model = encoder_model.to_decoder_model();

    let mut backward_decoder = encoder.into_seekable_decoder();
    let reset_snapshot = backward_decoder.pos_and_state();

    c.bench_function(&format!("backward_decoding_{}", label_suffix), |b| {
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
    let reset_snapshot = forward_decoder.pos_and_state();

    c.bench_function(&format!("forward_decoding_{}", label_suffix), |b| {
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
