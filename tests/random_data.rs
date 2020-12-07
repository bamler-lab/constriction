#![feature(min_const_generics)]
#![warn(rust_2018_idioms)]

use std::{cmp::max, cmp::min, ops::RangeInclusive};

use num::cast::AsPrimitive;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use statrs::distribution::Normal;

use constriction::{
    distributions::LeakyQuantizer, queue, stack, BitArray, Decode, Encode, IntoDecoder,
};

fn make_random_normal(
    amt: usize,
    domain: RangeInclusive<i32>,
) -> impl Iterator<Item = (i32, Normal)> + DoubleEndedIterator + Clone {
    (0..amt).map(move |i| {
        // Generate random numbers that can also be reproduced when iterating in reverse direction.
        let mut rng = Pcg64Mcg::new(i as u128);

        let mean = (200.0 / u32::MAX as f64) * rng.next_u32() as f64 - 100.0;
        let std_dev = (30.0 / u32::MAX as f64) * rng.next_u32() as f64 + 0.01;
        let distribution = Normal::new(mean, std_dev).expect("Parameters are valid.");

        let sample = distribution.sample(&mut rng).round() as i32;
        let symbol = max(min(sample, *domain.end()), *domain.start());

        (symbol, distribution)
    })
}

fn test_normal<Encoder, Probability, I, R, const PRECISION: usize, const REVERSE: bool>(
    amt: usize,
    inspect: I,
) -> R
where
    Encoder: Encode + Default + IntoDecoder,
    Probability: BitArray + Into<f64> + Into<Encoder::CompressedWord>,
    i32: AsPrimitive<Probability>,
    f64: AsPrimitive<Probability>,
    Encoder::CompressedWord: AsPrimitive<Probability>,
    I: FnOnce(&Encoder) -> R,
{
    const DOMAIN: RangeInclusive<i32> = -127..=127;
    let data = make_random_normal(amt, DOMAIN);
    let quantizer = LeakyQuantizer::<_, _, Probability, PRECISION>::new(DOMAIN);

    let mut encoder = Encoder::default();
    encoder
        .encode_symbols(
            data.clone()
                .map(|(symbol, distribution)| (symbol, quantizer.quantize(distribution))),
        )
        .unwrap();

    let result = inspect(&encoder);

    let mut decoder = encoder.into_decoder();

    if REVERSE {
        for (symbol, distribution) in data.rev() {
            let decoded = decoder
                .decode_symbol(quantizer.quantize(distribution))
                .unwrap();
            assert_eq!(decoded, symbol);
        }
    } else {
        for (symbol, distribution) in data {
            let decoded = decoder
                .decode_symbol(quantizer.quantize(distribution))
                .unwrap();
            assert_eq!(decoded, symbol);
        }
    }

    result
}

macro_rules! search_precision {
    ($encoder:ty; $probability:ty; $reverse:expr; $amt:expr; $($precision:expr),+) => {
        {
            [
                $(
                    test_normal::<$encoder, $probability, _,  _, $precision, $reverse>(
                        $amt, |encoder| encoder.num_bits()
                    )
                ),+
            ]
        }
    }
}

#[test]
fn grid() {
    fn compare(
        label: &str,
        precisions: &[usize],
        amt: usize,
        num_bits_stack: &[usize],
        num_bits_queue: &[usize],
    ) {
        for ((&num_bits_stack, &num_bits_queue), &precision) in
            num_bits_stack.iter().zip(num_bits_queue).zip(precisions)
        {
            println!(
                "{}; precision={}; amt={}: num_bits_stack={}; num_bits_queue={} ({} more than stack)",
                label,
                precision,
                amt,
                num_bits_stack,
                num_bits_queue,
                num_bits_queue as isize - num_bits_stack as isize,
            );
        }
    }

    for amt in [10, 100, 1000, 10000].iter().cloned() {
        {
            {
                let num_bits_stack =
                    search_precision!(stack::Coder<u32, u64>; u32; true; amt; 8, 12, 16, 24, 32);
                let num_bits_queue = search_precision!(queue::Encoder<u32, u64>; u32; false; amt; 8, 12, 16, 24, 32);
                compare(
                    "Coder<u32, u64>; Probability=u32",
                    &[8, 12, 16, 24, 32],
                    amt,
                    &num_bits_stack,
                    &num_bits_queue,
                );
            }
            {
                let num_bits_stack =
                    search_precision!(stack::Coder<u32, u64>; u16; true; amt; 8, 12, 16);
                let num_bits_queue =
                    search_precision!(queue::Encoder<u32, u64>; u16; false; amt; 8, 12, 16);
                compare(
                    "Coder<u32, u64>; Probability=u16",
                    &[8, 12, 16],
                    amt,
                    &num_bits_stack,
                    &num_bits_queue,
                );

                let num_bits_stack = search_precision!(stack::Coder<u32, u64>; u8; true; amt; 8);
                let num_bits_queue =
                    search_precision!(queue::Encoder<u32, u64>; u8; false; amt; 8);
                compare(
                    "Coder<u32, u64>; Probability=u8",
                    &[8],
                    amt,
                    &num_bits_stack,
                    &num_bits_queue,
                );
            }
            {
                let num_bits_stack =
                    search_precision!(stack::Coder<u16, u64>; u16; true; amt; 8, 12, 16);
                let num_bits_queue =
                    search_precision!(queue::Encoder<u16, u64>; u16; false; amt; 8, 12, 16);
                compare(
                    "Coder<u16, u64>; Probability=u16",
                    &[8, 12, 16],
                    amt,
                    &num_bits_stack,
                    &num_bits_queue,
                );

                let num_bits_stack = search_precision!(stack::Coder<u16, u64>; u8; true; amt; 8);
                let num_bits_queue =
                    search_precision!(queue::Encoder<u16, u64>; u8; false; amt; 8);
                compare(
                    "Coder<u16, u64>; Probability=u8",
                    &[8],
                    amt,
                    &num_bits_stack,
                    &num_bits_queue,
                );
            }
            {
                let num_bits_stack = search_precision!(stack::Coder<u8, u64>; u8; true; amt; 8);
                let num_bits_queue =
                    search_precision!(queue::Encoder<u8, u64>; u8; false; amt; 8);
                compare(
                    "Coder<u8, u64>; Probability=u8",
                    &[8],
                    amt,
                    &num_bits_stack,
                    &num_bits_queue,
                );
            }
            {
                {
                    let num_bits_stack =
                        search_precision!(stack::Coder<u16, u32>; u16; true; amt; 8, 12, 16);
                    let num_bits_queue =
                        search_precision!(queue::Encoder<u16, u32>; u16; false; amt; 8, 12, 16);
                    compare(
                        "Coder<u16, u32>; Probability=u16",
                        &[8, 12, 16],
                        amt,
                        &num_bits_stack,
                        &num_bits_queue,
                    );

                    let num_bits_stack =
                        search_precision!(stack::Coder<u16, u32>; u8; true; amt; 8);
                    let num_bits_queue =
                        search_precision!(queue::Encoder<u16, u32>; u8; false; amt; 8);
                    compare(
                        "Coder<u16, u32>; Probability=u8",
                        &[8],
                        amt,
                        &num_bits_stack,
                        &num_bits_queue,
                    );
                }
                {
                    let num_bits_stack =
                        search_precision!(stack::Coder<u8, u32>; u8; true; amt; 8);
                    let num_bits_queue =
                        search_precision!(queue::Encoder<u8, u32>; u8; false; amt; 8);
                    compare(
                        "Coder<u8, u32>; Probability=u8",
                        &[8],
                        amt,
                        &num_bits_stack,
                        &num_bits_queue,
                    );
                }
            }
            {
                {
                    let num_bits_stack =
                        search_precision!(stack::Coder<u8, u16>; u8; true; amt; 8);
                    let num_bits_queue =
                        search_precision!(queue::Encoder<u8, u16>; u8; false; amt; 8);
                    compare(
                        "Coder<u8, u16>; Probability=u8",
                        &[8],
                        amt,
                        &num_bits_stack,
                        &num_bits_queue,
                    );
                }
            }
        }
    }
}
