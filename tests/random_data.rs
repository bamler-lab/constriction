#![feature(min_const_generics)]
#![warn(rust_2018_idioms)]

use std::{cmp::max, cmp::min, ops::RangeInclusive};

use num::cast::AsPrimitive;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use statrs::distribution::Normal;

use constriction::{models::LeakyQuantizer, queue, stack, BitArray, Decode, Encode, IntoDecoder};

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
    Encoder: Encode<PRECISION> + Default + IntoDecoder<PRECISION>,
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

macro_rules! batch {
    ($stack_type:ty; $queue_type:ty; $probability:ty; $($precision:expr),+; $amt:expr) => {
        {
            $({
                let num_bits_stack = test_normal::<$stack_type, $probability, _,  _, $precision, true>(
                    $amt, |encoder| encoder.num_bits()
                );
                let num_bits_queue = test_normal::<$queue_type, $probability, _,  _, $precision, false>(
                    $amt, |encoder| encoder.num_bits()
                );
                let coder_label = stringify!($stack_type);
                let probability_label = stringify!($probability);
                let word_size = <<$stack_type as ::constriction::Code>::CompressedWord as ::constriction::BitArray>::BITS;

                compare(
                    coder_label,
                    probability_label,
                    $precision,
                    $amt,
                    num_bits_stack,
                    num_bits_queue,
                    word_size
                );
            })+
        }
    }
}

fn compare(
    coder_label: &str,
    probability_label: &str,
    precision: usize,
    amt: usize,
    num_bits_stack: usize,
    num_bits_queue: usize,
    word_size: usize,
) {
    println!(
        "{}; Probability={}; precision={}; amt={}: num_bits_stack={}; num_bits_queue={} ({} bits or {} words more than stack)",
        &coder_label[7..],
        probability_label,
        precision,
        amt,
        num_bits_stack,
        num_bits_queue,
        num_bits_queue as isize - num_bits_stack as isize,
        (num_bits_queue as isize - num_bits_stack as isize) / word_size as isize
    );
}

#[test]
fn grid() {
    let amts = [
        10,
        100,
        1000,
        #[cfg(not(debug_assertions))]
        10000,
    ];

    for amt in amts.iter().cloned() {
        {
            batch!(stack::Stack<u64, u128>; queue::Encoder<u64, u128>; u32; 8, 12, 16, 24, 32; amt);
            batch!(stack::Stack<u64, u128>; queue::Encoder<u64, u128>; u16; 8, 12, 16; amt);
            batch!(stack::Stack<u64, u128>; queue::Encoder<u64, u128>; u8; 8; amt);

            batch!(stack::Stack<u32, u128>; queue::Encoder<u32, u128>; u32; 8, 12, 16, 24, 32; amt);
            batch!(stack::Stack<u32, u128>; queue::Encoder<u32, u128>; u16; 8, 12, 16; amt);
            batch!(stack::Stack<u32, u128>; queue::Encoder<u32, u128>; u8; 8; amt);

            batch!(stack::Stack<u16, u128>; queue::Encoder<u16, u128>; u16; 8, 12, 16; amt);
            batch!(stack::Stack<u16, u128>; queue::Encoder<u16, u128>; u8; 8; amt);

            batch!(stack::Stack<u8, u128>; queue::Encoder<u8, u128>; u8; 8; amt);
        }
        {
            batch!(stack::Stack<u32, u64>; queue::Encoder<u32, u64>; u32; 8, 12, 16, 24, 32; amt);
            batch!(stack::Stack<u32, u64>; queue::Encoder<u32, u64>; u16; 8, 12, 16; amt);
            batch!(stack::Stack<u32, u64>; queue::Encoder<u32, u64>; u8; 8; amt);

            batch!(stack::Stack<u16, u64>; queue::Encoder<u16, u64>; u16; 8, 12, 16; amt);
            batch!(stack::Stack<u16, u64>; queue::Encoder<u16, u64>; u8; 8; amt);

            batch!(stack::Stack<u8, u64>; queue::Encoder<u8, u64>; u8; 8; amt);
        }
        {
            batch!(stack::Stack<u16, u32>; queue::Encoder<u16, u32>; u16; 8, 12, 16; amt);
            batch!(stack::Stack<u16, u32>; queue::Encoder<u16, u32>; u8; 8; amt);

            batch!(stack::Stack<u8, u32>; queue::Encoder<u8, u32>; u8; 8; amt);
        }
        {
            batch!(stack::Stack<u8, u16>; queue::Encoder<u8, u16>; u8; 8; amt);
        }
    }
}
