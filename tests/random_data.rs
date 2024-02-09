#![warn(rust_2018_idioms)]

use std::{
    cmp::max,
    cmp::min,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    ops::RangeInclusive,
};

use num_traits::AsPrimitive;
use probability::{
    distribution::{Gaussian, Sample},
    source::{Source, Xorshift128Plus},
};

use constriction::{
    stream::{model::LeakyQuantizer, queue, stack, Decode, Encode, IntoDecoder},
    BitArray,
};

fn make_random_normal(
    amt: usize,
    domain: RangeInclusive<i32>,
) -> impl DoubleEndedIterator<Item = (i32, Gaussian)> + Clone {
    let mut hasher = DefaultHasher::new();
    (amt as u64).hash(&mut hasher);

    (0..amt).map(move |i| {
        // Generate random numbers that can also be reproduced when iterating in reverse direction.
        let mut hasher = hasher.clone();
        (i as u64).hash(&mut hasher);
        let seed1 = hasher.finish();
        (amt as u64).hash(&mut hasher); // Add arbitrary additional data to make `seed2` different from `seed1`.
        let seed2 = hasher.finish();
        let mut rng = Xorshift128Plus::new([seed1, seed2]);

        let mean = 200.0 * rng.read_f64() - 100.0;
        let std_dev = 30.0 * rng.read_f64() + 0.01;
        let distribution = Gaussian::new(mean, std_dev);

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
    Probability: BitArray + Into<f64> + Into<Encoder::Word>,
    i32: AsPrimitive<Probability>,
    f64: AsPrimitive<Probability>,
    Encoder::Word: AsPrimitive<Probability>,
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
                let word_size = <<$stack_type as ::constriction::stream::Code>::Word as ::constriction::BitArray>::BITS;

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
        "{}; Probability={}; precision={}; amt={}: num_bits_stack={}; num_bits_queue={} ({} bits or {} words more than ANS coder)",
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
        #[cfg(not(miri))]
        1000,
        #[cfg(not(any(miri, debug_assertions)))]
        10000,
    ];

    for amt in amts.iter().cloned() {
        {
            batch!(stack::AnsCoder<u64, u128>; queue::RangeEncoder<u64, u128>; u32; 8, 12, 16, 24, 32; amt);
            batch!(stack::AnsCoder<u64, u128>; queue::RangeEncoder<u64, u128>; u16; 8, 12, 16; amt);
            batch!(stack::AnsCoder<u64, u128>; queue::RangeEncoder<u64, u128>; u8; 8; amt);

            batch!(stack::AnsCoder<u32, u128>; queue::RangeEncoder<u32, u128>; u32; 8, 12, 16, 24, 32; amt);
            batch!(stack::AnsCoder<u32, u128>; queue::RangeEncoder<u32, u128>; u16; 8, 12, 16; amt);
            batch!(stack::AnsCoder<u32, u128>; queue::RangeEncoder<u32, u128>; u8; 8; amt);

            batch!(stack::AnsCoder<u16, u128>; queue::RangeEncoder<u16, u128>; u16; 8, 12, 16; amt);
            batch!(stack::AnsCoder<u16, u128>; queue::RangeEncoder<u16, u128>; u8; 8; amt);

            batch!(stack::AnsCoder<u8, u128>; queue::RangeEncoder<u8, u128>; u8; 8; amt);
        }
        {
            batch!(stack::AnsCoder<u32, u64>; queue::RangeEncoder<u32, u64>; u32; 8, 12, 16, 24, 32; amt);
            batch!(stack::AnsCoder<u32, u64>; queue::RangeEncoder<u32, u64>; u16; 8, 12, 16; amt);
            batch!(stack::AnsCoder<u32, u64>; queue::RangeEncoder<u32, u64>; u8; 8; amt);

            batch!(stack::AnsCoder<u16, u64>; queue::RangeEncoder<u16, u64>; u16; 8, 12, 16; amt);
            batch!(stack::AnsCoder<u16, u64>; queue::RangeEncoder<u16, u64>; u8; 8; amt);

            batch!(stack::AnsCoder<u8, u64>; queue::RangeEncoder<u8, u64>; u8; 8; amt);
        }
        {
            batch!(stack::AnsCoder<u16, u32>; queue::RangeEncoder<u16, u32>; u16; 8, 12, 16; amt);
            batch!(stack::AnsCoder<u16, u32>; queue::RangeEncoder<u16, u32>; u8; 8; amt);

            batch!(stack::AnsCoder<u8, u32>; queue::RangeEncoder<u8, u32>; u8; 8; amt);
        }
        {
            batch!(stack::AnsCoder<u8, u16>; queue::RangeEncoder<u8, u16>; u8; 8; amt);
        }
    }
}
