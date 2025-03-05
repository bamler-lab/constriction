use core::{borrow::Borrow, marker::PhantomData};

use num_traits::{PrimInt, Unsigned, WrappingAdd, WrappingSub};

use super::{Codebook, DecoderCodebook, EncoderCodebook, SymbolCodeError};
use crate::{CoderError, DefaultEncoderError};

/// Zero-sized marker struct for Exponential-Golomb Coding.
///
/// See [Wikipedia](https://en.wikipedia.org/wiki/Exponential-Golomb_coding).
///
/// This is not a super practical code. It's provided here mainly to show that the symbol
/// code traits can be used for codes with (essentially) infinitely sized codebooks.
///
/// # Example
///
/// ```
/// use constriction::{
///     symbol::{DefaultQueueEncoder, WriteBitStream, ReadBitStream},
///     symbol::exp_golomb::ExpGolomb,
///     UnwrapInfallible,
/// };
///
/// let codebook = ExpGolomb::<u32>::new();
/// let mut encoder = DefaultQueueEncoder::new();
/// encoder.encode_iid_symbols(&[3, 7, 0, 1], &codebook).unwrap();
/// let mut decoder = encoder.into_decoder().unwrap_infallible();
/// let bit_string = decoder.clone().map(
///     |bit| if bit.unwrap_infallible() { '1' } else { '0' }
/// ).collect::<String>();
///
/// // (Note that the `DefaultQueueEncoder` pads to full words with zeros.
/// // This is not be a problem since we're using a prefix code.)
/// assert_eq!(bit_string, "00100000100010100000000000000000");
///
/// let decoded = decoder.decode_iid_symbols(4, &codebook).collect::<Result<Vec<_>, _>>().unwrap();
/// assert_eq!(decoded, [3, 7, 0, 1]);
/// ```
#[derive(Debug, Clone)]
pub struct ExpGolomb<N> {
    phantom: PhantomData<N>,
}

impl<N> ExpGolomb<N> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<N> Default for ExpGolomb<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N> Codebook for ExpGolomb<N> {
    type Symbol = N;
}

impl<N: Unsigned + PrimInt + WrappingAdd + WrappingSub> EncoderCodebook for ExpGolomb<N> {
    fn encode_symbol_prefix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        mut emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), DefaultEncoderError<BackendError>> {
        let n_plus1 = symbol.borrow().wrapping_add(&N::one());

        if n_plus1 == N::zero() {
            let len = N::zero().count_zeros();
            for _ in 0..len {
                emit(false)?;
            }
            emit(true)?;
            for _ in 0..len {
                emit(false)?;
            }
        } else {
            let len = N::zero().count_zeros() - n_plus1.leading_zeros() - 1;
            for _ in 0..len {
                emit(false)?;
            }
            let mut mask = N::one() << len as usize;
            while mask != N::zero() {
                emit(n_plus1 & mask != N::zero())?;
                mask = mask >> 1;
            }
        }

        Ok(())
    }

    fn encode_symbol_suffix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        mut emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), DefaultEncoderError<BackendError>> {
        let n_plus1 = symbol.borrow().wrapping_add(&N::one());

        if n_plus1 == N::zero() {
            let len = N::zero().count_zeros();
            for _ in 0..len {
                emit(false)?;
            }
            emit(true)?;
            for _ in 0..len {
                emit(false)?;
            }
        } else {
            let len = N::zero().count_zeros() - n_plus1.leading_zeros() - 1;
            let mut remaining = n_plus1;
            loop {
                emit(remaining & N::one() != N::zero())?;
                remaining = remaining >> 1;
                if remaining == N::zero() {
                    break;
                }
            }
            for _ in 0..len {
                emit(false)?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidCodeword;

impl<N: Unsigned + PrimInt + WrappingAdd + WrappingSub> DecoderCodebook for ExpGolomb<N> {
    type InvalidCodeword = InvalidCodeword;

    fn decode_symbol<BackendError>(
        &self,
        mut source: impl Iterator<Item = Result<bool, BackendError>>,
    ) -> Result<Self::Symbol, CoderError<SymbolCodeError<Self::InvalidCodeword>, BackendError>>
    {
        let mut len = 0u32;
        loop {
            match source.next().transpose()? {
                Some(false) => len += 1,
                Some(true) => break,
                None => {
                    return Err(
                        SymbolCodeError::InvalidCodeword(InvalidCodeword).into_coder_error()
                    );
                }
            }
        }

        if len > N::max_value().count_ones() {
            return Err(SymbolCodeError::InvalidCodeword(InvalidCodeword).into_coder_error());
        }

        let mut n_plus1 = N::one();
        for _ in 0..len {
            if let Some(bit) = source.next().transpose()? {
                n_plus1 = (n_plus1 << 1) | if bit { N::one() } else { N::zero() };
            } else {
                return Err(SymbolCodeError::InvalidCodeword(InvalidCodeword).into_coder_error());
            }
        }

        if len == N::max_value().count_ones() && n_plus1 != N::zero() {
            Err(SymbolCodeError::InvalidCodeword(InvalidCodeword).into_coder_error())
        } else {
            Ok(n_plus1.wrapping_sub(&N::one()))
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{RngCore, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;

    use super::{
        super::{DefaultQueueEncoder, DefaultStackCoder, ReadBitStream, WriteBitStream},
        *,
    };
    use crate::UnwrapInfallible;

    use alloc::vec::Vec;
    use core::convert::Infallible;

    #[test]
    fn explicit_examples() {
        fn test_one<N: Unsigned + PrimInt + WrappingAdd + WrappingSub>(symbol: N, expected: &[u8]) {
            let codebook = ExpGolomb::new();
            let mut index = 0;
            codebook
                .encode_symbol_prefix(symbol, |bit| {
                    assert_eq!(expected[index], if bit { b'1' } else { b'0' });
                    index += 1;
                    Result::<_, Infallible>::Ok(())
                })
                .unwrap();
            assert_eq!(index, expected.len());
            codebook
                .encode_symbol_suffix(symbol, |bit| {
                    index -= 1;
                    assert_eq!(expected[index], if bit { b'1' } else { b'0' });
                    Result::<_, Infallible>::Ok(())
                })
                .unwrap();
            assert_eq!(index, 0);
        }

        test_one(0u32, b"1");
        test_one(1u32, b"010");
        test_one(2u32, b"011");
        test_one(3u32, b"00100");
        test_one(4u32, b"00101");
        test_one(5u32, b"00110");
        test_one(6u32, b"00111");
        test_one(7u32, b"0001000");

        test_one(
            u32::MAX - 2,
            b"000000000000000000000000000000011111111111111111111111111111110",
        );
        test_one(
            u32::MAX - 1,
            b"000000000000000000000000000000011111111111111111111111111111111",
        );
        test_one(
            u32::MAX,
            b"00000000000000000000000000000000100000000000000000000000000000000",
        );
    }

    #[test]
    fn encode_decode_iid_queue() {
        let amt = 1000;
        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        let symbols = (0..amt).map(|_| rng.next_u32() % 8).collect::<Vec<_>>();
        let codebook = ExpGolomb::<u32>::new();

        let mut encoder = DefaultQueueEncoder::new();
        encoder.encode_iid_symbols(&symbols, &codebook).unwrap();
        let mut decoder = encoder.into_decoder().unwrap_infallible();
        let reconstructed = decoder
            .decode_iid_symbols(amt, &codebook)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(reconstructed, symbols);
        assert!(decoder.maybe_exhausted());
    }

    #[test]
    fn encode_decode_iid_stack() {
        let amt = 1000;
        let mut rng = Xoshiro256StarStar::seed_from_u64(123);
        let symbols = (0..amt).map(|_| rng.next_u32() % 8).collect::<Vec<_>>();
        let codebook = ExpGolomb::<u32>::new();

        let mut coder = DefaultStackCoder::new();
        coder
            .encode_iid_symbols_reverse(&symbols, &codebook)
            .unwrap();
        let reconstructed = coder
            .decode_iid_symbols(amt, &codebook)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(reconstructed, symbols);
        assert!(coder.is_empty());
    }
}
