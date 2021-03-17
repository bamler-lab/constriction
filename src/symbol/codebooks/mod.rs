pub mod exp_golomb;
pub mod huffman;

use core::{
    borrow::Borrow,
    convert::Infallible,
    fmt::{Debug, Display},
};

use smallvec::SmallVec;

use crate::{CoderError, EncoderError, UnwrapInfallible};

use super::{StackCoder, WriteBitStream};

type SmallBitStack = StackCoder<usize, SmallVec<[usize; 1]>>;

#[derive(Debug)]
pub enum SymbolCodeError<InvalidCodeword = Infallible> {
    /// The compressed data ended before the current codeword was complete.
    OutOfCompressedData,

    /// Found a code word that does not map to any symbol.
    InvalidCodeword(InvalidCodeword),
}

impl<InvalidCodeword> SymbolCodeError<InvalidCodeword> {
    pub fn into_coder_error<BackendError>(self) -> CoderError<Self, BackendError> {
        CoderError::Frontend(self)
    }
}

impl<InvalidCodeword: Display> Display for SymbolCodeError<InvalidCodeword> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfCompressedData => write!(
                f,
                "The compressed data ended before the current codeword was complete."
            ),
            Self::InvalidCodeword(err) => write!(f, "Invalid codeword for this codebook: {}", err),
        }
    }
}

#[cfg(feature = "std")]
impl<InvalidCodeword: std::error::Error + 'static> std::error::Error
    for SymbolCodeError<InvalidCodeword>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OutOfCompressedData => None,
            Self::InvalidCodeword(source) => Some(source),
        }
    }
}

pub trait Codebook {
    type Symbol;
}

pub trait EncoderCodebook: Codebook {
    fn encode_symbol_prefix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        mut emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), EncoderError<BackendError>> {
        let mut reverse_codeword = SmallBitStack::new();
        self.encode_symbol_suffix(symbol, |bit| reverse_codeword.write_bit(bit))
            .map_err(|err| CoderError::Frontend(err.into_frontend_error()))?;

        for bit in reverse_codeword {
            emit(bit.unwrap_infallible())?;
        }
        Ok(())
    }

    fn encode_symbol_suffix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        mut emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), EncoderError<BackendError>> {
        let mut reverse_codeword = SmallBitStack::new();
        self.encode_symbol_prefix(symbol, |bit| reverse_codeword.write_bit(bit))
            .map_err(|err| CoderError::Frontend(err.into_frontend_error()))?;

        for bit in reverse_codeword {
            emit(bit.unwrap_infallible())?;
        }
        Ok(())
    }
}

pub trait DecoderCodebook: Codebook {
    type InvalidCodeword;

    fn decode_symbol<BackendError>(
        &self,
        source: impl Iterator<Item = Result<bool, BackendError>>,
    ) -> Result<Self::Symbol, CoderError<SymbolCodeError<Self::InvalidCodeword>, BackendError>>;
}

impl<C: Codebook> Codebook for &C {
    type Symbol = C::Symbol;
}

impl<C: EncoderCodebook> EncoderCodebook for &C {
    #[inline(always)]
    fn encode_symbol_prefix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), EncoderError<BackendError>> {
        (*self).encode_symbol_prefix(symbol, emit)
    }

    #[inline(always)]
    fn encode_symbol_suffix<BackendError>(
        &self,
        symbol: impl Borrow<Self::Symbol>,
        emit: impl FnMut(bool) -> Result<(), BackendError>,
    ) -> Result<(), EncoderError<BackendError>> {
        (*self).encode_symbol_suffix(symbol, emit)
    }
}

impl<C: DecoderCodebook> DecoderCodebook for &C {
    type InvalidCodeword = C::InvalidCodeword;

    fn decode_symbol<BackendError>(
        &self,
        source: impl Iterator<Item = Result<bool, BackendError>>,
    ) -> Result<Self::Symbol, CoderError<SymbolCodeError<Self::InvalidCodeword>, BackendError>>
    {
        (*self).decode_symbol(source)
    }
}
