pub mod ans;
pub mod models;
pub mod range;

#[cfg(feature = "std")]
use std::error::Error;

use core::borrow::Borrow;

use crate::{BitArray, EncodingError};
use models::{DecoderModel, EncoderModel, EntropyModel};
use num::cast::AsPrimitive;

pub trait Code {
    type CompressedWord: BitArray;
    type State: Clone;

    /// Returns the current internal state of the coder.
    ///
    /// This method is usually used together with [`Seek::seek`].
    fn state(&self) -> Self::State;

    /// Check if there might be no compressed data available.
    ///
    /// This method is useful to check for consistency, e.g., when decoding data with a
    /// [`Decode`]. This method returns `true` if there *might* not be any compressed
    /// data. This can have several causes, e.g.:
    /// - the method is called on a newly constructed empty encoder or decoder; or
    /// - the user is decoding in-memory data and called `maybe_empty` after decoding
    ///   all available compressed data; or
    /// - it is unknown whether there is any compressed data left.
    ///
    /// The last item in the above list deserves further explanation. It is not always
    /// possible to tell whether any compressed data is available. For example, when
    /// encoding data onto or decoding data from a stream (like a network socket), then
    /// the coder is not required to keep track of whether or not any compressed data
    /// has already been emitted or can still be received, respectively. In such a case,
    /// when it is not known whether any compressed data is available, `maybe_empty`
    /// *must* return `true`.
    ///
    /// The contrapositive of the above requirement is that, when `maybe_empty` returns
    /// `false`, then some compressed data is definitely available. Therefore,
    /// `maybe_empty` can used to check for data corruption: if the user of this library
    /// believes that they have decoded all available compressed data but `maybe_empty`
    /// returns `false`, then the decoded data must have been corrupted. However, the
    /// converse is not true: if `maybe_empty` returns `true` then this is not
    /// necessarily a particularly strong guarantee of data integrity.
    ///
    /// Note that it is always legal to call [`decode_symbol`] even on an empty
    /// [`Decode`]. Some decoder implementations may even always return and `Ok(_)`
    /// value (with an arbitrary but deterministically constructed wrapped symbol) even
    /// if the decoder is empty,
    ///
    /// # Implementation Guide
    ///
    /// The default implementation always returns `true` since this is always a *valid*
    /// (albeit not necessarily the most useful) return value. If you overwrite this
    /// method, you may return `false` only if there is definitely some compressed data
    /// available. When in doubt, return `true`.
    ///
    /// [`decode_symbol`]: Decode::decode_symbol
    fn maybe_empty(&self) -> bool {
        true
    }
}

pub trait Encode<const PRECISION: usize>: Code {
    fn encode_symbol<D>(
        &mut self,
        symbol: impl Borrow<D::Symbol>,
        model: D,
    ) -> Result<(), EncodingError>
    where
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>;

    fn encode_symbols<S, D>(
        &mut self,
        symbols_and_models: impl IntoIterator<Item = (S, D)>,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        for (symbol, model) in symbols_and_models.into_iter() {
            self.encode_symbol(symbol, model)?;
        }

        Ok(())
    }

    fn try_encode_symbols<S, D, E>(
        &mut self,
        symbols_and_models: impl IntoIterator<Item = Result<(S, D), E>>,
    ) -> Result<(), TryCodingError<EncodingError, E>>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        for symbol_and_model in symbols_and_models.into_iter() {
            let (symbol, model) =
                symbol_and_model.map_err(|err| TryCodingError::InvalidEntropyModel(err))?;
            self.encode_symbol(symbol, model)?;
        }

        Ok(())
    }

    fn encode_iid_symbols<S, D>(
        &mut self,
        symbols: impl IntoIterator<Item = S>,
        model: &D,
    ) -> Result<(), EncodingError>
    where
        S: Borrow<D::Symbol>,
        D: EncoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        self.encode_symbols(symbols.into_iter().map(|symbol| (symbol, model)))
    }
}

pub trait Decode<const PRECISION: usize>: Code {
    /// The error type for [`decode_symbol`].
    ///
    /// This is an associated type because [`decode_symbol`] is infallible for some
    /// decoders (e.g., for a [`Ans`]). These decoders set the `DecodingError`
    /// type to [`core::convert::Infallible`] so that the compiler can optimize away
    /// error checks.
    ///
    /// [`decode_symbol`]: #tymethod.decode_symbol
    /// [`Ans`]: ans.Ans
    #[cfg(not(feature = "std"))]
    type DecodingError: Debug;

    /// The error type for [`decode_symbol`].
    ///
    /// This is an associated type because [`decode_symbol`] is infallible for some
    /// decoders (e.g., for a [`Ans`]). These decoders set the `DecodingError`
    /// type to [`core::convert::Infallible`] so that the compiler can optimize away
    /// error checks.
    ///
    /// [`decode_symbol`]: #tymethod.decode_symbol
    /// [`Ans`]: ans.Ans
    #[cfg(feature = "std")]
    type DecodingError: Error;

    fn decode_symbol<D>(&mut self, model: D) -> Result<D::Symbol, Self::DecodingError>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>;

    /// TODO: This would be much nicer to denote as
    /// `fn decode_symbols(...) -> impl Iterator`
    /// but existential return types are currently not allowed in trait methods.
    fn decode_symbols<'s, I, D>(
        &'s mut self,
        models: I,
    ) -> DecodeSymbols<'s, Self, I::IntoIter, PRECISION>
    where
        I: IntoIterator<Item = D> + 's,
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        DecodeSymbols {
            decoder: self,
            models: models.into_iter(),
        }
    }

    fn try_decode_symbols<'s, I, D, E>(
        &'s mut self,
        models: I,
    ) -> TryDecodeSymbols<'s, Self, I::IntoIter, PRECISION>
    where
        I: IntoIterator<Item = Result<D, E>> + 's,
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        TryDecodeSymbols {
            decoder: self,
            models: models.into_iter(),
        }
    }

    /// Decode a sequence of symbols using a fixed entropy model.
    ///
    /// This is a convenience wrapper around [`decode_symbols`], and the inverse of
    /// [`encode_iid_symbols`]. See example in the latter.
    ///
    /// [`decode_symbols`]: #method.decode_symbols
    /// [`encode_iid_symbols`]: #method.encode_iid_symbols
    fn decode_iid_symbols<'s, D>(
        &'s mut self,
        amt: usize,
        model: &'s D,
    ) -> DecodeIidSymbols<'s, Self, D, PRECISION>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
    {
        DecodeIidSymbols {
            decoder: self,
            model,
            amt,
        }
    }

    /// Decodes multiple symbols driven by a provided iterator.
    ///
    /// Iterates over all `x` in `iterator.into_iter()`. In each iteration, it decodes
    /// a symbol using the entropy model `model`, terminates on error and otherwise
    /// calls `callback(x, decoded_symbol)`.
    ///
    /// The default implementation literally just has a `for` loop over
    /// `iterator.into_iter()`, which contains a single line of code in its body. But,
    /// in a real-world example, it turns out to be significantly faster this way than
    /// if we would write the same `for` loop at the call site. This may hint at some
    /// issues with method inlining.
    fn map_decode_iid_symbols<'s, D, I>(
        &'s mut self,
        iterator: I,
        model: &'s D,
        mut callback: impl FnMut(I::Item, D::Symbol),
    ) -> Result<(), Self::DecodingError>
    where
        D: DecoderModel<PRECISION>,
        D::Probability: Into<Self::CompressedWord>,
        Self::CompressedWord: AsPrimitive<D::Probability>,
        I: IntoIterator,
    {
        for x in iterator.into_iter() {
            callback(x, self.decode_symbol(model)?)
        }

        Ok(())
    }
}

/// A trait for conversion into matching decoder type.
///
/// This is useful for generic code that encodes some data with a user provided
/// encoder of generic type, and then needs to obtain a compatible decoder.
///
/// This trait is similar to [`AsDecoder`], except that the conversion takes
/// ownership of `self` (typically an encoder). This means that the calling
/// function may return the resulting decoder or put it on the heap since it will
/// (typically) be free of any references into the current stack frame.
///
/// If you don't have ownership of the original encoder, or you want to reuse the
/// original encoder once you no longer need the returned decoder, then consider
/// using [`AsDecoder`] instead.
///
/// # Example
///
/// To be able to convert an encoder of generic type `Encoder` into a decoder,
/// declare a trait bound `Encoder: IntoDecoder<PRECISION>`. In the following
/// example, differences to the example for [`AsDecoder`] are marked by `// <--`.
///
/// ```
/// # #![feature(min_const_generics)]
/// # use constriction::stream::{
/// #     models::{EncoderModel, DecoderModel, LeakyQuantizer},
/// #     ans::DefaultAns,
/// #     Decode, Encode, IntoDecoder
/// # };
/// #
/// fn encode_and_decode<Encoder, D, const PRECISION: usize>(
///     mut encoder: Encoder, // <-- Needs ownership of `encoder`.
///     model: D
/// ) -> Encoder::IntoDecoder
/// where
///     Encoder: Encode<PRECISION> + IntoDecoder<PRECISION>, // <-- Different trait bound.
///     D: EncoderModel<PRECISION, Symbol=i32> + DecoderModel<PRECISION, Symbol=i32>,
///     D::Probability: Into<Encoder::CompressedWord>,
///     Encoder::CompressedWord: num::cast::AsPrimitive<D::Probability>
/// {
///     encoder.encode_symbol(137, &model);
///     let mut decoder = encoder.into_decoder();
///     let decoded = decoder.decode_symbol(&model).unwrap();
///     assert_eq!(decoded, 137);
///
///     // encoder.encode_symbol(42, &model); // <-- This would fail (we moved `encoder`).
///     decoder // <-- We can return `decoder` as it has no references to the current stack frame.
/// }
///
/// // Usage example:
/// let encoder = DefaultAns::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=200);
/// let model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 50.0).unwrap());
/// encode_and_decode(encoder, model);
/// ```
///
pub trait IntoDecoder<const PRECISION: usize>: Code + Sized {
    /// The target type of the conversion.
    ///
    /// This is the important part of the `IntoDecoder` trait. The actual conversion in
    /// [`into_decoder`] just delegates to `self.into()`. From a caller's perspective,
    /// the advantage of using the `IntoDecoder` trait rather than directly calling
    /// `self.into()` is that `IntoDecoder::IntoDecoder` defines a suitable return type
    /// so the caller doesn't need to specify one.
    ///
    /// [`into_decoder`]: Self::into_decoder
    type IntoDecoder: From<Self>
        + Code<CompressedWord = Self::CompressedWord, State = Self::State>
        + Decode<PRECISION>;

    /// Performs the conversion.
    ///
    /// The default implementation delegates to `self.into()`. There is usually no
    /// reason to overwrite the default implementation.
    fn into_decoder(self) -> Self::IntoDecoder {
        self.into()
    }
}

impl<Decoder: Decode<PRECISION>, const PRECISION: usize> IntoDecoder<PRECISION> for Decoder {
    type IntoDecoder = Self;
}

/// A trait for constructing a temporary matching decoder.
///
/// This is useful for generic code that encodes some data with a user provided
/// encoder of generic type, and then needs to obtain a compatible decoder.
///
/// This trait is similar to [`IntoDecoder`], but it has the following advantages
/// over it:
/// - it doesn't need ownership of `self` (typically an encoder); and
/// - `self` can be used again once the returned decoder is no longer used.
///
/// The disadvantage of `AsDecoder` compared to `IntoDecoder` is that the returned
/// decoder cannot outlive `self`, so it typically cannot be returned from the
/// calling function or put on the heap. If this would pose a problem for your use
/// case then use [`IntoDecoder`] instead.
///
/// # Example
///
/// To be able to temporarily convert an encoder of generic type `Encoder` into a
/// decoder, declare a trait bound `for<'a> Encoder: AsDecoder<'a, PRECISION>`. In
/// the following example, differences to the example for [`IntoDecoder`] are marked
/// by `// <--`.
///
/// ```
/// # #![feature(min_const_generics)]
/// # use constriction::stream::{
/// #     models::{EncoderModel, DecoderModel, LeakyQuantizer},
/// #     ans::DefaultAns,
/// #     Decode, Encode, AsDecoder
/// # };
/// #
/// fn encode_decode_encode<Encoder, D, const PRECISION: usize>(
///     encoder: &mut Encoder, // <-- Doesn't need ownership of `encoder`
///     model: D
/// )
/// where
///     Encoder: Encode<PRECISION>,
///     for<'a> Encoder: AsDecoder<'a, PRECISION>, // <-- Different trait bound.
///     D: EncoderModel<PRECISION, Symbol=i32> + DecoderModel<PRECISION, Symbol=i32>,
///     D::Probability: Into<Encoder::CompressedWord>,
///     Encoder::CompressedWord: num::cast::AsPrimitive<D::Probability>
/// {
///     encoder.encode_symbol(137, &model);
///     let mut decoder = encoder.as_decoder();
///     let decoded = decoder.decode_symbol(&model).unwrap(); // (Doesn't mutate `encoder`.)
///     assert_eq!(decoded, 137);
///
///     std::mem::drop(decoder); // <-- We have to explicitly drop `decoder` ...
///     encoder.encode_symbol(42, &model); // <-- ... before we can use `encoder` again.
/// }
///
/// // Usage example:
/// let mut encoder = DefaultAns::new();
/// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(0..=200);
/// let model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 50.0).unwrap());
/// encode_decode_encode(&mut encoder, model);
/// ```
pub trait AsDecoder<'a, const PRECISION: usize>: Code + Sized + 'a {
    /// The target type of the conversion.
    ///
    /// This is the important part of the `AsDecoder` trait. The actual conversion in
    /// [`as_decoder`] just delegates to `self.into()`. From a caller's perspective, the
    /// advantage of using the `AsDecoder` trait rather than directly calling
    /// `self.into()` is that `AsDecoder::AsDecoder` defines a suitable return type so
    /// the caller doesn't need to specify one.
    ///
    /// [`as_decoder`]: Self::as_decoder
    type AsDecoder: From<&'a Self>
        + Code<CompressedWord = Self::CompressedWord, State = Self::State>
        + Decode<PRECISION>
        + 'a;

    /// Performs the conversion.
    ///
    /// The default implementation delegates to `self.into()`. There is usually no
    /// reason to overwrite the default implementation.
    fn as_decoder(&'a self) -> Self::AsDecoder {
        self.into()
    }
}

/// A trait for entropy coders that keep track of their current position within the
/// compressed data.
///
/// This is the counterpart of [`Seek`]. Call [`Pos::pos_and_state`] to record
/// "snapshots" of an entropy coder, and then call [`Seek::seek`] at a later time
/// to jump back to these snapshots. See examples in the documentations of [`Seek`]
/// and [`Seek::seek`].
pub trait Pos: Code {
    /// Returns the position in the compressed data, in units of `CompressedWord`s.
    ///
    /// It is up to the entropy coder to define what constitutes the beginning and end
    /// positions within the compressed data (for example, a [`Ans`] begins encoding
    /// at position zero but it begins decoding at position `ans.buf().len()`).
    ///
    /// [`Ans`]: ans::Ans
    fn pos(&self) -> usize;

    /// Convenience method that returns both parts of a snapshot expected by
    /// [`Seek::seek`].
    ///
    /// The default implementation just delegates to [`Pos::pos`] and [`Code::state`].
    /// See documentation of [`Seek::seek`] for usage examples.
    fn pos_and_state(&self) -> (usize, Self::State) {
        (self.pos(), self.state())
    }
}

/// A trait for entropy coders that support random access.
///
/// This is the counterpart of [`Pos`]. While [`Pos::pos_and_state`] can be used to
/// record "snapshots" of an entropy coder, [`Seek::seek`] can be used to jump to these
/// recorded snapshots.
///
/// Not all entropy coders that implement `Pos` also implement `Seek`. For example,
/// [`DefaultAns`] implements `Pos` but it doesn't implement `Seek` because it
/// supports both encoding and decoding and therefore always operates at the head. In
/// such a case one can usually obtain a seekable entropy coder in return for
/// surrendering some other property. For example, `DefaultAns` provides the methods
/// [`seekable_decoder`] and [`into_seekable_decoder`] that return a decoder which
/// implements `Seek` but which can no longer be used for encoding (i.e., it doesn't
/// implement [`Encode`]).
///
/// # Example
///
/// ```
/// use constriction::stream::{models::Categorical, ans::DefaultAns, Decode, Pos, Seek};
///
/// // Create a `Ans` encoder and an entropy model:
/// let mut ans = DefaultAns::new();
/// let probabilities = vec![0.03, 0.07, 0.1, 0.1, 0.2, 0.2, 0.1, 0.15, 0.05];
/// let entropy_model = Categorical::<u32, 24>::from_floating_point_probabilities(&probabilities)
///     .unwrap();
///
/// // Encode some symbols in two chunks and take a snapshot after each chunk.
/// let symbols1 = vec![8, 2, 0, 7];
/// ans.encode_iid_symbols_reverse(&symbols1, &entropy_model).unwrap();
/// let snapshot1 = ans.pos_and_state();
///
/// let symbols2 = vec![3, 1, 5];
/// ans.encode_iid_symbols_reverse(&symbols2, &entropy_model).unwrap();
/// let snapshot2 = ans.pos_and_state();
///
/// // As discussed above, `DefaultAns` doesn't impl `Seek` but we can get a decoder that does:
/// let mut seekable_decoder = ans.seekable_decoder();
///
/// // `seekable_decoder` is still a `Ans`, so decoding would start with the items we encoded
/// // last. But since it implements `Seek` we can jump ahead to our first snapshot:
/// seekable_decoder.seek(snapshot1);
/// let decoded1 = seekable_decoder
///     .decode_iid_symbols(4, &entropy_model)
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
/// assert_eq!(decoded1, symbols1);
///
/// // We've reached the end of the compressed data ...
/// assert!(seekable_decoder.is_empty());
///
/// // ... but we can still jump to somewhere else and continue decoding from there:
/// seekable_decoder.seek(snapshot2);
///
/// // Creating snapshots didn't mutate the coder, so we can just decode through `snapshot1`:
/// let decoded_both = seekable_decoder.decode_iid_symbols(7, &entropy_model).map(Result::unwrap);
/// assert!(decoded_both.eq(symbols2.into_iter().chain(symbols1)));
/// assert!(seekable_decoder.is_empty()); // <-- We've reached the end again.
/// ```
///
/// [`DefaultAns`]: ans::DefaultAns
/// [`seekable_decoder`]: ans::Ans::seekable_decoder
/// [`into_seekable_decoder`]: ans::Ans::into_seekable_decoder
pub trait Seek: Code {
    /// Jumps to a given position in the compressed data.
    ///
    /// The argument `pos_and_state` is the same pair of values returned by
    /// [`Pos::pos_and_state`], i.e., it is a tuple of the position in the compressed
    /// data and the `State` to which the entropy coder should be restored. Both values
    /// are absolute (i.e., seeking happens independently of the current state or
    /// position of the entropy coder). The position is measured in units of
    /// `CompressedWord`s (see second example below where we manipulate a position
    /// obtained from `Pos::pos_and_state` in order to reflect a manual reordering of
    /// the `CompressedWord`s in the compressed data).
    ///
    /// # Examples
    ///
    /// The method takes the position and state as a tuple rather than as independent
    /// method arguments so that one can simply pass in the tuple obtained from
    /// [`Pos::pos_and_state`] as sketched below:
    ///
    /// ```
    /// // Step 1: Obtain an encoder and encode some data (omitted for brevity) ...
    /// # use constriction::stream::{ans::DefaultAns, Pos, Seek};
    /// # let encoder = DefaultAns::new();
    ///
    /// // Step 2: Take a snapshot by calling `Pos::pos_and_state`:
    /// let snapshot = encoder.pos_and_state(); // <-- Returns a tuple `(pos, state)`.
    ///
    /// // Step 3: Encode some more data and then obtain a decoder (omitted for brevity) ...
    /// # let mut decoder = encoder.seekable_decoder();
    ///
    /// // Step 4: Jump to snapshot by calling `Seek::seek`:
    /// decoder.seek(snapshot); // <-- No need to deconstruct `snapshot` into `(pos, state)`.
    /// ```
    ///
    /// For more fine-grained control, one may want to assemble the tuple
    /// `pos_and_state` manually. For example, a [`DefaultAns`] encodes data from
    /// front to back and then decodes the data in the reverse direction from back to
    /// front. Decoding from back to front may be inconvenient in some use cases, so one
    /// might prefer to instead reverse the order of the `CompressedWord`s once encoding
    /// is finished, and then decode them in the more natural direction from front to
    /// back. Reversing the compressed data changes the position of each
    /// `CompressedWord`, and so any positions obtained from `Pos` need to be adjusted
    /// accordingly before they may be passed to `seek`, as in the following example:
    ///
    /// ```
    /// use constriction::stream::{
    ///     models::LeakyQuantizer,
    ///     ans::{backend::ReadCursorForward, DefaultAns, Ans},
    ///     Decode, Pos, Seek
    /// };
    ///
    /// // Construct a `DefaultAns` for encoding and an entropy model:
    /// let mut encoder = DefaultAns::new();
    /// let quantizer = LeakyQuantizer::<_, _, u32, 24>::new(-100..=100);
    /// let entropy_model = quantizer.quantize(statrs::distribution::Normal::new(0.0, 10.0).unwrap());
    ///
    /// // Encode two chunks of symbols and take a snapshot in-between:
    /// encoder.encode_iid_symbols_reverse(-100..40, &entropy_model).unwrap();
    /// let (mut snapshot_pos, snapshot_state) = encoder.pos_and_state();
    /// encoder.encode_iid_symbols_reverse(50..101, &entropy_model).unwrap();
    ///
    /// // Obtain compressed data, reverse it, and create a decoder that reads it from front to back:
    /// let mut compressed = encoder.into_compressed();
    /// compressed.reverse();
    /// snapshot_pos = compressed.len() - snapshot_pos; // <-- Adjusts the snapshot position.
    /// let mut decoder = Ans::from_compressed(ReadCursorForward::new(compressed)).unwrap();
    ///
    /// // Since we chose to encode onto a stack, decoding yields the last encoded chunk first:
    /// assert_eq!(decoder.decode_symbol(&entropy_model).unwrap(), 50);
    /// assert_eq!(decoder.decode_symbol(&entropy_model).unwrap(), 51);
    ///
    /// // To jump to our snapshot, we have to use the adjusted `snapshot_pos`:
    /// decoder.seek((snapshot_pos, snapshot_state));
    /// assert!(decoder.decode_iid_symbols(140, &entropy_model).map(Result::unwrap).eq(-100..40));
    /// assert!(decoder.is_empty()); // <-- We've reached the end of the compressed data.
    /// ```
    ///
    /// [`DefaultAns`]: ans::DefaultAns
    fn seek(&mut self, pos_and_state: (usize, Self::State)) -> Result<(), ()>;
}

#[allow(missing_debug_implementations)] // Any useful debug output would have to mutate the decoder.
pub struct DecodeSymbols<'a, Decoder: ?Sized, I, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    models: I,
}

impl<'a, Decoder, I, D, const PRECISION: usize> Iterator
    for DecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = D>,
    D: DecoderModel<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item = Result<<I::Item as EntropyModel<PRECISION>>::Symbol, Decoder::DecodingError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.models
            .next()
            .map(|model| self.decoder.decode_symbol(model))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.models.size_hint()
    }
}

impl<'a, Decoder, I, D, const PRECISION: usize> ExactSizeIterator
    for DecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = D> + ExactSizeIterator,
    D: DecoderModel<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
}

#[allow(missing_debug_implementations)] // Any useful debug output would have to mutate the decoder.
pub struct TryDecodeSymbols<'a, Decoder: ?Sized, I, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    models: I,
}

impl<'a, Decoder, I, D, E, const PRECISION: usize> Iterator
    for TryDecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = Result<D, E>>,
    D: DecoderModel<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item = Result<D::Symbol, TryCodingError<Decoder::DecodingError, E>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.models.next().map(|model| {
            Ok(self
                .decoder
                .decode_symbol(model.map_err(|err| TryCodingError::InvalidEntropyModel(err))?)?)
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We don't terminate when we encounter an error, so the size doesn't change.
        self.models.size_hint()
    }
}

impl<'a, Decoder, I, D, E, const PRECISION: usize> ExactSizeIterator
    for TryDecodeSymbols<'a, Decoder, I, PRECISION>
where
    Decoder: Decode<PRECISION>,
    I: Iterator<Item = Result<D, E>> + ExactSizeIterator,
    D: DecoderModel<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
}

#[derive(Debug)]
pub struct DecodeIidSymbols<'a, Decoder: ?Sized, D, const PRECISION: usize> {
    decoder: &'a mut Decoder,
    model: &'a D,
    amt: usize,
}

impl<'a, Decoder, D, const PRECISION: usize> Iterator
    for DecodeIidSymbols<'a, Decoder, D, PRECISION>
where
    Decoder: Decode<PRECISION>,
    D: DecoderModel<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
    type Item = Result<D::Symbol, Decoder::DecodingError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.amt != 0 {
            self.amt -= 1;
            Some(self.decoder.decode_symbol(self.model))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.amt, Some(self.amt))
    }
}

impl<'a, Decoder, D, const PRECISION: usize> ExactSizeIterator
    for DecodeIidSymbols<'a, Decoder, D, PRECISION>
where
    Decoder: Decode<PRECISION>,
    D: DecoderModel<PRECISION>,
    Decoder::CompressedWord: AsPrimitive<D::Probability>,
    D::Probability: Into<Decoder::CompressedWord>,
{
}

#[derive(Debug)]
pub enum TryCodingError<CodingError, ModelError> {
    /// The iterator provided to [`Coder::try_push_symbols`] or
    /// [`Coder::try_pop_symbols`] yielded `Err(_)`.
    ///
    /// The variant wraps the original error, which can also be retrieved via
    /// [`source`](#method.source).
    ///
    /// [`Coder::try_push_symbols`]: struct.Coder.html#method.try_push_symbols
    /// [`Coder::try_pop_symbols`]: struct.Coder.html#method.try_pop_symbols
    InvalidEntropyModel(ModelError),

    CodingError(CodingError),
}

impl core::fmt::Display for EncodingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ImpossibleSymbol => write!(
                f,
                "Tried to encode symbol that has zero probability under the used entropy model."
            ),
            Self::CapacityExceeded => write!(f, "The encoder cannot accept any more symbols."),
        }
    }
}

#[cfg(feature = "std")]
impl Error for EncodingError {}

impl<CodingError: core::fmt::Display, ModelError: core::fmt::Display> core::fmt::Display
    for TryCodingError<CodingError, ModelError>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidEntropyModel(err) => {
                write!(f, "Error while constructing entropy model or data: {}", err)
            }
            Self::CodingError(err) => {
                write!(f, "Error while entropy coding: {}", err)
            }
        }
    }
}

#[cfg(feature = "std")]
impl<CodingError: Error + 'static, ModelError: Error + 'static> Error
    for TryCodingError<CodingError, ModelError>
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidEntropyModel(source) => Some(source),
            Self::CodingError(source) => Some(source),
        }
    }
}

impl<CodingError, ModelError> From<CodingError> for TryCodingError<CodingError, ModelError> {
    fn from(err: CodingError) -> Self {
        Self::CodingError(err)
    }
}
