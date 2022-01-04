//! Sources and sinks of compressed data
//!
//! This module declares traits for reading and writing sequences of bits in small chunks
//! ("words") of fixed size. When entropy coders provided by `constriction` read in or write
//! out a word of compressed data they do this through a generic type, typically called
//! `Backend`, that implements one or more of the traits from this module. Declaring traits
//! for entropy coder backends allows us to make entropy coders generic over the specific
//! backends they use used rather than forcing a single choice of backend implementation on
//! all users of a given entropy coder. This gives users with specialized needs fine grained
//! control over the behaviour of entropy coders without sacrificing ergonomics or
//! computational efficiency in the common use cases.
//!
//! # This module is meant for advanced use cases
//!
//! Most users of `constriction` don't have to worry much about backends since all entropy
//! coders default (at compile time) to reasonable backends based on what you use them for.
//! You will usually end up using a `Vec` backend for encoding, and you may use either a
//! `Vec` or a [`Cursor`] backend for decoding depending on whether your entropyc coder has
//! stack or queue semantics and whether you want to consume or retain the compressed data
//! after decoding. However, these automatically inferred default backends may not be
//! suitable for certain advanced use cases. For example, you may want to decode data
//! directly from a network socket, or you may be implementing a container format that
//! performs some additional operation like multiplexing directly after entropy coding. In
//! such a case, you may want to declare your own type for the source or sink of compressed
//! data and implement the relevant traits from this module for it so that you can use it as
//! a backend for an entropy coder.
//!
//! # Module Overview
//!
//! The main traits in this module are [`ReadWords`] and [`WriteWords`]. They express the
//! capability of a backend to be a source and/or sink of compressed data, respectively.
//! Both traits are generic over a type `Word`, which is usually a [`BitArray`], and which
//! represents the smallest unit of compressed data that an entropy coder will read and/or
//! write at a time (all provided entropy coders in `constriction` are generic over the
//! `Word` and default to `Word = u32`). Types that implement one of the backend traits
//! often also implement [`Pos`] and/or [`Seek`] from the parent module. The remaining
//! traits in this module specify further properties of the backend (see
//! [`BoundedReadWords`] and [`BoundedWriteWords`]) and provide permanent or temporary
//! conversions into backends with different capabilities (see [`IntoReadWords`],
//! [`IntoSeekReadWords`], [`AsReadWords`], and [`AsSeekReadWords`]).
//!
//! The backend traits are implemented for the standard library type `Vec` where applicable
//! and for a few new types defined in this module. The most important type defined in this
//! module is a [`Cursor`], which wraps a generic buffer of an in-memory slice of words
//! together with an index into the buffer (`constriction::backends::Cursor` is to the
//! backend traits roughly what [`std::io::Cursor`] is to the `std::io::{Read, Write}`
//! traits). A [`Cursor`] can hold either owned or borrowed data and can be used for reading
//! and/or writing (if the buffer is mutable) with both [`Queue`] and [`Stack`] read/write
//! semantics.
//!
//! # Read/Write Semantics
//!
//! The [`ReadWords`] trait has a second type parameter with trait bound [`Semantics`].
//! `Semantics` are (typically zero sized) marker types that indicate how reads from a
//! backend behave in relation to writes to the backend. This issue is moot for backends
//! that support only one of reading or writing but not both. Therefore, any backend that
//! does not implement `WriteWords` may implement `ReadWords` for multiple `Semantics`.
//! However, backends that implement both `ReadWords` and `WriteWords` must make sure to
//! implement `ReadWords` only for the one or more appropriate semantics. There are two
//! predefined `Semantics`: [`Queue`], which indicates that reads and writes operate in the
//! same linear direction ("first in first out") and [`Stack`], which is also a linear
//! sequence of words but one where reads and writes operate in opposite directions ("last
//! in first out"). You may define your own `Semantics` if you want to implement a backend
//! based on a more fancy abstract data type.
//!
//! For example, we implement `WriteWords<Word>` and `ReadWords<Word, Stack>` but not
//! `ReadWords<Word, Queue>` for the type `Vec<Word>` from the standard library because
//! [`Vec::push`] and [`Vec::pop`] have `Stack` semantics. By contrast, the type [`Cursor`]
//! declared in this module, which wraps a memory buffer and an index into the buffer,
//! implements `ReadWords` for both `Stack` and `Queue` semantics. The `Queue`
//! implementation increases the index after reading and the `Stack` implementation
//! decreases the index before reading (if you want the opposite interpretation then you can
//! wrap the `Cursor` in a [`Reverse`]). Thus, a `Cursor` can be used by entropy coders
//! with both stack and queue semantics, and both can use the `Cursor` in the way that is
//! correct for them. By contrast, while a stack-based entropy coder (like [`AnsCoder`]) can
//! use a `Vec<Word>` for both encoding and decoding, an entropy coder with queue semantics
//! (like a Range Coder) can use a `Vec` only for encoding but it has to wrap the `Vec` in a
//! `Cursor` for decoding, thus preventing accidental misuse.
//!
//! # Example of Entropy Coding With a Non-Standard Backend
//!
//! The following example encodes and decodes data to and from a file. It uses custom
//! backends that directly write each `Word` to, and read each `Word` from the file. This is
//! not a very practical example—if you encode all data at once then it's simpler and
//! possibly even more efficient to use the default backend, which writes to an in-memory
//! buffer, call `.get_compressed()` when you're done, and then flush the buffer to the file
//! in one go. But custom backends similar to the ones used in this example could also be
//! used to add additional processing to the compressed data, such as multiplexing or
//! demultiplexing for some container format.
//!
//! ```
//! use constriction::{
//!     backends::{FallibleCallbackWriteWords, FallibleIteratorReadWords},
//!     stream::{
//!         model::DefaultLeakyQuantizer,
//!         queue::{DefaultRangeDecoder, DefaultRangeEncoder},
//!         Decode, Encode,
//!     },
//! };
//! use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
//! use probability::distribution::Gaussian;
//! use std::{fs::File, io::{BufReader, BufWriter}};
//!
//! fn encode_to_file_on_the_fly(amt: u32) {
//!     // Some simple entropy model, just for demonstration purpose.
//!     let quantizer = DefaultLeakyQuantizer::new(-256..=255);
//!     let model = quantizer.quantize(Gaussian::new(0.0, 100.0));
//!
//!     // Some long-ish sequence of test symbols, made up in a reproducible way.
//!     let symbols = (0..amt).map(|i| {
//!         let cheap_hash = i.wrapping_mul(0x6979_E2F3).wrapping_add(0x0059_0E91);
//!         (cheap_hash >> (32 - 9)) as i32 - 256
//!     });
//!
//!     // Open a file and build a backend that writes to this file one word at a time.
//!     // (Wrapping the `File` in a `BufWriter` isn't strictly necessary here,
//!     // it's just good practice when writing to a file.)
//!     let mut file = BufWriter::new(File::create("backend_queue_example.tmp").unwrap());
//!     let backend =
//!         FallibleCallbackWriteWords::new(move |word| file.write_u32::<LittleEndian>(word));
//!
//!     // Wrap the backend in a `RangeEncoder` and encode (i.e., compress) the symbols.
//!     let mut encoder = DefaultRangeEncoder::with_backend(backend);
//!     encoder.encode_iid_symbols(symbols, &model).unwrap();
//!
//!     // Dropping the encoder doesn't automatically seal the compressed bit string because that
//!     // could fail. We explicitly have to seal it by calling `.into_compressed()`, which returns
//!     // the backend since that's what logically "holds" the compressed data, and then drop that.
//!     std::mem::drop(encoder.into_compressed().unwrap());
//! }
//!
//! fn decode_from_file_on_the_fly(amt: u32) {
//!     // Same toy entropy model that we used for encoding.
//!     let quantizer = DefaultLeakyQuantizer::new(-256..=255);
//!     let model = quantizer.quantize(Gaussian::new(0.0, 100.0));
//!
//!     // Open the file and iterate over its contents in `u32` words (wrapping it in a `BufReader`
//!     // is again just for good practice). We're deliberately being pedantic about the errors
//!     // here in order to show how backend errors can be reported to the encoder.
//!     let mut file = BufReader::new(File::open("backend_queue_example.tmp").unwrap());
//!     let word_iterator = std::iter::from_fn(move || match file.read_u32::<LittleEndian>() {
//!         Ok(word) => Some(Ok(word)),
//!         Err(err) => {
//!             if err.kind() == std::io::ErrorKind::UnexpectedEof {
//!                 None // Reached end of file, end iteration.
//!             } else {
//!                 Some(Err(err)) // Some other I/O error occurred. Propagate it up.
//!             }
//!         }
//!     });
//!
//!     // Create a decoder that decodes on the fly from our iterator.
//!     let backend = FallibleIteratorReadWords::new(word_iterator);
//!     let mut decoder = DefaultRangeDecoder::with_backend(backend).unwrap();
//!
//!     // Decode the symbols and verify their correctness.
//!     for (i, symbol) in decoder.decode_iid_symbols(amt as usize, &model).enumerate() {
//!         let cheap_hash = (i as u32).wrapping_mul(0x6979_E2F3).wrapping_add(0x0059_0E91);
//!         let expected = (cheap_hash >> (32 - 9)) as i32 - 256;
//!         assert_eq!(symbol.unwrap(), expected);
//!     }
//!
//!     // Recover the original iterator over compressed words and verify that it's been exhausted.
//!     let mut word_iterator = decoder.into_raw_parts().0.into_iter();
//!     assert!(word_iterator.next().is_none());
//!
//!     // `word_iterator` owns the file since we used a `move` clausure above to construct it.
//!     // So dropping it calls `std::fs::File`'s destructor, which releases the file handle.
//!     std::mem::drop(word_iterator);
//!     std::fs::remove_file("backend_queue_example.tmp").unwrap();
//! }
//!
//! encode_to_file_on_the_fly(1000);
//! decode_from_file_on_the_fly(1000);
//! ```
//!
//! [`BitArray`]: crate::BitArray
//! [`ChainCoder`]: crate::stream::chain::ChainCoder
//! [`AnsCoder`]: crate::stream::stack::AnsCoder

use alloc::{boxed::Box, vec::Vec};
use core::{
    convert::Infallible,
    fmt::{Debug, Display},
    marker::PhantomData,
};
use smallvec::SmallVec;

use crate::{Pos, PosSeek, Queue, Seek, Semantics, Stack};

// MAIN TRAITS FOR CAPABILITIES OF BACKENDS ===================================

/// A trait for sources of compressed data (mainly used by decoders).
///
/// See the [module-level documentation](self) for more information, in particular regarding
/// the type parameter `S: Semantics`.
pub trait ReadWords<Word, S: Semantics> {
    /// The error type that can occur when reading from the data source, or [`Infallible`].
    ///
    /// Note that "end of file" / "out of data" is *not* considered an error. The [`read`]
    /// method indicates "end of file" by returning `Ok(None)`, not `Err(...)`. If reading
    /// the data source cannot fail (except for "end of file") then `ReadError` should be
    /// [`Infallible`] so that the compiler can optimize out any error checks (see also
    /// [`UnwrapInfallible`]).
    ///
    /// [`read`]: Self::read
    /// [`UnwrapInfallible`]: crate::UnwrapInfallible
    type ReadError: Debug;

    /// Reads a single `Word` from the data source and advances the state of the data source
    /// accordingly (i.e., so that the next `read` won't read the same `Word` again).
    ///
    /// Returns
    /// - `Ok(Some(word))` if the read succeeded;
    /// - `Ok(None)` if the backend is exhausted (i.e., there's no more data left); or
    /// - `Err(err)` if an error `err` *other than "end of file"* occurred during reading
    ///   (e.g., a file system error)
    ///
    /// Note that `ReadWords::read` has stricter requirements than the standard library's
    /// [`Iterator::next`]. Once `ReadWords::read` indicates end of file by returning
    /// `Ok(None)`, it must never return `Ok(Some(_))` when called again (i.e., types that
    /// implement `ReadWords` have to be "fused", in iterator terminology). Entropy coders
    /// may rely on this contract for correctness of the encoded and decoded data but not
    /// for memory safety.
    fn read(&mut self) -> Result<Option<Word>, Self::ReadError>;

    /// Returns `true` if the data source *could* be out of data.
    ///
    /// The default implementation always returns `true` since returning `true` makes no
    /// statement. Overwrite the default implementation if you may in some cases be able to
    /// say with certainty that there is still data left to be read, and return `false` in
    /// these cases.
    ///
    /// If `maybe_exhausted()` returns `false` then the next call to `read` must return
    /// either `Ok(Some(_))` or `Err(_)` but not `Ok(None)`.
    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        true
    }
}

/// A trait for sinks of compressed data (mainly used by encoders).
///
/// See the [module-level documentation](self) for more information.
pub trait WriteWords<Word> {
    /// The error type that can occur when writing to the data sink, or [`Infallible`].
    ///
    /// This type should be [`Infallible`] if writing cannot fail, so that the compiler can
    /// optimize out any error checks (see also [`UnwrapInfallible`]).
    ///
    /// An example error could be that the data sink is "full", if that's a state that the
    /// data sink can be in. Note the asymmetry compared to [`ReadWords`]: while we consider
    /// an attempt to write to a "full" `WriteWords` as an error, we consider an attempt to
    /// read from an "empty" `ReadWords` as a normal operation (that returns `Ok(None)`).
    /// This is because it is a common task to read all data from a data source until it is
    /// empty (and attempting to read past the "empty" state is often the only way to detect
    /// emptyness) whereas it is not a common task to intentionally write to a data sink
    /// until it is full (and therefore attempting to write to a full data sink is typically
    /// an error).
    ///
    /// [`UnwrapInfallible`]: crate::UnwrapInfallible
    type WriteError: Debug;

    /// Writes a single `Word` to the data sink and advances the state of the data sink
    /// accordingly (i.e., so that the next `write` won't overwrite the current `Word`).
    fn write(&mut self, word: Word) -> Result<(), Self::WriteError>;

    /// Writes a sequence of `Word`s to the data sink, short-circuiting on error.
    ///
    /// The default implementation calls [`write`] for each word. You may want to overwrite
    /// this if your data sink can perform additional optimizations (e.g., by utilizing the
    /// provided iterator's `size_hint`).
    fn extend_from_iter(
        &mut self,
        iter: impl Iterator<Item = Word>,
    ) -> Result<(), Self::WriteError> {
        for word in iter {
            self.write(word)?;
        }
        Ok(())
    }

    /// Returns `true` if the data sink *could* be full
    ///
    /// It is always correct to return `true` from this method, even if the concept of being
    /// "full" doesn't apply to the data sink. The default implementation always returns
    /// `true`. The precise meaning of "full" may vary between data sinks. A data sink
    /// that's "not full" (i.e., where this method returns `false`) may still return an
    /// error when trying to [`write`] to it.
    ///
    /// [`write`]: Self::write
    #[inline(always)]
    fn maybe_full(&self) -> bool {
        true
    }
}

/// A trait for data sources that know how much data is left.
pub trait BoundedReadWords<Word, S: Semantics>: ReadWords<Word, S> {
    /// Returns the number of `Word`s that are left for reading.
    ///
    /// If `remaining()` returns `n` then the next `n` calls to [`read`] must not return
    /// `Ok(None)`, and any subsequent `read`s must not return `Ok(Some(_))`.
    ///
    /// [`read`]: ReadWords::read
    fn remaining(&self) -> usize;

    /// Whether or not there is no data left to read.
    ///
    /// You'll usually want to overwrite the default implementation of
    /// [`ReadWords::maybe_exhausted`] to call `is_exhausted`, although the only strict
    /// requirement is that `maybe_exhausted` must not return `false` if `is_exhausted`
    /// returns `true`.
    #[inline(always)]
    fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }
}
/// A trait for data sinks with a known finite capacity.
pub trait BoundedWriteWords<Word>: WriteWords<Word> {
    /// Returns the number of `Word`s that one can expect to still be able to write to the
    /// data sink.
    ///
    /// The precise interpretation of the return value depends on the specific data sink.
    /// Calling [`write`] may still fail even if `space_left` returns a nonzero value (since
    /// we want to allow for unpredictable I/O errors).
    ///
    /// [`write`]: WriteWords::write
    fn space_left(&self) -> usize;

    /// Whether or not there is expected to still be some space left to write.
    ///
    /// You'll usually want to overwrite the default implementation of
    /// [`WriteWords::maybe_full`] to call `is_full`, although the only strict requirement
    /// is that `maybe_full` must not return `false` if `is_full` returns `true`.
    #[inline(always)]
    fn is_full(&self) -> bool {
        self.space_left() == 0
    }
}

// TRAITS FOR CONVERSIONS BETWEEN BACKENDS WITH DIFFERENT CAPABILITIES ========

/// A trait for types that can be turned into a source of compressed data (for decoders).
///
/// This trait is roughly analogous to the standard library trait [`IntoIterator`], except
/// that it is generic over `Word` rather than having `Word` as an associated type. This
/// makes it possible to convert a single type into data sources of varying word sizes (for
/// example, one could imagine implementing both `IntoReadWords<u32>` and
/// `IntoReadWords<u16>` for `Vec<u8>` using the `byteorder` crate).
///
/// # See also
///
/// - [module level documentation](self) for more information on the concept of sources and
///   sinks of compressed data;
/// - [`AsReadWords`] for a simliar conversion that does not take ownership of the data;
/// - [`IntoSeekReadWords`] for a conversion with stronger guarantees.
pub trait IntoReadWords<Word, S: Semantics> {
    /// The type of the data source that will result from the conversion.
    type IntoReadWords: ReadWords<Word, S>;

    /// Performs the conversion.
    fn into_read_words(self) -> Self::IntoReadWords;
}

/// A trait for types that can be temporarily used by decoders as a source of compressed
/// data.
///
/// This trait is meant for situations where you want to use some data as a source of
/// `Word`s without consuming it. This allows you to decode the same compressed data several
/// times but it means that you typically won't be allowed to return the resulting data
/// source or any entropy coder that wraps it from the current function because it doesn't
/// take ownership of the data. If you want to take ownership of the data, use
/// [`IntoReadWords`] instead.
///
/// Note that, if you want to decode the same compressed data several times then you'll
/// probably want to decode *different parts* of that data each time. In this case, it's
/// likely you'll rather want to use [`AsSeekReadWords`].
///
/// # See also
///
/// - [module level documentation](self) for more information on the concept of sources and
///   sinks of compressed data;
/// - [`IntoReadWords`] for a simliar conversion that takes ownership of the data;
/// - [`AsSeekReadWords`] for a conversion with stronger guarantees.
pub trait AsReadWords<'a, Word, S: Semantics>: 'a {
    /// The type of the data source as which the original type can be used.
    type AsReadWords: ReadWords<Word, S>;

    /// Performs the (temporary) conversion.
    fn as_read_words(&'a self) -> Self::AsReadWords;
}

/// A trait for types that can be turned into a randomly accessible source of compressed
/// data (for decoders).
///
/// This trait is similar to [`IntoReadWords`] but it adds the additional guarantee that the
/// resulting data source implements [`Seek`], i.e., that it can be used by decoders that
/// support random access.
///
/// # See also
///
/// - [module level documentation](self) for more information on the concept of sources and
///   sinks of compressed data;
/// - [`AsSeekReadWords`] for a simliar conversion that does not take ownership of the data;
pub trait IntoSeekReadWords<Word, S: Semantics> {
    /// The type of the random-access data source that will result from the conversion.
    type IntoSeekReadWords: Seek + ReadWords<Word, S>;

    /// Performs the conversion.
    fn into_seek_read_words(self) -> Self::IntoSeekReadWords;
}

/// A trait for types that can be temporarily used by random-access decoders as a source of
/// compressed data.
///
/// This trait is meant for situations where you want to use some data as a source of
/// `Word`s without consuming it. This allows you to decode the same compressed data several
/// times but it means that you typically won't be allowed to return the resulting data
/// source or any entropy coder that wraps it from the current function because it doesn't
/// take ownership of the data. If you want to take ownership of the data, use
/// [`IntoReadWords`] instead.
///
/// This trait is similar to [`AsReadWords`] but it adds the additional guarantee that the
/// resulting data source implements [`Seek`], i.e., that it can be used by decoders that
/// support random access. This is likely what you want if you're going to construct several
/// decoders for the same compressed data because why would you want decode the *whole* data
/// several times?
///
/// # See also
///
/// - [module level documentation](self) for more information on the concept of sources and
///   sinks of compressed data;
/// - [`IntoSeekReadWords`] for a simliar conversion that takes ownership of the data;
pub trait AsSeekReadWords<'a, Word, S: Semantics>: 'a {
    /// The type of the random-access data source as which the original type can be used.
    type AsSeekReadWords: Seek + ReadWords<Word, S>;

    /// Performs the (temporary) conversion.
    fn as_seek_read_words(&'a self) -> Self::AsSeekReadWords;
}

// IMPLEMENTATIONS FOR `Vec<Word>` ============================================

impl<Word> WriteWords<Word> for Vec<Word> {
    /// The only way how writing to a `Vec<Word>` can fail is if a memory allocation fails,
    /// which is typically treated as a fatal error (i.e., aborts) in Rust.
    type WriteError = Infallible;

    /// Appends the word to the end of the vector (= top of the stack)
    #[inline(always)]
    fn write(&mut self, word: Word) -> Result<(), Self::WriteError> {
        self.push(word);
        Ok(())
    }

    fn extend_from_iter(
        &mut self,
        iter: impl Iterator<Item = Word>,
    ) -> Result<(), Self::WriteError> {
        self.extend(iter);
        Ok(())
    }

    fn maybe_full(&self) -> bool {
        false
    }
}

impl<Word> ReadWords<Word, Stack> for Vec<Word> {
    /// The only way how reading from a vector can fail is if the vector is empty, but
    /// that's not considered an error (it returns `Ok(None)` instead).
    type ReadError = Infallible;

    /// Pops the word off the end of the vector (= top of the stack). If you instead want to
    /// keep the data unchanged (e.g., because you want to reuse it later) then wrap either
    /// the vector `v` or or the slice `&v[..]` in a [`Cursor`].
    #[inline(always)]
    fn read(&mut self) -> Result<Option<Word>, Self::ReadError> {
        Ok(self.pop())
    }

    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        self.is_empty()
    }
}

impl<Word> BoundedReadWords<Word, Stack> for Vec<Word> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.len()
    }
}

impl<Word> PosSeek for Vec<Word> {
    type Position = usize;
}

impl<Word> Pos for Vec<Word> {
    /// Returns the length of the vector since that's the current read and write position
    /// (vectors have [`Stack`] semantics).
    ///
    /// If you have a `Vec` with name `v` and your intention is to read to or write from it
    /// at arbitrary positions rather than just at the end then you probably want to wrap
    /// either `v` or the slice `&v[..]` in a [`Cursor`].
    fn pos(&self) -> usize {
        self.len()
    }
}

impl<Word> Seek for Vec<Word> {
    /// Seeking in a `Vec<Word>` only succeeds if the provided position `pos` is smaller
    /// than or equal to the vector's current length. In this case, seeking will truncate
    /// the vector to length `pos`. This is because vectors have [`Stack`] semantics, and
    /// the current read/write position (i.e., the head of the stack) is always at the end
    /// of the vector.
    ///
    /// If you have a `Vec` with name `v` and your intention is to read to or write from it
    /// at arbitrary positions rather than just at the end then you probably want to wrap
    /// either `v` or the slice `&v[..]` in a [`Cursor`].
    fn seek(&mut self, pos: usize) -> Result<(), ()> {
        if pos <= self.len() {
            self.truncate(pos);
            Ok(())
        } else {
            Err(())
        }
    }
}

// IMPLEMENTATIONS FOR `SmallVec<Word>` =======================================

impl<Array> WriteWords<Array::Item> for SmallVec<Array>
where
    Array: smallvec::Array,
{
    /// The only way how writing to a `Vec<Word>` can fail is if a memory allocation fails,
    /// which is typically treated as a fatal error (i.e., aborts) in Rust.
    type WriteError = Infallible;

    /// Appends the word to the end of the vector (= top of the stack)
    #[inline(always)]
    fn write(&mut self, word: Array::Item) -> Result<(), Self::WriteError> {
        self.push(word);
        Ok(())
    }

    fn extend_from_iter(
        &mut self,
        iter: impl Iterator<Item = Array::Item>,
    ) -> Result<(), Self::WriteError> {
        self.extend(iter);
        Ok(())
    }

    fn maybe_full(&self) -> bool {
        false
    }
}

impl<Array> ReadWords<Array::Item, Stack> for SmallVec<Array>
where
    Array: smallvec::Array,
{
    /// The only way how reading from a vector can fail is if the vector is empty, but
    /// that's not considered an error (it returns `Ok(None)` instead).
    type ReadError = Infallible;

    /// Pops the word off the end of the vector (= top of the stack). If you instead want to
    /// keep the data unchanged (e.g., because you want to reuse it later) then wrap either
    /// the vector `v` or or the slice `&v[..]` in a [`Cursor`].
    #[inline(always)]
    fn read(&mut self) -> Result<Option<Array::Item>, Self::ReadError> {
        Ok(self.pop())
    }

    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        self.is_empty()
    }
}

impl<Array> BoundedReadWords<Array::Item, Stack> for SmallVec<Array>
where
    Array: smallvec::Array,
{
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.len()
    }
}

impl<Array> PosSeek for SmallVec<Array>
where
    Array: smallvec::Array,
{
    type Position = usize;
}

impl<Array> Pos for SmallVec<Array>
where
    Array: smallvec::Array,
{
    /// Returns the length of the `SmallVec` since that's the current read and write
    /// position (`SmallVec`s, like `Vec`s, have [`Stack`] semantics).
    ///
    /// If you have a `Vec` or `SmallVec` with name `v` and your intention is to read to or
    /// write from it at arbitrary positions rather than just at the end then you probably
    /// want to wrap either `v` or the slice `&v[..]` in a [`Cursor`].
    fn pos(&self) -> usize {
        self.len()
    }
}

impl<Array> Seek for SmallVec<Array>
where
    Array: smallvec::Array,
{
    /// Seeking in a `SmallVec` only succeeds if the provided position `pos` is smaller than
    /// or equal to the `SmallVec`'s current length. In this case, seeking will truncate the
    /// `SmallVec` to length `pos`. This is because `SmallVec`s, like `Vec`s, have [`Stack`]
    /// semantics, and the current read/write position (i.e., the head of the stack) is
    /// always at the end of the `SmallVec`.
    ///
    /// If you have a `Vec` or `SmallVec` with name `v` and your intention is to read to or
    /// write from it at arbitrary positions rather than just at the end then you probably
    /// want to wrap either `v` or the slice `&v[..]` in a [`Cursor`].
    fn seek(&mut self, pos: usize) -> Result<(), ()> {
        if pos <= self.len() {
            self.truncate(pos);
            Ok(())
        } else {
            Err(())
        }
    }
}

// ADAPTER FOR (SEMANTIC) REVERSING OF READING DIRECTION ======================

/// Wrapper that inverts the read/write directions of a data source and/or data sink.
///
/// # Motivation
///
/// This wrapper is usually used for a [`Cursor`]. The `Cursor` type implements
/// `ReadWords<Word, S>` for both semantics `S = Queue` and `S = Stack`, so you can read
/// `Word`s from a cursor in either forward or backward direction.
///
/// Reading from a `Cursor` with `Queue` semantics reads from the underlying slice `[Word]`
/// in the normal direction (from index `0` to index `.len() - 1`), which is useful for
/// decoding data from an entropy coder that has queue semantics (like [`RangeEncoder`]). By
/// contrast, reading from a `Cursor` with `Stack` semantics reads in the reverse direction.
/// This is usually a good thing: it is consistent with how `Vec<Word>` (necessarily)
/// implements reading with `Stack` semantics, so if you have a `Vec<Word>` of data that was
/// compressed with a stack entropy coder (like [`AnsCoder`]) then you are free to choose
/// whether you want to decode the data either directly from the `Vec` or from a `Cursor`
/// that wraps the `Vec` (if you don't want to consume the compressed data). Both approaches
/// will achieve the same result, as you can see in the following example:
///
/// ```
/// use constriction::{
///     backends::Cursor, stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode},
///     UnwrapInfallible,
/// };
///
/// // Some simple entropy model, just for demonstration purpose.
/// let quantizer = DefaultLeakyQuantizer::new(-100..=100);
/// let model = quantizer.quantize(probability::distribution::Gaussian::new(25.0, 10.0));
///
/// // Encode the symbols `0..50` using a stack entropy coder and get the compressed data.
/// let mut coder = DefaultAnsCoder::new();
/// coder.encode_iid_symbols_reverse(0..50, &model).unwrap();
/// # // Test the claims we make in the comments below, but don't document them since that
/// # // would distract from the point we're trying to make here.
/// # let mut d = coder.as_decoder();
/// # assert!(d.decode_iid_symbols(50, &model).map(UnwrapInfallible::unwrap_infallible).eq(0..50));
/// # let mut d = coder.clone().into_decoder();
/// # assert!(d.decode_iid_symbols(50, &model).map(UnwrapInfallible::unwrap_infallible).eq(0..50));
/// let compressed = coder.into_compressed().unwrap_infallible(); // `compressed` is a `Vec<u32>`.
/// dbg!(compressed.len()); // Prints "compressed.len() = 11".
///
/// // You can either decode directly from the `Vec` (could also just have used `coder` for that).
/// let mut c2 = DefaultAnsCoder::from_compressed(compressed.clone()).unwrap();
/// assert!(c2.decode_iid_symbols(50, &model).map(UnwrapInfallible::unwrap_infallible).eq(0..50));
/// # assert!(c2.is_empty());
///
/// // Or you can wrap the slice `[u32]` in a `Cursor` (could have used `coder.as_decoder()`).
/// let borrowing_cursor = Cursor::new_at_write_end(&compressed[..]);
/// let mut c3 = DefaultAnsCoder::from_compressed(borrowing_cursor).unwrap();
/// assert!(c3.decode_iid_symbols(50, &model).map(UnwrapInfallible::unwrap_infallible).eq(0..50));
/// # assert!(c3.is_empty());
///
/// // You can also wrap the `Vec<u32>` in a `Cursor` (could have used `coder.into_decoder()`).
/// let owning_cursor = Cursor::new_at_write_end(compressed);
/// let mut c4 = DefaultAnsCoder::from_compressed(owning_cursor).unwrap();
/// assert!(c4.decode_iid_symbols(50, &model).map(UnwrapInfallible::unwrap_infallible).eq(0..50));
/// # assert!(c4.is_empty());
/// ```
///
/// However, it can sometimes be awkward to read data in reverse direction, e.g., if you're
/// reading it from a file or socket. In these situations you may prefer to reverse the
/// entire sequence of `Word`s after encoding so that the decoder can read them in forward
/// direction. You can then still wrap the reversed sequence of `Words` in a `Cursor`, but
/// if you pass this cursor to an `AnsCoder` then the `AnsCoder` has to know somehow that
/// the top of this stack is at the beginning of the slice, i.e., that reading with `Stack`
/// semantics means reading in the forward direction. To express this, wrap the `Cursor` in
/// a `Reverse`, otherwise you'll get garbled data:
///
/// ```
/// # use constriction::{
/// #     backends::Cursor, stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode},
/// #     UnwrapInfallible,
/// # };
/// # let quantizer = DefaultLeakyQuantizer::new(-100..=100);
/// # let model = quantizer.quantize(probability::distribution::Gaussian::new(25.0, 10.0));
/// # let mut coder = DefaultAnsCoder::new();
/// # coder.encode_iid_symbols_reverse(0..50, &model).unwrap();
/// # let mut compressed = coder.into_compressed().unwrap_infallible(); // `compressed` is a `Vec<u32>`.
/// use constriction::backends::Reverse;
///
/// // ... obtain same `model` and `compressed` as in the above example ...
///
/// compressed.reverse(); // Reverses the sequence of `u32` words in place (mutates `compressed`).
///
/// // Naively decoding the reversed compressed data leads to garbled symbols.
/// let wrong_cursor = Cursor::new_at_write_end(&compressed[..]);
/// let mut c5 = DefaultAnsCoder::from_compressed(wrong_cursor).unwrap();
/// let bug = c5.decode_iid_symbols(50, &model).collect::<Result<Vec<_>, _>>().unwrap_infallible();
/// dbg!(bug); // Prints "bug = [39, 47, 40, ...]", not what we had encoded.
///
/// // We must set the initial cursor position "at_write_beginning" and wrap it in a `Reverse`.
/// let reversed_cursor = Reverse(Cursor::new_at_write_beginning(&compressed[..]));
/// let mut c6 = DefaultAnsCoder::from_compressed(reversed_cursor).unwrap();
/// assert!(c6.decode_iid_symbols(50, &model).map(UnwrapInfallible::unwrap_infallible).eq(0..50));
/// # assert!(c6.is_empty());
/// ```
///
/// Wrapping a `Cursor` (or any `ReadWords`) in a `Reverse` with the statement `let
/// reversed_cursor = Reverse(original_cursor);` is a no-op. It changes neither the order of
/// the underlying sequence of `Word`s nor the current cursor position, but it will reverse
/// the direction in which the cursor position moves when you read data from
/// `reversed_cursor`. Therefore you should already have reversed the the underlying
/// sequence of `Word`s before you wrap the cursor in `Reverse`, and you should have
/// initialitze the `Cursor` position at the beginning of the reversed sequence of `Word`s
/// rather than the end.
///
/// # Shortcut
///
/// The method [`Cursor::into_reversed`] does everything at once:
/// - it reverses the underlying sequence of `Word`s;
/// - it moves the current read position to its mirrored position (thus following where the
///   value that it originally pointed to moved upon reversing the sequence of `Word`s); and
/// - it wraps the thus modified `Cursor` in a `Reverse`.
///
/// ```
/// # use constriction::{
/// #     backends::Cursor, stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode},
/// #     UnwrapInfallible,
/// # };
/// # let quantizer = DefaultLeakyQuantizer::new(-100..=100);
/// # let model = quantizer.quantize(probability::distribution::Gaussian::new(25.0, 10.0));
/// # let mut coder = DefaultAnsCoder::new();
/// # coder.encode_iid_symbols_reverse(0..50, &model).unwrap();
/// # let compressed = coder.into_compressed().unwrap_infallible(); // `compressed` is a `Vec<u32>`.
/// # let mut compressed_clone = compressed.clone();
/// # let mutably_borrowing_cursor = Cursor::new_at_write_end(&mut compressed_clone[..]);
/// # let mut c = DefaultAnsCoder::from_compressed(mutably_borrowing_cursor.into_reversed()).unwrap();
/// # assert!(c.decode_iid_symbols(50, &model).map(UnwrapInfallible::unwrap_infallible).eq(0..50));
/// # assert!(c.is_empty());
/// # let owning_cursor = Cursor::new_at_write_end(compressed);
/// // ... obtain same `model` and `owning_cursor` as in the `c4` example above ...
/// // (could also use a mutably borrowing cursor instead)
///
/// let mut c7 = DefaultAnsCoder::from_compressed(owning_cursor.into_reversed()).unwrap();
/// assert!(c7.decode_iid_symbols(50, &model).map(UnwrapInfallible::unwrap_infallible).eq(0..50));
/// # assert!(c7.is_empty());
/// ```
///
/// [`RangeEncoder`]: crate::stream::queue::RangeEncoder
/// [`AnsCoder`]: crate::stream::stack::AnsCoder
#[derive(Debug)]
pub struct Reverse<Backend>(pub Backend);

impl<Word, B: ReadWords<Word, Stack>> ReadWords<Word, Queue> for Reverse<B> {
    type ReadError = B::ReadError;

    #[inline(always)]
    fn read(&mut self) -> Result<Option<Word>, Self::ReadError> {
        self.0.read()
    }

    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        self.0.maybe_exhausted()
    }
}

impl<Word, B: ReadWords<Word, Queue>> ReadWords<Word, Stack> for Reverse<B> {
    type ReadError = B::ReadError;

    #[inline(always)]
    fn read(&mut self) -> Result<Option<Word>, Self::ReadError> {
        self.0.read()
    }

    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        self.0.maybe_exhausted()
    }
}

impl<Word, B: BoundedReadWords<Word, Stack>> BoundedReadWords<Word, Queue> for Reverse<B> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.0.remaining()
    }

    #[inline(always)]
    fn is_exhausted(&self) -> bool {
        self.0.is_exhausted()
    }
}

impl<Word, B: BoundedReadWords<Word, Queue>> BoundedReadWords<Word, Stack> for Reverse<B> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.0.remaining()
    }

    #[inline(always)]
    fn is_exhausted(&self) -> bool {
        self.0.is_exhausted()
    }
}

impl<B: PosSeek> PosSeek for Reverse<B> {
    type Position = B::Position;
}

impl<B: Pos> Pos for Reverse<B> {
    /// Delegates the call to the wrapped backend and returns its result without doing any
    /// conversion. This is consistent with the implementaiton of `Seek::sek` for
    /// `Reverse`.
    #[inline(always)]
    fn pos(&self) -> B::Position {
        self.0.pos()
    }
}

impl<B: Seek> Seek for Reverse<B> {
    /// Passes `pos` through to the wrapped backend, i.e., doesn't do any conversion. This
    /// is consistent with the implementation of `Pos::pos` for `Reverse`.
    #[inline(always)]
    fn seek(&mut self, pos: B::Position) -> Result<(), ()> {
        self.0.seek(pos)
    }
}

// ADAPTER FOR IN-MEMORY BUFFERS ==============================================

/// Adapter that turns an in-memory buffer into an `impl ReadWords` and/or an `impl
/// WriteWords`.
///
/// A `Cursor<Word, Buf>` allows you to use an in-memory buffer `Buf` of a slice of `Word`s
/// as a source and/or sink of compressed data in an entropy coder. The type `Buf` must
/// implement `AsRef<[Word]>` to be used as a data source (i.e., an implementation of
/// [`ReadWords`]) and it must implement `AsMut<[Word]>` to be used as a data sink (i.e., an
/// implementation of [`WriteWords`]). In the most typical use cases, `Buf` is either a
/// `Vec<Word>` (if the entropy coder should own the compressed data) or a reference to a
/// slice of `Word`s, i.e., `&[Word]` (if the entropy coder should only have shared access
/// to the compressed data, e.g., because you want to keep the compressed data alive even
/// after the entropy coder gets dropped).
///
/// A `Cursor<Word, Buf>` implements `ReadWords` for both [`Queue`] and [`Stack`] semantics.
/// By convention, reading with `Queue` semantics incremenets the `Cursor`'s index into the
/// slice returned by `.as_ref()` whereas reading with `Stack` semantics decrements the
/// index. Whether `Queue` or `Stack` semantics will be used is usually decided by the
/// implementation of the entropy coder that uses the `Cursor` as its backend. If you want
/// to read in the opposite direction than what's the convention for your use case (e.g.,
/// because you've already manually reversed the order of the `Word`s in the buffer) then
/// wrap the `Cursor` in a [`Reverse`]. The implementation of `WriteWords<Word>` (if `Buf`
/// implements `AsMut<[Word]>`) always writes in the same direction in which
/// `ReadWords<Word, Queue>` reads.
///
/// # Examples
///
/// ## Sharing and Owning Cursors
///
/// The following example shows how a `Cursor` can be used to decode both shared and owned
/// compressed data with a [`RangeDecoder`]:
///
/// ```
/// use constriction::{
///     stream::{
///         model::DefaultLeakyQuantizer, queue::{DefaultRangeEncoder, DefaultRangeDecoder},
///         Encode, Decode
///     },
///     UnwrapInfallible,
/// };
///
/// // Some simple entropy model, just for demonstration purpose.
/// let quantizer = DefaultLeakyQuantizer::new(-100..=100);
/// let model = quantizer.quantize(probability::distribution::Gaussian::new(25.0, 10.0));
///
/// // Encode the symbols `0..100` using a `RangeEncoder` (uses the default `Vec` backend because
/// // we don't know the size of the compressed data upfront).
/// let mut encoder = DefaultRangeEncoder::new();
/// encoder.encode_iid_symbols(0..100, &model);
/// let compressed = encoder.into_compressed().unwrap_infallible(); // `compressed` is a `Vec<u32>`.
/// dbg!(compressed.len()); // Prints "compressed.len() = 40".
///
/// // Create a `RangeDecoder` with shared access to the compressed data. This constructs a
/// // `Cursor<u32, &[u32]>` that points to the beginning of the data and loads it in the decoder.
/// let mut sharing_decoder
///     = DefaultRangeDecoder::from_compressed(&compressed[..]).unwrap_infallible();
/// // `sharing_decoder` has type `RangeDecoder<u32, u64, Cursor<u32, &'a [u32]>`.
///
/// // Decode the data and verify correctness.
/// assert!(sharing_decoder.decode_iid_symbols(100, &model).map(Result::unwrap).eq(0..100));
/// assert!(sharing_decoder.maybe_exhausted());
///
/// // We can still use `compressed` because we gave the decoder only shared access to it. Thus,
/// // `sharing_decoder` contains a reference into `compressed`, so we couldn't return it from the
/// // current function. If we want to return a decoder, we have to give it ownership of the data:
/// let mut owning_decoder = DefaultRangeDecoder::from_compressed(compressed).unwrap_infallible();
/// // `owning_decoder` has type `RangeDecoder<u32, u64, Cursor<u32, Vec<u32>>`.
///
/// // Verify that we can decode the data again.
/// assert!(owning_decoder.decode_iid_symbols(100, &model).map(Result::unwrap).eq(0..100));
/// assert!(owning_decoder.maybe_exhausted());
/// ```
///
/// ## `Cursor`s automatically use the correct `Semantics`
///
/// You can use a `Cursor` also as a stack, e.g., for an [`AnsCoder`]. The `Cursor` will
/// automatically read data in the correct (i.e., reverse) direction when it is invoked with
/// `Stack` semantics. Note, however, that using a `Cursor` is not always necessary when you
/// decode with an `AnsCoder` because the `AnsCoder` can also decode directly from a `Vec`
/// (see last example below). However, you'll need a `Cursor` if you don't own the
/// compressed data:
///
/// ```
/// # use constriction::{
/// #     stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode},
/// #     CoderError, UnwrapInfallible,
/// # };
/// #
/// fn decode_shared_data(amt: usize, compressed: &[u32]) -> Vec<i32> {
///     // Some simple entropy model, just for demonstration purpose.
///     let quantizer = DefaultLeakyQuantizer::new(-100..=100);
///     let model = quantizer.quantize(probability::distribution::Gaussian::new(25.0, 10.0));
///
///     // `AnsCoder::from_compressed_slice` wraps the provided compressed data in a `Cursor` and
///     // initializes the cursor position at the end (= top of the stack; see documentation of
///     // `Reverse` if you want to read the data from the beginning instead).
///     let mut decoder = DefaultAnsCoder::from_compressed_slice(compressed).unwrap();
///     decoder.decode_iid_symbols(amt, &model).collect::<Result<Vec<_>, _>>().unwrap_infallible()
/// }
/// #
/// # let quantizer = DefaultLeakyQuantizer::new(-100..=100);
/// # let model = quantizer.quantize(probability::distribution::Gaussian::new(25.0, 10.0));
/// # let mut coder = DefaultAnsCoder::new();
/// # coder.encode_iid_symbols_reverse(0..100, &model).unwrap();
/// # let compressed = coder.into_compressed().unwrap_infallible();
/// # assert!(decode_shared_data(100, &compressed).iter().cloned().eq(0..100));
/// ```
///
/// ## Owning `Cursor`s vs `Vec`s
///
/// If you have ownership of the compressed data, then decoding it with an `AnsCoder`
/// doesn't always require a `Cursor`. An `AnsCoder` can also directly decode from a
/// `Vec<Word>` backend. The difference between `Vec<Word>` and an owning cursor
/// `Cursor<Word, Vec<Word>>` is that decoding from a `Vec` *consumes* the compressed data
/// (so you can interleave multiple encoding/decoding steps arbitrarily) whereas a `Cursor`
/// (whether it be sharing or owning) does not consume the compressed data that is read from
/// it. You can still interleave multiple encoding/decoding steps with an `AnsCoder` that
/// uses a `Cursor` instead of a `Vec` backend, but since a `Cursor` doesn't grow or shrink
/// the wrapped buffer you will typically either run out of buffer space at some point or
/// the final buffer will be padded to its original size with some partially overwritten
/// left-over compressed data (for older readers like myself: think of a `Cursor` as a
/// cassette recorder).
///
/// ```
/// use constriction::{
///     backends::Cursor, stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode},
///     CoderError, UnwrapInfallible,
/// };
///
/// // Some simple entropy model, just for demonstration purpose.
/// let quantizer = DefaultLeakyQuantizer::new(-100..=100);
/// let model = quantizer.quantize(probability::distribution::Gaussian::new(25.0, 10.0));
///
/// // Encode the symbols `0..50` using a stack entropy coder and get the compressed data.
/// let mut coder = DefaultAnsCoder::new();
/// coder.encode_iid_symbols_reverse(0..50, &model).unwrap();
/// let compressed = coder.into_compressed().unwrap_infallible(); // `compressed` is a `Vec<u32>`.
/// dbg!(compressed.len()); // Prints "compressed.len() = 11".
///
/// // We can either reconstruct (a clone of) the original `coder` with `Vec` backend and decode
/// // data and/or encode some more data, or even do both in any order.
/// let mut vec_coder = DefaultAnsCoder::from_compressed(compressed.clone()).unwrap();
/// // Decode the top half of the symbols off the stack and verify correctness.
/// assert!(
///     vec_coder.decode_iid_symbols(25, &model)
///         .map(UnwrapInfallible::unwrap_infallible)
///         .eq(0..25)
/// );
/// // Then encode some more symbols onto it.
/// vec_coder.encode_iid_symbols_reverse(50..75, &model).unwrap();
/// let compressed2 = vec_coder.into_compressed().unwrap_infallible();
/// dbg!(compressed2.len()); // Prints "compressed2.len() = 17"
/// // `compressed2` is longer than `compressed1` because the symbols we poped off had lower
/// // information content under the `model` than the symbols we replaced them with.
///
/// // In principle, we could have done the same with an `AnsCoder` that uses a `Cursor` backend.
/// let cursor = Cursor::new_at_write_end(compressed); // Could also use `&mut compressed[..]`.
/// let mut cursor_coder = DefaultAnsCoder::from_compressed(cursor).unwrap();
/// // Decode the top half of the symbols off the stack and verify correctness.
/// assert!(
///     cursor_coder.decode_iid_symbols(25, &model)
///         .map(UnwrapInfallible::unwrap_infallible)
///         .eq(0..25)
/// );
/// // Encoding *a few* more symbols works ...
/// cursor_coder.encode_iid_symbols_reverse(65..75, &model).unwrap();
/// // ... but at some point we'll run out of buffer space.
/// assert_eq!(
///     cursor_coder.encode_iid_symbols_reverse(50..65, &model),
///     Err(CoderError::Backend(constriction::backends::BoundedWriteError::OutOfSpace))
/// );
/// ```
///
/// [`RangeDecoder`]: crate::stream::queue::RangeDecoder
/// [`AnsCoder`]: crate::stream::stack::AnsCoder
#[derive(Clone, Debug)]
pub struct Cursor<Word, Buf> {
    buf: Buf,

    /// The index of the next word to be read with a `ReadWords<Word, Queue>` or written
    /// with a `WriteWords<Word>, and one plus the index of the next word to read with
    /// `ReadWords<Word, Stack>.
    ///
    /// Satisfies the invariant `pos <= buf.as_ref().len()` if `Buf: AsRef<[Word]>` (see
    /// unsafe trait `SafeBuf`).
    pos: usize,

    phantom: PhantomData<Word>,
}

/// Unsafe marker trait indicating sane implementation of `AsRef` (and possibly `AsMut`).
///
/// By implementing `SafeBuf<Word>` for a type `T`, you guarantee that
/// - calling `x.as_ref()` for some `x: T` several times in a row (with no other method
///   calls on `x` in-between) never returns slices of decreasing length; and
/// - if `T` implements `AsMut<[Word]>` then the above property must also hold for any
///   sequence of calls of `x.as_ref()` and `x.as_mut()`, and the lengths of slices returned
///   by either of these calls must not decrease.
///
/// This is very likely the behaviour you would expect anyway for `AsRef` and `AsMut`. This
/// guarantee allows the implementation of `ReadWords<Word, Stack>` for [`Cursor`] to elide
/// an additional pedantic bounds check by maintaining an in-bounds invariant on its index
/// into the buffer.
///
/// # Safety
///
/// If `SafeBuf` is implemented for a type `Buf` that violates the above contract then the
/// implementations of `ReadWords<Word, Stack>::read` for `Cursor<Word, Buf>` and of
/// `WriteWords<Word>` for `Reverse<Cursor<Word, Buf>>` may attempt to access the buffer out
/// of bounds without bounds checks.
pub unsafe trait SafeBuf<Word>: AsRef<[Word]> {}

unsafe impl<'a, Word> SafeBuf<Word> for &'a [Word] {}
unsafe impl<'a, Word> SafeBuf<Word> for &'a mut [Word] {}
unsafe impl<Word> SafeBuf<Word> for Vec<Word> {}
unsafe impl<Word> SafeBuf<Word> for Box<[Word]> {}

impl<Word, Buf> Cursor<Word, Buf> {
    /// Creates a `Cursor` for the buffer `buf` and initializes the cursor position to point
    /// at the beginning (i.e., index zero) of the buffer.
    ///
    /// You can use the resulting cursor, for decoding compressed data with `Queue`
    /// semantics (for example, calling [`RangeDecoder::from_compressed`] with a vector or
    /// slice of `Word`s will result in a call to `Cursor::new_at_write_beginning`).
    ///
    /// If you want to read from the resulting buffer with `Stack` semantics then you'll
    /// have to wrap it in a [`Reverse`], i.e., `let reversed_cursor =
    /// Reverse(Cursor::new_at_write_beginning(buf))`. This usually only makes sense if
    /// you've already manually reversed the order of `Word`s in `buf`. See documentation of
    /// [`Reverse`] for an example.
    ///
    /// This method is called `new_at_write_beginning` rather than simply `new_at_beginning`
    /// just to avoid confusion around the meaning of the word "beginning". This doesn't
    /// mean that you must (or even can, necessarily) use the resulting `Cursor` for
    /// writing. But the unqualified word "beginning" would be ambiguous since reading from
    /// a `Cursor` could start (i.e., "begin") at either boundary of the buffer (depending
    /// on the `Semantics`). By contrast, writing to a `Cursor` always "begins" at index
    /// zero, so "write_beginning" is unambiguous.
    ///
    /// [`RangeDecoder::from_compressed`]:
    /// crate::stream::queue::RangeDecoder::from_compressed
    #[inline(always)]
    pub fn new_at_write_beginning(buf: Buf) -> Self {
        Self {
            buf,
            pos: 0,
            phantom: PhantomData,
        }
    }

    /// Creates a `Cursor` for the buffer `buf` and initializes the cursor position to point
    /// at the end of the buffer.
    ///
    /// You can use the resulting cursor, for decoding compressed data with `Stack`
    /// semantics (for example, [`AnsCoder::from_compressed_slice`] calls
    /// `Cursor::new_at_write_end` internally).
    ///
    /// This method is called `new_at_write_end` rather than simply `new_at_end` just to
    /// avoid confusion around the meaning of the word "end". This doesn't mean that you
    /// must (or even can, necessarily) use the resulting `Cursor` for writing. But the
    /// unqualified word "end" would be ambiguous since reading from a `Cursor` could
    /// terminate (i.e., "end") at either boundary of the buffer (depending on the
    /// `Semantics`). By contrast, writing to a `Cursor` always "ends" at index
    /// `.as_ref().len()`, so "write_end" is unambiguous.
    ///
    /// [`AnsCoder::from_compressed_slice`]:
    /// crate::stream::stack::AnsCoder::from_compressed_slice
    #[inline(always)]
    pub fn new_at_write_end(buf: Buf) -> Self
    where
        Buf: AsRef<[Word]>,
    {
        let pos = buf.as_ref().len();
        Self {
            buf,
            pos,
            phantom: PhantomData,
        }
    }

    /// Same as [`new_at_write_end`] but for `Buf`s that implement `AsMut` but don't
    /// implement `AsRef`.
    ///
    /// You can usually just call `new_at_write_end`, it will still give you mutable access
    /// (i.e., implement `WriteWords`) if `Buf` implements `AsMut`.
    ///
    /// [`new_at_write_end`]: Self::new_at_write_end
    #[inline(always)]
    pub fn new_at_write_end_mut(mut buf: Buf) -> Self
    where
        Buf: AsMut<[Word]>,
    {
        let pos = buf.as_mut().len();
        Self {
            buf,
            pos,
            phantom: PhantomData,
        }
    }

    /// Creates a `Cursor` for the buffer `buf` and initializes the cursor position to point
    /// at the given index `pos`.
    ///
    /// You can use the resulting cursor for reading compressed data with both `Queue` and
    /// `Stack` semantics, or for writing data (if `Buf` implements `AsMut`). Reading will
    /// automatically advance the cursor position in the correct direction depending on
    /// whether the read uses `Queue` or `Stack` semantics.
    ///
    /// This method is only useful if you want to point the cursor somewhere in the middle
    /// of the buffer. If you want to initalize the cursor position at either end of the
    /// buffer then calling [`new_at_write_beginning`] or [`new_at_write_end`] expresses
    /// your intent more clearly.
    ///
    /// [`new_at_write_beginning`]: Self::new_at_write_beginning
    /// [`new_at_write_end`]: Self::new_at_write_end
    #[allow(clippy::result_unit_err)]
    pub fn new_at_pos(buf: Buf, pos: usize) -> Result<Self, ()>
    where
        Buf: AsRef<[Word]>,
    {
        if pos > buf.as_ref().len() {
            Err(())
        } else {
            Ok(Self {
                buf,
                pos,
                phantom: PhantomData,
            })
        }
    }

    /// Same as [`new_at_pos`] but for `Buf`s that implement `AsMut` but don't implement
    /// `AsRef`.
    ///
    /// You can usually just call `new_at_pos`, it will still give you mutable access (i.e.,
    /// implement `WriteWords`) if `Buf` implements `AsMut`.
    ///
    /// [`new_at_pos`]: Self::new_at_pos
    #[allow(clippy::result_unit_err)]
    pub fn new_at_pos_mut(mut buf: Buf, pos: usize) -> Result<Self, ()>
    where
        Buf: AsMut<[Word]>,
    {
        if pos > buf.as_mut().len() {
            Err(())
        } else {
            Ok(Self {
                buf,
                pos,
                phantom: PhantomData,
            })
        }
    }

    /// Returns a new (read-only) `Cursor` that shares its buffer with the current `Cursor`.
    ///
    /// The new `Cursor` is initialized to point at the same position where the current
    /// `Cursor` currently points to, but it can move around independently from the current
    /// `Cursor`. This is a cheaper variant of [`cloned`] since it doesn't copy the data in
    /// the buffer.
    ///
    /// Note that the lifetime of the new `Cursor` is tied to the liefetime of `&self`, so
    /// you won't be able to mutably access the current `Cursor` while the new `Cursor` is
    /// alive. Unfortunately, this excludes both reading and writing from the current
    /// `Cursor` (since reading and writing mutates the `Cursor` as it advances its
    /// position). If you want to create multiple cursors with the same buffer without
    /// copying the buffer, then create a `Cursor` for a slice `&[Word]` (e.g., by calling
    /// `.as_view()` once) and then `.clone()` that `Cursor` (which won't clone the contents
    /// of the buffer, only the pointer to it):
    ///
    /// ```
    /// use constriction::{backends::{Cursor, ReadWords}, Queue};
    /// let data = vec![1, 2, 3, 4];
    ///
    /// // Either directly create a `Cursor` for a slice and clone that ...
    /// let mut cursor = Cursor::new_at_write_beginning(&data[..]);
    /// assert_eq!(<_ as ReadWords<u32, Queue>>::read(&mut cursor), Ok(Some(1)));
    /// let mut cursor_clone = cursor.clone(); // Doesn't clone the data, only the pointer to it.
    /// // `cursor_clone` initially points to the same position as `cursor` but their positions
    /// // advance independently from each other:
    /// assert_eq!(<_ as ReadWords<u32, Queue>>::read(&mut cursor), Ok(Some(2)));
    /// assert_eq!(<_ as ReadWords<u32, Queue>>::read(&mut cursor), Ok(Some(3)));
    /// assert_eq!(<_ as ReadWords<u32, Queue>>::read(&mut cursor_clone), Ok(Some(2)));
    /// assert_eq!(<_ as ReadWords<u32, Queue>>::read(&mut cursor_clone), Ok(Some(3)));
    ///
    /// // ... or, if someone gave you a `Cursor` that owns its buffer, then you can call `.as_view()`
    /// // on it once to get a `Cursor` to a slice, which you can then clone cheaply again.
    /// let mut original = Cursor::new_at_write_beginning(data);
    /// assert_eq!(<_ as ReadWords<u32, Queue>>::read(&mut original), Ok(Some(1)));
    /// // let mut clone = original.clone(); // <-- This would clone the data, which could be expensive.
    /// let mut view = original.as_view();   // `view` is a `Cursor<u32, &[u32]>`
    /// let mut view_clone = view.clone();   // Doesn't clone the data, only the pointer to it.
    /// assert_eq!(<_ as ReadWords<u32, Queue>>::read(&mut view), Ok(Some(2)));
    /// assert_eq!(<_ as ReadWords<u32, Queue>>::read(&mut view_clone), Ok(Some(2)));
    /// ```
    ///
    /// If we had instead used `original` while `view` was still alive then the borrow
    /// checker would have complained:
    ///
    /// ```compile_fail
    /// use constriction::{backends::{Cursor, ReadWords}, Queue};
    /// let data = vec![1, 2, 3, 4];
    /// let mut original = Cursor::new_at_write_beginning(data);
    /// let mut view = original.as_view();
    ///
    /// <_ as ReadWords<u32, Queue>>::read(&mut original); // Error: mutable borrow occurs here
    /// <_ as ReadWords<u32, Queue>>::read(&mut view);     // immutable borrow later used here
    /// ```
    ///
    /// [`cloned`]: Self::cloned
    pub fn as_view(&self) -> Cursor<Word, &[Word]>
    where
        Buf: AsRef<[Word]>,
    {
        Cursor {
            buf: self.buf.as_ref(),
            pos: self.pos,
            phantom: PhantomData,
        }
    }

    /// Same as [`as_view`] except that the returned view also implements [`WriteWords`].
    ///
    /// [`as_view`]: Self::as_view
    pub fn as_mut_view(&mut self) -> Cursor<Word, &mut [Word]>
    where
        Buf: AsMut<[Word]>,
    {
        Cursor {
            buf: self.buf.as_mut(),
            pos: self.pos,
            phantom: PhantomData,
        }
    }

    /// Makes a deep copy of the Cursor, copying the data to a new, owned buffer.
    ///
    /// If you don't need ownership over the data then use [`as_view`] instead as it is
    /// cheaper.
    ///
    /// This method is different from [`Clone::clone`] because the return type isn't
    /// necessarily identical to `Self`. If you have a `Cursor` that doesn't own its data
    /// (for example, a `Cursor<Word, &[Word]>`), then calling `.clone()` on it is cheap
    /// since it doesn't copy the data (only the pointer to it), but calling `.cloned()` is
    /// expensive if the buffer is large.
    ///
    /// [`as_view`]: Self::as_view
    pub fn cloned(&self) -> Cursor<Word, Vec<Word>>
    where
        Word: Clone,
        Buf: AsRef<[Word]>,
    {
        Cursor {
            buf: self.buf.as_ref().to_vec(),
            pos: self.pos,
            phantom: PhantomData,
        }
    }

    /// Returns a reference to the generic buffer that the `Cursor` reads from or writes to.
    ///
    /// To get the actual slice of `Word`s, call `cursor.buf().as_ref()`.
    pub fn buf(&self) -> &Buf {
        &self.buf
    }

    /// Returns a mutable reference to the generic buffer that the `Cursor` reads from or
    /// writes to.
    ///
    /// Same as [`buf`](Self::buf) except that it requires mutable access to `self` and
    /// returns a mutable reference.
    ///
    /// To get the actual mutable slice of `Word`s, call `cursor.buf().as_mut()` (if `Buf`
    /// implements `AsMut`).
    pub fn buf_mut(&mut self) -> &mut Buf {
        &mut self.buf
    }

    /// Consumes the `Cursor` and returns the buffer and the current position.
    ///
    /// If you don't want to consume the `Cursor` then call [`buf`](Self::buf) or
    /// [`buf_mut`](Self::buf_mut) and [`pos`](Pos::pos) instead. You'll have to bring the
    /// [`Pos`] trait into scope for the last one to work (`use constriction::Pos;`).
    pub fn into_buf_and_pos(self) -> (Buf, usize) {
        (self.buf, self.pos)
    }

    /// Reverses both the data and the reading direction.
    ///
    /// This method consumes the original `Cursor`, reverses the order of the `Word`s
    /// in-place, updates the cursor position accordingly, and returns a `Cursor`-like
    /// backend that progresses in the opposite direction for reads and/or writes. Reading
    /// from and writing to the returned backend will have identical behavior as in the
    /// original `Cursor` backend, but the flipped directions will be observable through
    /// [`Pos::pos`], [`Seek::seek`], and [`Self::buf`].
    ///
    /// See documentation of [`Reverse`] for more information and a usage example.
    pub fn into_reversed(mut self) -> Reverse<Self>
    where
        Buf: AsMut<[Word]>,
    {
        self.buf.as_mut().reverse();
        self.pos = self.buf.as_mut().len() - self.pos;
        Reverse(self)
    }
}

impl<Word, Buf> Reverse<Cursor<Word, Buf>> {
    /// Reverses both the data and the reading direction.
    ///
    /// This is essentially the same as [`Cursor::into_reversed`], except that, rather than
    /// wrapping yet another `Reverse` around the `Cursor`, the last step of this method
    /// just removes the existing `Reverse` wrapper, which has the same effect.
    ///
    /// See documentation of [`Reverse`] for more information and a usage example.
    #[inline(always)]
    pub fn into_reversed(self) -> Cursor<Word, Buf>
    where
        Buf: AsMut<[Word]>,
    {
        // Accessing `.0` twice removes *two* `Reverse`, resulting in no semantic change.
        self.0.into_reversed().0
    }
}

impl<Word, Buf: AsMut<[Word]>> WriteWords<Word> for Cursor<Word, Buf> {
    type WriteError = BoundedWriteError;

    #[inline(always)]
    fn write(&mut self, word: Word) -> Result<(), Self::WriteError> {
        if let Some(target) = self.buf.as_mut().get_mut(self.pos) {
            *target = word;
            self.pos += 1;
            Ok(())
        } else {
            Err(BoundedWriteError::OutOfSpace)
        }
    }
}

impl<Word, Buf: AsMut<[Word]> + AsRef<[Word]>> BoundedWriteWords<Word> for Cursor<Word, Buf> {
    #[inline(always)]
    fn space_left(&self) -> usize {
        self.buf.as_ref().len() - self.pos
    }
}

impl<Word, Buf: SafeBuf<Word> + AsMut<[Word]>> WriteWords<Word> for Reverse<Cursor<Word, Buf>> {
    type WriteError = BoundedWriteError;

    #[inline(always)]
    fn write(&mut self, word: Word) -> Result<(), Self::WriteError> {
        if self.0.pos == 0 {
            Err(BoundedWriteError::OutOfSpace)
        } else {
            self.0.pos -= 1;
            unsafe {
                // SAFETY: We maintain the invariant `self.0.pos <= self.0.buf.as_mut().len()`
                // and we just decreased `self.0.pos` (and made sure that didn't wrap around),
                // so we now have `self.0.pos < self.0.buf.as_mut().len()`.
                *self.0.buf.as_mut().get_unchecked_mut(self.0.pos) = word;
                Ok(())
            }
        }
    }
}

impl<Word, Buf: SafeBuf<Word> + AsMut<[Word]>> BoundedWriteWords<Word>
    for Reverse<Cursor<Word, Buf>>
{
    #[inline(always)]
    fn space_left(&self) -> usize {
        self.0.buf.as_ref().len()
    }
}

/// Error type for data sinks with a finite capacity.
///
/// This is currently used as the `WriteError` in the [implementation of `WriteWords` for
/// `Cursor`](struct.Cursor.html#impl-WriteWords<Word>) but it should also be used in the
/// implementation of [`WriteWords`] for custom types where appropriate.
///
/// If you use this error type for a data sink then you may also want to implement
/// [`BoundedWriteWords`] for it (if the capacity is known upfront).
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum BoundedWriteError {
    /// Attempting to write compressed data failed because it would exceeded the finite
    /// capacity of the data sink.
    OutOfSpace,
}

impl Display for BoundedWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfSpace => write!(f, "Out of space."),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BoundedWriteError {}

impl<Word: Clone, Buf: SafeBuf<Word>> ReadWords<Word, Stack> for Cursor<Word, Buf> {
    type ReadError = Infallible;

    #[inline(always)]
    fn read(&mut self) -> Result<Option<Word>, Self::ReadError> {
        if self.pos == 0 {
            Ok(None)
        } else {
            self.pos -= 1;
            unsafe {
                // SAFETY: We maintain the invariant `self.pos <= self.buf.as_ref().len()`
                // and we just decreased `self.pos` (and made sure that didn't wrap around),
                // so we now have `self.pos < self.buf.as_ref().len()`.
                Ok(Some(self.buf.as_ref().get_unchecked(self.pos).clone()))
            }
        }
    }

    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        BoundedReadWords::<Word, Stack>::is_exhausted(self)
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>> ReadWords<Word, Queue> for Cursor<Word, Buf> {
    type ReadError = Infallible;

    #[inline(always)]
    fn read(&mut self) -> Result<Option<Word>, Self::ReadError> {
        let maybe_word = self.buf.as_ref().get(self.pos).cloned();
        if maybe_word.is_some() {
            self.pos += 1;
        }
        Ok(maybe_word)
    }

    #[inline(always)]
    fn maybe_exhausted(&self) -> bool {
        BoundedReadWords::<Word, Queue>::is_exhausted(self)
    }
}

impl<Word: Clone, Buf: SafeBuf<Word>> BoundedReadWords<Word, Stack> for Cursor<Word, Buf> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.pos
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>> BoundedReadWords<Word, Queue> for Cursor<Word, Buf> {
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.buf.as_ref().len() - self.pos
    }
}

impl<Word, Buf> PosSeek for Cursor<Word, Buf> {
    type Position = usize;
}

impl<Word, Buf: AsRef<[Word]>> Pos for Cursor<Word, Buf> {
    #[inline(always)]
    fn pos(&self) -> usize {
        self.pos
    }
}

impl<Word, Buf: AsRef<[Word]>> Seek for Cursor<Word, Buf> {
    #[inline(always)]
    fn seek(&mut self, pos: usize) -> Result<(), ()> {
        if pos > self.buf.as_ref().len() {
            // Note that `pos == buf.len()` is still a valid position (EOF for queues and
            // beginning for stacks).
            Err(())
        } else {
            self.pos = pos;
            Ok(())
        }
    }
}

impl<Word: Clone, Buf: SafeBuf<Word>> IntoReadWords<Word, Stack> for Buf {
    type IntoReadWords = Cursor<Word, Buf>;

    fn into_read_words(self) -> Self::IntoReadWords {
        Cursor::new_at_write_end(self)
    }
}

impl<Word: Clone, Buf: AsRef<[Word]>> IntoReadWords<Word, Queue> for Buf {
    type IntoReadWords = Cursor<Word, Buf>;

    fn into_read_words(self) -> Self::IntoReadWords {
        Cursor::new_at_write_beginning(self)
    }
}

impl<'a, Word: Clone + 'a, Buf: SafeBuf<Word> + 'a> AsReadWords<'a, Word, Stack> for Buf {
    type AsReadWords = Cursor<Word, &'a [Word]>;

    fn as_read_words(&'a self) -> Self::AsReadWords {
        Cursor::new_at_write_end(self.as_ref())
    }
}

impl<'a, Word: Clone + 'a, Buf: AsRef<[Word]> + 'a> AsReadWords<'a, Word, Queue> for Buf {
    type AsReadWords = Cursor<Word, &'a [Word]>;

    fn as_read_words(&'a self) -> Self::AsReadWords {
        Cursor::new_at_write_beginning(self.as_ref())
    }
}

impl<Word, Buf, S: Semantics> IntoSeekReadWords<Word, S> for Buf
where
    Buf: AsRef<[Word]> + IntoReadWords<Word, S, IntoReadWords = Cursor<Word, Buf>>,
    Cursor<Word, Buf>: ReadWords<Word, S>,
{
    type IntoSeekReadWords = Cursor<Word, Buf>;

    fn into_seek_read_words(self) -> Self::IntoSeekReadWords {
        self.into_read_words()
    }
}

impl<'a, Word: 'a, Buf, S: Semantics> AsSeekReadWords<'a, Word, S> for Buf
where
    Buf: AsReadWords<'a, Word, S, AsReadWords = Cursor<Word, &'a [Word]>>,
    Cursor<Word, &'a [Word]>: ReadWords<Word, S>,
{
    type AsSeekReadWords = Cursor<Word, &'a [Word]>;

    fn as_seek_read_words(&'a self) -> Self::AsSeekReadWords {
        self.as_read_words()
    }
}

// READ ADAPTER FOR ITERATORS =================================================

/// Adapter that turns an iterator over `Result<Word, ReadError>` into a data source.
///
/// Wraps an iterator over `Result<Word, ReadError>` and implements [`ReadWords<Word, S,
/// ReadError=ReadError>`](ReadWords) by pulling the iterator each time a client reads from
/// it. If the iterator implements [`ExactSizeIterator`] then this wrapper also implements
/// [`BoundedReadWords`].
///
/// Implements `ReadWord` for arbitrary [`Semantics`]. This is legal since it doesn't
/// implement `WriteWords`, so the question how reads relate to writes is moot.
///
/// See also [`InfallibleIteratorReadWords`], and [module-level documentation](self) for a
/// detailed usage example.
#[derive(Clone, Debug)]
pub struct FallibleIteratorReadWords<Iter: Iterator> {
    inner: core::iter::Fuse<Iter>,
}

impl<Iter: Iterator> FallibleIteratorReadWords<Iter> {
    /// Creates the adapter for the provided iterator.
    ///
    /// The provided iterator `iter` does *not* need to be fused (i.e., it may return `Some`
    /// after the first `None` even though a [`ReadWords`] wouldn't be allowed to do the
    /// equivalent of that). The adapter calls `iter.fuse()` to ensure correct behavior.
    ///
    /// You can get the (fused) iterator back by calling [IntoIterator::into_iter].
    pub fn new<I, Word, ReadError>(iter: I) -> Self
    where
        I: IntoIterator<IntoIter = Iter>,
        Iter: Iterator<Item = Result<Word, ReadError>>,
    {
        Self {
            inner: iter.into_iter().fuse(),
        }
    }
}

impl<Iter: Iterator> IntoIterator for FallibleIteratorReadWords<Iter> {
    type Item = Iter::Item;
    type IntoIter = core::iter::Fuse<Iter>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }
}

/// Since `FallibleIteratorReadWords` doesn't implement `WriteWords`, it is allowed to
/// implement `ReadWords` for all `ReadWriteLogic`s
impl<Iter, S, Word, ReadError> ReadWords<Word, S> for FallibleIteratorReadWords<Iter>
where
    Iter: Iterator<Item = Result<Word, ReadError>>,
    S: Semantics,
    ReadError: Debug,
{
    type ReadError = ReadError;

    #[inline(always)]
    fn read(&mut self) -> Result<Option<Word>, Self::ReadError> {
        self.inner.next().transpose()
    }
}

impl<Iter, S, Word> BoundedReadWords<Word, S> for FallibleIteratorReadWords<Iter>
where
    Self: ReadWords<Word, S>,
    Iter: ExactSizeIterator,
    S: Semantics,
{
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.inner.len()
    }
}

/// Adapter that turns an iterator over `Word` into a data source.
///
/// Wraps an iterator over `Word` and implements [`ReadWords<Word, S,
/// ReadError=Infallible>`](ReadWords) by pulling the iterator each time a client reads from
/// it. If the iterator implements [`ExactSizeIterator`] then this wrapper also implements
/// [`BoundedReadWords`].
///
/// Implements `ReadWord` for arbitrary [`Semantics`]. This is legal since it doesn't
/// implement `WriteWords`, so the question how reads relate to writes is moot.
///
/// See also [`FallibleIteratorReadWords`], and [module-level documentation](self) for a
/// detailed usage example.
#[derive(Clone, Debug)]
pub struct InfallibleIteratorReadWords<Iter: Iterator> {
    inner: core::iter::Fuse<Iter>,
}

impl<Iter: Iterator> InfallibleIteratorReadWords<Iter> {
    /// Creates the adapter for the provided iterator.
    ///
    /// The provided iterator `iter` does *not* need to be fused (i.e., it may return `Some`
    /// after the first `None` even though a [`ReadWords`] wouldn't be allowed to do the
    /// equivalent of that). The adapter calls `iter.fuse()` to ensure correct behavior.
    ///
    /// You can get the (fused) iterator back by calling [IntoIterator::into_iter].
    pub fn new<I, Word, ReadError>(iter: I) -> Self
    where
        I: IntoIterator<IntoIter = Iter>,
        Iter: Iterator<Item = Result<Word, ReadError>>,
    {
        Self {
            inner: iter.into_iter().fuse(),
        }
    }
}

impl<Iter: Iterator> IntoIterator for InfallibleIteratorReadWords<Iter> {
    type Item = Iter::Item;
    type IntoIter = core::iter::Fuse<Iter>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }
}

/// Since `InfallibleIteratorReadWords` doesn't implement `WriteWords`, it is allowed to
/// implement `ReadWords` for all `ReadWriteLogic`s
impl<Iter, S, Word> ReadWords<Word, S> for InfallibleIteratorReadWords<Iter>
where
    Iter: Iterator<Item = Word>,
    S: Semantics,
{
    type ReadError = Infallible;

    #[inline(always)]
    fn read(&mut self) -> Result<Option<Word>, Infallible> {
        Ok(self.inner.next())
    }
}

impl<Iter, S, Word> BoundedReadWords<Word, S> for InfallibleIteratorReadWords<Iter>
where
    Self: ReadWords<Word, S>,
    Iter: ExactSizeIterator,
    S: Semantics,
{
    #[inline(always)]
    fn remaining(&self) -> usize {
        self.inner.len()
    }
}

// WRITE ADAPTER FOR CALLBACKS ================================================

/// Adapter that turns a fallible callback into a fallible data sink.
///
/// Wraps a callback function from `Word` to `Result<(), Err>` and implements
/// [`WriteWords<Word, ReadError=Err>`](WriteWords) by calling the callback each time a
/// client writes to it.
///
/// See also [`InfallibleCallbackWriteWords`], and [module-level documentation](self) for a
/// detailed usage example.
#[derive(Clone, Debug)]
pub struct FallibleCallbackWriteWords<Callback> {
    write_callback: Callback,
}

impl<Callback> FallibleCallbackWriteWords<Callback> {
    /// Creates the adapter for the provided callback.
    pub fn new(write_callback: Callback) -> Self {
        Self { write_callback }
    }

    /// Consumes the adapter and returns the provided callback.
    pub fn into_inner(self) -> Callback {
        self.write_callback
    }
}

impl<Word, WriteError, Callback> WriteWords<Word> for FallibleCallbackWriteWords<Callback>
where
    Callback: FnMut(Word) -> Result<(), WriteError>,
    WriteError: Debug,
{
    type WriteError = WriteError;

    fn write(&mut self, word: Word) -> Result<(), Self::WriteError> {
        (self.write_callback)(word)
    }
}

/// Adapter that turns an infallible callback into an infallible data sink.
///
/// Wraps a callback function from `Word` to `()` and implements [`WriteWords<Word,
/// WriteError=Infallible>`](WriteWords) by calling the callback each time a client writes
/// to it.
///
/// See also [`FallibleCallbackWriteWords`], and [module-level documentation](self) for a
/// detailed usage example.
#[derive(Clone, Debug)]
pub struct InfallibleCallbackWriteWords<Callback> {
    write_callback: Callback,
}

impl<Callback> InfallibleCallbackWriteWords<Callback> {
    /// Creates the adapter for the provided callback.
    pub fn new(write_callback: Callback) -> Self {
        Self { write_callback }
    }

    /// Consumes the adapter and returns the provided callback.
    pub fn into_inner(self) -> Callback {
        self.write_callback
    }
}

impl<Word, Callback> WriteWords<Word> for InfallibleCallbackWriteWords<Callback>
where
    Callback: FnMut(Word),
{
    type WriteError = Infallible;

    fn write(&mut self, word: Word) -> Result<(), Infallible> {
        (self.write_callback)(word);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::stream::{model::DefaultLeakyQuantizer, stack::DefaultAnsCoder, Decode};
    use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
    use probability::distribution::Gaussian;
    use std::{
        fs::File,
        io::{BufReader, BufWriter},
    };

    #[test]
    #[cfg_attr(miri, ignore)]
    fn decode_on_the_fly_stack() {
        fn encode_to_file(amt: u32) {
            let quantizer = DefaultLeakyQuantizer::new(-256..=255);
            let model = quantizer.quantize(Gaussian::new(0.0, 100.0));

            let symbols = (0..amt).map(|i| {
                let cheap_hash = i.wrapping_mul(0x6979_E2F3).wrapping_add(0x0059_0E91);
                (cheap_hash >> (32 - 9)) as i32 - 256
            });

            let mut encoder = DefaultAnsCoder::new();
            encoder.encode_iid_symbols_reverse(symbols, &model).unwrap();
            let compressed = encoder.into_compressed().unwrap();

            let mut file = BufWriter::new(File::create("backend_stack_example.tmp").unwrap());
            for &word in compressed.iter().rev() {
                file.write_u32::<LittleEndian>(word).unwrap();
            }
        }

        fn decode_from_file_on_the_fly(amt: u32) {
            let quantizer = DefaultLeakyQuantizer::new(-256..=255);
            let model = quantizer.quantize(Gaussian::new(0.0, 100.0));

            let mut file = BufReader::new(File::open("backend_stack_example.tmp").unwrap());
            let word_iterator = std::iter::from_fn(move || match file.read_u32::<LittleEndian>() {
                Ok(word) => Some(Ok(word)),
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::UnexpectedEof {
                        None
                    } else {
                        Some(Err(err))
                    }
                }
            });

            let mut decoder =
                DefaultAnsCoder::from_reversed_compressed_iter(word_iterator).unwrap();

            for (i, symbol) in decoder.decode_iid_symbols(amt as usize, &model).enumerate() {
                let cheap_hash = (i as u32)
                    .wrapping_mul(0x6979_E2F3)
                    .wrapping_add(0x0059_0E91);
                let expected = (cheap_hash >> (32 - 9)) as i32 - 256;
                assert_eq!(symbol.unwrap(), expected);
            }
            assert!(decoder.is_empty());

            let mut word_iterator = decoder.into_raw_parts().0.into_iter();
            assert!(word_iterator.next().is_none());

            std::mem::drop(word_iterator);
            std::fs::remove_file("backend_stack_example.tmp").unwrap();
        }

        encode_to_file(1000);
        decode_from_file_on_the_fly(1000);
    }
}
