use numpy::{PyArray1, PyReadonlyArray1};

use pyo3::prelude::*;

use super::distributions::{Categorical, LeakyQuantizer};
use statrs::distribution::Normal;

#[pymodule]
fn ans(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Coder>()?;
    Ok(())
}

/// The type of compressed words.
///
/// The rust library is generic over this type. For the python bindings, we'll
/// probably eventually want to provide coders for different word sizes in one
/// binary but we're not there yet.
type W = u32;

/// An entropy coder based on Asymmetric Numeral Systems (ANS).
///
/// Note that this entropy coder is a stack (a "last in first out" data
/// structure). You can push symbols on the stack using the methods
/// `push_gaussian_symbols` or `push_iid_categorical_symbols`, and then pop
/// them off *in reverse order* using the methods `pop_gaussian_symbols` or
/// `pop_iid_categorical_symbols`, respectively.
///
/// To retrieve the compressed data that is currently on the stack, first
/// query for the size of the compressed data using the method `num_words()`,
/// then allocate a numpy array of this size and dtype `uint32`, and finally
/// pass this array to the method `copy_compressed`.
///
/// To decompress data, pass the compressed data to the constructor and then
/// pop off the symbols in reverse order.
///
/// # Constructor
///
/// Coder(compressed)
///
/// Arguments:
/// compressed (optional) -- initial compressed data, as a numpy array with
///     dtype `uint32`.
#[pyclass]
#[text_signature = "(compressed)"]
pub struct Coder {
    inner: super::Coder<W>,
}

#[pymethods]
impl Coder {
    /// Constructs a new entropy coder, optionally passing initial compressed data.
    #[new]
    pub fn new(compressed: Option<PyReadonlyArray1<u32>>) -> Self {
        let inner = if let Some(compressed) = compressed {
            let mut compressed_vec = Vec::new();
            compressed_vec.extend_from_slice(compressed.as_slice().unwrap());
            super::Coder::with_compressed_data(compressed_vec)
        } else {
            super::Coder::new()
        };

        Self { inner }
    }

    /// Resets the coder for compression.
    ///
    /// After calling this method, the method `is_empty` will return `True`.
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// The current size of the compressed data, in `np.uint32` words.
    #[getter]
    pub fn num_words(&self) -> usize {
        self.inner.num_words()
    }

    /// The current size of the compressed data, in bits.
    #[getter]
    pub fn num_bits(&self) -> usize {
        self.inner.num_bits()
    }

    /// Returns `True` iff the coder is in its default initial state.
    ///
    /// The default initial state is the state returned by the constructor when
    /// called without arguments, or the state to which the coder is set when
    /// calling `clear`.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Copies the compressed data to the provided numpy array.
    ///
    /// The argument `destination` must by a one-dimensional numpy array with
    /// dtype `uint32` and with the exact correct size. Use the method `num_words`
    /// to find out the correct size.
    ///
    /// Example:
    ///
    /// ```python
    /// coder = ans.Coder()
    /// # ... push some symbols on coder ...
    /// compressed_len = coder.num_words()
    /// compressed = np.empty((compressed_len,), dtype=np.uint32)
    /// coder.copy_compressed(compressed)
    ///
    /// # Optional: write the compressed data to a file in
    /// #           platform-independent byte ordering.
    /// if sys.byteorder == "big":
    ///     compressed.byteswap()
    /// with open("path/to/file", "wb") as file:
    ///     compressed.tofile(file)
    /// ```
    pub fn get_compressed<'p>(&mut self, py: Python<'p>) -> &'p PyArray1<u32> {
        let mut compressed = Vec::new();
        compressed.extend_from_slice(&*self.inner.to_compressed());
        PyArray1::from_vec(py, compressed)
    }

    /// Encodes a sequence of symbols using (leaky) Gaussian entropy models.
    ///
    /// The provided numpy arrays `symbols`, `means`, and `stds` must all have the
    /// same size.
    ///
    /// See method `pop_gaussian_symbols` for a usage example.
    ///
    /// Arguments:
    /// min_supported_symbol -- lower bound of the domain for argument `symbols`
    ///     (inclusively).
    /// max_supported_symbol -- upper bound of the domain for argument `symbols`
    ///     (inclusively).
    /// symbols -- the symbols to be encoded. Must be a contiguous one-dimensional
    ///     numpy array (call `.copy()` on it if it is not contiguous) with dtype
    ///     `np.int32`. Each value in the array must be no smaller than
    ///     `min_supported_symbol` and no larger than `max_supported_symbol`.
    /// means -- the mean values of the Gaussian entropy models for each symbol.
    ///     Must be a contiguous one-dimensional numpy array with dtype `np.float64`
    ///     and with the exact same length as the argument `symbols`.
    /// stds -- the standard deviations of the Gaussian entropy models for each
    ///     symbol. Must be a contiguous one-dimensional numpy array with dtype
    ///     `np.float64` and with the exact same length as the argument `symbols`.
    ///     All entries must be strictly positive (i.e., nonzero and nonnegative)
    ///     and finite.
    #[text_signature = "(symbols, min_supported_symbol, max_supported_symbol, means, stds)"]
    pub fn push_gaussian_symbols(
        &mut self,
        symbols: PyReadonlyArray1<i32>,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        means: PyReadonlyArray1<f64>,
        stds: PyReadonlyArray1<f64>,
    ) {
        let quantizer = LeakyQuantizer::new(min_supported_symbol..=max_supported_symbol);
        self.inner
            .push_symbols(
                symbols
                    .iter()
                    .unwrap()
                    .zip(means.iter().unwrap())
                    .zip(stds.iter().unwrap())
                    .map(|((&symbol, &mean), &std)| {
                        (symbol, quantizer.quantize(Normal::new(mean, std).unwrap()))
                    }),
            )
            .unwrap();
    }

    /// Decodes a sequence of symbols *in reverse order* using (leaky) Gaussian entropy
    /// models.
    ///
    /// The provided numpy arrays `means`, `stds`, and `symbols_out` must all have
    /// the same size. The provided `means`, `stds`, `min_supported_symbol`,
    /// `max_supported_symbol`, and `leaky` must be the exact same values that were
    /// used for encoding. Even a tiny modification of these arguments can cause the
    /// coder to decode *completely* different symbols.
    ///
    /// The symbols will be popped off the stack and written to the target array in
    /// reverseorder so as to simplify usage, e.g.:
    ///
    /// ```python
    /// coder = ans.Coder()
    /// symbols = np.array([2, 8, -5], dtype=np.int32)
    /// decoded = np.empty((3,), dtype=np.int32)
    /// means = np.array([0.1, 10.3, -3.2], dtype=np.float64)
    /// stds = np.array([3.2, 1.3, 1.9], dtype=np.float64)
    ///
    /// # Push symbols on the stack:
    /// coder.push_gaussian_symbols(symbols, -10, 10, means, stds, True)
    ///
    /// # Pop symbols off the stack in reverse order:
    /// coder.pop_gaussian_symbols(-10, 10, means, stds, decoded, True)
    ///
    /// # Verify that the decoded symbols match the encoded ones.
    /// assert np.all(symbols == decoded)
    /// assert coder.is_empty()
    /// ```
    ///
    /// Arguments:
    /// min_supported_symbol -- lower bound of the domain supported by the entropy
    ///     model (inclusively). Must be the same value that was used for encoding.
    /// max_supported_symbol -- upper bound of the domain supported by the entropy
    ///     model (inclusively). Must be the same value that was used for encoding.
    /// means -- the mean values of the Gaussian entropy models for each symbol.
    ///     Must be a contiguous one-dimensional numpy array with dtype `float64`
    ///     and with the exact same length as the argument `symbols_out`.
    /// stds -- the standard deviations of the Gaussian entropy models for each
    ///     symbol. Must be a contiguous one-dimensional numpy array with dtype
    ///     `float64` and with the exact same length as the argument `symbols_out`.
    pub fn pop_gaussian_symbols<'p>(
        &mut self,
        min_supported_symbol: i32,
        max_supported_symbol: i32,
        means: PyReadonlyArray1<f64>,
        stds: PyReadonlyArray1<f64>,
        py: Python<'p>,
    ) -> &'p PyArray1<i32> {
        let quantizer = LeakyQuantizer::new(min_supported_symbol..=max_supported_symbol);
        let symbols = self
            .inner
            .pop_symbols(
                means
                    .as_slice()
                    .unwrap()
                    .iter()
                    .zip(stds.as_slice().unwrap().iter())
                    .map(|(&mean, &std)| quantizer.quantize(Normal::new(mean, std).unwrap())),
            )
            .unwrap();

        PyArray1::from_vec(py, symbols)

        // TODO: use PyArray1::from_iter instead.
        // --> reverse order in `push_symbols` rather than `pop_symbols`
        //     (which is much easier to do anyway)
    }

    /// Encodes a sequence of symbols using a fixed categorical distribution.
    ///
    /// This method is analogous to the method `push_gaussian_symbols` except that
    /// - all symbols are encoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian distribution.
    ///
    /// In detail, the categorical entropy model is constructed as follows:
    /// - each symbol from `min_supported_symbol` to `max_supported_symbol`
    ///   (inclusively) gets assigned at least the smallest nonzero probability
    ///   that is representable within the internally used precision.
    /// - the remaining probability mass is distributed among the symbols from
    ///   `min_provided_symbol` to `min_provided_symbol + len(probabilities) - 1`
    ///   (inclusively), in the proportions specified by the provided probabilities
    ///   (as far as this is possible within the internally used fixed point
    ///   accuracy). The provided probabilities do not need to be normalized (i.e.,
    ///   the do not need to add up to one) but they must all be nonnegative.
    pub fn push_iid_categorical_symbols(
        &mut self,
        symbols: PyReadonlyArray1<i32>,
        min_supported_symbol: i32,
        probabilities: PyReadonlyArray1<f64>,
    ) {
        let distribution = Categorical::from_continuous_probabilities(
            probabilities.as_slice().unwrap(),
            min_supported_symbol,
        );
        self.inner
            .push_iid_symbols(symbols.iter().unwrap().cloned(), &distribution)
            .unwrap();
    }

    /// Decodes a sequence of categorically distributed symbols *in reverse order*.
    ///
    /// This method is analogous to the method `pop_gaussian_symbols` except that
    /// - all symbols are decoded with the same entropy model; and
    /// - the entropy model is a categorical rather than a Gaussian distribution.
    ///
    /// See documentation of `push_iid_categorical_symbols` for details of the
    /// categorical entropy model. See documentation of `pop_gaussian_symbols` for a
    /// discussion of the reverse order of decoding, and for a related usage
    /// example.
    pub fn pop_iid_categorical_symbols<'p>(
        &mut self,
        amt: usize,
        min_supported_symbol: i32,
        probabilities: PyReadonlyArray1<f64>,
        py: Python<'p>,
    ) -> &'p PyArray1<i32> {
        let distribution = Categorical::from_continuous_probabilities(
            probabilities.as_slice().unwrap(),
            min_supported_symbol,
        );
        let symbols = self.inner.pop_iid_symbols(amt, &distribution).unwrap();
        PyArray1::from_vec(py, symbols)
    }
}
