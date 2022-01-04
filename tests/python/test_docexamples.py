import constriction
import numpy as np
import sys
import scipy


def test_module_example1():
    # Create an empty Asymmetric Numeral Systems (ANS) Coder:
    coder = constriction.stream.stack.AnsCoder()

    # Some made up data and entropy models for demonstration purpose:
    min_supported_symbol, max_supported_symbol = -100, 100  # both inclusively
    symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
    means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
    stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)

    # Encode the data (in reverse order, since ANS is a stack):
    coder.encode_leaky_gaussian_symbols_reverse(
        symbols, min_supported_symbol, max_supported_symbol, means, stds)

    print(f"Compressed size: {coder.num_bits()} bits")
    print(
        f"(without unnecessary trailing zeros: {coder.num_valid_bits()} bits)")

    # Get the compressed bit string, convert it into an architecture-independent
    # byte order, and write it to a binary file:
    compressed = coder.get_compressed()
    if sys.byteorder == "big":
        compressed.byteswap(inplace=True)

    # We won't write it to a file her, let's just directly continue decoding.

    if sys.byteorder == "big":
        compressed.byteswap(inplace=True)

    # Initialize an ANS coder from the compressed bit string:
    coder = constriction.stream.stack.AnsCoder(compressed)

    # Use the same entropy models that we used for encoding:
    min_supported_symbol, max_supported_symbol = -100, 100  # both inclusively
    means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
    stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)

    # Decode and print the data:
    reconstructed = coder.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert coder.is_empty()
    assert np.all(reconstructed == symbols)


def test_module_example2():
    # Create an empty Range Encoder:
    encoder = constriction.stream.queue.RangeEncoder()

    # Same made up data and entropy models as in the ANS Coding example above:
    min_supported_symbol, max_supported_symbol = -100, 100  # both inclusively
    symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
    means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
    stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)

    # Encode the data (this time in normal order, since Range Coding is a queue):
    encoder.encode_leaky_gaussian_symbols(
        symbols, min_supported_symbol, max_supported_symbol, means, stds)

    print(f"Compressed size: {encoder.num_bits()} bits")

    # Get the compressed bit string (sealed up to full words):
    compressed = encoder.get_compressed()

    # ... writing and reading from file same as above (skipped here) ...

    # Initialize a Range Decoder from the compressed bit string:
    decoder = constriction.stream.queue.RangeDecoder(compressed)

    # Decode the data and verify it's correct:
    reconstructed = decoder.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert decoder.maybe_exhausted()
    assert np.all(reconstructed == symbols)


def test_ans_example():
    ans = constriction.stream.stack.AnsCoder()  # No arguments => empty ANS coder

    symbols = np.array([2, -1, 0, 2, 3], dtype=np.int32)
    min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
    means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype=np.float64)
    stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype=np.float64)

    ans.encode_leaky_gaussian_symbols_reverse(
        symbols, min_supported_symbol, max_supported_symbol, means, stds)

    print(f"Compressed size: {ans.num_valid_bits()} bits")

    compressed = ans.get_compressed()
    if sys.byteorder == "big":
        # Convert native byte order to a consistent one (here: little endian).
        compressed.byteswap(inplace=True)

    if sys.byteorder == "big":
        # Convert little endian byte order to native byte order.
        compressed.byteswap(inplace=True)

    ans = constriction.stream.stack.AnsCoder(compressed)

    min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
    means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype=np.float64)
    stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype=np.float64)

    reconstructed = ans.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert ans.is_empty()
    assert np.all(reconstructed == symbols)


def test_custom_model_ans():
    def fixed_model_params():
        model_scipy = scipy.stats.cauchy(loc=10.3, scale=5.8)
        # Wrap the scipy-model in a `CustomModel`, which will implicitly
        # quantize it to integers in the given range from -100 to 100 (both
        # ends inclusively).
        model = constriction.stream.model.CustomModel(
            model_scipy.cdf, model_scipy.ppf, -100, 100)

        symbols = np.array([5, 14, -1, 21], dtype=np.int32)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(symbols, model)
        assert np.all(coder.decode(model, 4) == symbols)

    def variable_model_params():
        # The optional argument `params` will receive a 1-d python array when
        # the model is used for encoding or decoding.
        model = constriction.stream.model.CustomModel(
            lambda x, loc, scale: scipy.stats.cauchy.cdf(x, loc, scale),
            lambda x, loc, scale: scipy.stats.cauchy.ppf(x, loc, scale),
            -100, 100)

        model_parameters = np.array([
            (7.3, 3.9),  # Location and scale of entropy model for 1st symbol.
            (11.5, 5.2),  # Location and scale of entropy model for 2nd symbol.
            (-3.2, 4.9),  # and so on ...
            (25.9, 7.1),
        ])

        symbols = np.array([5, 14, -1, 21], dtype=np.int32)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(symbols, model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy())
        assert np.all(
            coder.decode(model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy()) == symbols)

    def discrete_distribution():
        model = constriction.stream.model.CustomModel(
            lambda x, params: scipy.stats.binom.cdf(x, n=10, p=params),
            lambda x, params: scipy.stats.binom.ppf(x, n=10, p=params),
            0, 10)

        success_probabilities = np.array([0.3, 0.7, 0.2, 0.6])

        symbols = np.array([4, 8, 1, 5], dtype=np.int32)
        coder = constriction.stream.stack.AnsCoder()
        coder.encode_reverse(
            symbols, model, success_probabilities)
        assert np.all(
            coder.decode(model, success_probabilities) == symbols)

    fixed_model_params()
    variable_model_params()
    discrete_distribution()


def test_custom_model_range():
    def fixed_model_params():
        model_scipy = scipy.stats.cauchy(loc=10.3, scale=5.8)
        # Wrap the scipy-model in a `CustomModel`, which will implicitly
        # quantize it to integers in the given range from -100 to 100 (both
        # ends inclusively).
        model = constriction.stream.model.CustomModel(
            model_scipy.cdf, model_scipy.ppf, -100, 100)

        symbols = np.array([5, 14, -1, 21], dtype=np.int32)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(symbols, model)
        compressed = encoder.get_compressed()
        decoder = constriction.stream.queue.RangeDecoder(compressed)
        assert np.all(decoder.decode(model, 4) == symbols)

    def variable_model_params():
        # The optional argument `params` will receive a 1-d python array when
        # the model is used for encoding or decoding.
        model = constriction.stream.model.CustomModel(
            lambda x, loc, scale: scipy.stats.cauchy.cdf(x, loc, scale),
            lambda x, loc, scale: scipy.stats.cauchy.ppf(x, loc, scale),
            -100, 100)

        model_parameters = np.array([
            (7.3, 3.9),  # Location and scale of entropy model for 1st symbol.
            (11.5, 5.2),  # Location and scale of entropy model for 2nd symbol.
            (-3.2, 4.9),  # and so on ...
            (25.9, 7.1),
        ])

        symbols = np.array([5, 14, -1, 21], dtype=np.int32)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(symbols, model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy())
        compressed = encoder.get_compressed()
        decoder = constriction.stream.queue.RangeDecoder(compressed)
        assert np.all(
            decoder.decode(model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy()) == symbols)

    def discrete_distribution():
        model = constriction.stream.model.CustomModel(
            lambda x, params: scipy.stats.binom.cdf(x, n=10, p=params),
            lambda x, params: scipy.stats.binom.ppf(x, n=10, p=params),
            0, 10)

        success_probabilities = np.array([0.3, 0.7, 0.2, 0.6])

        symbols = np.array([4, 8, 1, 5], dtype=np.int32)
        encoder = constriction.stream.queue.RangeEncoder()
        encoder.encode(symbols, model, success_probabilities)
        compressed = encoder.get_compressed()
        decoder = constriction.stream.queue.RangeDecoder(compressed)
        assert np.all(
            decoder.decode(model, success_probabilities) == symbols)

    fixed_model_params()
    variable_model_params()
    discrete_distribution()


def test_custom_model_chain():
    compressed = np.array(
        [0xa5dd25f7, 0xfaef49b5, 0xd5b12228, 0x156ceb98, 0x71a0a92b,
         0x99e6d365, 0x2eebfadb, 0x404a567b, 0xf6cbdc09, 0xe63f3848],
        dtype=np.uint32)

    def fixed_model_params():
        model_scipy = scipy.stats.cauchy(loc=10.3, scale=5.8)
        # Wrap the scipy-model in a `CustomModel`, which will implicitly
        # quantize it to integers in the given range from -100 to 100 (both
        # ends inclusively).
        model = constriction.stream.model.CustomModel(
            model_scipy.cdf, model_scipy.ppf, -100, 100)

        coder = constriction.stream.chain.ChainCoder(compressed, False, False)
        symbols = coder.decode(model, 4)
        assert np.all(symbols == np.array([18, 6, 33, 59]))
        coder.encode_reverse(symbols, model)
        assert np.all(np.hstack(coder.get_data()) == compressed)

    def variable_model_params():
        # The optional argument `params` will receive a 1-d python array when
        # the model is used for encoding or decoding.
        model = constriction.stream.model.CustomModel(
            lambda x, loc, scale: scipy.stats.cauchy.cdf(x, loc, scale),
            lambda x, loc, scale: scipy.stats.cauchy.ppf(x, loc, scale),
            -100, 100)

        model_parameters = np.array([
            (7.3, 3.9),  # Location and scale of entropy model for 1st symbol.
            (11.5, 5.2),  # Location and scale of entropy model for 2nd symbol.
            (-3.2, 4.9),  # and so on ...
            (25.9, 7.1),
        ])

        coder = constriction.stream.chain.ChainCoder(compressed, False, False)
        symbols = coder.decode(model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy())
        assert np.all(symbols == np.array([13, 7, 16, 85]))
        coder.encode_reverse(symbols, model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy())
        assert np.all(np.hstack(coder.get_data()) == compressed)

    def discrete_distribution():
        model = constriction.stream.model.CustomModel(
            lambda x, params: scipy.stats.binom.cdf(x, n=10, p=params),
            lambda x, params: scipy.stats.binom.ppf(x, n=10, p=params),
            0, 10)

        success_probabilities = np.array([0.3, 0.7, 0.2, 0.6])

        coder = constriction.stream.chain.ChainCoder(compressed, False, False)
        symbols = coder.decode(model, success_probabilities)
        assert np.all(symbols == np.array([4, 6, 4, 9]))
        coder.encode_reverse(
            symbols, model, success_probabilities)
        assert np.all(np.hstack(coder.get_data()) == compressed)

    fixed_model_params()
    variable_model_params()
    discrete_distribution()
