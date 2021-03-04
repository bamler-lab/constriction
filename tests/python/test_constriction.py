import constriction
import numpy as np


def test_ans():
    encoder = constriction.stream.ans.Ans()

    min_supported_symbol, max_supported_symbol = -100, 100
    symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
    means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float64)
    stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float64)

    encoder.encode_leaky_gaussian_symbols_reverse(
        symbols, min_supported_symbol, max_supported_symbol, means, stds)
    assert encoder.num_bits() == 64
    assert encoder.num_valid_bits() == 51
    compressed = encoder.get_compressed()
    assert np.all(compressed == np.array(
        [1109163715, 757457], dtype=np.uint32))

    decoder = constriction.stream.ans.Ans(compressed)
    reconstructed = decoder.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert decoder.is_empty()
    assert np.all(reconstructed == symbols)
