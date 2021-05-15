import constriction
import numpy as np
import scipy.stats


def test_ans_gaussian():
    encoder = constriction.stream.ans.AnsCoder()

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

    decoder = constriction.stream.ans.AnsCoder(compressed)
    reconstructed = decoder.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert decoder.is_empty()
    assert np.all(reconstructed == symbols)


def test_custom_model():
    symbols = np.array([3, 2, 6, -51, -19, 5, 87], dtype=np.int32)

    # Encode and decode i.i.d. symbols
    model_py = scipy.stats.norm(1.2, 4.9)
    model = constriction.stream.model.CustomModel(
        model_py.cdf, model_py.ppf, -100, 100)
    encoder = constriction.stream.ans.AnsCoder()
    encoder.encode_iid_custom_model_reverse(symbols, model)
    compressed = encoder.get_compressed()
    print(compressed)
    assert np.all(compressed == np.array(
        [3187671595, 2410106987,  48580], dtype=np.uint32))
    decoder = constriction.stream.ans.AnsCoder(compressed)
    reconstructed = decoder.decode_iid_custom_model(len(symbols), model)
    assert decoder.is_empty()
    assert np.all(reconstructed == symbols)

    # Encode and decode i.i.d. symbols, but with parameterized custom model.
    model_parameters = (
        np.array([[1.2, 4.9]], dtype=np.float64) +
        np.array([[0.0]]*len(symbols), dtype=np.float64)
    )
    model = constriction.stream.model.CustomModel(
        lambda x, params: scipy.stats.norm.cdf(
            x, loc=params[0], scale=params[1]),
        lambda x, params: scipy.stats.norm.ppf(
            x, loc=params[0], scale=params[1]),
        -100, 100)
    encoder = constriction.stream.ans.AnsCoder()
    encoder.encode_custom_model_reverse(symbols, model, model_parameters)
    compressed = encoder.get_compressed()
    print(compressed)
    assert np.all(compressed == np.array(
        [3187671595, 2410106987,  48580], dtype=np.uint32))
    decoder = constriction.stream.ans.AnsCoder(compressed)
    reconstructed = decoder.decode_custom_model(model, model_parameters)
    assert decoder.is_empty()
    assert np.all(reconstructed == symbols)

    # Encode and decode non-i.i.d. symbols.
    model_parameters = np.array([[i, 4.9] for i in symbols], dtype=np.float64)
    model = constriction.stream.model.CustomModel(
        lambda x, params: scipy.stats.norm.cdf(
            x, loc=params[0], scale=params[1]),
        lambda x, params: scipy.stats.norm.ppf(
            x, loc=params[0], scale=params[1]),
        -100, 100)
    encoder = constriction.stream.ans.AnsCoder()
    encoder.encode_custom_model_reverse(symbols, model, model_parameters)
    compressed = encoder.get_compressed()
    print(compressed)
    assert np.all(compressed == np.array(
        [3397926478, 6042], dtype=np.uint32))
    decoder = constriction.stream.ans.AnsCoder(compressed)
    reconstructed = decoder.decode_custom_model(model, model_parameters)
    assert decoder.is_empty()
    assert np.all(reconstructed == symbols)


def test_huffman_queue():
    probabilities = np.array([0.3,0.28,0.12, 0.1,0.2], dtype=np.float32)
    symbols = [1, 3, 2, 4, 0, 1, 4, 0, 2, 1]

    encoder = constriction.symbol.QueueEncoder()
    encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(
        probabilities)
    for symbol in symbols:
        encoder.encode_symbol(symbol, encoder_codebook)
    compressed, compressed_len = encoder.get_compressed()
    print(compressed, compressed_len)
    assert compressed_len == 23
    assert np.all(compressed == np.array([3873993], dtype=np.uint32))

    decoder = encoder.get_decoder()
    decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(
        probabilities)
    reconstructed = [decoder.decode_symbol(decoder_codebook) for _ in range(len(symbols))]
    assert reconstructed == symbols
