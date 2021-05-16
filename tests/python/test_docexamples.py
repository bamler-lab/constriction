import constriction
import numpy as np
import sys


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
    print(f"(without unnecessary trailing zeros: {coder.num_valid_bits()} bits)")

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
   
    symbols = np.array([2, -1, 0, 2, 3], dtype = np.int32)
    min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
    means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype = np.float64)
    stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype = np.float64)
   
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
    means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype = np.float64)
    stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype = np.float64)
   
    reconstructed = ans.decode_leaky_gaussian_symbols(
        min_supported_symbol, max_supported_symbol, means, stds)
    assert ans.is_empty()
    assert np.all(reconstructed == symbols)
