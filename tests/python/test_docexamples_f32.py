import constriction
import numpy as np
import sys
import scipy


def test_module_example1():
    message = np.array([6, 10, -4, 2, 5, 2, 1, 0, 2], dtype=np.int32)

    # Define an i.i.d. entropy model (see below for more complex models):
    entropy_model = constriction.stream.model.QuantizedGaussian(
        -50, 50, 3.2, 9.6)

    # Let's use an ANS coder in this example. See below for a Range Coder example.
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)

    compressed = encoder.get_compressed()
    print(f"compressed representation: {compressed}")
    print(f"(in binary: {[bin(word) for word in compressed]})")
    assert np.all(compressed == np.array([3114258274, 357938615], dtype=np.uint32))

    decoder = constriction.stream.stack.AnsCoder(compressed)
    decoded = decoder.decode(entropy_model, 9)  # (decodes 9 symbols)
    assert np.all(decoded == message)


def test_module_example2():
    # Same representation of message and entropy model as in the previous example:
    message = np.array([6, 10, -4, 2, 5, 2, 1, 0, 2], dtype=np.int32)
    entropy_model = constriction.stream.model.QuantizedGaussian(
        -50, 50, 3.2, 9.6)

    # Let's use a Range coder now:
    encoder = constriction.stream.queue.RangeEncoder()         # <-- CHANGED LINE
    # <-- (slightly) CHANGED LINE
    encoder.encode(message, entropy_model)

    compressed = encoder.get_compressed()
    print(f"compressed representation: {compressed}")
    print(f"(in binary: {[bin(word) for word in compressed]})")
    assert np.all(compressed == np.array([2682585243, 513522013], dtype=np.uint32))

    decoder = constriction.stream.queue.RangeDecoder(
        compressed)  # <--CHANGED LINE
    decoded = decoder.decode(entropy_model, 9)  # (decodes 9 symbols)
    assert np.all(decoded == message)


def test_old_module_example1():
    # Create an empty Asymmetric Numeral Systems (ANS) Coder:
    coder = constriction.stream.stack.AnsCoder()

    # Some made up data and entropy models for demonstration purpose:
    model = constriction.stream.model.QuantizedGaussian(-100, 100)
    symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
    means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float32)
    stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float32)

    # Encode the data (in reverse order, since ANS is a stack):
    coder.encode_reverse(symbols, model, means, stds)

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
    means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float32)
    stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float32)

    # Decode and print the data:
    reconstructed = coder.decode(model, means, stds)
    assert coder.is_empty()
    assert np.all(reconstructed == symbols)


def test_module_example3():
    # Same message as above, but a complex entropy model consisting of two parts:
    message = np.array(
        [6,   10,   -4,   2,   5,    2, 1, 0, 2], dtype=np.int32)
    means = np.array([2.3,  6.1, -8.5, 4.1, 1.3], dtype=np.float32)
    stds = np.array([6.2,  5.3,  3.8, 3.2, 4.7], dtype=np.float32)
    entropy_model1 = constriction.stream.model.QuantizedGaussian(-50, 50)
    entropy_model2 = constriction.stream.model.Categorical(
        np.array([0.2, 0.5, 0.3], dtype=np.float32), # Probabilities of the symbols 0,1,2.
        perfect=False
    ) 

    # Simply encode both parts in sequence with their respective models:
    encoder = constriction.stream.queue.RangeEncoder()
    # per-symbol params.
    encoder.encode(message[0:5], entropy_model1, means, stds)
    encoder.encode(message[5:9], entropy_model2)

    compressed = encoder.get_compressed()
    print(f"compressed representation: {compressed}")
    print(f"(in binary: {[bin(word) for word in compressed]})")
    assert np.all(compressed == np.array([3176507206], dtype=np.uint32))

    decoder = constriction.stream.queue.RangeDecoder(compressed)
    decoded_part1 = decoder.decode(entropy_model1, means, stds)
    decoded_part2 = decoder.decode(entropy_model2, 4)
    assert np.all(np.concatenate((decoded_part1, decoded_part2)) == message)


def test_chain1():
   # Parameters for a few example Gaussian entropy models:
    leaky_gaussian = constriction.stream.model.QuantizedGaussian(-100, 100)
    means = np.array([3.2, -14.3, 5.7], dtype=np.float32)
    stds = np.array([6.4, 4.2, 3.9], dtype=np.float32)

    def run_encoder_part(side_information):
        # Construct a `ChainCoder` for *decoding*:
        coder = constriction.stream.chain.ChainCoder(
            side_information,    # Provided bit string.
            is_remainders=False, # Bit string is *not* remaining data after decoding.
            seal=True            # Bit string comes from an external source here.
        )
        # Decode side information into a sequence of symbols as usual in bits-back coding:
        symbols = coder.decode(leaky_gaussian, means, stds)
        # Obtain what's *remaining* on the coder after decoding the symbols:
        remaining1, remaining2 = coder.get_remainders()
        return symbols, np.concatenate([remaining1, remaining2])

    def run_decoder_part(symbols, remaining):
        # Construct a `ChainCoder` for *encoding*:
        coder = constriction.stream.chain.ChainCoder(
            remaining,           # Provided bit string.
            is_remainders=True,  # Bit string *is* remaining data after decoding.
            seal=False           # Bit string comes from a `ChainCoder`, no need to seal it.
        )
        # Re-encode the symbols to recover the side information:
        coder.encode_reverse(symbols, leaky_gaussian, means, stds)
        # Obtain the reconstructed data
        data1, data2 = coder.get_data(unseal=True)
        return np.concatenate([data1, data2])

    np.random.seed(123)
    sample_side_information = np.random.randint(2**32, size=10, dtype=np.uint32)
    symbols, remaining = run_encoder_part(sample_side_information)
    recovered = run_decoder_part(symbols, remaining)
    assert np.all(recovered == sample_side_information)


def test_chain2():
    # Some sample binary data and sample probabilities for our entropy models
    data = np.array(
        [0x80d14131, 0xdda97c6c, 0x5017a640, 0x01170a3e], np.uint32)
    probabilities = np.array(
        [[0.1, 0.7, 0.1, 0.1],  # (<-- probabilities for first decoded symbol)
         [0.2, 0.2, 0.1, 0.5],  # (<-- probabilities for second decoded symbol)
         [0.2, 0.1, 0.4, 0.3]], dtype=np.float32)  # (<-- probabilities for third decoded symbol)
    model_family = constriction.stream.model.Categorical(perfect=False)

    # Decoding `data` with an `AnsCoder` results in the symbols `[0, 0, 2]`:
    ansCoder = constriction.stream.stack.AnsCoder(data, seal=True)
    assert np.all(ansCoder.decode(model_family, probabilities)
                  == np.array([0, 0, 2], dtype=np.int32))

    # Even if we change only the first entropy model (slightly), *all* decoded
    # symbols can change:
    probabilities[0, :] = np.array([0.09, 0.71, 0.1, 0.1], dtype=np.float32)
    ansCoder = constriction.stream.stack.AnsCoder(data, seal=True)
    assert np.all(ansCoder.decode(model_family, probabilities)
                  == np.array([1, 0, 0], dtype=np.int32))


def test_chain3():
    # Same compressed data and original entropy models as in our first example
    data = np.array(
        [0x80d14131, 0xdda97c6c, 0x5017a640, 0x01170a3e], np.uint32)
    probabilities = np.array(
        [[0.1, 0.7, 0.1, 0.1],
         [0.2, 0.2, 0.1, 0.5],
         [0.2, 0.1, 0.4, 0.3]], dtype=np.float32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    # Decode with the original entropy models, this time using a `ChainCoder`:
    chainCoder = constriction.stream.chain.ChainCoder(data, seal=True)
    assert np.all(chainCoder.decode(model_family, probabilities)
                  == np.array([0, 3, 3], dtype=np.int32))

    # We obtain different symbols than for the `AnsCoder`, of course, but that's
    # not the point here. Now let's change the first model again:
    probabilities[0, :] = np.array([0.09, 0.71, 0.1, 0.1], dtype=np.float32)
    chainCoder = constriction.stream.chain.ChainCoder(data, seal=True)
    assert np.all(chainCoder.decode(model_family, probabilities)
                  == np.array([1, 3, 3], dtype=np.int32))


def test_stack1():
    # Define the two parts of the message and their respective entropy models:
    message_part1 = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
    probabilities_part1 = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
    model_part1 = constriction.stream.model.Categorical(probabilities_part1, perfect=False)
    # `model_part1` is a categorical distribution over the (implied) alphabet
    # {0,1,2,3} with P(X=0) = 0.2, P(X=1) = 0.4, P(X=2) = 0.1, and P(X=3) = 0.3;
    # we will use it below to encode each of the 7 symbols in `message_part1`.

    message_part2 = np.array([6,   10,   -4,    2], dtype=np.int32)
    means_part2 = np.array([2.5, 13.1, -1.1, -3.0], dtype=np.float32)
    stds_part2 = np.array([4.1,  8.7,  6.2,  5.4], dtype=np.float32)
    model_family_part2 = constriction.stream.model.QuantizedGaussian(-100, 100)
    # `model_family_part2` is a *family* of Gaussian distributions, quantized to
    # bins of width 1 centered at the integers -100, -99, ..., 100. We could
    # have provided a fixed mean and standard deviation to the constructor of
    # `QuantizedGaussian` but we'll instead provide individual means and standard
    # deviations for each symbol when we encode and decode `message_part2` below.

    print(
        f"Original message: {np.concatenate([message_part1, message_part2])}")

    # Encode both parts of the message in sequence (in reverse order):
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(
        message_part2, model_family_part2, means_part2, stds_part2)
    coder.encode_reverse(message_part1, model_part1)

    # Get and print the compressed representation:
    compressed = coder.get_compressed()
    print(f"compressed representation: {compressed}")
    print(f"(in binary: {[bin(word) for word in compressed]})")

    # You could save `compressed` to a file using `compressed.tofile("filename")`,
    # read it back in: `compressed = np.fromfile("filename", dtype=np.uint32) and
    # then re-create `coder = constriction.stream.stack.AnsCoder(compressed)`.

    # Decode the message:
    decoded_part1 = coder.decode(model_part1, 7)  # (decodes 7 symbols)
    decoded_part2 = coder.decode(model_family_part2, means_part2, stds_part2)
    print(f"Decoded message: {np.concatenate([decoded_part1, decoded_part2])}")
    assert np.all(decoded_part1 == message_part1)
    assert np.all(decoded_part2 == message_part2)


def test_stack2():
    ans = constriction.stream.stack.AnsCoder()  # No arguments => empty ANS coder

    symbols = np.array([2, -1, 0, 2, 3], dtype=np.int32)
    min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
    model = constriction.stream.model.QuantizedGaussian(
        min_supported_symbol, max_supported_symbol)
    means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype=np.float32)
    stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype=np.float32)

    ans.encode_reverse(symbols, model, means, stds)

    print(f"Compressed size: {ans.num_valid_bits()} bits")

    compressed = ans.get_compressed()
    # if sys.byteorder == "big":
    #     # Convert native byte order to a consistent one (here: little endian).
    #     compressed.byteswap(inplace=True)
    # compressed.tofile("compressed.bin")

    # compressed = np.fromfile("compressed.bin", dtype=np.uint32)
    # if sys.byteorder == "big":
    #     # Convert little endian byte order to native byte order.
    #     compressed.byteswap(inplace=True)

    ans = constriction.stream.stack.AnsCoder(compressed)

    min_supported_symbol, max_supported_symbol = -10, 10  # both inclusively
    model = constriction.stream.model.QuantizedGaussian(
        min_supported_symbol, max_supported_symbol)
    means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype=np.float32)
    stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype=np.float32)

    reconstructed = ans.decode(model, means, stds)
    assert ans.is_empty()
    assert np.all(reconstructed == symbols)


def test_ans_decode1():
    # Define a concrete categorical entropy model over the (implied)
    # alphabet {0, 1, 2}:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)

    # Decode a single symbol from some example compressed data:
    compressed = np.array([2514924296, 114], dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed)
    symbol = coder.decode(model)
    assert symbol == 2


def test_ans_decode2():
    # Use the same concrete entropy model as in the previous example:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)

    # Decode 9 symbols from some example compressed data, using the
    # same (fixed) entropy model defined above for all symbols:
    compressed = np.array([2514924296, 114], dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed)
    symbols = coder.decode(model, 9)
    assert np.all(symbols == np.array(
        [2, 0, 0, 1, 2, 2, 1, 2, 2], dtype=np.int32))


def test_ans_decode3():
    # Define a generic quantized Gaussian distribution for all integers
    # in the range from -100 to 100 (both ends inclusive):
    model_family = constriction.stream.model.QuantizedGaussian(-100, 100)

    # Specify the model parameters for each symbol:
    means = np.array([10.3, -4.7, 20.5], dtype=np.float32)
    stds = np.array([5.2, 24.2,  3.1], dtype=np.float32)

    # Decode a message from some example compressed data:
    compressed = np.array([597775281, 3], dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed)
    symbols = coder.decode(model_family, means, stds)
    assert np.all(symbols == np.array([12, -13, 25], dtype=np.int32))


def test_ans_decode4():
    # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for first decoded symbol)
         [0.3, 0.2, 0.2, 0.2, 0.1]],  # (for second decoded symbol)
        dtype=np.float32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    # Decode 2 symbols:
    compressed = np.array([2142112014, 31], dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed)
    symbols = coder.decode(model_family, probabilities)
    assert np.all(symbols == np.array([3, 1], dtype=np.int32))


def test_ans_encode_reverse1():
    # Define a concrete categorical entropy model over the (implied)
    # alphabet {0, 1, 2}:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)

    # Encode a single symbol with this entropy model:
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(2, model)  # Encodes the symbol `2`.


def test_ans_encode_reverse2():
    # Use the same concrete entropy model as in the previous example:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)

    # Encode an example message using the above `model` for all symbols:
    symbols = np.array([0, 2, 1, 2, 0, 2, 0, 2, 1], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(symbols, model)
    assert np.all(coder.get_compressed() == np.array(
        [1276732052, 172], dtype=np.uint32))


def test_ans_encode_reverse3():
    # Define a generic quantized Gaussian distribution for all integers
    # in the range from -100 to 100 (both ends inclusive):
    model_family = constriction.stream.model.QuantizedGaussian(-100, 100)

    # Specify the model parameters for each symbol:
    means = np.array([10.3, -4.7, 20.5], dtype=np.float32)
    stds = np.array([5.2, 24.2,  3.1], dtype=np.float32)

    # Encode an example message:
    # (needs `len(symbols) == len(means) == len(stds)`)
    symbols = np.array([12, -13, 25], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(symbols, model_family, means, stds)
    assert np.all(coder.get_compressed() == np.array(
        [597775281, 3], dtype=np.uint32))


def test_ans_encode_reverse4():
    # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for symbols[0])
         [0.3, 0.2, 0.2, 0.2, 0.1]],  # (for symbols[1])
        dtype=np.float32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    # Encode 2 symbols (needs `len(symbols) == probabilities.shape[0]`):
    symbols = np.array([3, 1], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(symbols, model_family, probabilities)
    assert np.all(coder.get_compressed() == np.array(
        [45298482], dtype=np.uint32))


def test_ans_seek():
    probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)
    message_part1 = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
    message_part2 = np.array([2, 2, 0, 1, 3], dtype=np.int32)

    # Encode both parts of the message (in reverse order, because ANS
    # operates as a stack) and record a checkpoint in-between:
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(message_part2, model)
    (position, state) = coder.pos()  # Records a checkpoint.
    coder.encode_reverse(message_part1, model)

    # We could now call `coder.get_compressed()` but we'll just decode
    # directly from the original `coder` for simplicity.

    # Decode first symbol:
    assert coder.decode(model) == 1

    # Jump to part 2 and decode it:
    coder.seek(position, state)
    decoded_part2 = coder.decode(model, 5)
    assert np.all(decoded_part2 == message_part2)


def test_range_coding_mod():
    # Define the two parts of the message and their respective entropy models:
    message_part1 = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
    probabilities_part1 = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
    model_part1 = constriction.stream.model.Categorical(probabilities_part1, perfect=False)
    # `model_part1` is a categorical distribution over the (implied) alphabet
    # {0,1,2,3} with P(X=0) = 0.2, P(X=1) = 0.4, P(X=2) = 0.1, and P(X=3) = 0.3;
    # we will use it below to encode each of the 7 symbols in `message_part1`.

    message_part2 = np.array([6,   10,   -4,    2], dtype=np.int32)
    means_part2 = np.array([2.5, 13.1, -1.1, -3.0], dtype=np.float32)
    stds_part2 = np.array([4.1,  8.7,  6.2,  5.4], dtype=np.float32)
    model_family_part2 = constriction.stream.model.QuantizedGaussian(-100, 100)
    # `model_family_part2` is a *family* of Gaussian distributions, quantized to
    # bins of width 1 centered at the integers -100, -99, ..., 100. We could
    # have provided a fixed mean and standard deviation to the constructor of
    # `QuantizedGaussian` but we'll instead provide individual means and standard
    # deviations for each symbol when we encode and decode `message_part2` below.

    print(
        f"Original message: {np.concatenate([message_part1, message_part2])}")

    # Encode both parts of the message in sequence:
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(message_part1, model_part1)
    encoder.encode(message_part2, model_family_part2, means_part2, stds_part2)

    # Get and print the compressed representation:
    compressed = encoder.get_compressed()
    print(f"compressed representation: {compressed}")
    print(f"(in binary: {[bin(word) for word in compressed]})")

    # You could save `compressed` to a file using `compressed.tofile("filename")`
    # and read it back in: `compressed = np.fromfile("filename", dtype=np.uint32).

    # Decode the message:
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    decoded_part1 = decoder.decode(model_part1, 7)  # (decodes 7 symbols)
    decoded_part2 = decoder.decode(model_family_part2, means_part2, stds_part2)
    print(f"Decoded message: {np.concatenate([decoded_part1, decoded_part2])}")
    assert np.all(decoded_part1 == message_part1)
    assert np.all(decoded_part2 == message_part2)


def test_old_module_example2():
    # Create an empty Range Encoder:
    encoder = constriction.stream.queue.RangeEncoder()

    # Same made up data and entropy models as in the ANS Coding example above:
    model = constriction.stream.model.QuantizedGaussian(-100, 100)
    symbols = np.array([23, -15, 78, 43, -69], dtype=np.int32)
    means = np.array([35.2, -1.7, 30.1, 71.2, -75.1], dtype=np.float32)
    stds = np.array([10.1, 25.3, 23.8, 35.4, 3.9], dtype=np.float32)

    # Encode the data (this time in normal order, since Range Coding is a queue):
    encoder.encode(symbols, model, means, stds)

    print(f"Compressed size: {encoder.num_bits()} bits")

    # Get the compressed bit string (sealed up to full words):
    compressed = encoder.get_compressed()

    # ... writing and reading from file same as above (skipped here) ...

    # Initialize a Range Decoder from the compressed bit string:
    decoder = constriction.stream.queue.RangeDecoder(compressed)

    # Decode the data and verify it's correct:
    reconstructed = decoder.decode(model, means, stds)
    assert decoder.maybe_exhausted()
    assert np.all(reconstructed == symbols)


def test_ans_example():
    ans = constriction.stream.stack.AnsCoder()  # No arguments => empty ANS coder

    model = constriction.stream.model.QuantizedGaussian(-10, 10)
    symbols = np.array([2, -1, 0, 2, 3], dtype=np.int32)
    means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype=np.float32)
    stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype=np.float32)

    ans.encode_reverse(symbols, model, means, stds)

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
    means = np.array([2.3, -1.7, 0.1, 2.2, -5.1], dtype=np.float32)
    stds = np.array([1.1, 5.3, 3.8, 1.4, 3.9], dtype=np.float32)

    reconstructed = ans.decode(model, means, stds)
    assert ans.is_empty()
    assert np.all(reconstructed == symbols)


def test_range_coder_encode1():
    # Define a concrete categorical entropy model over the (implied)
    # alphabet {0, 1, 2}:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)

    # Encode a single symbol with this entropy model:
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(2, model)  # Encodes the symbol `2`.
    # ... then encode some more symbols ...


def test_range_coder_encode2():
    # Use the same concrete entropy model as in the previous example:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)

    # Encode an example message using the above `model` for all symbols:
    symbols = np.array([0, 2, 1, 2, 0, 2, 0, 2, 1], dtype=np.int32)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols, model)
    assert np.all(encoder.get_compressed() ==
                  np.array([369323598], dtype=np.uint32))


def test_range_coder_encode3():
    # Define a generic quantized Gaussian distribution for all integers
    # in the range from -100 to 100 (both ends inclusive):
    model_family = constriction.stream.model.QuantizedGaussian(-100, 100)

    # Specify the model parameters for each symbol:
    means = np.array([10.3, -4.7, 20.5], dtype=np.float32)
    stds = np.array([5.2, 24.2,  3.1], dtype=np.float32)

    # Encode an example message:
    # (needs `len(symbols) == len(means) == len(stds)`)
    symbols = np.array([12, -13, 25], dtype=np.int32)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols, model_family, means, stds)
    assert np.all(encoder.get_compressed() ==
                  np.array([2655472005], dtype=np.uint32))


def test_range_coder_encode4():
    # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for first encoded symbol)
         [0.3, 0.2, 0.2, 0.2, 0.1]],  # (for second encoded symbol)
        dtype=np.float32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    # Encode 2 symbols (needs `len(symbols) == probabilities.shape[0]`):
    symbols = np.array([3, 1], dtype=np.int32)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols, model_family, probabilities)
    assert np.all(encoder.get_compressed() ==
                  np.array([2705829510], dtype=np.uint32))


def test_range_coding_decode1():
    # Define a concrete categorical entropy model over the (implied)
    # alphabet {0, 1, 2}:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)

    # Decode a single symbol from some example compressed data:
    compressed = np.array([3089773345, 1894195597], dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    symbol = decoder.decode(model)
    assert symbol == 2


def test_range_coding_decode2():
    # Use the same concrete entropy model as in the previous example:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)

    # Decode 9 symbols from some example compressed data, using the
    # same (fixed) entropy model defined above for all symbols:
    compressed = np.array([369323598], dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    symbols = decoder.decode(model, 9)
    assert np.all(symbols == np.array(
        [0, 2, 1, 2, 0, 2, 0, 2, 1], dtype=np.int32))


def test_range_coding_seek():
    probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)
    message_part1 = np.array([1, 2, 0, 3, 2, 3, 0], dtype=np.int32)
    message_part2 = np.array([2, 2, 0, 1, 3], dtype=np.int32)

    # Encode both parts of the message and record a checkpoint in-between:
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(message_part1, model)
    (position, state) = encoder.pos()  # Records a checkpoint.
    encoder.encode(message_part2, model)

    compressed = encoder.get_compressed()
    decoder = constriction.stream.queue.RangeDecoder(compressed)

    # Decode first symbol:
    assert decoder.decode(model) == 1

    # Jump to part 2 and decode it:
    decoder.seek(position, state)
    decoded_part2 = decoder.decode(model, 5)
    assert np.all(decoded_part2 == message_part2)


def test_range_coding_decode3():
    # Define a generic quantized Gaussian distribution for all integers
    # in the range from -100 to 100 (both ends inclusive):
    model_family = constriction.stream.model.QuantizedGaussian(-100, 100)

    # Specify the model parameters for each symbol:
    means = np.array([10.3, -4.7, 20.5], dtype=np.float32)
    stds = np.array([5.2, 24.2,  3.1], dtype=np.float32)

    # Decode a message from some example compressed data:
    compressed = np.array([2655472005], dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    symbols = decoder.decode(model_family, means, stds)
    assert np.all(symbols == np.array([12, -13, 25], dtype=np.int32))


def test_range_coding_decode4():
    # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for first decoded symbol)
         [0.3, 0.2, 0.2, 0.2, 0.1]],  # (for second decoded symbol)
        dtype=np.float32)
    model_family = constriction.stream.model.Categorical(perfect=False)

    # Decode 2 symbols:
    compressed = np.array([2705829510], dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    symbols = decoder.decode(model_family, probabilities)
    assert np.all(symbols == np.array([3, 1], dtype=np.int32))


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
        coder.encode_reverse(
            symbols, model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy())
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


def test_model_mod1():
    model = constriction.stream.model.QuantizedGaussian(-100, 100, 12.6, 7.3)

    # Encode and decode an example message:
    symbols = np.array([12, 15, 4, -2, 18, 5], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()  # (RangeEncoder also works)
    coder.encode_reverse(symbols, model)
    assert np.all(coder.get_compressed() == np.array(
        [745994372, 25704], dtype=np.uint32))

    reconstructed = coder.decode(model, 6)  # (decodes 6 i.i.d. symbols)
    assert np.all(reconstructed == symbols)  # (verify correctness)


def test_model_mod2():
    model_family = constriction.stream.model.QuantizedGaussian(-100, 100)
    # Note: we omitted the mean and standard deviation, but the quantization range
    #       {-100, ..., 100} must always be specified when constructing the model.

    # Define arrays of model parameters (means and standard deviations):
    symbols = np.array([12,   15,   4,   -2,   18,   5], dtype=np.int32)
    means = np.array([13.2, 17.9, 7.3, -4.2, 25.1, 3.2], dtype=np.float32)
    stds = np.array([3.2,  4.7, 5.2,  3.1,  6.3, 2.9], dtype=np.float32)

    # Encode and decode an example message:
    coder = constriction.stream.stack.AnsCoder()  # (RangeEncoder also works)
    coder.encode_reverse(symbols, model_family, means, stds)
    assert np.all(coder.get_compressed() == np.array(
        [2051912079, 1549], dtype=np.uint32))

    reconstructed = coder.decode(model_family, means, stds)
    assert np.all(reconstructed == symbols)  # (verify correctness)


def test_categorical1():
    # Define a categorical distribution over the (implied) alphabet {0,1,2,3}
    # with P(X=0) = 0.2, P(X=1) = 0.4, P(X=2) = 0.1, and P(X=3) = 0.3:
    probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, perfect=False)

    # Encode and decode an example message:
    symbols = np.array([0, 3, 2, 3, 2, 0, 2, 1], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()  # (RangeEncoder also works)
    coder.encode_reverse(symbols, model)
    assert np.all(coder.get_compressed() == np.array(
        [2484720979, 175], dtype=np.uint32))

    reconstructed = coder.decode(model, 8)  # (decodes 8 i.i.d. symbols)
    assert np.all(reconstructed == symbols)  # (verify correctness)


def test_categorical2():
    # Define 3 categorical distributions, each over the alphabet {0,1,2,3,4}:
    model_family = constriction.stream.model.Categorical(perfect=False)
    probabilities = np.array(
        [[0.3, 0.1, 0.1, 0.3, 0.2],  # (for symbols[0])
         [0.1, 0.4, 0.2, 0.1, 0.2],  # (for symbols[1])
         [0.4, 0.2, 0.1, 0.2, 0.1]],  # (for symbols[2])
        dtype=np.float32)

    symbols = np.array([0, 4, 1], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()  # (RangeEncoder also works)
    coder.encode_reverse(symbols, model_family, probabilities)
    assert np.all(coder.get_compressed() == np.array(
        [104018743], dtype=np.uint32))

    reconstructed = coder.decode(model_family, probabilities)
    assert np.all(reconstructed == symbols)  # (verify correctness)


def test_custom_model1():
    model = constriction.stream.model.CustomModel(
        lambda x: 0.5 + 0.5 * np.tanh(x * 0.1),  # define your CDF here
        lambda xi: xi,  # provide an approximate inverse of the CDF
        -100, 100)  # (or whichever range your model has)

    # Encode and decode an example message:
    symbols = np.array([-3, 2, 5, 5, 6], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()  # (RangeEncoder also works)
    coder.encode_reverse(symbols, model)
    print(coder.get_compressed())

    reconstructed = coder.decode(model, 5)  # (decodes 5 i.i.d. symbols)
    assert np.all(reconstructed == symbols)  # (verify correctness)


def test_custom_model2():
    model_family = constriction.stream.model.CustomModel(
        lambda x, a, b: 0.5 + 0.5 * np.tanh(a + x * b),  # define your CDF here
        lambda xi, a, b: xi,  # provide an approximate inverse of the CDF
        -100, 100)  # (or whichever range your model has)

    # Encode and decode an example message with per-symbol model parameters:
    symbols = np.array([-2, 1, 4], dtype=np.int32)
    model_params1 = np.array([1, 10, -3], dtype=np.float32)
    model_params2 = np.array([0.01, 0.04, 0.2], dtype=np.float32)
    coder = constriction.stream.stack.AnsCoder()  # (RangeEncoder also works)
    coder.encode_reverse(symbols, model_family, model_params1, model_params2)
    print(coder.get_compressed())

    reconstructed = coder.decode(model_family, model_params1, model_params2)
    assert np.all(reconstructed == symbols)  # (verify correctness)


def test_scipy_model1():
    import scipy.stats

    scipy_model = scipy.stats.cauchy(loc=6.7, scale=12.4)
    model = constriction.stream.model.ScipyModel(scipy_model, -100, 100)

    # Encode and decode an example message:
    symbols = np.array([22, 14, 5, -3, 19, 7], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()  # (RangeEncoder also works)
    coder.encode_reverse(symbols, model)
    assert np.all(coder.get_compressed() == np.array(
        [3569876501, 1944098], dtype=np.uint32))

    reconstructed = coder.decode(model, 6)  # (decodes 6 i.i.d. symbols)
    assert np.all(reconstructed == symbols)  # (verify correctness)


def test_scipy_model2():
    import scipy.stats

    scipy_model_family = scipy.stats.cauchy
    model_family = constriction.stream.model.ScipyModel(
        scipy_model_family, -100, 100)

    # Encode and decode an example message with per-symbol model parameters:
    symbols = np.array([22,   14,   5,   -3,   19,   7], dtype=np.int32)
    locs = np.array([26.2, 10.9, 8.7, -6.3, 25.1, 8.9], dtype=np.float32)
    scales = np.array([4.3, 7.4,  2.9,  4.1,  9.7, 3.4], dtype=np.float32)
    coder = constriction.stream.stack.AnsCoder()  # (RangeEncoder also works)
    coder.encode_reverse(symbols, model_family, locs, scales)
    assert np.all(coder.get_compressed() == np.array(
        [3611353862, 17526], dtype=np.uint32))

    reconstructed = coder.decode(model_family, locs, scales)
    assert np.all(reconstructed == symbols)  # (verify correctness)


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
        encoder.encode(
            symbols, model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy())
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


def test_old_custom_model_chain():
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
        symbols = coder.decode(
            model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy())
        assert np.all(symbols == np.array([13, 7, 16, 85]))
        coder.encode_reverse(
            symbols, model, model_parameters[:, 0].copy(), model_parameters[:, 1].copy())
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


def test_huffman1():
    # Define an entropy model over the (implied) alphabet {0, 1, 2, 3}:
    probabils = np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float32)

    # Encode some example message, using the same model for each symbol here:
    message = [1, 3, 2, 3, 0, 1, 3, 0, 2, 1, 1, 3, 3, 1, 2, 0, 1, 3, 1]
    encoder = constriction.symbol.QueueEncoder()
    encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(
        probabils)
    for symbol in message:
        encoder.encode_symbol(symbol, encoder_codebook)

    # Obtain the compressed representation and the bitrate:
    compressed, bitrate = encoder.get_compressed()
    print(compressed, bitrate)  # (prints: [3756389791, 61358], 48)
    print(f"(in binary: {[bin(word) for word in compressed]}")
    assert np.all(compressed == np.array([3756389791, 61358], dtype=np.uint32))
    assert bitrate == 48

    # Decode the message
    decoder = constriction.symbol.QueueDecoder(compressed)
    decoded = []
    decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(
        probabils)
    for symbol in range(19):
        decoded.append(decoder.decode_symbol(decoder_codebook))

    assert decoded == message  # (verifies correctness)


def test_huffman2():
    # Define an entropy model over the (implied) alphabet {0, 1, 2, 3}:
    probabils = np.array([0.3, 0.2, 0.4, 0.1], dtype=np.float32)

    # Encode some example message, using the same model for each symbol here:
    message = [1, 3, 2, 3, 0, 1, 3, 0, 2, 1, 1, 3, 3, 1, 2, 0, 1, 3, 1]
    coder = constriction.symbol.StackCoder()
    encoder_codebook = constriction.symbol.huffman.EncoderHuffmanTree(
        probabils)
    for symbol in reversed(message):  # Note: reversed
        coder.encode_symbol(symbol, encoder_codebook)

    # Obtain the compressed representation and the bitrate:
    compressed, bitrate = coder.get_compressed()
    print(compressed, bitrate)  # (prints: [[2818274807, 129455] 48)
    print(f"(in binary: {[bin(word) for word in compressed]}")
    assert np.all(compressed == np.array([2818274807, 129455], dtype=np.uint32))
    assert bitrate == 48

    # Decode the message (we could explicitly construct a decoder:
    # `decoder = constritcion.symbol.StackCoder(compressed)`
    # but we can also also reuse our existing `coder` for decoding):
    decoded = []
    decoder_codebook = constriction.symbol.huffman.DecoderHuffmanTree(
        probabils)
    for symbol in range(19):
        decoded.append(coder.decode_symbol(decoder_codebook))

    assert decoded == message  # (verifies correctness)
