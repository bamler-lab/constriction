import constriction
import numpy as np
import sys
import scipy

def test_chain_independence():
    data = np.array([0x80d1_4131, 0xdda9_7c6c,
                    0x5017_a640, 0x0117_0a3e], np.uint32)
    probabilities = np.array([
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.2, 0.1, 0.5],
        [0.2, 0.1, 0.4, 0.3],
    ])
    model = constriction.stream.model.Categorical(lazy=True)

    ansCoder = constriction.stream.stack.AnsCoder(data, True)
    assert np.all(ansCoder.decode(model, probabilities) == [0, 0, 2])

    probabilities[0, :] = np.array([0.09, 0.71, 0.1, 0.1])
    ansCoder = constriction.stream.stack.AnsCoder(data, True)
    assert np.all(ansCoder.decode(model, probabilities) == [1, 0, 0])

    probabilities[0, :] = np.array([0.1, 0.7, 0.1, 0.1])
    chainCoder = constriction.stream.chain.ChainCoder(data, False, True)
    assert np.all(chainCoder.decode(model, probabilities) == [0, 3, 3])

    probabilities[0, :] = np.array([0.09, 0.71, 0.1, 0.1])
    chainCoder = constriction.stream.chain.ChainCoder(data, False, True)
    assert np.all(chainCoder.decode(model, probabilities) == [1, 3, 3])



def test_module_example3():
    # Same message as above, but a complex entropy model consisting of two parts:
    message = np.array(
        [6,   10,   -4,   2,   5,    2, 1, 0, 2], dtype=np.int32)
    means = np.array([2.3,  6.1, -8.5, 4.1, 1.3], dtype=np.float32)
    stds = np.array([6.2,  5.3,  3.8, 3.2, 4.7], dtype=np.float32)
    entropy_model1 = constriction.stream.model.QuantizedGaussian(-50, 50)
    entropy_model2 = constriction.stream.model.Categorical(
        np.array([0.2, 0.5, 0.3], dtype=np.float32), # Probabilities of the symbols 0,1,2.
        lazy=True
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


def test_chain2():
    # Some sample binary data and sample probabilities for our entropy models
    data = np.array(
        [0x80d14131, 0xdda97c6c, 0x5017a640, 0x01170a3e], np.uint32)
    probabilities = np.array(
        [[0.1, 0.7, 0.1, 0.1],  # (<-- probabilities for first decoded symbol)
         [0.2, 0.2, 0.1, 0.5],  # (<-- probabilities for second decoded symbol)
         [0.2, 0.1, 0.4, 0.3]], dtype=np.float32)  # (<-- probabilities for third decoded symbol)
    model_family = constriction.stream.model.Categorical(lazy=True)

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
    model_family = constriction.stream.model.Categorical(lazy=True)

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
    model_part1 = constriction.stream.model.Categorical(probabilities_part1, lazy=True)
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


def test_ans_decode1():
    # Define a concrete categorical entropy model over the (implied)
    # alphabet {0, 1, 2}:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)

    # Decode a single symbol from some example compressed data:
    compressed = np.array([2514924296, 114], dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed)
    symbol = coder.decode(model)
    assert symbol == 2


def test_ans_decode2():
    # Use the same concrete entropy model as in the previous example:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)

    # Decode 9 symbols from some example compressed data, using the
    # same (fixed) entropy model defined above for all symbols:
    compressed = np.array([2514924296, 114], dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed)
    symbols = coder.decode(model, 9)
    assert np.all(symbols == np.array(
        [2, 0, 0, 1, 2, 2, 1, 2, 2], dtype=np.int32))



def test_ans_decode4():
    # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for first decoded symbol)
         [0.3, 0.2, 0.2, 0.2, 0.1]],  # (for second decoded symbol)
        dtype=np.float32)
    model_family = constriction.stream.model.Categorical(lazy=True)

    # Decode 2 symbols:
    compressed = np.array([2142112014, 31], dtype=np.uint32)
    coder = constriction.stream.stack.AnsCoder(compressed)
    symbols = coder.decode(model_family, probabilities)
    assert np.all(symbols == np.array([3, 1], dtype=np.int32))


def test_ans_encode_reverse1():
    # Define a concrete categorical entropy model over the (implied)
    # alphabet {0, 1, 2}:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)

    # Encode a single symbol with this entropy model:
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(2, model)  # Encodes the symbol `2`.


def test_ans_encode_reverse2():
    # Use the same concrete entropy model as in the previous example:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)

    # Encode an example message using the above `model` for all symbols:
    symbols = np.array([0, 2, 1, 2, 0, 2, 0, 2, 1], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(symbols, model)
    assert np.all(coder.get_compressed() == np.array(
        [1276732052, 172], dtype=np.uint32))




def test_ans_encode_reverse4():
    # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for symbols[0])
         [0.3, 0.2, 0.2, 0.2, 0.1]],  # (for symbols[1])
        dtype=np.float32)
    model_family = constriction.stream.model.Categorical(lazy=True)

    # Encode 2 symbols (needs `len(symbols) == probabilities.shape[0]`):
    symbols = np.array([3, 1], dtype=np.int32)
    coder = constriction.stream.stack.AnsCoder()
    coder.encode_reverse(symbols, model_family, probabilities)
    assert np.all(coder.get_compressed() == np.array(
        [45298482], dtype=np.uint32))


def test_ans_seek():
    probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)
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
    model_part1 = constriction.stream.model.Categorical(probabilities_part1, lazy=True)
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


def test_range_coder_encode1():
    # Define a concrete categorical entropy model over the (implied)
    # alphabet {0, 1, 2}:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)

    # Encode a single symbol with this entropy model:
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(2, model)  # Encodes the symbol `2`.
    # ... then encode some more symbols ...


def test_range_coder_encode2():
    # Use the same concrete entropy model as in the previous example:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)

    # Encode an example message using the above `model` for all symbols:
    symbols = np.array([0, 2, 1, 2, 0, 2, 0, 2, 1], dtype=np.int32)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(symbols, model)
    assert np.all(encoder.get_compressed() ==
                  np.array([369323598], dtype=np.uint32))



def test_range_coder_encode4():
    # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for first encoded symbol)
         [0.3, 0.2, 0.2, 0.2, 0.1]],  # (for second encoded symbol)
        dtype=np.float32)
    model_family = constriction.stream.model.Categorical(lazy=True)

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
    model = constriction.stream.model.Categorical(probabilities, lazy=True)

    # Decode a single symbol from some example compressed data:
    compressed = np.array([3089773345, 1894195597], dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    symbol = decoder.decode(model)
    assert symbol == 2


def test_range_coding_decode2():
    # Use the same concrete entropy model as in the previous example:
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)

    # Decode 9 symbols from some example compressed data, using the
    # same (fixed) entropy model defined above for all symbols:
    compressed = np.array([369323598], dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    symbols = decoder.decode(model, 9)
    assert np.all(symbols == np.array(
        [0, 2, 1, 2, 0, 2, 0, 2, 1], dtype=np.int32))


def test_range_coding_seek():
    probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)
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


def test_range_coding_decode4():
    # Define 2 categorical models over the alphabet {0, 1, 2, 3, 4}:
    probabilities = np.array(
        [[0.1, 0.2, 0.3, 0.1, 0.3],  # (for first decoded symbol)
         [0.3, 0.2, 0.2, 0.2, 0.1]],  # (for second decoded symbol)
        dtype=np.float32)
    model_family = constriction.stream.model.Categorical(lazy=True)

    # Decode 2 symbols:
    compressed = np.array([2705829510], dtype=np.uint32)
    decoder = constriction.stream.queue.RangeDecoder(compressed)
    symbols = decoder.decode(model_family, probabilities)
    assert np.all(symbols == np.array([3, 1], dtype=np.int32))



def test_categorical1():
    # Define a categorical distribution over the (implied) alphabet {0,1,2,3}
    # with P(X=0) = 0.2, P(X=1) = 0.4, P(X=2) = 0.1, and P(X=3) = 0.3:
    probabilities = np.array([0.2, 0.4, 0.1, 0.3], dtype=np.float32)
    model = constriction.stream.model.Categorical(probabilities, lazy=True)

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
    model_family = constriction.stream.model.Categorical(lazy=True)
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
