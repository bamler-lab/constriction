import constriction
import numpy as np


def test_vbq_example1():
    rng = np.random.default_rng(123)
    unquantized = rng.normal(size=(4, 5)).astype(np.float32)
    print(f"unquantized values:\n{unquantized}\n")

    prior = constriction.quant.EmpiricalDistribution(unquantized)
    assert np.allclose(prior.entropy_base2(), np.log2(20))

    # Allow larger quantization errors in upper left corner of the matrix by using a high variance:
    posterior_variance = np.array([
        [10.0, 10.0, 1.0, 1.0, 1.0],
        [10.0, 10.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32)

    quantized_fine = constriction.quant.vbq(
        unquantized, prior, posterior_variance, 0.01)
    quantized_coarse = constriction.quant.vbq(
        unquantized, prior, posterior_variance, 0.1)

    quantized_fine_expected = np.array([
        [-0.67108965, -0.36778665, 1.1921661, 0.13632113, 0.9202309],
        [0.13632113, -0.36778665, 0.5419522, -0.36778665, -0.36778665],
        [0.13632113, -1.5259304, 1.1921661, -0.67108965, 0.9202309],
        [0.13632113, 1.5320331, -0.67108965, -0.36778665, 0.33776912],
    ])

    quantized_coarse_expected = np.array([
        [0.13632113, 0.13632113, 0.9202309, 0.13632113, 0.9202309],
        [0.13632113, 0.13632113, 0.13632113, -0.36778665, -0.36778665],
        [0.13632113, -1.5259304, 0.9202309, -0.36778665, 0.9202309],
        [0.13632113, 1.1921661, -0.36778665, 0.13632113, 0.13632113],
    ])

    assert np.allclose(quantized_fine, quantized_fine_expected)
    assert np.allclose(quantized_coarse, quantized_coarse_expected)


def test_vbq_example2():
    rng = np.random.default_rng(123)
    unquantized = rng.normal(size=(4, 5)).astype(np.float32)
    print(f"unquantized values:\n{unquantized}\n")

    prior = constriction.quant.EmpiricalDistribution(
        unquantized, specialize_along_axis=0)
    assert np.allclose(prior.entropy_base2(), np.array([np.log2(5)] * 4))

    # Allow larger quantization errors in upper left corner of the matrix by using a high variance:
    posterior_variance = np.array([
        [10.0, 10.0, 1.0, 1.0, 1.0],
        [10.0, 10.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32)

    quantized_fine = constriction.quant.vbq(
        unquantized, prior, posterior_variance, 0.01)
    quantized_coarse = constriction.quant.vbq(
        unquantized, prior, posterior_variance, 0.1)

    quantized_fine_expected = np.array([
        [-0.9891214, -0.36778665, 1.2879252, 0.19397442, 0.9202309],
        [0.5419522, -0.31659546, 0.5419522, -0.31659546, -0.31659546],
        [0.09716732, -1.5259304, 1.1921661, -0.67108965, 1.0002694],
        [0.13632113, 1.5320331, -0.6599694, -0.31179485, 0.33776912],
    ])

    quantized_coarse_expected = np.array([
        [0.19397442, 0.19397442, 0.9202309, 0.19397442, 0.9202309],
        [-0.31659546, -0.31659546, 0.5419522, -0.31659546, -0.31659546],
        [0.09716732, -1.5259304, 1.0002694, -0.67108965, 1.0002694],
        [0.13632113, 1.5320331, -0.31179485, -0.31179485, 0.13632113],
    ])

    assert np.allclose(quantized_fine, quantized_fine_expected)
    assert np.allclose(quantized_coarse, quantized_coarse_expected)


def test_vbq_inplace_example1():
    rng = np.random.default_rng(123)
    unquantized = rng.normal(size=(4, 5)).astype(np.float32)
    print(f"unquantized values:\n{unquantized}\n")

    prior = constriction.quant.EmpiricalDistribution(unquantized)
    assert np.allclose(prior.entropy_base2(), np.log2(20))

    # Allow larger quantization errors in upper left corner of the matrix by using a high variance:
    posterior_variance = np.array([
        [10.0, 10.0, 1.0, 1.0, 1.0],
        [10.0, 10.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32)

    quantized_fine = unquantized.copy()
    constriction.quant.vbq_(quantized_fine, prior, posterior_variance, 0.01)
    quantized_coarse = unquantized.copy()
    constriction.quant.vbq_(quantized_coarse, prior, posterior_variance, 0.1)

    quantized_fine_expected = np.array([
        [-0.67108965, -0.36778665, 1.1921661, 0.13632113, 0.9202309],
        [0.13632113, -0.36778665, 0.5419522, -0.36778665, -0.36778665],
        [0.13632113, -1.5259304, 1.1921661, -0.67108965, 0.9202309],
        [0.13632113, 1.5320331, -0.67108965, -0.36778665, 0.33776912],
    ])

    quantized_coarse_expected = np.array([
        [0.13632113, 0.13632113, 0.9202309, 0.13632113, 0.9202309],
        [0.13632113, 0.13632113, 0.13632113, -0.36778665, -0.36778665],
        [0.13632113, -1.5259304, 0.9202309, -0.36778665, 0.9202309],
        [0.13632113, 1.1921661, -0.36778665, 0.13632113, 0.13632113],
    ])

    assert np.allclose(quantized_fine, quantized_fine_expected)
    assert np.allclose(quantized_coarse, quantized_coarse_expected)


def test_vbq_inplace_example2():
    rng = np.random.default_rng(123)
    unquantized = rng.normal(size=(4, 5)).astype(np.float32)
    print(f"unquantized values:\n{unquantized}\n")

    prior = constriction.quant.EmpiricalDistribution(
        unquantized, specialize_along_axis=0)
    assert np.allclose(prior.entropy_base2(), np.array([np.log2(5)] * 4))

    # Allow larger quantization errors in upper left corner of the matrix by using a high variance:
    posterior_variance = np.array([
        [10.0, 10.0, 1.0, 1.0, 1.0],
        [10.0, 10.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32)

    quantized_fine = unquantized.copy()
    constriction.quant.vbq_(quantized_fine, prior, posterior_variance, 0.01)
    quantized_coarse = unquantized.copy()
    constriction.quant.vbq_(quantized_coarse, prior, posterior_variance, 0.1)

    quantized_fine_expected = np.array([
        [-0.9891214, -0.36778665, 1.2879252, 0.19397442, 0.9202309],
        [0.5419522, -0.31659546, 0.5419522, -0.31659546, -0.31659546],
        [0.09716732, -1.5259304, 1.1921661, -0.67108965, 1.0002694],
        [0.13632113, 1.5320331, -0.6599694, -0.31179485, 0.33776912],
    ])

    quantized_coarse_expected = np.array([
        [0.19397442, 0.19397442, 0.9202309, 0.19397442, 0.9202309],
        [-0.31659546, -0.31659546, 0.5419522, -0.31659546, -0.31659546],
        [0.09716732, -1.5259304, 1.0002694, -0.67108965, 1.0002694],
        [0.13632113, 1.5320331, -0.31179485, -0.31179485, 0.13632113],
    ])

    assert np.allclose(quantized_fine, quantized_fine_expected)
    assert np.allclose(quantized_coarse, quantized_coarse_expected)


def test_empirical_distribution_example1():
    rng = np.random.default_rng(123)
    matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)

    distribution = constriction.quant.EmpiricalDistribution(matrix)
    points, counts = distribution.points_and_counts()
    assert np.all(points == [1.0, 2.0, 3.0, 4.0, 5.0])
    assert np.all(counts == [1, 7, 3, 6, 3])
    assert np.allclose(distribution.entropy_base2(), 2.088376522064209)


def test_empirical_distribution_example2():
    rng = np.random.default_rng(123)
    matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    distribution = constriction.quant.EmpiricalDistribution(
        matrix, specialize_along_axis=0)
    entropies = distribution.entropy_base2()
    points, counts = distribution.points_and_counts()

    points_expected = [
        [1.0, 2.0, 4.0],
        [2.0, 4.0, 5.0],
        [2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
    ]
    counts_expected = [
        [1, 3, 1],
        [1, 2, 2],
        [2, 1, 2],
        [1, 2, 1, 1],
    ]

    for p, pe in zip(points, points_expected):
        assert np.all(p == pe)
    for c, ce in zip(counts, counts_expected):
        assert np.all(c == ce)
    assert np.all(np.abs(entropies - [1.37, 1.52, 1.52, 1.92]) < 0.01)


def test_entropy():
    rng = np.random.default_rng(123)
    matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    print(f"matrix = {matrix}\n")

    marginal_distribution = constriction.quant.EmpiricalDistribution(matrix)
    specialized_distribution = constriction.quant.EmpiricalDistribution(
        matrix, specialize_along_axis=0)

    assert np.allclose(
        marginal_distribution.entropy_base2(), 2.088376522064209)
    assert np.allclose(
        specialized_distribution.entropy_base2(), [1.3709505, 1.5219281, 1.5219281, 1.921928])
    assert np.allclose(
        specialized_distribution.entropy_base2(2), 1.521928071975708)


def test_insert_example1():
    rng = np.random.default_rng(123)
    matrix1 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    matrix2 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)

    distribution = constriction.quant.EmpiricalDistribution(matrix1)
    points, counts = distribution.points_and_counts()
    points_expected = [1.0, 2.0, 3.0, 4.0, 5.0]
    counts_expected = [1, 6, 1, 5, 2]
    assert np.all(points == points_expected)
    assert np.all(counts == counts_expected)

    distribution.insert(matrix2)
    points, counts = distribution.points_and_counts()
    points_expected = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    counts_expected = [1, 1, 10, 7, 7, 4]
    assert np.all(points == points_expected)
    assert np.all(counts == counts_expected)


def test_insert_example2():
    rng = np.random.default_rng(123)
    matrix1 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    matrix2 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)

    distribution = constriction.quant.EmpiricalDistribution(
        matrix1, specialize_along_axis=0)
    points, counts = distribution.points_and_counts()
    points_expected = [
        [1.0, 2.0, 4.0],
        [2.0, 4.0, 5.0],
        [2.0, 3.0, 4.0]
    ]
    counts_expected = [
        [1, 3, 1],
        [1, 2, 2],
        [2, 1, 2]
    ]
    for p, pe in zip(points, points_expected):
        assert np.all(p == pe)
    for c, ce in zip(counts, counts_expected):
        assert np.all(c == ce)

    distribution.insert(matrix2)
    distribution.insert(4.0, index=2)
    points, counts = distribution.points_and_counts()
    points_expected = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0],
        [0.0, 2.0, 3.0, 4.0, 5.0]
    ]
    counts_expected = [
        [1, 4, 2, 2, 1],
        [4, 2, 2, 2],
        [1, 2, 3, 4, 1]
    ]
    print(counts)
    for p, pe in zip(points, points_expected):
        assert np.all(p == pe)
    for c, ce in zip(counts, counts_expected):
        assert np.all(c == ce)


def test_total():
    rng = np.random.default_rng(123)
    matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)

    marginal_distribution = constriction.quant.EmpiricalDistribution(matrix)
    specialized_distribution = constriction.quant.EmpiricalDistribution(
        matrix, specialize_along_axis=0)

    # Insert only into the slice at index=2.
    specialized_distribution.insert(10., index=2)

    assert marginal_distribution.total() == 20
    assert np.all(specialized_distribution.total() == [5, 5, 6, 5])
    assert specialized_distribution.total(2) == 6


def test_points_and_counts_example1():
    rng = np.random.default_rng(123)
    matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    distribution = constriction.quant.EmpiricalDistribution(matrix)
    original_entropy = distribution.entropy_base2()
    assert np.allclose(original_entropy, 2.088376522064209)

    points, counts = distribution.points_and_counts()
    reconstructed_distribution = constriction.quant.EmpiricalDistribution(
        points, counts=counts)
    reconstructed_entropy = reconstructed_distribution.entropy_base2()
    assert reconstructed_entropy == original_entropy


def test_points_and_counts_example2a():
    rng = np.random.default_rng(123)
    matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    distribution = constriction.quant.EmpiricalDistribution(
        matrix, specialize_along_axis=0)
    expected_entropies = np.array([1.3709505, 1.5219281, 1.5219281, 1.921928])
    original_entropies = distribution.entropy_base2()
    assert np.allclose(original_entropies, expected_entropies)

    points, counts = distribution.points_and_counts()
    reconstructed_distribution = constriction.quant.EmpiricalDistribution(
        points, counts=counts, specialize_along_axis=0)
    reconstructed_entropies = reconstructed_distribution.entropy_base2()
    assert np.all(reconstructed_entropies == original_entropies)


def test_points_and_counts_example2b():
    rng = np.random.default_rng(123)
    matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    distribution = constriction.quant.EmpiricalDistribution(
        matrix, specialize_along_axis=1)
    expected_entropies = np.array([1., 1.5, 0.8112781, 1., 2.])
    original_entropies = distribution.entropy_base2()
    assert np.allclose(original_entropies, expected_entropies)

    points, counts = distribution.points_and_counts()
    reconstructed_distribution = constriction.quant.EmpiricalDistribution(
        points, counts=counts, specialize_along_axis=1)
    reconstructed_entropies = reconstructed_distribution.entropy_base2()
    assert np.all(reconstructed_entropies == original_entropies)
