import constriction
import numpy as np


def test_update_marginal():
    def assert_distributions_equal(distribution1, distribution2):
        points1, counts1 = distribution1.points_and_counts()
        points2, counts2 = distribution2.points_and_counts()
        assert np.all(points1 == points2)
        assert np.all(counts1 == counts2)

    rng = np.random.default_rng(1231)
    matrix1 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    matrix2 = rng.binomial(10, 0.3, size=(2, 5)).astype(np.float32)

    distribution1_1 = constriction.quant.EmpiricalDistribution(matrix1)
    distribution1_2 = constriction.quant.EmpiricalDistribution(matrix1)
    distribution2 = constriction.quant.EmpiricalDistribution(matrix2)
    distribution2.insert(matrix1[0, :])

    distribution1_1.remove(matrix1[1:, :])
    distribution1_1.insert(matrix2)
    assert_distributions_equal(distribution1_1, distribution2)

    distribution1_2.update(matrix1[1:, :], matrix2)
    assert_distributions_equal(distribution1_2, distribution2)


def test_update_rows():
    def assert_distributions_equal(distribution1, distribution2):
        points1, counts1 = distribution1.points_and_counts()
        points2, counts2 = distribution2.points_and_counts()
        for p1, p2 in zip(points1, points2):
            assert np.all(p1 == p2)
        for c1, c2 in zip(counts1, counts2):
            assert np.all(c1 == c2)

    rng = np.random.default_rng(1232)
    matrix1 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    matrix2 = rng.binomial(10, 0.3, size=(3, 2)).astype(np.float32)

    distribution1_1 = constriction.quant.EmpiricalDistribution(
        matrix1, specialize_along_axis=0)
    distribution1_2 = constriction.quant.EmpiricalDistribution(
        matrix1, specialize_along_axis=0)
    distribution2 = constriction.quant.EmpiricalDistribution(
        matrix2, specialize_along_axis=0)
    distribution2.insert(matrix1[:, :3])

    distribution1_1.remove(matrix1[:, 3:])
    distribution1_1.insert(matrix2)
    assert_distributions_equal(distribution1_1, distribution2)

    distribution1_2.update(matrix1[:, 3:], matrix2)
    assert_distributions_equal(distribution1_2, distribution2)


def test_update_cols():
    def assert_distributions_equal(distribution1, distribution2):
        points1, counts1 = distribution1.points_and_counts()
        points2, counts2 = distribution2.points_and_counts()
        for p1, p2 in zip(points1, points2):
            assert np.all(p1 == p2)
        for c1, c2 in zip(counts1, counts2):
            assert np.all(c1 == c2)

    rng = np.random.default_rng(1233)
    matrix1 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    matrix2 = rng.binomial(10, 0.3, size=(2, 5)).astype(np.float32)

    distribution1_1 = constriction.quant.EmpiricalDistribution(
        matrix1, specialize_along_axis=1)
    distribution1_2 = constriction.quant.EmpiricalDistribution(
        matrix1, specialize_along_axis=1)
    distribution2 = constriction.quant.EmpiricalDistribution(
        matrix2, specialize_along_axis=1)
    distribution2.insert(matrix1[:1, :])

    distribution1_1.remove(matrix1[1:, :])
    distribution1_1.insert(matrix2)
    assert_distributions_equal(distribution1_1, distribution2)

    distribution1_2.update(matrix1[1:, :], matrix2)
    assert_distributions_equal(distribution1_2, distribution2)
