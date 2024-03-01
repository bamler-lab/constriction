import constriction
import numpy as np


def test_empirical_distribution_update_marginal():
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


def test_empirical_distribution_update_rows():
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


def test_empirical_distribution_update_cols():
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


def test_rated_grid_update_marginal():
    def assert_grids_equal(grid1, grid2):
        points1, rates1 = grid1.points_and_rates()
        points2, rates2 = grid2.points_and_rates()
        assert np.all(points1 == points2)
        assert np.all(rates1 == rates2)

    rng = np.random.default_rng(1231)
    matrix1 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    matrix2 = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    combined = np.vstack((matrix1, matrix2))

    grid1_1 = constriction.quant.RatedGrid(combined)
    grid1_2 = constriction.quant.RatedGrid(combined)
    grid2 = constriction.quant.RatedGrid(matrix2[1:, :])

    grid1_1.remove(matrix1)
    grid1_1.insert(matrix2[1:, :])
    grid1_1.remove(matrix2)
    assert_grids_equal(grid1_1, grid2)

    grid1_2.update(matrix1, matrix2[1:, :])
    grid1_2.remove(matrix2)
    assert_grids_equal(grid1_2, grid2)


def test_rated_grid_update_rows():
    def assert_grids_equal(grid1, grid2):
        points1, rates1 = grid1.points_and_rates()
        points2, rates2 = grid2.points_and_rates()
        for p1, p2 in zip(points1, points2):
            assert np.all(p1 == p2)
        for r1, r2 in zip(rates1, rates2):
            assert np.all(r1 == r2)

    rng = np.random.default_rng(1232)
    matrix1 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    matrix2 = rng.binomial(10, 0.3, size=(3, 7)).astype(np.float32)
    combined = np.hstack((matrix1, matrix2))

    grid1_1 = constriction.quant.RatedGrid(
        combined, specialize_along_axis=0)
    grid1_2 = constriction.quant.RatedGrid(
        combined, specialize_along_axis=0)
    grid2 = constriction.quant.RatedGrid(
        matrix2[:, :5], specialize_along_axis=0)

    grid1_1.remove(matrix1)
    grid1_1.insert(matrix2[:, :5])
    grid1_1.remove(matrix2)
    assert_grids_equal(grid1_1, grid2)

    grid1_2.update(matrix1, matrix2[:, :5])
    grid1_2.remove(matrix2)
    assert_grids_equal(grid1_2, grid2)


def test_rated_grid_update_cols():
    def assert_grids_equal(grid1, grid2):
        points1, rates1 = grid1.points_and_rates()
        points2, rates2 = grid2.points_and_rates()
        for p1, p2 in zip(points1, points2):
            assert np.all(p1 == p2)
        for r1, r2 in zip(rates1, rates2):
            assert np.all(r1 == r2)

    rng = np.random.default_rng(1233)
    matrix1 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    matrix2 = rng.binomial(10, 0.3, size=(7, 5)).astype(np.float32)
    combined = np.vstack((matrix1, matrix2))

    grid1_1 = constriction.quant.RatedGrid(
        combined, specialize_along_axis=1)
    grid1_2 = constriction.quant.RatedGrid(
        combined, specialize_along_axis=1)
    grid2 = constriction.quant.RatedGrid(
        matrix2[4:, :], specialize_along_axis=1)

    grid1_1.remove(matrix1)
    grid1_1.insert(matrix2[4:, :])
    grid1_1.remove(matrix2)
    assert_grids_equal(grid1_1, grid2)

    grid1_2.update(matrix1,matrix2[4:, :])
    grid1_2.remove(matrix2)
    assert_grids_equal(grid1_2, grid2)
