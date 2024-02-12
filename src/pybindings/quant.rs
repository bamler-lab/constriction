use alloc::format;
use alloc::vec::Vec;
use core::{borrow::Borrow, convert::Infallible};
use pyo3::types::PyList;

use ndarray::parallel::prelude::*;
use ndarray::{ArrayBase, Axis, Dimension, IxDyn};
use numpy::{PyArray, PyArrayDyn, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;

use crate::quant::UnnormalizedDistribution;
use crate::F32;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<EmpiricalDistribution>()?;
    module.add_function(wrap_pyfunction!(vbq, module)?)?;
    module.add_function(wrap_pyfunction!(vbq_, module)?)?;
    Ok(())
}

#[derive(Debug, Clone, FromPyObject)]
pub enum PyReadonlyF32ArrayOrScalar<'py> {
    Array(PyReadonlyArrayDyn<'py, f32>),
    Scalar(f32),
}
use PyReadonlyF32ArrayOrScalar::*;

/// Dynamic data structure for frequencies of values within one or numpy arrays.
///
/// An `EmpiricalDistribution` counts how many times each value appears within within one or numpy
/// arrays. `EmpiricalDistribution` is a dynamic data structure, i.e., it provides methods to
/// efficiently [`remove`](#constriction.quant.EmpiricalDistribution.remove) and
/// [`update`](#constriction.quant.EmpiricalDistribution.update) existing points, and to
/// [`insert`](#constriction.quant.EmpiricalDistribution.insert) new points.
///
/// The constructor initializes the distribution from the provided numpy array `points`. If
/// `specialize_along_axis` is set to an integer, then the `EmpiricalDistribution` represents each
/// slice of `points` along the specified axis by an individual distribution, see Example 2 below.
///
/// ## Example 1: Marginal Distribution
///
/// Without setting `specialize_along_axis`, we obtain a distribution that counts occurrences over
/// the entire provided array:
///
/// ```python
/// rng = np.random.default_rng(123)
/// matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
/// print(f"matrix = {matrix}\n")
///
/// distribution = constriction.quant.EmpiricalDistribution(matrix)
/// points, counts = distribution.points_and_counts()
/// for point, count in zip(points, counts):
///     print(f"The matrix contains {count} instances of the value {point}.")
/// print(f"Entropy per entry: {distribution.entropy_base2()} bit")
/// ```
///
/// This prints:
///
/// ```text
/// matrix = [[4. 1. 2. 2. 2.]
///  [4. 5. 2. 4. 5.]
///  [3. 2. 4. 2. 4.]
///  [3. 5. 2. 4. 3.]]
///
/// The matrix contains 1 instances of the value 1.0.
/// The matrix contains 7 instances of the value 2.0.
/// The matrix contains 3 instances of the value 3.0.
/// The matrix contains 6 instances of the value 4.0.
/// The matrix contains 3 instances of the value 5.0.
/// Entropy per entry: 2.088376522064209 bit
/// ```
///
/// ## Example 2: Specialization Along an Axis
///
/// The example below uses the same matrix as Example 1 above, but constructs the
/// `EmpiricalDistribution` with argument `specialize_along_axis=0`. This creates a separate
/// distribution for each row (axis zero) of the matrix.
///
/// ```python
/// distribution = constriction.quant.EmpiricalDistribution(
///     matrix, specialize_along_axis=0)
/// entropies = distribution.entropy_base2()
/// points, counts = distribution.points_and_counts()
/// for i, (entropy, points, counts) in enumerate(zip(entropies, points, counts)):
///     summary = ", ".join(f"{p} ({c}x)" for (p, c) in zip(points, counts))
///     print(f"Row {i} has entropy {entropy:.2f} per entry and contains: {summary}.")
/// print(f"Mean entropy per entry: {entropies.mean()}")
/// ```
///
/// This prints:
///
/// ```text
/// Row 0 has entropy 1.37 per entry and contains: 1.0 (1x), 2.0 (3x), 4.0 (1x).
/// Row 1 has entropy 1.52 per entry and contains: 2.0 (1x), 4.0 (2x), 5.0 (2x).
/// Row 2 has entropy 1.52 per entry and contains: 2.0 (2x), 3.0 (1x), 4.0 (2x).
/// Row 3 has entropy 1.92 per entry and contains: 2.0 (1x), 3.0 (2x), 4.0 (1x), 5.0 (1x).
/// Mean entropy per entry: 1.5841835737228394
/// ```
///
/// Note that the mean entropy per entry is lower (or equal) if we model each row individually. This
/// is true in general.
#[pyclass]
#[derive(Debug)]
pub struct EmpiricalDistribution(EmpiricalDistributionImpl);

#[derive(Debug)]
enum EmpiricalDistributionImpl {
    Single(crate::quant::EmpiricalDistribution),
    Multiple {
        distributions: Vec<crate::quant::EmpiricalDistribution>,
        axis: usize,
    },
}

#[pymethods]
impl EmpiricalDistribution {
    #[new]
    pub fn new(
        py: Python<'_>,
        points: PyReadonlyArrayDyn<'_, f32>,
        specialize_along_axis: Option<usize>,
    ) -> PyResult<Py<Self>> {
        let points = points.as_array();
        let distribution = if let Some(axis) = specialize_along_axis {
            let distributions = points
                .axis_iter(Axis(axis))
                .into_par_iter()
                .map(|points| {
                    crate::quant::EmpiricalDistribution::try_from_points(points.iter().copied())
                })
                .collect::<Result<Vec<_>, _>>()?;
            EmpiricalDistributionImpl::Multiple {
                distributions,
                axis,
            }
        } else {
            let distribution =
                crate::quant::EmpiricalDistribution::try_from_points(points.iter().copied())?;
            EmpiricalDistributionImpl::Single(distribution)
        };

        Py::new(py, Self(distribution))
    }

    /// Add one or more values to the distribution.
    ///
    /// The argument `new` can be a scalar or a python array. If the distribution was constructed
    /// with `specialize_along_axis=i` for some `i` then `insert` must either be called with
    /// argument `index=j` (in which case, the point(s) in `new` will be inserted in the `j`th
    /// distribution), or `new` must be an array whose dimension along axis `i` equals the dimension
    /// of axis `i` of the array provided to the constructor (in this case, the array `new` is
    /// logically split into slices along axis `i`, and each slice is inserted into the
    /// corresponding distribution).
    ///
    /// ## Example 1: Adding Points to a Marginal Distribution
    ///
    /// ```python
    /// rng = np.random.default_rng(123)
    /// matrix1 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    /// matrix2 = rng.binomial(10, 0.3, size=(3, 5)).astype(np.float32)
    /// print(f"matrix1 = {matrix1}\n")
    /// print(f"matrix2 = {matrix2}\n")
    ///
    /// distribution = constriction.quant.EmpiricalDistribution(matrix1)
    /// points, counts = distribution.points_and_counts()
    /// for point, count in zip(points, counts):
    ///     print(f"matrix1 contains {count} instances of the value {point}.")
    /// print()
    ///
    /// distribution.insert(matrix2)
    /// points, counts = distribution.points_and_counts()
    /// for point, count in zip(points, counts):
    ///     print(f"both matrices combined contain {count} instances of the value {point}.")
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// matrix1 = [[4. 1. 2. 2. 2.]
    ///  [4. 5. 2. 4. 5.]
    ///  [3. 2. 4. 2. 4.]]
    ///
    /// matrix2 = [[3. 5. 2. 4. 3.]
    ///  [2. 2. 3. 3. 2.]
    ///  [0. 3. 4. 5. 3.]]
    ///
    /// matrix1 contains 1 instances of the value 1.0.
    /// matrix1 contains 6 instances of the value 2.0.
    /// matrix1 contains 1 instances of the value 3.0.
    /// matrix1 contains 5 instances of the value 4.0.
    /// matrix1 contains 2 instances of the value 5.0.
    ///
    /// both matrices combined contain 1 instances of the value 0.0.
    /// both matrices combined contain 1 instances of the value 1.0.
    /// both matrices combined contain 10 instances of the value 2.0.
    /// both matrices combined contain 7 instances of the value 3.0.
    /// both matrices combined contain 7 instances of the value 4.0.
    /// both matrices combined contain 4 instances of the value 5.0.
    /// ```
    ///
    /// ## Example 2: Specialization Along an Axis
    ///
    /// The example below uses the same matrices as Example 1 above, but it constructs the
    /// `EmpiricalDistribution` with argument `specialize_along_axis=0`. This creates a separate
    /// distribution for each row (axis zero) of the matrix. Calling `insert` inserts the contents
    /// of each row of the provided matrix to its corresponding distribution.
    ///
    /// ```python
    /// distribution = constriction.quant.EmpiricalDistribution(
    ///     matrix1, specialize_along_axis=0)
    /// points, counts = distribution.points_and_counts()
    /// for i, (points, counts) in enumerate(zip(points, counts)):
    ///     summary = ", ".join(f"{p} ({c}x)" for (p, c) in zip(points, counts))
    ///     print(f"Row {i} of matrix1 contains: {summary}.")
    /// print()
    ///
    /// # Insert each row of `matrix2` its corresponding distribution:
    /// distribution.insert(matrix2)
    ///
    /// # Insert a point into only the distribution with index 2 (instead of a single value `4.0`,
    /// # you could also provide a numpy array with arbitrary shape here):
    /// distribution.insert(4.0, index=2)
    ///
    /// points, counts = distribution.points_and_counts()
    /// for i, (points, counts) in enumerate(zip(points, counts)):
    ///     summary = ", ".join(f"{p} ({c}x)" for (p, c) in zip(points, counts))
    ///     print(f"Rows {i} of matrix1 and matrix2 contain: {summary}.")
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// Row 0 of matrix1 contains: 1.0 (1x), 2.0 (3x), 4.0 (1x).
    /// Row 1 of matrix1 contains: 2.0 (1x), 4.0 (2x), 5.0 (2x).
    /// Row 2 of matrix1 contains: 2.0 (2x), 3.0 (1x), 4.0 (2x).
    ///
    /// Rows 0 of matrix1 and 2 contain: 1.0 (1x), 2.0 (4x), 3.0 (2x), 4.0 (2x), 5.0 (1x).
    /// Rows 1 of matrix1 and 2 contain: 2.0 (4x), 3.0 (2x), 4.0 (2x), 5.0 (2x).
    /// Rows 2 of matrix1 and 2 contain: 0.0 (1x), 2.0 (2x), 3.0 (3x), 4.0 (4x), 5.0 (1x).
    /// ```
    pub fn insert(
        &mut self,
        new: PyReadonlyF32ArrayOrScalar<'_>,
        index: Option<usize>,
    ) -> PyResult<()> {
        if let Some(index) = index {
            let EmpiricalDistributionImpl::Multiple { distributions, .. } = &mut self.0 else {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ));
            };

            let distribution = distributions
                .get_mut(index)
                .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("`index` out of bounds"))?;

            match new {
                Scalar(new) => distribution.insert(F32::new(new)?, 1),
                Array(new) => {
                    for &new in new.as_array() {
                        distribution.insert(F32::new(new)?, 1);
                    }
                }
            }
        } else {
            match new {
                Scalar(new) => match &mut self.0 {
                    EmpiricalDistributionImpl::Single(distribution) => {
                        distribution.insert(F32::new(new)?, 1);
                    }
                    EmpiricalDistributionImpl::Multiple { .. } => {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(
                            "Scalar updates with an `EmpiricalDistribution` that was created with \
                            argument `specialize_along_axis` require argument `index`.",
                        ));
                    }
                },
                Array(new) => {
                    let new = new.as_array();

                    match &mut self.0 {
                        EmpiricalDistributionImpl::Single(distribution) => {
                            for &new in &new {
                                distribution.insert(F32::new(new)?, 1);
                            }
                        }
                        EmpiricalDistributionImpl::Multiple {
                            distributions,
                            axis,
                        } => {
                            let new = new.axis_iter(Axis(*axis));
                            if new.len() != distributions.len() {
                                return Err(pyo3::exceptions::PyIndexError::new_err(
                                    alloc::format!(
                                        "Axis {} has wrong dimension: expected {} but found {}.",
                                        axis,
                                        distributions.len(),
                                        new.len()
                                    ),
                                ));
                            }

                            new.into_par_iter().zip(distributions).try_for_each(
                                |(new, distribution)| {
                                    for &new in &new {
                                        distribution.insert(F32::new(new)?, 1);
                                    }
                                    Ok::<(), PyErr>(())
                                },
                            )?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Remove one or more values from the distribution.
    ///
    /// The argument `old` can be a scalar or a python array. If the distribution was constructed
    /// with `specialize_along_axis=i` for some `i` then `insert` must either be called with
    /// argument `index=j` (in which case, the point(s) in `old` will be removed from the `j`th
    /// distribution), or `old` must be an array whose dimension along axis `i` equals the dimension
    /// of axis `i` of the array provided to the constructor (in this case, the array `old` is
    /// logically split into slices along axis `i`, and the points from each slice are removed from
    /// the corresponding distribution).
    ///
    /// Returns an
    ///
    /// For code examples, see documentation of the method
    /// [`insert`](#constriction.quant.EmpiricalDistribution.insert), which has an analogous API.
    pub fn remove(
        &mut self,
        old: PyReadonlyF32ArrayOrScalar<'_>,
        index: Option<usize>,
    ) -> PyResult<()> {
        if let Some(index) = index {
            let EmpiricalDistributionImpl::Multiple { distributions, .. } = &mut self.0 else {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ));
            };

            let distribution = distributions
                .get_mut(index)
                .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("`index` out of bounds"))?;

            match old {
                Scalar(old) => {
                    distribution.remove(F32::new(old)?, 1)?;
                }
                Array(old) => {
                    for &old in old.as_array() {
                        distribution.remove(F32::new(old)?, 1)?;
                    }
                }
            }
        } else {
            match old {
                Scalar(old) => match &mut self.0 {
                    EmpiricalDistributionImpl::Single(distribution) => {
                        distribution.remove(F32::new(old)?, 1)?;
                    }
                    EmpiricalDistributionImpl::Multiple { .. } => {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(
                            "Scalar updates with an `EmpiricalDistribution` that was created with \
                        argument `specialize_along_axis` require argument `index`.",
                        ));
                    }
                },
                Array(old) => {
                    let old = old.as_array();

                    match &mut self.0 {
                        EmpiricalDistributionImpl::Single(distribution) => {
                            for &old in &old {
                                distribution.remove(F32::new(old)?, 1)?;
                            }
                        }
                        EmpiricalDistributionImpl::Multiple {
                            distributions,
                            axis,
                        } => {
                            let old = old.axis_iter(Axis(*axis));
                            if old.len() != distributions.len() {
                                return Err(pyo3::exceptions::PyIndexError::new_err(
                                    alloc::format!(
                                        "Axis {} has wrong dimension: expected {} but found {}.",
                                        axis,
                                        distributions.len(),
                                        old.len()
                                    ),
                                ));
                            }

                            old.into_par_iter().zip(distributions).try_for_each(
                                |(old, distribution)| {
                                    for &old in &old {
                                        distribution.remove(F32::new(old)?, 1)?;
                                    }
                                    Ok::<(), PyErr>(())
                                },
                            )?;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Update the positions of one or more values in the distribution.
    ///
    /// Calling `empirical_distribution.update(old, new)` is equivalent to:
    ///
    /// ```python
    /// empirical_distribution.remove(old)
    /// empirical_distribution.insert(new)
    /// ```
    ///
    /// However, `update` checks that `old` and `new` have matching dimensions. Also, if the values
    /// change only by small amounts then calling `update` might be faster because it should require
    /// fewer restructurings of the internal tree structure.
    ///
    /// The optional argument `index` has the same meaning as in the methods
    /// [`insert`](#constriction.quant.EmpiricalDistribution.insert) and
    /// [`remove`](#constriction.quant.EmpiricalDistribution.remove).
    ///
    /// For code examples, see documentation of the method
    /// [`insert`](#constriction.quant.EmpiricalDistribution.insert), which has an analogous API.
    pub fn update(
        &mut self,
        old: PyReadonlyF32ArrayOrScalar<'_>,
        new: PyReadonlyF32ArrayOrScalar<'_>,
        index: Option<usize>,
    ) -> PyResult<()> {
        if let Some(index) = index {
            let EmpiricalDistributionImpl::Multiple { distributions, .. } = &mut self.0 else {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ));
            };

            let distribution = distributions
                .get_mut(index)
                .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("`index` out of bounds"))?;

            match (old, new) {
                (Scalar(old), Scalar(new)) => {
                    distribution.remove(F32::new(old)?, 1)?;
                    distribution.insert(F32::new(new)?, 1);
                }
                (Array(old), Array(new)) if old.dims() == new.dims() => {
                    for (&old, &new) in old.as_array().iter().zip(&new.as_array()) {
                        distribution.remove(F32::new(old)?, 1)?;
                        distribution.insert(F32::new(new)?, 1);
                    }
                }
                _ => {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "`old` and `new` must have the same shape.",
                    ))
                }
            }
        } else {
            match (old, new) {
                (Scalar(old), Scalar(new)) => match &mut self.0 {
                    EmpiricalDistributionImpl::Single(distribution) => {
                        distribution.remove(F32::new(old)?, 1)?;
                        distribution.insert(F32::new(new)?, 1);
                    }
                    EmpiricalDistributionImpl::Multiple { .. } => {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(
                            "Scalar updates with an `EmpiricalDistribution` that was created with \
                            argument `specialize_along_axis` require argument `index`.",
                        ));
                    }
                },
                (Array(old), Array(new)) if old.dims() == new.dims() => {
                    let old = old.as_array();
                    let new = new.as_array();

                    match &mut self.0 {
                        EmpiricalDistributionImpl::Single(distribution) => {
                            for (&old, &new) in old.iter().zip(&new) {
                                distribution.remove(F32::new(old)?, 1)?;
                                distribution.insert(F32::new(new)?, 1);
                            }
                        }
                        EmpiricalDistributionImpl::Multiple {
                            distributions,
                            axis,
                        } => {
                            let old = old.axis_iter(Axis(*axis));
                            if old.len() != distributions.len() {
                                return Err(pyo3::exceptions::PyIndexError::new_err(
                                    alloc::format!(
                                        "Axis {} has wrong dimension: expected {} but found {}.",
                                        axis,
                                        distributions.len(),
                                        old.len()
                                    ),
                                ));
                            }

                            old.into_par_iter()
                                .zip(new.axis_iter(Axis(*axis)))
                                .zip(distributions)
                                .try_for_each(|((old, new), distribution)| {
                                    for (&old, &new) in old.iter().zip(&new) {
                                        distribution.remove(F32::new(old)?, 1)?;
                                        distribution.insert(F32::new(new)?, 1);
                                    }
                                    Ok::<(), PyErr>(())
                                })?;
                        }
                    }
                }
                _ => {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "`old` and `new` must have the same shape.",
                    ))
                }
            }
        }

        Ok(())
    }

    /// Move all entries with a given `old` value to a new `value`.
    ///
    /// The argument `old` can be a scalar or a python array. If the distribution was constructed
    /// with `specialize_along_axis=i` for some `i` then `shift` must either be called with argument
    /// `index=j` (in which case, only the `j`th distribution will be affected), or `old` and `new`
    /// must both be a list of scalars or rank 1 numpy arrays. In the latter case, the length of the
    /// list must equal the dimension of axis `i` of the array provided to the constructor, and each
    /// list entry specifies how the corresponding distribution is updated.
    ///
    /// TODO: examples
    pub fn shift(&mut self, old: &PyAny, new: &PyAny, index: Option<usize>) -> PyResult<()> {
        if let Some(index) = index {
            let EmpiricalDistributionImpl::Multiple { distributions, .. } = &mut self.0 else {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ));
            };

            let distribution = distributions
                .get_mut(index)
                .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("`index` out of bounds"))?;

            return shift_single(old, new, distribution, || {
                pyo3::exceptions::PyAssertionError::new_err(
                    "If `index` is set then `old` and `new` must both be scalars or rank-1 \
                        arrays (with the same shape).",
                )
            });
        } else {
            // No `index` argument provided.
            match &mut self.0 {
                EmpiricalDistributionImpl::Single(distribution) => {
                    return shift_single(old, new, distribution, || {
                        pyo3::exceptions::PyAssertionError::new_err(
                            "`old` and `new` must both be scalars or rank-1 tensors (with same shape).",
                        )
                    });
                }
                EmpiricalDistributionImpl::Multiple { distributions, .. } => {
                    if let Ok(old) = old.extract::<&PyList>() {
                        if let Ok(new) = new.extract::<&PyList>() {
                            if old.len() != distributions.len() || new.len() != distributions.len()
                            {
                                return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                                    "Lists `old` and `new` must have length {} but have \
                                    lengths {} and {}, respectively.",
                                    distributions.len(),
                                    old.len(),
                                    new.len(),
                                )));
                            }

                            return old.iter()
                                .zip(new)
                                .zip(distributions)
                                .try_for_each(|((old, new), distribution)| {
                                    shift_single(
                                        old,
                                        new,
                                        distribution,
                                        || pyo3::exceptions::PyAssertionError::new_err(
                                            "Each element of the lists `old` and `new` must be a scalar or a rank-1 array, and dimensions of corresponding entries must match."
                                        )
                                    )
                                });
                        }
                    }

                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "If no `index` argument is provided then `old` and `new` must both be lits."
                    ));
                }
            }
        }

        fn shift_single(
            old: &PyAny,
            new: &PyAny,
            distribution: &mut crate::quant::EmpiricalDistribution,
            mk_err: impl Fn() -> PyErr,
        ) -> Result<(), PyErr> {
            if let Ok(old) = old.extract::<f32>() {
                if let Ok(new) = new.extract::<f32>() {
                    let count = distribution.remove_all(F32::new(old)?);
                    distribution.insert(F32::new(new)?, count);
                    return Ok(());
                }
            } else if let Ok(old) = old.extract::<PyReadonlyArrayDyn<'_, f32>>() {
                if let Ok(new) = new.extract::<PyReadonlyArrayDyn<'_, f32>>() {
                    if old.dims() == new.dims() && old.dims().ndim() == 1 {
                        let old = old.as_array();
                        let new = new.as_array();

                        for (&old, &new) in old.iter().zip(&new) {
                            let count = distribution.remove_all(F32::new(old)?);
                            distribution.insert(F32::new(new)?, count);
                        }
                        return Ok(());
                    }
                }
            }
            Err(mk_err())
        }
    }

    /// Returns the total number of points that are represented by the distribution.
    ///
    /// For a new `EmpiricalDistribution`, the method `total` returns the number of elements in the
    /// numpy array that was provided to the constructor. Calling `insert` or `remove` increases or
    /// decreases, respectively, the value returned by `total`. Calling `update` leaves it
    /// invariant.
    ///
    /// If the `EmpiricalDistribution` was constructed with argument `specialize_along_axis` set,
    /// then `total` returns the number of points in slice specified by `index` (if provided), or a
    /// list of totals for each slice (if no `index` is provided).
    ///
    /// ## Examples
    ///
    /// ```python
    /// rng = np.random.default_rng(123)
    /// matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    /// print(f"matrix = {matrix}\n")
    ///
    /// marginal_distribution = constriction.quant.EmpiricalDistribution(matrix)
    /// specialized_distribution = constriction.quant.EmpiricalDistribution(
    ///     matrix, specialize_along_axis=0)
    ///
    /// specialized_distribution.insert(10., index=2) # Insert only into the slice at index=2.
    ///
    /// print(f"marginal_distribution.total() = {marginal_distribution.total()}")
    /// print(f"specialized_distribution.total() = {specialized_distribution.total()}")
    /// print(f"specialized_distribution.total(2) = {specialized_distribution.total(2)}")
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// matrix = [[4. 1. 2. 2. 2.]
    ///  [4. 5. 2. 4. 5.]
    ///  [3. 2. 4. 2. 4.]
    ///  [3. 5. 2. 4. 3.]]
    ///
    /// marginal_distribution.total() = 20
    /// specialized_distribution.total() = [5 5 6 5]
    /// specialized_distribution.total(2) = 6
    /// ```
    pub fn total(&self, py: Python<'_>, index: Option<usize>) -> PyResult<PyObject> {
        match (index, &self.0) {
            (Some(_), EmpiricalDistributionImpl::Single { .. }) => {
                Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ))
            }
            (
                Some(index),
                EmpiricalDistributionImpl::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let distribution = distributions.get(index).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("`index` out of bounds")
                })?;
                Ok(distribution.total().to_object(py))
            }
            (None, EmpiricalDistributionImpl::Single(distribution)) => {
                Ok(distribution.total().to_object(py))
            }
            (
                None,
                EmpiricalDistributionImpl::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let totals = distributions.iter().map(|d| d.total()).collect::<Vec<_>>();
                Ok(PyArray::from_vec(py, totals).to_object(py))
            }
        }
    }

    /// Returns the Shannon entropy per entry, to base 2.
    ///
    /// If the `EmpiricalDistribution` was constructed with argument `specialize_along_axis` set,
    /// then this method returns the entropy of the slice specified by `index` (if provided), or a
    /// list of entropies for each slice (if no `index` is provided).
    ///
    /// ## Example
    ///
    /// ```python
    /// rng = np.random.default_rng(123)
    /// matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    /// print(f"matrix = {matrix}\n")
    ///
    /// marginal_distribution = constriction.quant.EmpiricalDistribution(matrix)
    /// specialized_distribution = constriction.quant.EmpiricalDistribution(
    ///     matrix, specialize_along_axis=0)
    ///
    /// print(f"marginal_distribution.entropy_base2() = {marginal_distribution.entropy_base2()}")
    /// print(f"specialized_distribution.entropy_base2() = {specialized_distribution.entropy_base2()}")
    /// print(f"specialized_distribution.entropy_base2(2) = {specialized_distribution.entropy_base2(2)}")
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// matrix = [[4. 1. 2. 2. 2.]
    ///  [4. 5. 2. 4. 5.]
    ///  [3. 2. 4. 2. 4.]
    ///  [3. 5. 2. 4. 3.]]
    ///
    /// marginal_distribution.entropy_base2() = 2.088376522064209
    /// specialized_distribution.entropy_base2() = [1.3709505 1.5219281 1.5219281 1.921928 ]
    /// specialized_distribution.entropy_base2(2) = 1.521928071975708
    /// ```
    pub fn entropy_base2(&self, py: Python<'_>, index: Option<usize>) -> PyResult<PyObject> {
        match (index, &self.0) {
            (Some(_), EmpiricalDistributionImpl::Single { .. }) => {
                Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ))
            }
            (
                Some(index),
                EmpiricalDistributionImpl::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let distribution = distributions.get(index).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("`index` out of bounds")
                })?;
                Ok(distribution.entropy_base2::<f32>().to_object(py))
            }
            (None, EmpiricalDistributionImpl::Single(distribution)) => {
                Ok(distribution.entropy_base2::<f32>().to_object(py))
            }
            (
                None,
                EmpiricalDistributionImpl::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let entropies = distributions
                    .iter()
                    .map(|d| d.entropy_base2::<f32>())
                    .collect::<Vec<_>>();
                Ok(PyArray::from_vec(py, entropies).to_object(py))
            }
        }
    }

    /// Returns a tuple `(points, counts)`, which are both 1d numpy arrays of equal length, where
    /// `points` contains a sorted list of all *distinct* values represented by the
    /// `EmpiricalDistribution`, and `counts` contains their respective multiplicities, i.e., how
    /// often each point occurs in the data.
    ///
    /// If the `EmpiricalDistribution` was constructed with argument `specialize_along_axis` set,
    /// then `points_and_counts` returns the points and counts for the slice specified by `index`
    /// (if provided), or a list of tuples `(points, counts)`, with one tuple per slice.
    ///
    /// See code examples in the top-level documentation of the class `EmpiricalDistribution`.
    pub fn points_and_counts(
        &self,
        py: Python<'_>,
        index: Option<usize>,
    ) -> PyResult<(PyObject, PyObject)> {
        match (index, &self.0) {
            (Some(_), EmpiricalDistributionImpl::Single { .. }) => {
                Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ))
            }
            (
                Some(index),
                EmpiricalDistributionImpl::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let distribution = distributions.get(index).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("`index` out of bounds")
                })?;
                let (points, counts): (Vec<_>, Vec<_>) = distribution
                    .iter()
                    .map(|(value, count)| (value.get(), count))
                    .unzip();
                let points = PyArray::from_vec(py, points).to_object(py);
                let counts = PyArray::from_vec(py, counts).to_object(py);
                Ok((points, counts))
            }
            (None, EmpiricalDistributionImpl::Single(distribution)) => {
                let (points, counts): (Vec<_>, Vec<_>) = distribution
                    .iter()
                    .map(|(value, count)| (value.get(), count))
                    .unzip();
                let points = PyArray::from_vec(py, points).to_object(py);
                let counts = PyArray::from_vec(py, counts).to_object(py);
                Ok((points, counts))
            }
            (
                None,
                EmpiricalDistributionImpl::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let vecs = distributions
                    .par_iter()
                    .map(|distribution| {
                        distribution
                            .iter()
                            .map(|(value, count)| (value.get(), count))
                            .unzip()
                    })
                    .collect::<Vec<(Vec<_>, Vec<_>)>>();
                let (points, counts): (Vec<_>, Vec<_>) = vecs
                    .into_iter()
                    .map(|(points, counts)| {
                        let points = PyArray::from_vec(py, points).to_object(py);
                        let counts = PyArray::from_vec(py, counts).to_object(py);
                        (points, counts)
                    })
                    .unzip();
                let points = PyArray::from_vec(py, points).to_object(py);
                let counts = PyArray::from_vec(py, counts).to_object(py);
                Ok((points, counts))
            }
        }
    }
}

/// Quantizes an array of values using [Variational Bayesian Quantization (VBQ)].
///
/// Returns an array of quantized values with the same shape and dtype as the argument
/// `unquantized`. If you want to instead overwrite the original array with its quantized values
/// then use `vbq_` (with trailing underscore) instead.
///
/// VBQ is a quantization method that takes into account (i) the "prior" distribution of
/// unquantized points (putting a higher density of grid points in regions of high prior density)
/// and (ii) the saliency of the value that we quantize, i.e., how much a given distortion of a
/// single value would hurt the overall quality of the quantization of a collection of values.
///
/// ## Overview of the VBQ Method
///
/// For each entry in the argument `unquantized`, VBQ returns a point `quantized` by minimizing the
/// following objective function:
///
/// `loss(quantized) = distortion(quantized - unquantized) + coarseness * rate_estimate(quantized)`
///
/// where the Python API is currently restricted to a quadratic distortion,
///
/// `distortion(quantized - unquantized) = (quantized - unquantized)**2 / (2 * posterior_variance)`.
///
/// Future versions of `constriction` might provide support for more general distortion metrics
/// (the Rust API of constriction
/// [already does](https://docs.rs/constriction/latest/constriction/quant/fn.vbq.html)).
///
/// Here, the `rate_estimate` in the above objective function is calculated based on the provided
/// `prior` distribution and on some theoretical considerations of how the VBQ algorithm works. You
/// will get better estimates (and therefore better  quantization results) if the `prior`
/// approximates the distribution of quantized points. Since you don't have access to the quantized
/// points before running VBQ, it is recommended to run VBQ on a given set of points several times
/// in a row. In the first run, set the `prior` to the empirical distribution of *unquantized*
/// points. In  subsequent runs of VBQ, set the prior to the empirical distribution of the quantized
/// points you obtained in the previous run).
///
/// Here, `coarseness` is a nonnegative scalar that controls how much distortion is acceptable
/// (higher values for `coarseness` lead to a sparser quantization grid and lower entropy of the
/// quantized values, while lower values for `coarseness` ensure that quantized values stay close
/// to their unquantized counterparts). The argument `posterior_variance` is typically a numpy array
/// of the same dimension as `unquantized` (although it can also be a scalar to support edge cases).
///
///
///  and the scalar `bit_penalty` are provided by the caller, and
/// `rate_estimate` is an estimate of the information content that the result `quantized` will have
/// under the empirical distribution of the collection of all quantized points (see below). To use
/// VBQ as described in the [original paper] (Eq. 9), set `bit_penalty = 2.0 * lambda *
/// posterior_variance`, where `lambda > 0` is a parameter that trades off between rate and
/// distortion (`lambda` is the same for all points you want to quantize) and `posterior_variance`
/// is an estimate of the variance of the point you currently want to quantize under its Bayesian
/// posterior distribution (`posterior_variance` will typically be different for each point you
/// want to quantize, or else a different quantization method may be more suitable).
///
/// ## Arguments
///
/// - `unquantized`: a numpy array of values that you want to quantize.
/// - `prior`: an `EmpiricalDistribution` that influences how VBQ distributes its grid points.
///   You typically want to use `prior = EmpiricalDistribution(unquantized)` or
///   `prior = EmpiricalDistribution(unquantized, specialize_along_axis=some_axis)` here, so that
///   VBQ uses the empirical distribution of unquantized values to inform its positioning of grid
///   points. However, if you already have a better idea of where grid points will likely end up
///   (e.g., because you already ran VBQ once with argument `update_prior=True`, then it can be
///   better to provide the distribution over estimated quantized points instead).
/// - `posterior_variance`: typically a numpy array ith the same dimensions as `unquantized`
///   (although it can also be a scalar to support edge cases). The `posterior_variance` controls
///   how important a faithful quantization of each entry of `unquantized` is, relative to the other
///   entries. See objective function stated above. Entries with higher `posterior_variance` will
///   generally be quantized to grid points that can be further away from their unquantized values
///   than entries with lower `posterior_variance`.
/// - `coarseness`: a nonnegative scalar that controls how much distortion is acceptable globally
///   (higher values for `coarseness` lead to a sparser quantization grid and lower entropy of the
///   quantized values, while lower values for `coarseness` ensure that quantized values stay close
///   to their unquantized counterparts). The argument `coarseness` is a convenience. Setting
///   `coarseness` to a value different from `1.0` has the same effect as multiplying all entries of
///   `posterior_variance` by `coarseness`.
/// - `update_prior`: optional boolean that decides whether the provided `prior` will be updated
///   after quantizing each entry of `unquantized`. Defaults to `false` if now `reference` is
///   provided. Providing a `reverence` implies `update_prior=True.`
///   Setting `update_prior=True` has two effects:
///   (i) once `vbq` terminates, all `unquantized` (or `reference`) points are removed from `prior`
///   and replaced by the (returned) quantized points; and
///   (ii) since the updates occur by piecemeal immediately once each entry was quantized, entries
///   towards the end of the array `unquantized` are quantized with a better estimate of the final
///   distribution of quantized points. However, this also means that each entry of `unquantized`
///   gets quantized with a different prior, and therefore potentially to a slightly different grid,
///   which can result in spurious clusters of grid points that lie very close to each other. For
///   this reason, setting `update_prior=True` is recommended only for intermediate runs of VBQ that
///   are part of some convergence process. Any final run of VBQ should set `update_prior=False`.
/// - `reference`: an optional array with same dimensions as `unquantized`. Providing a `reference`
///   implies `update_prior=True`, and it changes how prior updates are carried out: *without* a
///   `reference`, prior updates are carried out under the assumption that `prior` is the
///   distribution of values in the array `unquantized`; thus, prior updates remove an entry of
///   `unquantized` from the `prior` and replace it with the corresponding quantized value. By
///   contrast, if a `reference` is provided, then prior updates are carried out under the
///   assumption that `prior` is the distribution of values in `reference`; thus, prior updates
///   remove an entry of `reference` from the `prior` and replace it with the quantized value of the
///   corresponding entry of `unquantized`.
///
/// ## Example 1: quantization with a *global* prior (i.e. without `specialize_along_axis`)
///
/// ```python
/// rng = np.random.default_rng(123)
/// unquantized = rng.normal(size=(4, 5)).astype(np.float32)
/// print(f"unquantized values:\n{unquantized}\n")
///
/// prior = constriction.quant.EmpiricalDistribution(unquantized)
///
/// print(f"entropy before quantization: {prior.entropy_base2()}")
/// print(f"(this is simply log_2(20) = {np.log2(20)} since all matrix entries are different)")
/// print()
///
/// # Allow larger quantization errors in upper left corner of the matrix by using a high variance:
/// posterior_variance = np.array([
///     [10.0, 10.0, 1.0, 1.0, 1.0],
///     [10.0, 10.0, 1.0, 1.0, 1.0],
///     [1.0, 1.0, 1.0, 1.0, 1.0],
///     [1.0, 1.0, 1.0, 1.0, 1.0],
/// ], dtype=np.float32)
///
/// quantized_fine = constriction.quant.vbq(unquantized, prior, posterior_variance, 0.01)
/// quantized_coarse = constriction.quant.vbq(unquantized, prior, posterior_variance, 0.1)
///
/// print(f"quantized to a fine grid:\n{quantized_fine}\n")
/// print(f"quantization errors:\n{np.abs(unquantized - quantized_fine)}\n")
///
/// print(f"quantized to a coarse grid:\n{quantized_coarse}\n")
/// print(f"quantization errors:\n{np.abs(unquantized - quantized_coarse)}\n")
/// ```
///
/// This prints:
///
/// ```text
/// unquantized values:
/// [[-0.9891214  -0.36778665  1.2879252   0.19397442  0.9202309 ]
///  [ 0.5771038  -0.63646364  0.5419522  -0.31659546 -0.32238913]
///  [ 0.09716732 -1.5259304   1.1921661  -0.67108965  1.0002694 ]
///  [ 0.13632113  1.5320331  -0.6599694  -0.31179485  0.33776912]]
///
/// entropy before quantization: 4.321928024291992
/// (this is simply log_2(20) = 4.321928094887363 since all matrix entries are different)
///
/// quantized to a fine grid:
/// [[-0.67108965 -0.36778665  1.1921661   0.13632113  0.9202309 ]
///  [ 0.13632113 -0.36778665  0.5419522  -0.36778665 -0.36778665]
///  [ 0.13632113 -1.5259304   1.1921661  -0.67108965  0.9202309 ]
///  [ 0.13632113  1.5320331  -0.67108965 -0.36778665  0.33776912]]
///
/// quantization errors:
/// [[0.31803173 0.         0.09575915 0.05765329 0.        ]
///  [0.44078267 0.268677   0.         0.05119118 0.04539752]
///  [0.03915381 0.         0.         0.         0.08003849]
///  [0.         0.         0.01112026 0.0559918  0.        ]]
///
/// quantized to a coarse grid:
/// [[ 0.13632113  0.13632113  0.9202309   0.13632113  0.9202309 ]
///  [ 0.13632113  0.13632113  0.13632113 -0.36778665 -0.36778665]
///  [ 0.13632113 -1.5259304   0.9202309  -0.36778665  0.9202309 ]
///  [ 0.13632113  1.1921661  -0.36778665  0.13632113  0.13632113]]
///
/// quantization errors:
/// [[1.1254425  0.5041078  0.36769432 0.05765329 0.        ]
///  [0.44078267 0.77278477 0.40563107 0.05119118 0.04539752]
///  [0.03915381 0.         0.27193516 0.303303   0.08003849]
///  [0.         0.339867   0.29218274 0.44811597 0.201448  ]]
/// ```
///
/// Note that, with both low and high `coarseness`, the quantization error tends to be larger in the
/// upper left 2x2 block of the matrix. This is because we set the `posterior_variance` high in this
/// block, which tells `vbq` to care less about distortion for the corresponding values.
///
/// ## Example 2: quantization with `specialize_along_axis`
///
/// We can quantize to a different grid for each row of the matrix by replacing the following line
/// in Example 1 above:
///
/// ```python
/// prior = constriction.quant.EmpiricalDistribution(unquantized)
/// ```
///
/// with
///
/// ```python
/// prior = constriction.quant.EmpiricalDistribution(unquantized, specialize_along_axis=0)
/// ```
///
/// With this change, we obtain the following output:
///
/// ```text
/// unquantized values:
/// [[-0.9891214  -0.36778665  1.2879252   0.19397442  0.9202309 ]
///  [ 0.5771038  -0.63646364  0.5419522  -0.31659546 -0.32238913]
///  [ 0.09716732 -1.5259304   1.1921661  -0.67108965  1.0002694 ]
///  [ 0.13632113  1.5320331  -0.6599694  -0.31179485  0.33776912]]
///
/// Row entropies before quantization: [2.321928 2.321928 2.321928 2.321928]
/// (this is simply log_2(5) = 2.321928094887362 since each row contains 5 distinct values)
///
/// quantized to a fine grid:
/// [[-0.9891214  -0.36778665  1.2879252   0.19397442  0.9202309 ]
///  [ 0.5419522  -0.31659546  0.5419522  -0.31659546 -0.31659546]
///  [ 0.09716732 -1.5259304   1.1921661  -0.67108965  1.0002694 ]
///  [ 0.13632113  1.5320331  -0.6599694  -0.31179485  0.33776912]]
///
/// quantization errors:
/// [[0.         0.         0.         0.         0.        ]
///  [0.0351516  0.31986818 0.         0.         0.00579366]
///  [0.         0.         0.         0.         0.        ]
///  [0.         0.         0.         0.         0.        ]]
///
/// quantized to a coarse grid:
/// [[ 0.19397442  0.19397442  0.9202309   0.19397442  0.9202309 ]
///  [-0.31659546 -0.31659546  0.5419522  -0.31659546 -0.31659546]
///  [ 0.09716732 -1.5259304   1.0002694  -0.67108965  1.0002694 ]
///  [ 0.13632113  1.5320331  -0.31179485 -0.31179485  0.13632113]]
///
/// quantization errors:
/// [[1.1830958  0.5617611  0.36769432 0.         0.        ]
///  [0.8936993  0.31986818 0.         0.         0.00579366]
///  [0.         0.         0.19189668 0.         0.        ]
///  [0.         0.         0.34817454 0.         0.201448  ]]
/// ```
///
/// Notice that both quantized matrices have repeating entries within rows, but no repeats within
/// columns. This is because each row of the matrix is quantized using a different slice of the
/// `prior`, and thus VBQ constructs a different grid for each row.
///
/// ## References
///
/// VBQ was originally proposed and empirically evaluated for the compression of images and word
/// embeddings by [Yang et al., ICML 2020]. For an empirical evaluation of VBQ for the compression
/// of neural network weights, see [Tan and Bamler, Deploy & Monitor ML Workshop @ NeurIPS 2022].
///
/// [Variational Bayesian Quantization (VBQ)]: http://proceedings.mlr.press/v119/yang20a/yang20a.pdf
/// [Yang et al., ICML 2020]: http://proceedings.mlr.press/v119/yang20a/yang20a.pdf
/// [original paper]: http://proceedings.mlr.press/v119/yang20a/yang20a.pdf
/// [Tan and Bamler, Deploy & Monitor ML Workshop @ NeurIPS 2022]:
///   https://raw.githubusercontent.com/dmml-workshop/dmml-neurips-2022/main/accepted-papers/paper_21.pdf
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn vbq<'a>(
    py: Python<'a>,
    unquantized: PyReadonlyArrayDyn<'a, f32>,
    prior: Py<EmpiricalDistribution>,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<&'a PyArrayDyn<f32>> {
    match &mut prior.borrow_mut(py).0 {
        EmpiricalDistributionImpl::Single(distribution) => vbq_single(
            py,
            unquantized,
            distribution,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
        ),
        EmpiricalDistributionImpl::Multiple {
            distributions,
            axis,
        } => vbq_multiplexed(
            py,
            unquantized,
            distributions,
            *axis,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
        ),
    }
}

fn vbq_single<'a>(
    py: Python<'a>,
    unquantized: PyReadonlyArrayDyn<'a, f32>,
    prior: &mut crate::quant::EmpiricalDistribution,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<&'a PyArrayDyn<f32>> {
    let len = unquantized.len();
    let mut quantized: Vec<f32> = Vec::with_capacity(len);
    let unquantized = unquantized.as_array();
    let dim = unquantized.raw_dim();

    {
        let mut quantized = ArrayBase::<ndarray::ViewRepr<&mut _>, _>::from_shape(
            dim.clone(),
            quantized.spare_capacity_mut(),
        )
        .expect("len was chosen to match shape");

        match (update_prior, reference.is_some()) {
            (Some(false), true) => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "providing a `reference` implies `update_prior=True`",
                ))
            }
            (None | Some(false), false) => {
                // Prior doesn't get mutated, so we can run VBQ in parallel.
                let src_and_dst = par_azip!(&unquantized, &mut quantized);
                match posterior_variance {
                    Array(posterior_variance) => {
                        if posterior_variance.dims() != dim {
                            return Err(pyo3::exceptions::PyAssertionError::new_err(
                                "`posterior_variance` must have the same shape as `unquantized`.",
                            ));
                        }
                        let posterior_variance = posterior_variance.as_array();
                        let two_coarseness = 2.0 * coarseness;
                        let src_dst_penalty = src_and_dst
                            .and(&posterior_variance)
                            .into_par_iter()
                            .map(|(src, dst, var)| ((src, dst), F32::new(two_coarseness * *var)));
                        vbq_parallel(
                            src_dst_penalty,
                            prior,
                            |(src, _dst)| **src,
                            |(_src, dst), value| {
                                dst.write(value);
                            },
                        )
                    }
                    Scalar(posterior_variance) => {
                        let bit_penalty = Result::<_, Infallible>::Ok(F32::new(
                            2.0 * coarseness * posterior_variance,
                        )?);
                        let src_dst_penalty =
                            src_and_dst.into_par_iter().map(|sd| (sd, bit_penalty));
                        vbq_parallel(
                            src_dst_penalty,
                            prior,
                            |(src, _dst)| **src,
                            |(_src, dst), value| {
                                dst.write(value);
                            },
                        )
                    }
                }?;
            }
            _ => {
                // We mutate the prior after each update, and therefore the result depends on the
                // order in which we iterate over items. Fall back to a sequential implementation
                // since running VBQ in parallel would make it nondeterministic here.
                vbq_sequential(
                    unquantized.iter().zip(quantized.iter_mut()),
                    dim,
                    prior,
                    posterior_variance,
                    coarseness,
                    reference,
                    |_src, dst, value| {
                        dst.write(value);
                    },
                )?
            }
        }
    }
    unsafe {
        // SAFETY: We created `quantized` with the required capacity and initialized all of its
        // entries above.
        quantized.set_len(len);
    }
    let quantized = ArrayBase::from_shape_vec(unquantized.dim(), quantized)
        .expect("Vec should have correct len");
    Ok(PyArray::from_owned_array(py, quantized))
}

#[allow(clippy::too_many_arguments)]
fn vbq_multiplexed<'a>(
    py: Python<'a>,
    unquantized: PyReadonlyArrayDyn<'a, f32>,
    priors: &mut [crate::quant::EmpiricalDistribution],
    axis: usize,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<&'a PyArrayDyn<f32>> {
    let len = unquantized.len();

    let shape = unquantized.shape();
    let unquantized = unquantized.as_array();
    let unquantized = unquantized.axis_iter(Axis(axis));
    if unquantized.len() != priors.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err(alloc::format!(
            "Axis {} has wrong dimension: expected {} but found {}.",
            axis,
            priors.len(),
            unquantized.len()
        )));
    }

    let mut quantized: Vec<f32> = Vec::with_capacity(len);

    {
        let mut quantized = ArrayBase::<ndarray::ViewRepr<&mut _>, _>::from_shape(
            shape,
            quantized.spare_capacity_mut(),
        )
        .expect("len was chosen to match shape");

        let unquantized = unquantized.into_par_iter();
        let quantized = quantized.axis_iter_mut(Axis(axis));
        let quantized = quantized.into_par_iter();

        vbq_multiplexed_generic(
            shape,
            unquantized,
            quantized,
            priors,
            axis,
            posterior_variance,
            coarseness,
            |_src, dst, new| {
                dst.write(new);
            },
            update_prior,
            reference,
        )?;
    }
    unsafe {
        // SAFETY: We created `quantized` with the required capacity and initialized all of its
        // entries above.
        quantized.set_len(len);
    }
    let quantized =
        ArrayBase::from_shape_vec(shape, quantized).expect("Vec should have correct len");
    Ok(PyArray::from_owned_array(py, quantized))
}

#[allow(clippy::too_many_arguments)]
fn vbq_multiplexed_inplace<'a>(
    mut unquantized: PyReadwriteArrayDyn<'a, f32>,
    priors: &mut [crate::quant::EmpiricalDistribution],
    axis: usize,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<()> {
    let shape = unquantized.shape().to_vec();
    let mut unquantized = unquantized.as_array_mut();
    let unquantized = unquantized.axis_iter_mut(Axis(axis));
    if unquantized.len() != priors.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err(alloc::format!(
            "Axis {} has wrong dimension: expected {} but found {}.",
            axis,
            priors.len(),
            unquantized.len()
        )));
    }

    let unquantized = unquantized.into_par_iter();

    vbq_multiplexed_generic(
        &shape,
        unquantized,
        rayon::iter::repeat(core::iter::repeat(())).take(priors.len()),
        priors,
        axis,
        posterior_variance,
        coarseness,
        |src, _dst, new| {
            *src = new;
        },
        update_prior,
        reference,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn vbq_multiplexed_generic<'a, I1, I2, U, Src, Dst>(
    shape: &[usize],
    unquantized: I1,
    quantized: I2,
    priors: &mut [crate::quant::EmpiricalDistribution],
    axis: usize,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'a>,
    coarseness: f32,
    update: U,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'a, f32>>,
) -> PyResult<()>
where
    I1: IndexedParallelIterator,
    <I1 as ParallelIterator>::Item: IntoIterator<Item = Src>,
    I2: IndexedParallelIterator,
    <I2 as ParallelIterator>::Item: IntoIterator<Item = Dst>,
    Src: Borrow<f32>,
    U: Fn(Src, Dst, f32) + Send + Sync,
{
    fn inner<I1, I2, I3, R, E, U1, U2, Src, Dst>(
        shape: &[usize],
        unquantized: I1,
        quantized: I2,
        priors: &mut [crate::quant::EmpiricalDistribution],
        axis: usize,
        posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
        coarseness: f32,
        update: U1,
        update_prior: U2,
        reference: I3,
    ) -> PyResult<()>
    where
        I1: IndexedParallelIterator,
        <I1 as ParallelIterator>::Item: IntoIterator<Item = Src>,
        I2: IndexedParallelIterator,
        <I2 as ParallelIterator>::Item: IntoIterator<Item = Dst>,
        I3: IndexedParallelIterator,
        <I3 as ParallelIterator>::Item: IntoIterator<Item = R>,
        Src: Borrow<f32>,
        U1: Fn(Src, Dst, f32) + Send + Sync,
        U2: Fn(&mut crate::quant::EmpiricalDistribution, F32, F32, R) -> Result<(), E>
            + Send
            + Sync,
        E: Send + Sync,
        PyErr: From<E>,
    {
        match posterior_variance {
            Array(posterior_variance) => {
                if posterior_variance.shape() != shape {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "`posterior_variance` must have the same shape as `unquantized`.",
                    ));
                }
                let posterior_variance = posterior_variance.as_array();
                let two_coarseness = 2.0 * coarseness;

                unquantized
                    .zip(quantized)
                    .zip(posterior_variance.axis_iter(Axis(axis)))
                    .zip(priors)
                    .zip(reference)
                    .try_for_each(|((((src, dst), var), prior), reference)| {
                        for (((src, dst), &var), reference) in
                            src.into_iter().zip(dst).zip(&var).zip(reference)
                        {
                            let old = F32::new(*src.borrow())?;
                            let new = crate::quant::vbq(
                                old,
                                prior,
                                |x| x * x,
                                F32::new(two_coarseness * var)?,
                            );
                            update(src, dst, new.get());
                            update_prior(prior, old, new, reference)?;
                        }
                        Ok::<(), PyErr>(())
                    })?;
            }
            Scalar(posterior_variance) => {
                let bit_penalty = F32::new(2.0 * coarseness * posterior_variance)?;
                unquantized
                    .zip(quantized)
                    .zip(priors)
                    .zip(reference)
                    .try_for_each(|(((src, dst), prior), reference)| {
                        for ((src, dst), reference) in src.into_iter().zip(dst).zip(reference) {
                            let old = F32::new(*src.borrow())?;
                            let new = crate::quant::vbq(old, prior, |x| x * x, bit_penalty);
                            update(src, dst, new.get());
                            update_prior(prior, old, new, reference)?
                        }
                        Ok::<(), PyErr>(())
                    })?;
            }
        }

        Ok(())
    }

    match (update_prior, reference) {
        (Some(false), Some(_)) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "providing a `reference` implies `update_prior=True`",
            ));
        }
        (None | Some(false), None) => {
            // The caller set `update_prior=False` (either implicitly or explicitly).
            inner(
                shape,
                unquantized,
                quantized,
                priors,
                axis,
                posterior_variance,
                coarseness,
                update,
                |_prior, _old, _new, _reference| Ok::<(), Infallible>(()),
                rayon::iter::repeat(core::iter::repeat(())).take(priors.len()),
            )?;
        }
        (Some(true), None) => {
            // The caller set `update_prior=False` without providing a reference
            inner(
                shape,
                unquantized,
                quantized,
                priors,
                axis,
                posterior_variance,
                coarseness,
                update,
                |prior, old, new, _reference| {
                    prior.remove(old, 1)?;
                    prior.insert(new, 1);
                    Ok::<(), PyErr>(())
                },
                rayon::iter::repeat(core::iter::repeat(())).take(priors.len()),
            )?;
        }
        (None | Some(true), Some(mut reference)) => {
            // The caller provided a reference for prior updates.
            if reference.shape() != shape {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "`reference` must have the same shape as `unquantized`.",
                ));
            }
            let mut reference = reference.as_array_mut();

            inner(
                shape,
                unquantized,
                quantized,
                priors,
                axis,
                posterior_variance,
                coarseness,
                update,
                |prior, _old, new, reference| {
                    prior.remove(F32::new(*reference)?, 1)?;
                    prior.insert(new, 1);
                    *reference = new.get();
                    Ok::<(), PyErr>(())
                },
                reference.axis_iter_mut(Axis(axis)).into_par_iter(),
            )?;
        }
    }

    Ok(())
}

/// In-place variant of `vbq`
///
/// This function is equivalent to `vbq`, except that it quantizes in-place. Thus, instead of
/// returning an array of quantized values, the entries of the argument `unquantized` get
/// overwritten with their quantized values. This avoids allocating a new array in memory.
///
/// The use of a trailing underscore in the function name to indicate in-place operation follows the
/// convention used by the pytorch machine-learning framework.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn vbq_(
    py: Python<'_>,
    unquantized: PyReadwriteArrayDyn<'_, f32>,
    prior: Py<EmpiricalDistribution>,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
) -> PyResult<()> {
    match &mut prior.borrow_mut(py).0 {
        EmpiricalDistributionImpl::Single(distribution) => vbq_single_inplace(
            unquantized,
            distribution,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
        ),
        EmpiricalDistributionImpl::Multiple {
            distributions,
            axis,
        } => vbq_multiplexed_inplace(
            unquantized,
            distributions,
            *axis,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
        ),
    }
}

fn vbq_single_inplace(
    mut unquantized: PyReadwriteArrayDyn<'_, f32>,
    prior: &mut crate::quant::EmpiricalDistribution,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
) -> PyResult<()> {
    let mut unquantized = unquantized.as_array_mut();
    let dim: ndarray::prelude::Dim<ndarray::IxDynImpl> = unquantized.raw_dim();

    match (update_prior, reference.is_some()) {
        (Some(false), true) => Err(pyo3::exceptions::PyValueError::new_err(
            "providing a `reference` implies `update_prior=True`",
        )),
        (None | Some(false), false) => {
            // Prior doesn't get mutated, so we can run VBQ in parallel.
            let src_and_dst = par_azip!(&mut unquantized);
            match posterior_variance {
                Array(posterior_variance) => {
                    if posterior_variance.dims() != dim {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(
                            "`posterior_variance` must have the same shape as `unquantized`.",
                        ));
                    }
                    let posterior_variance = posterior_variance.as_array();
                    let two_coarseness = 2.0 * coarseness;
                    let src_dst_penalty = src_and_dst
                        .and(&posterior_variance)
                        .into_par_iter()
                        .map(|(sd, var)| (sd, F32::new(two_coarseness * *var)));
                    vbq_parallel(src_dst_penalty, prior, |sd| **sd, |sd, value| *sd = value)
                }
                Scalar(posterior_variance) => {
                    let bit_penalty = Result::<_, Infallible>::Ok(F32::new(
                        2.0 * coarseness * posterior_variance,
                    )?);
                    let src_dst_penalty = src_and_dst.into_par_iter().map(|sd| (sd, bit_penalty));
                    vbq_parallel(
                        src_dst_penalty,
                        prior,
                        |(sd,)| **sd,
                        |(sd,), value| *sd = value,
                    )
                }
            }
        }
        _ => {
            // We mutate the prior after each update, and therefore the result depends on the
            // order in which we iterate over items. Fall back to a sequential implementation
            // since running VBQ in parallel would make it nondeterministic here.
            vbq_sequential(
                unquantized.iter_mut().map(|src| (src, ())),
                dim,
                prior,
                posterior_variance,
                coarseness,
                reference,
                |src, _dst, value| *src = value,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn vbq_sequential<Src, Dst>(
    src_and_dst: impl ExactSizeIterator<Item = (Src, Dst)>,
    dim: IxDyn,
    prior: &mut crate::quant::EmpiricalDistribution,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    update: impl FnMut(Src, Dst, f32),
) -> PyResult<()>
where
    Src: Borrow<f32>,
{
    return match posterior_variance {
        Array(posterior_variance) => {
            if posterior_variance.dims() != dim {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "`posterior_variance` must have the same shape as `unquantized`.",
                ));
            }
            let two_coarseness = 2.0 * coarseness;
            let bit_penalty = posterior_variance.as_array();
            let bit_penalty = bit_penalty.iter().map(|&x| F32::new(two_coarseness * x));
            internal(src_and_dst.zip(bit_penalty), dim, prior, reference, update)
        }
        Scalar(posterior_variance) => {
            let bit_penalty =
                Result::<_, Infallible>::Ok(F32::new(2.0 * coarseness * posterior_variance)?);
            internal(
                src_and_dst.map(|(src, dst)| ((src, dst), bit_penalty)),
                dim,
                prior,
                reference,
                update,
            )
        }
    };

    fn internal<Src, Dst, E>(
        src_dst_penalty: impl ExactSizeIterator<Item = ((Src, Dst), Result<F32, E>)>,
        dim: IxDyn,
        prior: &mut crate::quant::EmpiricalDistribution,
        reference: Option<PyReadwriteArrayDyn<'_, f32>>,
        mut update: impl FnMut(Src, Dst, f32),
    ) -> PyResult<()>
    where
        Src: Borrow<f32>,
        PyErr: From<E>,
    {
        if let Some(mut reference) = reference {
            if reference.dims() != dim {
                return Err(pyo3::exceptions::PyAssertionError::new_err(
                    "`reference` must have the same shape as `unquantized`.",
                ));
            }

            let mut reference = reference.as_array_mut();

            for (((src, dst), bit_penalty), reference) in src_dst_penalty.zip(&mut reference) {
                let unquantized = F32::new(*src.borrow())?;
                let reference_val = F32::new(*reference)?;
                let quantized = crate::quant::vbq(unquantized, prior, |x| x * x, bit_penalty?);
                prior.remove(reference_val, 1)?;
                prior.insert(quantized, 1);
                update(src, dst, quantized.get());
                *reference = quantized.get();
            }
        } else {
            for ((src, dst), bit_penalty) in src_dst_penalty {
                let unquantized = F32::new(*src.borrow())?;
                let quantized = crate::quant::vbq(unquantized, prior, |x| x * x, bit_penalty?);
                update(src, dst, quantized.get());
                prior.remove(unquantized, 1)?;
                prior.insert(quantized, 1);
            }
        }

        Ok(())
    }
}

fn vbq_parallel<SrcAndDst, E>(
    src_dst_penalty: impl ParallelIterator<Item = (SrcAndDst, Result<F32, E>)>,
    prior: &mut crate::quant::EmpiricalDistribution,
    extract: impl Fn(&SrcAndDst) -> f32 + Sync + Send,
    update: impl Fn(SrcAndDst, f32) + Sync + Send,
) -> PyResult<()>
where
    SrcAndDst: Send + Sync,
    PyErr: From<E>,
{
    src_dst_penalty.try_for_each(move |(sd, bit_penalty)| {
        let unquantized = F32::new(extract(&sd))?;
        let quantized = crate::quant::vbq(unquantized, prior, |x| x * x, bit_penalty?);
        update(sd, quantized.get());
        PyResult::Ok(())
    })
}

impl From<crate::quant::NotFoundError> for PyErr {
    fn from(_err: crate::quant::NotFoundError) -> Self {
        pyo3::exceptions::PyKeyError::new_err(
            "Attempted to remove a value from an `EmpiricalDistribution` that does not exist in \
            the distribution.",
        )
    }
}
