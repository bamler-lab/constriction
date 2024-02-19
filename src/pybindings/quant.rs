#![allow(clippy::too_many_arguments)]

use alloc::format;
use alloc::vec::Vec;
use core::{borrow::Borrow, convert::Infallible};
use pyo3::types::PyList;

use ndarray::parallel::prelude::*;
use ndarray::{ArrayBase, Axis, Dimension, IxDyn};
use numpy::{PyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::*;

use crate::quant::{
    FromPoints, NotFoundError, QuantizationMethod, RateDistortionQuantization,
    UnnormalizedDistribution, Vbq,
};
use crate::F32;

pub fn init_module(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<EmpiricalDistribution>()?;
    module.add_class::<RatedGrid>()?;
    module.add_function(wrap_pyfunction!(vbq, module)?)?;
    module.add_function(wrap_pyfunction!(vbq_, module)?)?;
    module.add_function(wrap_pyfunction!(rate_distortion_quantization, module)?)?;
    module.add_function(wrap_pyfunction!(rate_distortion_quantization_, module)?)?;
    Ok(())
}

#[derive(Debug, Clone, FromPyObject)]
pub enum PyReadonlyF32ArrayOrScalar<'py> {
    Array(PyReadonlyArrayDyn<'py, f32>),
    Scalar(f32),
}
use PyReadonlyF32ArrayOrScalar::*;

/// Dynamic data structure for frequencies of values within one or more numpy arrays.
///
/// An `EmpiricalDistribution` counts how many times each value appears within one or more numpy
/// arrays. `EmpiricalDistribution` is a dynamic data structure, i.e., it provides methods to
/// efficiently [`remove`](#constriction.quant.EmpiricalDistribution.remove) and
/// [`update`](#constriction.quant.EmpiricalDistribution.update) existing points, and to
/// [`insert`](#constriction.quant.EmpiricalDistribution.insert) new points.
///
/// The constructor initializes the distribution from the provided numpy array `points`. If
/// `specialize_along_axis` is set to an integer, then the `EmpiricalDistribution` represents each
/// slice of `points` along the specified axis by an individual distribution, see Example 2 below.
///
/// The optional argument `counts` can be used to provide a multiplicity for each inserted point.
/// If `counts` is provided, then `points` and `counts` must be in exactly the format that is
/// returned by the method
/// [`points_and_counts`](#constriction.quant.EmpiricalDistribution.points_and_counts) (except that
/// `points` don't need to be sorted). Thus, `counts` must have `dtype=np.uint32`, and `points` and
/// `counts` must either be equally sized rank-1 numpy arrays (if `specialize_along_axis` is not
/// set) or equal length lists of rank-1 numpy arrays where corresponding entries match (if
/// `specialize_along_axis` is set). See example in the documentation of
/// [`points_and_counts`](#constriction.quant.EmpiricalDistribution.points_and_counts).
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
/// distribution = constriction.quant.EmpiricalDistribution(matrix, specialize_along_axis=0)
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
pub struct EmpiricalDistribution(MaybeMultiplexed<crate::quant::EmpiricalDistribution>);

/// A finite grid of (not necessarily one-dimensional) points, each weighted with a "bit rate".
///
/// The "rate" for each grid point `x` is calculated from the information content `-log_2(f(x))`
/// of the empirical frequency `f(x)` of point `x` in either some pre-quantized numpy array of
/// points or in an `EmpiricalDistribution` (see
/// [`EmpiricalDistribution.rated_grid`](#constriction.quant.EmpiricalDistribution.rated_grid)).
///
/// The constructor initializes the grid from the provided numpy array `points`. If
/// `specialize_along_axis` is set to an integer, then the constructor creates a individual grid
/// with individual weights for each slice of `points` along the specified axis by an individual distribution, see Example 2 below.
///
/// The optional argument `counts` can be used to provide a multiplicity for each inserted point.
/// If `counts` is provided, then `points` and `counts` must be in exactly the format that is
/// returned by the method
/// [`EmpiricalDistribution.points_and_counts`](#constriction.quant.EmpiricalDistribution.points_and_counts)
/// (except that `points` don't need to be sorted). Thus, `counts` must have `dtype=np.uint32`, and
/// `points` and `counts` must either be equally sized rank-1 numpy arrays (if
/// `specialize_along_axis` is not set) or equal length lists of rank-1 numpy arrays where
/// corresponding entries match (if `specialize_along_axis` is set).
///
/// ## Example 1: Global Grid
///
/// Without setting `specialize_along_axis`, we obtain a weighted grid that contains all values that
/// occur in the provided array:
///
/// ```python
/// rng = np.random.default_rng(123)
/// matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
/// print(f"matrix = {matrix}\n")
///
/// grid = constriction.quant.RatedGrid(matrix)
/// points, rates = grid.points_and_rates()
/// print(f"Grid points and their respective rates, sorted by increasing rate:")
/// print("\n".join(f"({point}, {rate})" for point, rate in zip(points, rates)))
/// print(f"Entropy per item: {grid.entropy_base2()} bit")
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
/// Grid points and their respective rates, sorted by increasing rate:
/// (2.0, 1.5145732164382935)
/// (4.0, 1.736965537071228)
/// (3.0, 2.7369654178619385)
/// (5.0, 2.7369654178619385)
/// (1.0, 4.321928024291992)
/// Entropy per item: 2.088376522064209 bit
/// ```
///
/// ## Example 2: Specialization Along an Axis
///
/// The example below uses the same matrix as Example 1 above, but constructs the
/// `EmpiricalDistribution` with argument `specialize_along_axis=0`. This creates a separate
/// distribution for each row (axis zero) of the matrix.
///
/// ```python
/// grid = constriction.quant.RatedGrid(matrix, specialize_along_axis=0)
/// points, rates = grid.points_and_rates()
/// entropies = grid.entropy_base2()
/// print(f"Grid points and their respective rates for each row, each sorted by increasing rate:")
/// for i, (entropy, points, rates) in enumerate(zip(entropies, points, rates)):
///     summary = ", ".join(f"({p}, {r:.3f})" for (p, r) in zip(points, rates))
///     print(f"Row {i} (entropy {entropy:.3f}): {summary}.")
/// ```
///
/// This prints:
///
/// ```text
/// Grid points and their respective rates for each row, each sorted by increasing rate:
/// Row 0 (entropy 1.371): (2.0, 0.737), (1.0, 2.322), (4.0, 2.322).
/// Row 1 (entropy 1.522): (4.0, 1.322), (5.0, 1.322), (2.0, 2.322).
/// Row 2 (entropy 1.522): (2.0, 1.322), (4.0, 1.322), (3.0, 2.322).
/// Row 3 (entropy 1.922): (3.0, 1.322), (2.0, 2.322), (4.0, 2.322), (5.0, 2.322).
/// ```
///
/// Note that the mean entropy per entry is lower (or equal) if we model each row individually. This
/// is true in general.
#[pyclass]
#[derive(Debug)]
pub struct RatedGrid(MaybeMultiplexed<crate::quant::RatedGrid>);

#[derive(Debug)]
enum MaybeMultiplexed<T> {
    Single(T),
    Multiple { distributions: Vec<T>, axis: usize },
}

impl<T> MaybeMultiplexed<T> {
    pub fn new(
        points: &PyAny,
        counts: Option<&PyAny>,
        specialize_along_axis: Option<usize>,
    ) -> PyResult<Self>
    where
        T: FromPoints<F32, u32> + Send,
    {
        let result = if let Some(counts) = counts {
            if let Some(axis) = specialize_along_axis {
                let points = points.extract::<&PyList>()?;
                let counts = counts.extract::<&PyList>()?;
                if points.len() != counts.len() {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "Lists `points` and `counts` must have the same length.",
                    ));
                }

                // This currently can't easily be parallelized due to a limitation of `pyo3`.
                let distributions = points
                    .iter()
                    .zip(counts)
                    .map(|(points, counts)| {
                        let points = points.extract::<PyReadonlyArray1<'_, f32>>()?;
                        let points = points.as_array();
                        let counts = counts.extract::<PyReadonlyArray1<'_, u32>>()?;
                        let counts = counts.as_array();
                        if points.len() != counts.len() {
                            return Err(pyo3::exceptions::PyAssertionError::new_err(
                                "The lengths of corresponding entries of `points` and `counts` must match.",
                            ));
                        }
                        Ok(T::try_from_points_and_counts(
                            points.iter().copied().zip(counts.iter().copied()),
                        )?)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Self::Multiple {
                    distributions,
                    axis,
                }
            } else {
                let points = points.extract::<PyReadonlyArray1<'_, f32>>()?;
                let points = points.as_array();
                let counts = counts.extract::<PyReadonlyArray1<'_, u32>>()?;
                let counts = counts.as_array();
                if points.len() != counts.len() {
                    return Err(pyo3::exceptions::PyAssertionError::new_err(
                        "`points` and `counts` must have equal length.",
                    ));
                }
                let distribution = T::try_from_points_and_counts(
                    points.iter().copied().zip(counts.iter().copied()),
                )?;
                Self::Single(distribution)
            }
        } else {
            let points = points.extract::<PyReadonlyArrayDyn<'_, f32>>()?;
            let points = points.as_array();
            if let Some(axis) = specialize_along_axis {
                let distributions = points
                    .axis_iter(Axis(axis))
                    .into_par_iter()
                    .map(|points| T::try_from_points(points.iter().copied()))
                    .collect::<Result<Vec<_>, _>>()?;
                Self::Multiple {
                    distributions,
                    axis,
                }
            } else {
                let distribution = T::try_from_points(points.iter().copied())?;
                Self::Single(distribution)
            }
        };

        Ok(result)
    }

    fn extract_data<'a, A, I>(
        &'a self,
        py: Python<'_>,
        index: Option<usize>,
        into_iter: impl Fn(&'a T) -> I + Sync,
    ) -> PyResult<(PyObject, PyObject)>
    where
        T: Send + Sync,
        A: numpy::Element,
        I: Iterator<Item = (F32, A)> + 'a,
    {
        match (index, self) {
            (Some(_), MaybeMultiplexed::Single { .. }) => {
                Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an object that \
                was created with argument `specialize_along_axis`.",
                ))
            }
            (
                Some(index),
                MaybeMultiplexed::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let distribution = distributions.get(index).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("`index` out of bounds")
                })?;
                let (points, counts): (Vec<_>, Vec<_>) = into_iter(distribution)
                    .map(|(value, count)| (value.get(), count))
                    .unzip();
                let points = PyArray::from_vec(py, points).to_object(py);
                let counts = PyArray::from_vec(py, counts).to_object(py);
                Ok((points, counts))
            }
            (None, MaybeMultiplexed::Single(distribution)) => {
                let (points, counts): (Vec<_>, Vec<_>) = into_iter(distribution)
                    .map(|(value, count)| (value.get(), count))
                    .unzip();
                let points = PyArray::from_vec(py, points).to_object(py);
                let counts = PyArray::from_vec(py, counts).to_object(py);
                Ok((points, counts))
            }
            (
                None,
                MaybeMultiplexed::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let vecs = distributions
                    .par_iter()
                    .map(|distribution| {
                        into_iter(distribution)
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
                let points = PyList::new(py, points).to_object(py);
                let counts = PyList::new(py, counts).to_object(py);
                Ok((points, counts))
            }
        }
    }

    fn entropy_base2(
        &self,
        py: Python<'_>,
        index: Option<usize>,
        get_entropy: impl Fn(&T) -> f32,
    ) -> PyResult<PyObject> {
        match (index, self) {
            (Some(_), MaybeMultiplexed::Single { .. }) => {
                Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ))
            }
            (
                Some(index),
                MaybeMultiplexed::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let distribution = distributions.get(index).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("`index` out of bounds")
                })?;
                Ok(get_entropy(distribution).to_object(py))
            }
            (None, MaybeMultiplexed::Single(distribution)) => {
                Ok(get_entropy(distribution).to_object(py))
            }
            (
                None,
                MaybeMultiplexed::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let entropies = distributions.iter().map(get_entropy).collect::<Vec<_>>();
                Ok(PyArray::from_vec(py, entropies).to_object(py))
            }
        }
    }
}

#[pymethods]
impl EmpiricalDistribution {
    #[new]
    pub fn new(
        py: Python<'_>,
        points: &PyAny,
        counts: Option<&PyAny>,
        specialize_along_axis: Option<usize>,
    ) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self(MaybeMultiplexed::new(
                points,
                counts,
                specialize_along_axis,
            )?),
        )
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
            let MaybeMultiplexed::Multiple { distributions, .. } = &mut self.0 else {
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
                    MaybeMultiplexed::Single(distribution) => {
                        distribution.insert(F32::new(new)?, 1);
                    }
                    MaybeMultiplexed::Multiple { .. } => {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(
                            "Scalar updates with an `EmpiricalDistribution` that was created with \
                            argument `specialize_along_axis` require argument `index`.",
                        ));
                    }
                },
                Array(new) => {
                    let new = new.as_array();

                    match &mut self.0 {
                        MaybeMultiplexed::Single(distribution) => {
                            for &new in &new {
                                distribution.insert(F32::new(new)?, 1);
                            }
                        }
                        MaybeMultiplexed::Multiple {
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
            let MaybeMultiplexed::Multiple { distributions, .. } = &mut self.0 else {
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
                    MaybeMultiplexed::Single(distribution) => {
                        distribution.remove(F32::new(old)?, 1)?;
                    }
                    MaybeMultiplexed::Multiple { .. } => {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(
                            "Scalar updates with an `EmpiricalDistribution` that was created with \
                        argument `specialize_along_axis` require argument `index`.",
                        ));
                    }
                },
                Array(old) => {
                    let old = old.as_array();

                    match &mut self.0 {
                        MaybeMultiplexed::Single(distribution) => {
                            for &old in &old {
                                distribution.remove(F32::new(old)?, 1)?;
                            }
                        }
                        MaybeMultiplexed::Multiple {
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
            let MaybeMultiplexed::Multiple { distributions, .. } = &mut self.0 else {
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
                    MaybeMultiplexed::Single(distribution) => {
                        distribution.remove(F32::new(old)?, 1)?;
                        distribution.insert(F32::new(new)?, 1);
                    }
                    MaybeMultiplexed::Multiple { .. } => {
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
                        MaybeMultiplexed::Single(distribution) => {
                            for (&old, &new) in old.iter().zip(&new) {
                                distribution.remove(F32::new(old)?, 1)?;
                                distribution.insert(F32::new(new)?, 1);
                            }
                        }
                        MaybeMultiplexed::Multiple {
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
    /// ## Example 1: shifting a single point
    ///
    /// ```python
    /// rng = np.random.default_rng(123)
    /// matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    /// print(f"matrix = {matrix}\n")
    ///
    /// distribution = constriction.quant.EmpiricalDistribution(matrix)
    /// points, counts = distribution.points_and_counts()
    /// print(f"points and counts before shifting: {points}, {counts}")
    /// print(f"entropy before shifting: {distribution.entropy_base2()}\n")
    ///
    /// distribution.shift(2., 2.5)
    /// points, counts = distribution.points_and_counts()
    /// print(f"points and counts after first shift: {points}, {counts}")
    /// print(f"entropy after first shift: {distribution.entropy_base2()}\n")
    ///
    /// distribution.shift(3., 2.5)
    /// points, counts = distribution.points_and_counts()
    /// print(f"points and counts after second shift: {points}, {counts}")
    /// print(f"entropy after second shift: {distribution.entropy_base2()}")
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
    /// points and counts before shifting: [1. 2. 3. 4. 5.], [1 7 3 6 3]
    /// entropy before shifting: 2.088376522064209
    ///
    /// points and counts after first shift: [1.  2.5 3.  4.  5. ], [1 7 3 6 3]
    /// entropy after first shift: 2.088376522064209
    ///
    /// points and counts after second shift: [1.  2.5 4.  5. ], [ 1 10  6  3]
    /// entropy after second shift: 1.647730827331543
    /// ```
    ///
    /// Notice that the second shift merges two grid points, thus reducing the entropy.
    ///
    /// ## Example 2: shifting multiple points, without `specialize_along_axis`
    ///
    /// We can carry out the same two shifts from Example 1 with a single method call:
    ///
    /// ```python
    /// distribution.shift(
    ///     np.array([2., 3.], dtype=np.float32),
    ///     np.array([2.5, 2.5], dtype=np.float32)
    /// )
    /// ```
    ///
    /// Multiple shifts are carried out independently (i.e., we first remove *all* grid points that
    /// are listed in `old` from the distribution, record their respective counts, and we then
    /// insert the recorded counts at the positions listed in `new`). Thus, `shift` can also be used
    /// to swap entries:
    ///
    /// ```python
    /// # (using same `matrix` as in Example 1 above)
    ///
    /// distribution = constriction.quant.EmpiricalDistribution(matrix)
    /// points, counts = distribution.points_and_counts()
    /// print(f"points before swapping: [{', '.join(str(p) for p in points)}]")
    /// print(f"counts before swapping: [{', '.join(str(c) for c in counts)}]\n")
    ///
    /// # Swap the 7 entries at position `2` with the 6 entries at position `4`:
    /// distribution.shift(
    ///     np.array([2., 4.], dtype=np.float32),
    ///     np.array([4., 2.], dtype=np.float32)
    /// )
    ///
    /// points, counts = distribution.points_and_counts()
    /// print(f"points after swapping: [{', '.join(str(p) for p in points)}]")
    /// print(f"counts after swapping: [{', '.join(str(c) for c in counts)}]")
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// points before swapping: [1.0, 2.0, 3.0, 4.0, 5.0]
    /// counts before swapping: [1, 7, 3, 6, 3]
    ///
    /// points after swapping: [1.0, 2.0, 3.0, 4.0, 5.0]
    /// counts after swapping: [1, 6, 3, 7, 3]
    /// ```
    ///
    /// ## Example 3: shifting multiple points, with `specialize_along_axis`
    ///
    /// The following example still uses the same `matrix` as Examples 1 and 2 above:
    ///
    /// ```python
    /// distribution = constriction.quant.EmpiricalDistribution(matrix, specialize_along_axis=0)
    /// points, counts = distribution.points_and_counts()
    /// print(f"points before shifting: [{', '.join(str(p) for p in points)}]")
    /// print(f"counts before shifting: [{', '.join(str(c) for c in counts)}]\n")
    ///
    /// original_positions = [
    ///     2.,                                     # Move all `2.`s in the first row ...
    ///     np.array([], dtype=np.float32),         # Move nothing in the second row ...
    ///     np.array([3., 2.], dtype=np.float32),   # Swap all `3.`s and `2.`s in the third row ...
    ///     np.array([3., 4.], dtype=np.float32),   # Move all `3.`s and `4.`s in the fourth row ...
    /// ]
    /// target_positions = [
    ///     2.1,                                    # ... to `2.1`.
    ///     np.array([], dtype=np.float32),         # ... to nothing.
    ///     np.array([2., 3.], dtype=np.float32),   # ... with the same positions in reverse order.
    ///     np.array([30., 4.4], dtype=np.float32), # ... to `30.` and `4.4`, respectively.
    /// ]
    ///
    /// distribution.shift(original_positions, target_positions)
    ///
    /// points, counts = distribution.points_and_counts()
    /// print(f"points after shifting: [{', '.join(str(p) for p in points)}]")
    /// print(f"counts after shifting: [{', '.join(str(c) for c in counts)}]")
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// points before shifting: [[1. 2. 4.], [2. 4. 5.], [2. 3. 4.], [2. 3. 4. 5.]]
    /// counts before shifting: [[1 3 1], [1 2 2], [2 1 2], [1 2 1 1]]
    ///
    /// points after shifting: [[1.  2.1 4. ], [2. 4. 5.], [2. 3. 4.], [ 2.   4.4  5.  30. ]]
    /// counts after shifting: [[1 3 1], [1 2 2], [1 2 2], [1 1 1 2]]
    /// ```
    pub fn shift(&mut self, old: &PyAny, new: &PyAny, index: Option<usize>) -> PyResult<()> {
        if let Some(index) = index {
            let MaybeMultiplexed::Multiple { distributions, .. } = &mut self.0 else {
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ));
            };

            let distribution = distributions
                .get_mut(index)
                .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("`index` out of bounds"))?;

            return shift_single(old, new, &mut Vec::new(), distribution, || {
                pyo3::exceptions::PyAssertionError::new_err(
                    "If `index` is set then `old` and `new` must both be scalars or rank-1 \
                        arrays (with the same shape).",
                )
            });
        } else {
            // No `index` argument provided.
            match &mut self.0 {
                MaybeMultiplexed::Single(distribution) => {
                    return shift_single(old, new, &mut Vec::new(), distribution, || {
                        pyo3::exceptions::PyAssertionError::new_err(
                            "`old` and `new` must both be scalars or rank-1 tensors (with same shape).",
                        )
                    });
                }
                MaybeMultiplexed::Multiple { distributions, .. } => {
                    if let Ok(old) = old.extract::<&PyList>() {
                        if let Ok(new) = new.extract::<&PyList>() {
                            if old.len() != distributions.len() || new.len() != distributions.len()
                            {
                                return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                                    "Lists `old` and `new` must both have length {} but they have \
                                    lengths {} and {}, respectively.",
                                    distributions.len(),
                                    old.len(),
                                    new.len(),
                                )));
                            }

                            // This currently can't easily be parallelized due to a limitation of `pyo3`.
                            let mut buf = Vec::new();
                            return old.iter().zip(new).zip(distributions).try_for_each(
                                |((old, new), distribution)| {
                                    shift_single(old, new, &mut buf, distribution, || {
                                        pyo3::exceptions::PyAssertionError::new_err(
                                            "Each element of the lists `old` and `new` must be a \
                                            scalar or a rank-1 array, and dimensions of \
                                            corresponding entries must match.",
                                        )
                                    })
                                },
                            );
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
            buf: &mut Vec<u32>,
            distribution: &mut crate::quant::EmpiricalDistribution,
            mk_err: impl Fn() -> PyErr,
        ) -> Result<(), PyErr> {
            if let Ok(old) = old.extract::<PyReadonlyArrayDyn<'_, f32>>() {
                if let Ok(new) = new.extract::<PyReadonlyArrayDyn<'_, f32>>() {
                    if old.dims() == new.dims() && old.dims().ndim() == 1 {
                        let old = old.as_array();
                        let new = new.as_array();
                        buf.clear();
                        buf.reserve(old.len());

                        for &old in old.iter() {
                            buf.push(distribution.remove_all(F32::new(old)?));
                        }
                        for (&new, &count) in new.iter().zip(&*buf) {
                            distribution.insert(F32::new(new)?, count);
                        }
                        return Ok(());
                    }
                }
            } else if let Ok(old) = old.extract::<f32>() {
                if let Ok(new) = new.extract::<f32>() {
                    let count = distribution.remove_all(F32::new(old)?);
                    distribution.insert(F32::new(new)?, count);
                    return Ok(());
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
            (Some(_), MaybeMultiplexed::Single { .. }) => {
                Err(pyo3::exceptions::PyIndexError::new_err(
                    "The `index` argument can only be used with an `EmpiricalDistribution` that \
                    was created with argument `specialize_along_axis`.",
                ))
            }
            (
                Some(index),
                MaybeMultiplexed::Multiple {
                    distributions,
                    axis: _,
                },
            ) => {
                let distribution = distributions.get(index).ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("`index` out of bounds")
                })?;
                Ok(distribution.total().to_object(py))
            }
            (None, MaybeMultiplexed::Single(distribution)) => {
                Ok(distribution.total().to_object(py))
            }
            (
                None,
                MaybeMultiplexed::Multiple {
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
    /// specialized_distribution = constriction.quant.EmpiricalDistribution(matrix, specialize_along_axis=0)
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
        self.0
            .entropy_base2(py, index, |distribution| distribution.entropy_base2())
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
    /// A possible use case of this method is to serialize an `EmpiricalDistribution`. You can save
    /// the returned `points` and `counts` in a file, read them back later, and then pass them as
    /// arguments `points` and `counts` to the constructor of `EmpiricalDistribution` to reconstruct
    /// the distribution.
    ///
    /// ## Example 1: Without `specialize_along_axis`
    ///
    /// ```python
    /// rng = np.random.default_rng(123)
    /// matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    /// print(f"matrix = {matrix}\n")
    ///
    /// distribution = constriction.quant.EmpiricalDistribution(matrix)
    /// print(f"entropy = {distribution.entropy_base2()}")
    /// points, counts = distribution.points_and_counts()
    /// print(f"points = {points}")
    /// print(f"counts = {counts}")
    ///
    /// # ... save `points` and `counts` to a file and load them back later ...
    ///
    /// reconstructed_distribution = constriction.quant.EmpiricalDistribution(
    ///     points, counts=counts)
    /// print(f"reconstructed_distribution = {reconstructed_distribution.entropy_base2()}")
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
    /// entropy = 2.088376522064209
    /// points = [1. 2. 3. 4. 5.]
    /// counts = [1 7 3 6 3]
    /// reconstructed_distribution = 2.088376522064209
    /// ```
    ///
    /// Note that the reconstructed distribution has the same entropy as the original distribution
    /// because it's the same distribution.
    ///
    /// ## Example 2: With `specialize_along_axis`
    ///
    /// The following example uses the same `matrix` as Example 1 above:
    ///
    /// ```python
    /// distribution = constriction.quant.EmpiricalDistribution(matrix, specialize_along_axis=0)
    /// print(f"entropies = {distribution.entropy_base2()}")
    /// points, counts = distribution.points_and_counts()
    /// print(f"points = [{', '.join(str(p) for p in points)}]")
    /// print(f"counts = [{', '.join(str(c) for c in counts)}]")
    ///
    /// # ... save `points` and `counts` to a file and load them back later ...
    ///
    /// reconstructed_distribution = constriction.quant.EmpiricalDistribution(
    ///     points, counts=counts, specialize_along_axis=0)
    /// print(f"reconstructed_distribution = {reconstructed_distribution.entropy_base2()}")
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// entropies = [1.3709505 1.5219281 1.5219281 1.921928 ]
    /// points = [[1. 2. 4.], [2. 4. 5.], [2. 3. 4.], [2. 3. 4. 5.]]
    /// counts = [[1 3 1], [1 2 2], [2 1 2], [1 2 1 1]]
    /// reconstructed_distribution = [1.3709505 1.5219281 1.5219281 1.921928 ]
    /// ```
    ///
    /// Note that the reconstructed distribution has the same per-row entropies as the original
    /// distribution because it's the same distribution.
    pub fn points_and_counts(
        &self,
        py: Python<'_>,
        index: Option<usize>,
    ) -> PyResult<(PyObject, PyObject)> {
        self.0
            .extract_data(py, index, move |distribution| distribution.iter())
    }

    /// Create a `RatedGrid` based on this `EmpiricalDistribution`.
    ///
    /// Returns a `RatedGrid` whose grid points are all points represented by this distribution, and
    /// whose bit rates are the information contents of the empirical frequencies of the points.
    ///
    /// You'll usually only want to call this method on an `EmpiricalDistribution` over points that
    /// were already quantized by some other method. Calling this method on an
    /// `EmpiricalDistribution` over, `N` *distinct* points (as one would usually obtain for, e.g.,
    /// unquantized weights of a trained neural network) will result in a `RatedGrid` over all `N`
    /// grid points, each with bit rate `log_2(N)`, which is hardly useful.
    pub fn rated_grid(&self) -> RatedGrid {
        let rated_grid = match &self.0 {
            MaybeMultiplexed::Single(distribution) => {
                MaybeMultiplexed::Single(distribution.rated_grid())
            }
            MaybeMultiplexed::Multiple {
                distributions,
                axis,
            } => MaybeMultiplexed::Multiple {
                distributions: distributions
                    .iter()
                    .map(|distribution| distribution.rated_grid())
                    .collect(),
                axis: *axis,
            },
        };
        RatedGrid(rated_grid)
    }
}

#[pymethods]
impl RatedGrid {
    #[new]
    pub fn new(
        py: Python<'_>,
        points: &PyAny,
        counts: Option<&PyAny>,
        specialize_along_axis: Option<usize>,
    ) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self(MaybeMultiplexed::new(
                points,
                counts,
                specialize_along_axis,
            )?),
        )
    }

    /// Returns the Shannon entropy per entry, to base 2.
    ///
    /// If the `RatedGrid` was constructed with argument `specialize_along_axis` set,
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
    /// marginal_grid = constriction.quant.RatedGrid(matrix)
    /// specialized_grid = constriction.quant.RatedGrid(matrix, specialize_along_axis=0)
    ///
    /// print(f"marginal_grid.entropy_base2() = {marginal_grid.entropy_base2()}")
    /// print(f"specialized_grid.entropy_base2() = {specialized_grid.entropy_base2()}")
    /// print(f"specialized_grid.entropy_base2(2) = {specialized_grid.entropy_base2(2)}")
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
    /// marginal_grid.entropy_base2() = 2.088376522064209
    /// specialized_grid.entropy_base2() = [1.3709505 1.5219281 1.5219281 1.921928 ]
    /// specialized_grid.entropy_base2(2) = 1.521928071975708
    /// ```
    pub fn entropy_base2(&self, py: Python<'_>, index: Option<usize>) -> PyResult<PyObject> {
        self.0.entropy_base2(py, index, |grid| grid.entropy_base2())
    }

    /// Returns a tuple `(points, rates)`, which are both 1d numpy arrays of equal length, where
    /// `points` contains a sorted list of all grid points, and `rates` contains their respective
    /// rates.
    ///
    /// If the `RatedGrid` was constructed with argument `specialize_along_axis` set, then
    /// `points_and_rates` returns the points and rates for the slice specified by `index` (if
    /// provided), or a list of tuples `(points, rates)`, with one tuple per slice.
    ///
    /// ## Example 1: Without `specialize_along_axis`
    ///
    /// ```python
    /// rng = np.random.default_rng(123)
    /// matrix = rng.binomial(10, 0.3, size=(4, 5)).astype(np.float32)
    /// print(f"matrix = {matrix}\n")
    ///
    /// grid = constriction.quant.RatedGrid(matrix)
    /// print(f"entropy = {grid.entropy_base2()}")
    /// points, rates = grid.points_and_rates()
    /// print(f"points = {points}")
    /// print(f"rates = {rates}")
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
    /// entropy = 2.088376522064209
    /// points = [2. 4. 3. 5. 1.]
    /// rates = [1.5145732 1.7369655 2.7369654 2.7369654 4.321928 ]
    /// ```
    ///
    /// ## Example 2: With `specialize_along_axis`
    ///
    /// The following example uses the same `matrix` as Example 1 above:
    ///
    /// ```python
    /// grid = constriction.quant.RatedGrid(matrix, specialize_along_axis=0)
    /// print(f"entropies = {grid.entropy_base2()}")
    /// points, rates = grid.points_and_rates()
    /// print(f"rates = [\n    {',\n    '.join(str(c) for c in rates)}\n]")
    /// ```
    ///
    /// This prints:
    ///
    /// ```text
    /// entropies = [1.3709506 1.5219281 1.5219281 1.921928 ]
    /// points = [[2. 1. 4.], [4. 5. 2.], [2. 4. 3.], [3. 2. 4. 5.]]
    /// rates = [
    ///     [0.73696554 2.321928   2.321928  ],
    ///     [1.321928 1.321928 2.321928],
    ///     [1.321928 1.321928 2.321928],
    ///     [1.321928 2.321928 2.321928 2.321928]
    /// ]
    /// ```
    pub fn points_and_rates(
        &self,
        py: Python<'_>,
        index: Option<usize>,
    ) -> PyResult<(PyObject, PyObject)> {
        self.0.extract_data(py, index, move |grid| {
            grid.points_and_rates().iter().copied()
        })
    }
}

/// Quantizes an array of values using [Variational Bayesian Quantization (VBQ)].
///
/// Returns an array of quantized values with the same shape and dtype as the argument
/// `unquantized`. If you want to instead overwrite the original array with its quantized values
/// then use `vbq_` (with a trailing underscore) instead.
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
/// (the Rust API of constriction [already does](
/// https://docs.rs/constriction/latest/constriction/quant/struct.EmpiricalDistribution.html#method.vbq)).
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
/// - `posterior_variance`: typically a numpy array with the same dimensions as `unquantized`
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
#[pyfunction]
fn vbq<'p>(
    py: Python<'p>,
    unquantized: PyReadonlyArrayDyn<'p, f32>,
    prior: Py<EmpiricalDistribution>,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'p>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'p, f32>>,
) -> PyResult<&'p PyArrayDyn<f32>> {
    quantize_out_of_place(
        Vbq,
        py,
        unquantized,
        &mut prior.borrow_mut(py).0,
        posterior_variance,
        coarseness,
        update_prior,
        reference,
        |prior, old, new| {
            prior.remove(old, 1)?;
            prior.insert(new, 1);
            Ok(())
        },
    )
}

/// In-place variant of `vbq`
///
/// This function is equivalent to `vbq`, except that it quantizes in-place. Thus, instead of
/// returning an array of quantized values, the entries of the argument `unquantized` get
/// overwritten with their quantized values. This avoids allocating a new array in memory.
///
/// The use of a trailing underscore in the function name to indicate in-place operation follows the
/// convention used by the pytorch machine-learning framework.
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
    quantize_in_place(
        Vbq,
        unquantized,
        &mut prior.borrow_mut(py).0,
        posterior_variance,
        coarseness,
        update_prior,
        reference,
        |prior, old, new| {
            prior.remove(old, 1)?;
            prior.insert(new, 1);
            Ok(())
        },
    )
}

/// Quantizes an array of values to grid points by optimizing a rate/distortion-tradeoff.
///
/// Returns an array of quantized values with the same shape as the argument `unquantized`. If you
/// want to instead overwrite the original array with its quantized values then use
/// `rate_distortion_quantization_` (with a trailing underscore) instead.
///
/// ## Rate/Distortion Quantization
///
/// For each point, the quantization method minimizes the following objective:
///
/// `loss(quantized) = distortion(quantized - unquantized) + rate_penalty * rate(quantized)`
///
/// where `rate(quantized)` is the rate that the `RatedGrid` associates with the grid point
/// `quantized`, and the minimization runs over all grid points `quantized`. The Python API is
/// currently restricted to a quadratic distortion,
///
/// `distortion(quantized - unquantized) = (quantized - unquantized)**2 / (2 * posterior_variance)`.
///
/// Future versions of `constriction` might provide support for more general distortion metrics
/// (the Rust API of constriction [already does](
/// https://docs.rs/constriction/latest/constriction/quant/struct.RatedGrid.html#method.quantize)).
///
/// ## Arguments
///
/// - `unquantized`: a numpy array of values that you want to quantize.
/// - `grid`: a `RatedGrid` that to which the values should be quantized. The should usually be
///   constructed from some existing (preliminary) quantized version of the points in `unquantized`
///   so that the grid positions grid points at good positions and with good rate estimates for
///   these `unquantized` points.
/// - `posterior_variance`: typically a numpy array with the same dimensions as `unquantized`
///   (although it can also be a scalar to support edge cases). The `posterior_variance` controls
///   how important a faithful quantization of each entry of `unquantized` is, relative to the other
///   entries. See objective function stated above. Entries with higher `posterior_variance` will
///   generally be quantized to grid points that can be further away from their unquantized values
///   than entries with lower `posterior_variance`.
/// - `rate_penalty`: a nonnegative scalar that controls how much distortion is acceptable globally
///   (higher values for `rate_penalty` lead to a lower entropy of the quantized values, while lower
///   values for `rate_penalty` ensure that quantized values stay closer to their unquantized
///   counterparts). The argument `rate_penalty` is a convenience. Setting `rate_penalty` to a value
///   different from `1.0` has the same effect as multiplying all entries of `posterior_variance` by
///   `rate_penalty`.
///
/// ## Example 1: quantization with a *global* grid (i.e. without `specialize_along_axis`)
///
/// ```python
/// rng = np.random.default_rng(123)
/// unquantized = rng.normal(scale=5.0, size=(4, 5)).astype(np.float32)
/// print(f"Unquantized values:\n{unquantized}\n")
///
/// naively_quantized = unquantized.round()
/// print(f"Naively quantized values (rounding to uniform grid):\n{naively_quantized}\n")
///
/// grid = constriction.quant.RatedGrid(naively_quantized)
/// print(f"Entropy of naively quantized values: {grid.entropy_base2()}\n")
///
/// # Allow larger quantization errors in upper left corner of the matrix by using a high variance:
/// posterior_variance = np.array([
///     [10.0, 10.0, 1.0, 1.0, 1.0],
///     [10.0, 10.0, 1.0, 1.0, 1.0],
///     [1.0, 1.0, 1.0, 1.0, 1.0],
///     [1.0, 1.0, 1.0, 1.0, 1.0],
/// ], dtype=np.float32)
///
/// quantized_low_penalty = constriction.quant.rate_distortion_quantization(
///     unquantized, grid, posterior_variance, 0.01)
/// quantized_high_penalty = constriction.quant.rate_distortion_quantization(
///     unquantized, grid, posterior_variance, 1.0)
///
/// print(f"Quantized with low rate_penalty:\n{quantized_low_penalty}\n")
/// print(f"(same as naive quantization since the rate hardly matters here)\n")
///
/// print(f"Quantized with high rate_penalty:\n{quantized_high_penalty}\n")
/// print(f"Differences to naive quantization:\n{quantized_high_penalty - naively_quantized}\n")
/// ```
///
/// This prints:
///
/// ```text
/// Unquantized values:
/// [[-4.9456067  -1.8389332   6.439626    0.9698721   4.6011543 ]
///  [ 2.885519   -3.1823182   2.7097611  -1.5829773  -1.6119456 ]
///  [ 0.4858366  -7.629652    5.9608307  -3.3554485   5.001347  ]
///  [ 0.68160564  7.6601653  -3.2998471  -1.5589743   1.6888456 ]]
///
/// Naively quantized values (rounding to uniform grid):
/// [[-5. -2.  6.  1.  5.]
///  [ 3. -3.  3. -2. -2.]
///  [ 0. -8.  6. -3.  5.]
///  [ 1.  8. -3. -2.  2.]]
///
/// Entropy of naively quantized values: 3.2841837406158447
///
/// Quantized with low rate_penalty:
/// [[-5. -2.  6.  1.  5.]
///  [ 3. -3.  3. -2. -2.]
///  [ 0. -8.  6. -3.  5.]
///  [ 1.  8. -3. -2.  2.]]
///
/// (same as naive quantization since the rate hardly matters here)
///
/// Quantized with high rate_penalty:
/// [[-2. -2.  6.  1.  5.]
///  [ 3. -2.  3. -2. -2.]
///  [ 1. -8.  6. -3.  5.]
///  [ 1.  8. -3. -2.  1.]]
///
/// Differences to naive quantization:
/// [[ 3.  0.  0.  0.  0.]
///  [ 0.  1.  0.  0.  0.]
///  [ 1.  0.  0.  0.  0.]
///  [ 0.  0.  0.  0. -1.]]
/// ```
///
/// Note that, with high `rate_penalty`, we obtain the largest quantization error in the upper left
/// 2x2 block of the matrix. This is because we set the `posterior_variance` high in this block,
/// which tells the rate/distortion quantization to care less about distortion for the corresponding
/// values.
///
/// ## Example 2: quantization with `specialize_along_axis`
///
/// We can quantize to a different grid for each row of the matrix by replacing the following line
/// in Example 1 above:
///
/// ```python
/// grid = constriction.quant.RatedGrid(naively_quantized)
/// ```
///
/// with
///
/// ```python
/// grid = constriction.quant.RatedGrid(naively_quantized, specialize_along_axis=0)
/// ```
///
/// With this change, we obtain the following output:
///
/// ```text
/// Unquantized values:
/// [[-4.9456067  -1.8389332   6.439626    0.9698721   4.6011543 ]
///  [ 2.885519   -3.1823182   2.7097611  -1.5829773  -1.6119456 ]
///  [ 0.4858366  -7.629652    5.9608307  -3.3554485   5.001347  ]
///  [ 0.68160564  7.6601653  -3.2998471  -1.5589743   1.6888456 ]]
///
/// Naively quantized values (rounding to uniform grid1):
/// [[-5. -2.  6.  1.  5.]
///  [ 3. -3.  3. -2. -2.]
///  [ 0. -8.  6. -3.  5.]
///  [ 1.  8. -3. -2.  2.]]
///
/// Entropy of naively quantized values: [2.321928  1.5219281 2.321928  2.321928 ]
///
/// Quantized with low rate_penalty:
/// [[-5. -2.  6.  1.  5.]
///  [ 3. -3.  3. -2. -2.]
///  [ 0. -8.  6. -3.  5.]
///  [ 1.  8. -3. -2.  2.]]
///
/// (same as naive quantization since the rate hardly matters here)
///
/// Quantized with high rate_penalty:
/// [[-5. -2.  6.  1.  5.]
///  [ 3. -2.  3. -2. -2.]
///  [ 0. -8.  6. -3.  5.]
///  [ 1.  8. -3. -2.  2.]]
///
/// Differences to naive quantization:
/// [[0. 0. 0. 0. 0.]
///  [0. 1. 0. 0. 0.]
///  [0. 0. 0. 0. 0.]
///  [0. 0. 0. 0. 0.]]
/// ```
///
/// For this small toy example, we hardly see an effect of the rate term here because the
/// quantization grids for each row are constructed based on so few (five) sample points that they
/// are all almost degenerate (i.e., have the same weight for almost all grid points). Specializing
/// along an axis only makes sense if the original data still contains lots of elements along the
/// remaining axes
#[pyfunction]
fn rate_distortion_quantization<'p>(
    py: Python<'p>,
    unquantized: PyReadonlyArrayDyn<'p, f32>,
    grid: Py<RatedGrid>,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'p>,
    rate_penalty: f32,
) -> PyResult<&'p PyArrayDyn<f32>> {
    quantize_out_of_place(
        RateDistortionQuantization,
        py,
        unquantized,
        &mut grid.borrow_mut(py).0,
        posterior_variance,
        rate_penalty,
        None,
        None,
        |_grid, _old, _new| Ok(()),
    )
}

/// In-place variant of `rate_distortion_quantization`
///
/// This function is equivalent to `rate_distortion_quantization`, except that it quantizes
/// in-place. Thus, instead of returning an array of quantized values, the entries of the argument
/// `unquantized` get overwritten with their quantized values. This avoids allocating a new array in
/// memory.
///
/// The use of a trailing underscore in the function name to indicate in-place operation follows the
/// convention used by the pytorch machine-learning framework.
#[pyfunction]
fn rate_distortion_quantization_(
    py: Python<'_>,
    unquantized: PyReadwriteArrayDyn<'_, f32>,
    grid: Py<RatedGrid>,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    rate_penalty: f32,
) -> PyResult<()> {
    quantize_in_place(
        RateDistortionQuantization,
        unquantized,
        &mut grid.borrow_mut(py).0,
        posterior_variance,
        rate_penalty,
        None,
        None,
        |_grid, _old, _new| Ok(()),
    )
}

fn quantize_out_of_place<'p, Grid: Send + Sync>(
    qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
    py: Python<'p>,
    unquantized: PyReadonlyArrayDyn<'p, f32>,
    grid: &mut MaybeMultiplexed<Grid>,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'p>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'p, f32>>,
    grid_update: impl Fn(&mut Grid, F32, F32) -> Result<(), NotFoundError> + Send + Sync,
) -> PyResult<&'p PyArrayDyn<f32>> {
    match grid {
        MaybeMultiplexed::Single(distribution) => quantize_single(
            qm,
            py,
            unquantized,
            distribution,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
            grid_update,
        ),
        MaybeMultiplexed::Multiple {
            distributions,
            axis,
        } => quantize_multiplexed(
            qm,
            py,
            unquantized,
            distributions,
            *axis,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
            grid_update,
        ),
    }
}

fn quantize_single<'p, Grid: Send + Sync>(
    qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
    py: Python<'p>,
    unquantized: PyReadonlyArrayDyn<'p, f32>,
    grid: &mut Grid,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'p, f32>>,
    grid_update: impl Fn(&mut Grid, F32, F32) -> Result<(), NotFoundError> + Send + Sync,
) -> PyResult<&'p PyArrayDyn<f32>> {
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
                            .map(|(src, dst, var)| ((src, dst), two_coarseness * *var));
                        quantize_parallel(
                            qm,
                            src_dst_penalty,
                            grid,
                            |(src, _dst)| **src,
                            |(_src, dst), value| {
                                dst.write(value);
                            },
                        )
                    }
                    Scalar(posterior_variance) => {
                        let bit_penalty = 2.0 * coarseness * posterior_variance;
                        let src_dst_penalty =
                            src_and_dst.into_par_iter().map(|sd| (sd, bit_penalty));
                        quantize_parallel(
                            qm,
                            src_dst_penalty,
                            grid,
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
                quantize_sequential(
                    qm,
                    unquantized.iter().zip(quantized.iter_mut()),
                    dim,
                    grid,
                    posterior_variance,
                    coarseness,
                    reference,
                    |_src, dst, value| {
                        dst.write(value);
                    },
                    grid_update,
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

fn quantize_multiplexed<'p, Grid: Send + Sync>(
    qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
    py: Python<'p>,
    unquantized: PyReadonlyArrayDyn<'p, f32>,
    grids: &mut [Grid],
    axis: usize,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'p>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'p, f32>>,
    grid_update: impl Fn(&mut Grid, F32, F32) -> Result<(), NotFoundError> + Send + Sync,
) -> PyResult<&'p PyArrayDyn<f32>> {
    let len = unquantized.len();

    let shape = unquantized.shape();
    let unquantized = unquantized.as_array();
    let unquantized = unquantized.axis_iter(Axis(axis));
    if unquantized.len() != grids.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err(alloc::format!(
            "Axis {} has wrong dimension: expected {} but found {}.",
            axis,
            grids.len(),
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

        quantize_multiplexed_generic(
            qm,
            shape,
            unquantized,
            quantized,
            grids,
            axis,
            posterior_variance,
            coarseness,
            |_src, dst, new| {
                dst.write(new);
            },
            update_prior,
            reference,
            grid_update,
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

fn quantize_multiplexed_inplace<Grid: Send + Sync>(
    qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
    mut unquantized: PyReadwriteArrayDyn<'_, f32>,
    grids: &mut [Grid],
    axis: usize,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    grid_update: impl Fn(&mut Grid, F32, F32) -> Result<(), NotFoundError> + Send + Sync,
) -> PyResult<()> {
    let shape = unquantized.shape().to_vec();
    let mut unquantized = unquantized.as_array_mut();
    let unquantized = unquantized.axis_iter_mut(Axis(axis));
    if unquantized.len() != grids.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err(alloc::format!(
            "Axis {} has wrong dimension: expected {} but found {}.",
            axis,
            grids.len(),
            unquantized.len()
        )));
    }

    let unquantized = unquantized.into_par_iter();

    quantize_multiplexed_generic(
        qm,
        &shape,
        unquantized,
        rayon::iter::repeat(core::iter::repeat(())).take(grids.len()),
        grids,
        axis,
        posterior_variance,
        coarseness,
        |src, _dst, new| {
            *src = new;
        },
        update_prior,
        reference,
        grid_update,
    )?;

    Ok(())
}

fn quantize_multiplexed_generic<Grid: Send + Sync, I1, I2, U, Src, Dst>(
    qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
    shape: &[usize],
    unquantized: I1,
    quantized: I2,
    grids: &mut [Grid],
    axis: usize,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    update: U,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    grid_update: impl Fn(&mut Grid, F32, F32) -> Result<(), NotFoundError> + Send + Sync,
) -> PyResult<()>
where
    I1: IndexedParallelIterator,
    <I1 as ParallelIterator>::Item: IntoIterator<Item = Src>,
    I2: IndexedParallelIterator,
    <I2 as ParallelIterator>::Item: IntoIterator<Item = Dst>,
    Src: Borrow<f32>,
    U: Fn(Src, Dst, f32) + Send + Sync,
{
    fn inner<Grid: Send + Sync, I1, I2, I3, R, E, U1, U2, Src, Dst>(
        qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
        shape: &[usize],
        unquantized: I1,
        quantized: I2,
        grids: &mut [Grid],
        axis: usize,
        posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
        coarseness: f32,
        update: U1,
        grid_update: U2,
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
        U2: Fn(&mut Grid, F32, F32, R) -> Result<(), E> + Send + Sync,
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
                    .zip(grids)
                    .zip(reference)
                    .try_for_each(|((((src, dst), var), grid), reference)| {
                        for (((src, dst), &var), reference) in
                            src.into_iter().zip(dst).zip(&var).zip(reference)
                        {
                            let old = F32::new(*src.borrow())?;
                            let new = qm.quantize(
                                old,
                                grid,
                                |x, y| ((x - y) * (x - y)).get(),
                                two_coarseness * var,
                            );
                            update(src, dst, new.get());
                            grid_update(grid, old, new, reference)?;
                        }
                        Ok::<(), PyErr>(())
                    })?;
            }
            Scalar(posterior_variance) => {
                let bit_penalty = 2.0 * coarseness * posterior_variance;
                unquantized
                    .zip(quantized)
                    .zip(grids)
                    .zip(reference)
                    .try_for_each(|(((src, dst), grid), reference)| {
                        for ((src, dst), reference) in src.into_iter().zip(dst).zip(reference) {
                            let old = F32::new(*src.borrow())?;
                            let new = qm.quantize(
                                old,
                                grid,
                                |x, y| ((x - y) * (x - y)).get(),
                                bit_penalty,
                            );
                            update(src, dst, new.get());
                            grid_update(grid, old, new, reference)?
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
                qm,
                shape,
                unquantized,
                quantized,
                grids,
                axis,
                posterior_variance,
                coarseness,
                update,
                |_prior, _old, _new, _reference| Ok::<(), Infallible>(()),
                rayon::iter::repeat(core::iter::repeat(())).take(grids.len()),
            )?;
        }
        (Some(true), None) => {
            // The caller set `update_prior=True` without providing a reference
            inner(
                qm,
                shape,
                unquantized,
                quantized,
                grids,
                axis,
                posterior_variance,
                coarseness,
                update,
                move |grid, old, new, _reference| grid_update(grid, old, new),
                rayon::iter::repeat(core::iter::repeat(())).take(grids.len()),
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
                qm,
                shape,
                unquantized,
                quantized,
                grids,
                axis,
                posterior_variance,
                coarseness,
                update,
                |grid, _old, new, reference| {
                    grid_update(grid, F32::new(*reference)?, new)?;
                    *reference = new.get();
                    Ok::<(), PyErr>(())
                },
                reference.axis_iter_mut(Axis(axis)).into_par_iter(),
            )?;
        }
    }

    Ok(())
}

fn quantize_in_place<Grid: Send + Sync>(
    qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
    unquantized: PyReadwriteArrayDyn<'_, f32>,
    grid: &mut MaybeMultiplexed<Grid>,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    grid_update: impl Fn(&mut Grid, F32, F32) -> Result<(), NotFoundError> + Send + Sync,
) -> PyResult<()> {
    match grid {
        MaybeMultiplexed::Single(distribution) => quantize_single_inplace(
            qm,
            unquantized,
            distribution,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
            grid_update,
        ),
        MaybeMultiplexed::Multiple {
            distributions,
            axis,
        } => quantize_multiplexed_inplace(
            qm,
            unquantized,
            distributions,
            *axis,
            posterior_variance,
            coarseness,
            update_prior,
            reference,
            grid_update,
        ),
    }
}

fn quantize_single_inplace<Grid: Send + Sync>(
    qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
    mut unquantized: PyReadwriteArrayDyn<'_, f32>,
    grid: &mut Grid,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    update_prior: Option<bool>,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    grid_update: impl Fn(&mut Grid, F32, F32) -> Result<(), NotFoundError> + Send + Sync,
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
                        .map(|(sd, var)| (sd, two_coarseness * *var));
                    quantize_parallel(
                        qm,
                        src_dst_penalty,
                        grid,
                        |sd| **sd,
                        |sd, value| *sd = value,
                    )
                }
                Scalar(posterior_variance) => {
                    let bit_penalty = 2.0 * coarseness * posterior_variance;
                    let src_dst_penalty = src_and_dst.into_par_iter().map(|sd| (sd, bit_penalty));
                    quantize_parallel(
                        qm,
                        src_dst_penalty,
                        grid,
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
            quantize_sequential(
                qm,
                unquantized.iter_mut().map(|src| (src, ())),
                dim,
                grid,
                posterior_variance,
                coarseness,
                reference,
                |src, _dst, value| *src = value,
                grid_update,
            )
        }
    }
}

fn quantize_sequential<Grid: Send + Sync, Src, Dst>(
    qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
    src_and_dst: impl ExactSizeIterator<Item = (Src, Dst)>,
    dim: IxDyn,
    grid: &mut Grid,
    posterior_variance: PyReadonlyF32ArrayOrScalar<'_>,
    coarseness: f32,
    reference: Option<PyReadwriteArrayDyn<'_, f32>>,
    update: impl FnMut(Src, Dst, f32),
    grid_update: impl Fn(&mut Grid, F32, F32) -> Result<(), NotFoundError> + Send + Sync,
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
            let bit_penalty = bit_penalty.iter().map(|&x| two_coarseness * x);
            internal(
                qm,
                src_and_dst.zip(bit_penalty),
                dim,
                grid,
                reference,
                update,
                grid_update,
            )
        }
        Scalar(posterior_variance) => {
            let bit_penalty = 2.0 * coarseness * posterior_variance;
            internal(
                qm,
                src_and_dst.map(|(src, dst)| ((src, dst), bit_penalty)),
                dim,
                grid,
                reference,
                update,
                grid_update,
            )
        }
    };

    fn internal<Grid: Send + Sync, Src, Dst>(
        qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
        src_dst_penalty: impl ExactSizeIterator<Item = ((Src, Dst), f32)>,
        dim: IxDyn,
        grid: &mut Grid,
        reference: Option<PyReadwriteArrayDyn<'_, f32>>,
        mut update: impl FnMut(Src, Dst, f32),
        grid_update: impl Fn(&mut Grid, F32, F32) -> Result<(), NotFoundError> + Send + Sync,
    ) -> PyResult<()>
    where
        Src: Borrow<f32>,
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
                let quantized = qm.quantize(
                    unquantized,
                    grid,
                    |x, y| ((x - y) * (x - y)).get(),
                    bit_penalty,
                );
                grid_update(grid, reference_val, quantized)?;
                update(src, dst, quantized.get());
                *reference = quantized.get();
            }
        } else {
            for ((src, dst), bit_penalty) in src_dst_penalty {
                let unquantized = F32::new(*src.borrow())?;
                let quantized = qm.quantize(
                    unquantized,
                    grid,
                    |x, y| ((x - y) * (x - y)).get(),
                    bit_penalty,
                );
                grid_update(grid, unquantized, quantized)?;
                update(src, dst, quantized.get());
            }
        }

        Ok(())
    }
}

fn quantize_parallel<SrcAndDst, Grid: Send + Sync>(
    qm: impl QuantizationMethod<Grid, F32, f32> + Send + Sync,
    src_dst_penalty: impl ParallelIterator<Item = (SrcAndDst, f32)>,
    grid: &mut Grid,
    extract: impl Fn(&SrcAndDst) -> f32 + Sync + Send,
    update: impl Fn(SrcAndDst, f32) + Sync + Send,
) -> PyResult<()>
where
    SrcAndDst: Send + Sync,
{
    src_dst_penalty.try_for_each(move |(sd, bit_penalty)| {
        let unquantized = F32::new(extract(&sd))?;
        let quantized = qm.quantize(
            unquantized,
            grid,
            |x, y| ((x - y) * (x - y)).get(),
            bit_penalty,
        );
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
