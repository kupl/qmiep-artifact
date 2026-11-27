//! NumPy boundaries for native spectrum solving and circuit synthesis.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use qmiep_synthesis::{Complex64 as C, Instruction, Mat, Options, QMIEPSolver};

type NativeSynthesis<'py> = (
    Bound<'py, PyList>,
    f64,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<C>>,
);

type NativeSpectrum<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<C>>,
    bool,
    f64,
    usize,
    usize,
    usize,
);

/// Return left row selection, diagonal and independently verified diagnostics.
/// Try candidates lazily until success or exhaustion unless a positive cap is supplied.
#[pyfunction]
#[pyo3(signature = (unitary, *, candidates=None, max_permutations=None, restarts=3, maxiter=500, atol=1e-10, seed=None))]
fn solve_spectrum<'py>(
    py: Python<'py>,
    unitary: PyReadonlyArray2<'py, C>,
    candidates: Option<Py<PyAny>>,
    max_permutations: Option<usize>,
    restarts: usize,
    maxiter: usize,
    atol: f64,
    seed: Option<u64>,
) -> PyResult<NativeSpectrum<'py>> {
    let array = unitary.as_array();
    let matrix = Mat::from_fn(array.nrows(), array.ncols(), |i, j| array[(i, j)]);
    drop(unitary);
    let solver = QMIEPSolver {
        max_permutations,
        restarts,
        maxiter,
        atol,
        seed,
    };
    let iterator = candidates
        .map(|value| value.bind(py).try_iter().map(Bound::unbind))
        .transpose()?;
    let result = py.detach(move || {
        let mut callback_error = None;
        let result = if let Some(iterator) = iterator {
            // Request rows lazily, so a successful solve never consumes another
            // Python candidate. Only the permutation callback needs the GIL.
            let rows = std::iter::from_fn(|| {
                if callback_error.is_some() {
                    return None;
                }
                Python::attach(|py| {
                    match iterator
                        .bind(py)
                        .clone()
                        .next()?
                        .and_then(|row| row.extract())
                    {
                        Ok(rows) => Some(rows),
                        Err(error) => {
                            callback_error = Some(error);
                            None
                        }
                    }
                })
            });
            solver.solve_with_permutation_candidates(matrix.as_ref(), rows)
        } else {
            solver.solve(matrix.as_ref())
        };
        if let Some(error) = callback_error {
            return Err(error);
        }
        result.map_err(|error| PyValueError::new_err(error.to_string()))
    })?;
    py.check_signals()?;
    Ok((
        result.multiplier.rows.into_pyarray(py),
        result.multiplier.diagonal.into_pyarray(py),
        result.success,
        result.antipodal_error,
        result.permutations_tried,
        result.attempts,
        result.nfev,
    ))
}

/// Return (instructions, global_phase, permutation_images, diagonal_entries).
#[pyfunction]
#[pyo3(signature = (unitary, *, algorithm=None, mode=None, kind=None, method=None, seed=None, atol=1e-10, check_input=true))]
fn synthesize<'py>(
    py: Python<'py>,
    unitary: PyReadonlyArray2<'py, C>,
    algorithm: Option<&str>,
    mode: Option<&str>,
    kind: Option<&str>,
    method: Option<&str>,
    // Seed for the QMIEP solver's random starts.
    seed: Option<u64>,
    // Absolute tolerance for input unitarity and numerical checks during synthesis.
    atol: f64,
    // Check U†U ≈ I before synthesis. Shape and finite entries are always checked.
    check_input: bool,
) -> PyResult<NativeSynthesis<'py>> {
    let selection = Options::from_names(algorithm, mode, kind, method)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    // Own the matrix before releasing the GIL; strided/read-only arrays also work.
    let array = unitary.as_array();
    let matrix = Mat::from_fn(array.nrows(), array.ncols(), |i, j| array[(i, j)]);
    drop(unitary);
    let options = Options {
        solver: QMIEPSolver {
            seed,
            ..Default::default()
        },
        atol,
        check_input,
        ..selection
    };
    let dimension = matrix.nrows();
    let result = py
        .detach(move || qmiep_synthesis::synthesize(matrix.as_ref(), options))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    py.check_signals()?;

    let instructions = PyList::empty(py);
    for instruction in result.circuit.instructions {
        match instruction {
            Instruction::U {
                qubit,
                theta,
                phi,
                lambda,
            } => instructions.append(("u", qubit, theta, phi, lambda))?,
            Instruction::Rz { qubit, angle } => instructions.append(("rz", qubit, angle))?,
            Instruction::H { qubit } => instructions.append(("h", qubit))?,
            Instruction::Cx { control, target } => {
                instructions.append(("cx", control, target))?;
            }
        }
    }
    // The Python tuple represents an exact result with identity arrays.
    let (permutation, diagonal) = match result.multiplier {
        Some(multiplier) => (
            multiplier.permutation.images().collect(),
            multiplier.diagonal,
        ),
        None => (
            (0..dimension).collect::<Vec<_>>(),
            vec![C::new(1.0, 0.0); dimension],
        ),
    };
    Ok((
        instructions,
        result.circuit.global_phase,
        permutation.into_pyarray(py),
        diagonal.into_pyarray(py),
    ))
}

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(synthesize, module)?)?;
    module.add_function(wrap_pyfunction!(solve_spectrum, module)?)
}
