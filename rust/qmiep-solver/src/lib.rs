//! QMIEP spectrum solvers, independent of circuit synthesis.
//!
//! The solvers seek a left multiplier `D P` such that `U_PD = D P U` has
//! antipodal spectrum.
//! The analytic solvers cover one and two qubits; the heuristic solver searches
//! dimensions `2^n` for `n >= 3` and verifies each numerical witness spectrally.

pub mod analytic;
mod dp_multiplier;
pub mod heuristic;
mod validation;

pub use analytic::{solve_one_qubit_spectrum, solve_two_qubit_spectrum};
pub use dp_multiplier::DPMultiplier;
pub use faer::{Mat, MatRef};
pub use heuristic::{QMIEPResult, QMIEPSolver, permutation_candidates};
pub use num_complex::Complex64;

pub(crate) type C = Complex64;
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}
