//! Closed-form QMIEP solvers for one and two qubits.

mod one_qubit;
mod two_qubit;

pub use one_qubit::solve_one_qubit_spectrum;
pub use two_qubit::solve_two_qubit_spectrum;
