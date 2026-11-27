//! General QMIEP search with lazy permutation enumeration, restarts and spectral checks.

mod generic;
mod newton;

pub use generic::{QMIEPResult, QMIEPSolver, permutation_candidates};
