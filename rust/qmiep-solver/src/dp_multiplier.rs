use crate::C;
use serde::Serialize;

/// Factors of the left multiplier `D P` in `U_PD = D P U`.
/// The solver seeks factors for which the eigenvalues of `U_PD` occur in
/// pairs `(λ, -λ)`, counting multiplicity.
#[derive(Clone, Debug, Serialize)]
pub struct DPMultiplier {
    /// Row `i` of `P U` is row `rows[i]` of `U`.
    pub rows: Vec<usize>,
    /// Unit-modulus diagonal entries of D: `U_PD[i, j] = diagonal[i] * U[rows[i], j]`.
    pub diagonal: Vec<C>,
}
