use crate::{C, DPMultiplier, Error, MatRef, Result, validation::validate_unitary};

/// Analytic U(2) QMIEP: equal diagonal magnitudes permit zero trace with P = I.
pub fn solve_one_qubit_spectrum(u: MatRef<'_, C>, atol: f64) -> Result<DPMultiplier> {
    validate_unitary(u, atol)?;
    if u.nrows() != 2 {
        return Err(Error(
            "the one-qubit QMIEP solver requires a 2 by 2 unitary".into(),
        ));
    }
    Ok(DPMultiplier {
        rows: vec![0, 1],
        diagonal: vec![
            C::from_polar(1.0, -u[(0, 0)].arg()),
            -C::from_polar(1.0, -u[(1, 1)].arg()),
        ],
    })
}
