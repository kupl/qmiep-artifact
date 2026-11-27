use crate::{C, Error, Mat, MatRef, Result};

pub(crate) fn validate_unitary(u: MatRef<'_, C>, atol: f64) -> Result<()> {
    if !atol.is_finite() || atol <= 0.0 {
        return Err(Error("atol must be finite and positive".into()));
    }
    let n = u.nrows();
    if n == 0 || n != u.ncols() {
        return Err(Error("matrix must be nonempty and square".into()));
    }
    if (0..n).any(|i| (0..n).any(|j| !u[(i, j)].is_finite())) {
        return Err(Error("matrix entries must be finite".into()));
    }
    if (u.adjoint() * u - Mat::<C>::identity(n, n)).norm_max() > atol {
        return Err(Error("matrix must be unitary within atol".into()));
    }
    Ok(())
}
