// Adapted in part from Qiskit 2.5.1.
// (C) Copyright IBM 2025, 2026
// Apache License, Version 2.0; see ../LICENSE-QISKIT and ../NOTICE.
// Modified for native matrices and optional QMIEP eigenvalue pairing.
//
// Unitarity check (validate)
// https://github.com/Qiskit/qiskit/blob/5d6ad05757a76c2a14074b27fa1c0b9d737e215f/crates/synthesis/src/linalg/mod.rs#L354-L361
// Eigenbasis projection (eigenbasis and polar)
// https://github.com/Qiskit/qiskit/blob/5d6ad05757a76c2a14074b27fa1c0b9d737e215f/crates/synthesis/src/linalg/mod.rs#L300-L317
// Block-ZXZ decomposition (left_polar and block_zxz)
// https://github.com/Qiskit/qiskit/blob/5d6ad05757a76c2a14074b27fa1c0b9d737e215f/crates/synthesis/src/qsd.rs#L334-L357
// Demultiplexing factors (demultiplex)
// https://github.com/Qiskit/qiskit/blob/5d6ad05757a76c2a14074b27fa1c0b9d737e215f/crates/synthesis/src/qsd.rs#L430-L435
use crate::{C, Error, Mat, MatRef, Result};
use std::f64::consts::TAU;

pub(crate) fn validate(u: MatRef<'_, C>, atol: f64, check_unitary: bool) -> Result<()> {
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
    if check_unitary && (u.adjoint() * u - Mat::<C>::identity(n, n)).norm_max() > atol {
        return Err(Error("matrix must be unitary within atol".into()));
    }
    Ok(())
}

pub(crate) fn polar(u: MatRef<'_, C>) -> Result<Mat<C>> {
    let svd = u.svd().map_err(|e| Error(format!("SVD: {e:?}")))?;
    Ok(svd.U() * svd.V().adjoint())
}

fn left_polar(u: MatRef<'_, C>) -> Result<(Mat<C>, Mat<C>)> {
    let svd = u.svd().map_err(|e| Error(format!("SVD: {e:?}")))?;
    let mut singular = Mat::<C>::zeros(u.nrows(), u.ncols());
    singular.diagonal_mut().copy_from(svd.S());
    Ok((
        svd.U() * singular * svd.U().adjoint(),
        svd.U() * svd.V().adjoint(),
    ))
}

pub(crate) struct BlockZxz {
    pub a1: Mat<C>,
    pub a2: Mat<C>,
    pub b: Mat<C>,
    pub c: Mat<C>,
}

/// U = (A1 ⊕ A2) H (I ⊕ B) H (I ⊕ C), with normalized H.
pub(crate) fn block_zxz(u: MatRef<'_, C>) -> Result<BlockZxz> {
    let m = u.nrows() / 2;
    let x = u.submatrix(0, 0, m, m);
    let y = u.submatrix(0, m, m, m);
    let (sx, ux) = left_polar(x)?;
    let (sy, uy) = left_polar(y)?;
    let c = ((uy.adjoint() * &ux) * faer::Scale(C::i()))
        .adjoint()
        .to_owned();
    let a1 = (sx + sy * faer::Scale(C::i())) * &ux;
    let a2 = u.submatrix(m, 0, m, m)
        + (u.submatrix(m, m, m, m) * (uy.adjoint() * ux)) * faer::Scale(C::i());
    let b = (a1.adjoint() * x) * faer::Scale(C::new(2.0, 0.0)) - Mat::<C>::identity(m, m);
    Ok(BlockZxz { a1, a2, b, c })
}

pub(crate) struct Demultiplex {
    pub v: Mat<C>,
    pub diagonal: Vec<C>,
    pub w: Mat<C>,
}

/// For already polar-projected inputs, U0 = V D W and U1 = V D† W.
/// This uses Qiskit's U0 U1† convention and dense diagonal multiplication.
pub(crate) fn demultiplex(
    u0: MatRef<'_, C>,
    u1: MatRef<'_, C>,
    paired: bool,
    atol: f64,
) -> Result<Demultiplex> {
    let (values, v) = eigenbasis((u0 * u1.adjoint()).as_ref(), paired, atol)?;
    let diagonal: Vec<_> = values.iter().map(|z| z.sqrt()).collect();
    let mut d = Mat::<C>::zeros(diagonal.len(), diagonal.len());
    for (i, &value) in diagonal.iter().enumerate() {
        d[(i, i)] = value;
    }
    let w = d * v.adjoint() * u1;
    Ok(Demultiplex { v, diagonal, w })
}

/// Eigenvectors are projected to a unitary basis, including degenerate spectra.
pub(crate) fn eigenbasis(u: MatRef<'_, C>, paired: bool, atol: f64) -> Result<(Vec<C>, Mat<C>)> {
    let eigen = u
        .eigen()
        .map_err(|e| Error(format!("eigendecomposition: {e:?}")))?;
    let q = polar(eigen.U())?;
    let values: Vec<C> = eigen.S().column_vector().iter().copied().collect();
    if !paired {
        return Ok((values, q));
    }
    let n = values.len();
    let angles: Vec<_> = values.iter().map(|z| z.arg().rem_euclid(TAU)).collect();
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| angles[a].total_cmp(&angles[b]));
    let gap = (0..n)
        .max_by(|&a, &b| {
            let delta = |j: usize| {
                if j + 1 == n {
                    angles[order[0]] + TAU - angles[order[j]]
                } else {
                    angles[order[j + 1]] - angles[order[j]]
                }
            };
            delta(a).total_cmp(&delta(b))
        })
        .unwrap();
    order.rotate_left((gap + 1) % n);
    let mut paired_order = Vec::with_capacity(n);
    let mut paired_values = Vec::with_capacity(n);
    for i in 0..n / 2 {
        let (a, b) = (order[i], order[i + n / 2]);
        let mismatch = (values[a] + values[b]).norm();
        if mismatch > atol {
            return Err(Error(format!(
                "QMIEP spectrum in dimension {n} is not antipodal within atol: {mismatch:.3e} > {atol:.3e}"
            )));
        }
        paired_order.extend([a, b]);
        paired_values.extend([values[a], -values[a]]);
    }
    Ok((
        paired_values,
        Mat::from_fn(n, n, |i, j| q[(i, paired_order[j])]),
    ))
}
