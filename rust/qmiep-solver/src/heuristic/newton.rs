use crate::{Error, Mat, Result};
use faer::{Side, linalg::solvers::Solve};

// Fixed regularization and Armijo backtracking parameters.
const DAMPING: f64 = 1e-8;
const INITIAL_STEP: f64 = 1.0;
const STEP_SHRINK: f64 = 0.5;
const ARMIJO: f64 = 1e-4;
const MAX_LINE_SEARCH_STEPS: usize = 12;

fn squared_norm(x: &[f64]) -> f64 {
    x.iter().map(|x| x * x).sum()
}

fn evaluate(
    x: &[f64],
    residual: &mut impl FnMut(&[f64]) -> Result<(Vec<f64>, Mat<f64>)>,
    nfev: &mut usize,
) -> Result<(Vec<f64>, Mat<f64>)> {
    let (r, j) = residual(x)?;
    *nfev += 1;
    if r.iter().any(|v| !v.is_finite()) || !j.norm_max().is_finite() {
        return Err(Error("nonfinite Newton residual or Jacobian".into()));
    }
    Ok((r, j))
}

/// Solve for p = -J^T (J J^T + lambda I)^-1 r using Cholesky.
fn damped_direction(r: &[f64], j: &Mat<f64>) -> Option<Mat<f64>> {
    let mut normal = j * j.transpose();
    for i in 0..r.len() {
        normal[(i, i)] += DAMPING;
    }
    let rhs = Mat::from_fn(r.len(), 1, |i, _| -r[i]);
    Some(j.transpose() * normal.llt(Side::Lower).ok()?.solve(&rhs))
}

/// Solve (J J^T + lambda I) y = -r, then take p = J^T y.
/// This permits more variables than independent equations and avoids an
/// artificial full-rank Hessian. Acceptance is by residual decrease; callers
/// must independently verify the actual spectral target.
pub(crate) fn minimize(
    mut x: Vec<f64>,
    maxiter: usize,
    tolerance: f64,
    mut residual: impl FnMut(&[f64]) -> Result<(Vec<f64>, Mat<f64>)>,
) -> Result<(Vec<f64>, usize)> {
    let mut nfev = 0;
    let (mut r, mut j) = evaluate(&x, &mut residual, &mut nfev)?;
    for _ in 0..maxiter {
        if r.iter().map(|r| r.abs()).fold(0.0, f64::max) <= tolerance {
            break;
        }
        let Some(direction) = damped_direction(&r, &j) else {
            break;
        };
        let jp = &j * &direction;
        let slope: f64 = r.iter().enumerate().map(|(i, r)| r * jp[(i, 0)]).sum();
        if !slope.is_finite() || slope >= 0.0 {
            break;
        }
        let value = 0.5 * squared_norm(&r);
        let mut step = INITIAL_STEP;
        let mut accepted = None;
        for _ in 0..MAX_LINE_SEARCH_STEPS {
            let next: Vec<_> = x
                .iter()
                .enumerate()
                .map(|(i, x)| x + step * direction[(i, 0)])
                .collect();
            if next == x {
                break;
            }
            let (next_r, next_j) = evaluate(&next, &mut residual, &mut nfev)?;
            if 0.5 * squared_norm(&next_r) <= value + ARMIJO * step * slope {
                accepted = Some((next, next_r, next_j));
                break;
            }
            step *= STEP_SHRINK;
        }
        let Some((next, next_r, next_j)) = accepted else {
            break;
        };
        x = next;
        r = next_r;
        j = next_j;
    }
    Ok((x, nfev))
}
