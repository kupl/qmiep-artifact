//! Numerical search for antipodal eigenvalue pairs in `U_PD = D P U`.

use super::newton;
use crate::{C, DPMultiplier, Error, Mat, MatRef, Result, validation::validate_unitary};
use nalgebra::{DMatrix, Schur};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Clone, Debug, Serialize)]
pub struct QMIEPResult {
    /// On success, U_PD = D P U passed the antipodal check.
    #[serde(flatten)]
    pub multiplier: DPMultiplier,
    /// Recording failure case, only for debugging usage
    pub success: bool,
    pub antipodal_error: f64,
    pub permutations_tried: usize,
    /// Number of phase optimization starts across all permutations tried.
    /// Each candidate starts at zero phases if needed, then uses random restarts.
    pub attempts: usize,
    /// Number of function evaluations across all starts and Newton stages.
    /// Each joint residual and Jacobian evaluation counts once, including line
    /// search trials. Separate eigenphase verification calls are excluded.
    pub nfev: usize,
}

impl QMIEPResult {
    /// Extract the multiplier if the eigenphase error of `U_PD` met the requested tolerance.
    pub fn verified(self) -> Result<DPMultiplier> {
        if !self.success {
            return Err(Error(format!(
                "No verified antipodal solution found within the search budget \
                 ({} permutations, {} starts; best eigenphase error {:.3e} rad). \
                 Yet, this does not establish nonexistence - tentatively due to \
                 misleading method in numerical heuristic.",
                self.permutations_tried, self.attempts, self.antipodal_error
            )));
        }
        Ok(self.multiplier)
    }
}

/// Bounded damped Newton search with independent spectral verification.
/// The first phase is fixed to zero. `atol` measures eigenphase error in radians.
/// Equal Python/Rust seeds need not return the same witness.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QMIEPSolver {
    /// Optional cap on candidates tried. By default search until success or exhaustion.
    pub max_permutations: Option<usize>,
    pub restarts: usize,
    pub maxiter: usize,
    /// Accept sorted eigenphases θ of DPU (radians) when `max_j |θ[j + N/2] - θ[j] - π| <= atol`.
    pub atol: f64,
    pub seed: Option<u64>,
}

impl Default for QMIEPSolver {
    fn default() -> Self {
        Self {
            max_permutations: None,
            restarts: 3,
            maxiter: 500,
            atol: 1e-10,
            seed: None,
        }
    }
}

/// Lazy lexicographic row selections, identity first, using O(dimension) memory.
/// Callers can filter or rank a bounded batch before `solve_with_permutation_candidates`.
pub fn permutation_candidates(dimension: usize) -> impl Iterator<Item = Vec<usize>> {
    std::iter::successors(Some((0..dimension).collect::<Vec<_>>()), |previous| {
        let mut rows = previous.clone();
        let i = (0..rows.len().saturating_sub(1))
            .rev()
            .find(|&i| rows[i] < rows[i + 1])?;
        let j = (i + 1..rows.len())
            .rev()
            .find(|&j| rows[i] < rows[j])
            .unwrap();
        rows.swap(i, j);
        rows[i + 1..].reverse();
        Some(rows)
    })
}

impl QMIEPSolver {
    pub fn validate_options(&self) -> Result<()> {
        if self.max_permutations == Some(0) || self.maxiter == 0 {
            return Err(Error(
                "max_permutations and maxiter must be positive".into(),
            ));
        }
        if !self.atol.is_finite() || self.atol <= 0.0 {
            return Err(Error("solver atol must be finite and positive".into()));
        }
        Ok(())
    }

    fn validate_input(&self, u: MatRef<'_, C>) -> Result<()> {
        self.validate_options()?;
        validate_unitary(u, 1e-10)?;
        if u.nrows() < 8 || !u.nrows().is_power_of_two() {
            return Err(Error(
                "heuristic QMIEP dimension must be 2^n with n >= 3".into(),
            ));
        }
        Ok(())
    }

    fn rng(&self) -> StdRng {
        self.seed
            .map_or_else(StdRng::from_os_rng, StdRng::seed_from_u64)
    }

    // In exact arithmetic, the analytic U(4) solution yields a traceless
    // matrix DPU, so its spectrum is antipodal. If some numerical issues causes
    // the eigenphase pairing error to exceed the tolerance, we refine the
    // diagonal phases using the same Newton solver.
    //
    // Example (suggested by an LLM): for A with eigenvalues e^{it}, e^{-it},
    // -1, -1, we have \operatorname{tr}(A) = 2\cos t - 2 = -t^2 + O(t^4).
    // At t = 10^{-8}, the trace magnitude is about 10^{-16}, while the
    // eigenphase pairing error is 10^{-8} radians.
    /// Verify the supplied matrix and refine its diagonal phases only if needed.
    /// An exhausted search returns its best candidate with `success = false`.
    pub fn polish_two_qubit(&self, u: MatRef<'_, C>) -> Result<QMIEPResult> {
        self.validate_options()?;
        validate_unitary(u, 1e-10)?;
        if u.nrows() != 4 {
            return Err(Error(
                "two-qubit polishing requires a 4 by 4 unitary".into(),
            ));
        }
        let error = antipodal_error(u)?;
        if error <= self.atol {
            return Ok(QMIEPResult {
                multiplier: DPMultiplier {
                    rows: (0..4).collect(),
                    diagonal: vec![C::new(1.0, 0.0); 4],
                },
                success: true,
                antipodal_error: error,
                permutations_tried: 1,
                attempts: 0,
                nfev: 0,
            });
        }
        self.optimize_from_error(u, error, &mut self.rng())
    }

    /// Return the left multiplier for `U_PD = D P U` and its search statistics.
    /// An exhausted search returns a result with `success = false`
    /// - records only fore debugging usage and in synthesis this will turn in fauilure;
    pub fn solve(&self, u: MatRef<'_, C>) -> Result<QMIEPResult> {
        if u.nrows() == 4 {
            self.validate_options()?;
            let analytic = crate::solve_two_qubit_spectrum(u, 1e-10)?;
            let a = Mat::from_fn(4, 4, |i, j| analytic.diagonal[i] * u[(analytic.rows[i], j)]);
            let mut result = self.polish_two_qubit(a.as_ref())?;
            if result.attempts == 0 {
                // Polishing verified the analytic multiplier as supplied.
                result.multiplier = analytic;
                return Ok(result);
            }
            for (phase, initial) in result.multiplier.diagonal.iter_mut().zip(analytic.diagonal) {
                *phase *= initial;
            }
            result.multiplier.rows = analytic.rows;
            let verified = Mat::from_fn(4, 4, |i, j| {
                result.multiplier.diagonal[i] * u[(result.multiplier.rows[i], j)]
            });
            result.antipodal_error = antipodal_error(verified.as_ref())?;
            result.success = result.antipodal_error <= self.atol;
            return Ok(result);
        }
        self.solve_with_permutation_candidates(u, permutation_candidates(u.nrows()))
    }

    /// Try permutations in the supplied order. The default candidate sequence
    /// used by `solve()` starts with the identity permutation (`P = I`).
    /// Stop at the first verified solution, the end of the iterator, or an explicit cap.
    /// Invalid candidates return errors before indexing.
    pub fn solve_with_permutation_candidates(
        &self,
        u: MatRef<'_, C>,
        candidates: impl IntoIterator<Item = Vec<usize>>,
    ) -> Result<QMIEPResult> {
        self.validate_input(u)?;
        let mut rng = self.rng();
        let mut fallback_candidate: Option<QMIEPResult> = None;
        let (mut tried, mut attempts, mut nfev) = (0usize, 0usize, 0usize);
        let mut candidates = candidates.into_iter();
        while self.max_permutations.is_none_or(|limit| tried < limit) {
            let Some(rows) = candidates.next() else { break };
            let mut seen = vec![false; u.nrows()];
            if rows.len() != u.nrows()
                || rows
                    .iter()
                    .any(|&i| i >= seen.len() || std::mem::replace(&mut seen[i], true))
            {
                return Err(Error(
                    "candidate must be a permutation matching the matrix dimension".into(),
                ));
            }
            let a = Mat::from_fn(u.nrows(), u.ncols(), |i, j| u[(rows[i], j)]);
            let mut result = self.optimize(a.as_ref(), &mut rng)?;
            result.multiplier.rows = rows;
            tried = tried.saturating_add(1);
            attempts = attempts.saturating_add(result.attempts);
            nfev = nfev.saturating_add(result.nfev);
            if result.success {
                return Ok(QMIEPResult {
                    permutations_tried: tried,
                    attempts,
                    nfev,
                    ..result
                });
            }
            if fallback_candidate
                .as_ref()
                .is_none_or(|previous| result.antipodal_error < previous.antipodal_error)
            {
                fallback_candidate = Some(result);
            }
        }
        let failure = fallback_candidate
            .ok_or_else(|| Error("permutation policy returned no candidates".into()))?;
        Ok(QMIEPResult {
            permutations_tried: tried,
            attempts,
            nfev,
            ..failure
        })
    }

    /// Optimize D for a supplied P U, returning identity rows relative to it.
    pub fn optimize_diagonal(&self, u: MatRef<'_, C>) -> Result<QMIEPResult> {
        self.validate_input(u)?;
        self.optimize(u, &mut self.rng())
    }

    fn optimize(&self, a: MatRef<'_, C>, rng: &mut StdRng) -> Result<QMIEPResult> {
        self.optimize_from_error(a, antipodal_error(a)?, rng)
    }

    fn optimize_from_error(
        &self,
        a: MatRef<'_, C>,
        mut best_error: f64,
        rng: &mut StdRng,
    ) -> Result<QMIEPResult> {
        let n = a.nrows();
        let mut best_phases = vec![0.0; n - 1];
        let (mut attempts, mut nfev) = (0, 0);
        let basis_tolerance = 1e-10_f64.min(self.atol * 1e-2);
        let newton_tolerance = 1e-11_f64.min(self.atol * 1e-2);
        let pairing_tolerance = 1e-11_f64.min(self.atol * 0.1);
        for start in 0..=self.restarts {
            if best_error <= self.atol {
                break;
            }
            attempts += 1;
            let mut phases = if start == 0 {
                vec![0.0; n - 1]
            } else {
                (1..n).map(|_| rng.random_range(-PI..PI)).collect()
            };
            let mut previous = None;
            // An approximate basis is enough for the initial Newton direction.
            // Fresh eigenvalues still enforce self.atol before accepting a witness.
            for stage @ (stopping_tolerance, stage_basis_tolerance) in [
                (pairing_tolerance, 1e-10),
                (newton_tolerance, basis_tolerance),
            ] {
                // Include the basis tolerance: strict polishing must remain
                // available even when both Newton stopping thresholds coincide.
                if previous == Some(stage) {
                    continue;
                }
                previous = Some(stage);
                let (next, evaluations) = newton::minimize(
                    phases,
                    self.maxiter,
                    // Check pairing before backtracking at the roundoff floor;
                    // the next stage retains the stricter polishing tolerance.
                    stopping_tolerance,
                    |x| residual_and_jacobian(x, a, stage_basis_tolerance),
                )?;
                phases = next;
                nfev += evaluations;
                let error = antipodal_error(transformed(&phases, a).as_ref())?;
                if error < best_error {
                    best_error = error;
                    best_phases.clone_from(&phases);
                }
                if best_error <= self.atol {
                    break;
                }
            }
        }
        Ok(QMIEPResult {
            multiplier: DPMultiplier {
                rows: (0..n).collect(),
                diagonal: std::iter::once(C::new(1.0, 0.0))
                    .chain(best_phases.iter().map(|&phase| C::from_polar(1.0, phase)))
                    .collect(),
            },
            success: best_error <= self.atol,
            antipodal_error: best_error,
            permutations_tried: 1,
            attempts,
            nfev,
        })
    }
}

fn transformed(phases: &[f64], a: MatRef<'_, C>) -> Mat<C> {
    let diagonal: Vec<_> = std::iter::once(C::new(1.0, 0.0))
        .chain(phases.iter().map(|&phase| C::from_polar(1.0, phase)))
        .collect();
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| diagonal[i] * a[(i, j)])
}

/// Try a Hermitian reduction, then a general EVD with normalized columns.
/// The latter need not return an orthogonal basis at repeated eigenvalues.
fn normal_eigenbasis(u: MatRef<'_, C>, tolerance: f64) -> Option<(Vec<f64>, Mat<C>)> {
    for imaginary in [false, true] {
        if let Some(basis) = hermitian_eigenbasis(u, tolerance, imaginary) {
            return Some(basis);
        }
    }
    let eigen = u.eigen().ok()?;
    let values: Vec<_> = eigen.S().column_vector().iter().copied().collect();
    if values.iter().any(|z| !z.is_finite()) {
        return None;
    }
    let n = u.nrows();
    let mut q = eigen.U().to_owned();
    for j in 0..n {
        let norm = q.col(j).norm_l2();
        if !norm.is_finite() || norm == 0.0 {
            return None;
        }
        for i in 0..n {
            q[(i, j)] /= norm;
        }
    }
    let mut gram = q.adjoint() * &q;
    let mut residual = u * &q;
    for j in 0..n {
        gram[(j, j)] -= C::new(1.0, 0.0);
        for i in 0..n {
            residual[(i, j)] -= q[(i, j)] * values[j];
        }
    }
    if gram.norm_max() <= 1e-10 && residual.norm_max() <= tolerance {
        Some((values.iter().map(|z| z.arg()).collect(), q))
    } else {
        None
    }
}

/// For unitary U, both (U + U†)/2 and (U - U†)/(2i) commute with U. A
/// Hermitian solver supplies orthonormal vectors cheaply, but cos(alpha) or
/// sin(alpha) collisions can mix distinct U eigenvalues. Check the residual.
fn hermitian_eigenbasis(
    u: MatRef<'_, C>,
    tolerance: f64,
    imaginary: bool,
) -> Option<(Vec<f64>, Mat<C>)> {
    let n = u.nrows();
    let h = Mat::from_fn(n, n, |i, j| {
        if imaginary {
            (u[(i, j)] - u[(j, i)].conj()) * C::new(0.0, -0.5)
        } else {
            (u[(i, j)] + u[(j, i)].conj()) * 0.5
        }
    });
    let eigen = h.self_adjoint_eigen(faer::Side::Lower).ok()?;
    let q = eigen.U().to_owned();
    let mut residual = u * &q;
    let mut angles = Vec::with_capacity(n);
    for j in 0..n {
        let value: C = (0..n).map(|i| q[(i, j)].conj() * residual[(i, j)]).sum();
        if !value.is_finite() || (value.norm() - 1.0).abs() > tolerance {
            return None;
        }
        angles.push(value.arg());
        for i in 0..n {
            residual[(i, j)] -= q[(i, j)] * value;
        }
    }
    (residual.norm_max() <= tolerance).then_some((angles, q))
}

fn pairing_eigenbasis(u: MatRef<'_, C>, tolerance: f64) -> Result<(Vec<f64>, Mat<C>)> {
    let n = u.nrows();
    if n >= 8
        && let Some(basis) = normal_eigenbasis(u, tolerance)
    {
        return Ok(basis);
    }
    // Schur vectors preserve orthogonality even at a repeated spectrum.
    let (q, t) = Schur::try_new(
        DMatrix::from_fn(n, n, |i, j| u[(i, j)]),
        f64::EPSILON,
        1000 * n,
    )
    .ok_or_else(|| Error("pairing Schur decomposition did not converge".into()))?
    .unpack();
    Ok((
        (0..n).map(|i| t[(i, i)].arg()).collect(),
        Mat::from_fn(n, n, |i, j| q[(i, j)]),
    ))
}

fn residual_and_jacobian(
    phases: &[f64],
    a: MatRef<'_, C>,
    basis_tolerance: f64,
) -> Result<(Vec<f64>, Mat<f64>)> {
    let u = transformed(phases, a);
    let n = u.nrows();
    let mut residuals = vec![0.0; n / 2];
    let mut jacobian = Mat::<f64>::zeros(n / 2, n - 1);
    // Pairing is only piecewise smooth at eigenvalue collisions.
    let (angles, q) = pairing_eigenbasis(u.as_ref(), basis_tolerance)?;
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&i, &j| angles[i].total_cmp(&angles[j]));
    for i in 0..n / 2 {
        let (first, second) = (order[i], order[i + n / 2]);
        residuals[i] = angles[second] - angles[first] - PI;
        for j in 1..n {
            jacobian[(i, j - 1)] = q[(j, second)].norm_sqr() - q[(j, first)].norm_sqr();
        }
    }
    Ok((residuals, jacobian))
}

/// For a unitary U of dimension 2m, write its eigenvalues as λ_k = exp(iα_k)
/// and sort their principal phases in radians: α_0 ≤ ⋯ ≤ α_{2m−1}.
/// An antipodal spectrum pairs index j with j+m, so λ_{j+m} = −λ_j and
/// α_{j+m} − α_j = π. The maximum deviation from this condition is
/// `ε(U) = max_{0 ≤ j < m} |α_{j+m} − α_j − π|`.
/// Verification recomputes the eigenvalues independently of the Newton residual.
fn antipodal_error(u: MatRef<'_, C>) -> Result<f64> {
    let values = u
        .eigenvalues()
        .map_err(|e| Error(format!("spectral verification: {e:?}")))?;
    if values.iter().any(|z| !z.is_finite()) {
        return Err(Error(
            "nonfinite eigenvalue in spectral verification".into(),
        ));
    }
    let mut angles: Vec<_> = values.iter().map(|z| z.arg()).collect();
    angles.sort_by(f64::total_cmp);
    let half = angles.len() / 2;
    Ok((0..half)
        .map(|i| (angles[i + half] - angles[i] - PI).abs())
        .fold(0.0, f64::max))
}
