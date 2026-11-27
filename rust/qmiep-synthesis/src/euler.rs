//! Standard U-gate coordinates, following Qiskit's U3/ZYZ phase convention.
//!
//! Reference: Qiskit 2.5.1, crates/synthesis/src/euler_one_qubit_decomposer.rs.
// (C) Copyright IBM 2025, 2026
// Apache License, Version 2.0; see ../LICENSE-QISKIT and ../NOTICE.
// Modified to return native U parameters and an explicit phase without gate simplification.

use crate::{C, Instruction};

/// Return `[theta, phi, lambda]` and alpha such that `m = exp(i alpha) U`.
fn decompose(m: [[C; 2]; 2]) -> ([f64; 3], f64) {
    let det_phase = (m[0][0] * m[1][1] - m[0][1] * m[1][0]).arg();
    let theta = 2.0 * m[1][0].norm().atan2(m[0][0].norm());
    let (a, b) = (m[1][1].arg(), m[1][0].arg());
    let (phi, lambda) = (a + b - det_phase, a - b);
    ([theta, phi, lambda], 0.5 * (det_phase - phi - lambda))
}

/// Synthesize a one-qubit unitary as exp(i phase) times a U gate.
pub(crate) fn synthesize(m: [[C; 2]; 2], qubit: usize) -> (Instruction, f64) {
    let ([theta, phi, lambda], phase) = decompose(m);
    (
        Instruction::U {
            qubit,
            theta,
            phi,
            lambda,
        },
        phase,
    )
}

pub(crate) fn matrix(theta: f64, phi: f64, lambda: f64) -> [[C; 2]; 2] {
    let (s, c) = (theta / 2.0).sin_cos();
    [
        [C::new(c, 0.0), -C::from_polar(s, lambda)],
        [C::from_polar(s, phi), C::from_polar(c, phi + lambda)],
    ]
}
