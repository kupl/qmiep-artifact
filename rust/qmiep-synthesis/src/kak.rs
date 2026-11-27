// Adapted from Qiskit 2.5.1 two_qubit_decompose/{weyl_decomposition,basis_decomposer}.rs.
// (C) Copyright IBM 2025, 2026
// Licensed under the Apache License, Version 2.0 (see ../LICENSE-QISKIT).
// Modified for this standalone crate: no Python bindings, exact Weyl factors only,
// deterministic diagonalization retries, and a CX-only basis with explicit phase.

use crate::{C, Error, Mat, MatRef, Result, linalg};
use nalgebra::{Matrix2, Matrix4, Vector4};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, PI, TAU};
use std::sync::OnceLock;

type M2 = Matrix2<C>;
type M4 = Matrix4<C>;
const ZERO: C = C::new(0.0, 0.0);
const ONE: C = C::new(1.0, 0.0);
const M_ONE: C = C::new(-1.0, 0.0);
const IM: C = C::new(0.0, 1.0);
const M_IM: C = C::new(0.0, -1.0);
const fn c64(re: f64, im: f64) -> C {
    C::new(re, im)
}
static IPX: M2 = M2::new(ZERO, IM, IM, ZERO);
static IPY: M2 = M2::new(ZERO, ONE, M_ONE, ZERO);
static IPZ: M2 = M2::new(IM, ZERO, ZERO, M_IM);
static MAGIC: M4 = M4::new(
    ONE, IM, ZERO, ZERO, ZERO, ZERO, IM, ONE, ZERO, ZERO, IM, M_ONE, ONE, M_IM, ZERO, ZERO,
);

static K12R: Matrix2<C> = Matrix2::new(
    c64(0., FRAC_1_SQRT_2),
    c64(FRAC_1_SQRT_2, 0.),
    c64(-FRAC_1_SQRT_2, 0.),
    c64(0., -FRAC_1_SQRT_2),
);

static K12R_DG: Matrix2<C> = Matrix2::new(
    c64(0., -FRAC_1_SQRT_2),
    c64(-FRAC_1_SQRT_2, 0.),
    c64(FRAC_1_SQRT_2, 0.),
    c64(0., FRAC_1_SQRT_2),
);

static K12L: Matrix2<C> =
    Matrix2::new(c64(0.5, 0.5), c64(0.5, 0.5), c64(-0.5, 0.5), c64(0.5, -0.5));

static K12L_DG: Matrix2<C> = Matrix2::new(
    c64(0.5, -0.5),
    c64(-0.5, -0.5),
    c64(0.5, -0.5),
    c64(0.5, 0.5),
);

static K22L: Matrix2<C> = Matrix2::new(
    c64(FRAC_1_SQRT_2, 0.),
    c64(-FRAC_1_SQRT_2, 0.),
    c64(FRAC_1_SQRT_2, 0.),
    c64(FRAC_1_SQRT_2, 0.),
);
static K22R: Matrix2<C> = Matrix2::new(ZERO, ONE, M_ONE, ZERO);

fn rz(theta: f64) -> M2 {
    M2::new(
        C::from_polar(1.0, -theta / 2.0),
        ZERO,
        ZERO,
        C::from_polar(1.0, theta / 2.0),
    )
}

fn product(u: M4) -> Result<(M2, M2, f64)> {
    let mut r = u.fixed_view::<2, 2>(0, 0).into_owned();
    if r.determinant().norm() < 0.1 {
        r = u.fixed_view::<2, 2>(2, 0).into_owned();
    }
    if r.determinant().norm() < 0.1 {
        return Err(Error("KAK tensor factor R is singular".into()));
    }
    r /= r.determinant().sqrt();
    let temp = u * M2::identity().kronecker(&r.adjoint());
    let mut l = M2::from_fn(|i, j| temp[(2 * i, 2 * j)]);
    let det = l.determinant();
    if det.norm() < 0.9 {
        return Err(Error("KAK tensor factor L is singular".into()));
    }
    l /= det.sqrt();
    Ok((l, r, det.arg() / 2.0))
}

struct Weyl {
    a: f64,
    b: f64,
    c: f64,
    phase: f64,
    k1l: M2,
    k1r: M2,
    k2l: M2,
    k2r: M2,
}

impl Weyl {
    fn new(mut u: M4) -> Result<Self> {
        let det = u.determinant();
        u *= det.powf(-0.25);
        let mut global_phase = det.arg() / 4.0;
        let magic_dagger = MAGIC.adjoint() * c64(0.5, 0.0);
        let u_p = magic_dagger * u * MAGIC;
        let m2 = u_p.transpose() * u_p;
        let mut eig = None;
        for trial in 0..100 {
            let (a, b) = if trial == 0 {
                (1.2602066112249388, 0.22317849046722027)
            } else {
                let angle = trial as f64 * 2.399963229728653;
                (angle.cos(), angle.sin())
            };
            let real = Mat::from_fn(4, 4, |i, j| a * m2[(i, j)].re + b * m2[(i, j)].im);
            let e = real
                .self_adjoint_eigen(faer::Side::Lower)
                .map_err(|e| Error(format!("KAK eigen: {e:?}")))?;
            let p = M4::from_fn(|i, j| c64(e.U()[(i, j)], 0.0));
            let d = (p.transpose() * m2 * p).diagonal();
            if (p * M4::from_diagonal(&d) * p.transpose() - m2)
                .iter()
                .all(|z| z.norm() < 1e-13)
            {
                eig = Some((p, d));
                break;
            }
        }
        let (mut p, d) = eig.ok_or_else(|| {
            Error("KAK could not jointly diagonalize the magic-basis matrix".into())
        })?;
        let mut d = -d.map(|x| x.arg() / 2.0);
        d[3] = -d[0] - d[1] - d[2];
        let mut cs = std::array::from_fn::<_, 3, _>(|i| ((d[i] + d[3]) / 2.0).rem_euclid(TAU));
        let folded = cs.map(|x| {
            let t = x.rem_euclid(FRAC_PI_2);
            t.min(FRAC_PI_2 - t)
        });
        let mut order = [0, 1, 2];
        order.sort_by(|&a, &b| folded[a].total_cmp(&folded[b]));
        order.rotate_left(1);
        cs = order.map(|i| cs[i]);
        (d[0], d[1], d[2]) = (d[order[0]], d[order[1]], d[order[2]]);
        let old_p = p;
        for (j, &original) in order.iter().enumerate() {
            p.set_column(j, &old_p.column(original));
        }
        if p.determinant().re < 0.0 {
            p.set_column(3, &(-p.column(3)));
        }
        let phases = M4::from_diagonal(&Vector4::from_fn(|i, _| C::from_polar(1.0, d[i])));
        let k1 = MAGIC * (u_p * p * phases) * magic_dagger;
        let k2 = MAGIC * p.transpose() * magic_dagger;
        let (mut k1l, mut k1r, phase_l) = product(k1)?;
        let (k2l, mut k2r, phase_r) = product(k2)?;
        global_phase += phase_l + phase_r;
        // Flip into Weyl chamber
        if cs[0] > FRAC_PI_2 {
            cs[0] -= 3.0 * FRAC_PI_2;
            k1l *= IPY;
            k1r *= IPY;
            global_phase += FRAC_PI_2;
        }
        if cs[1] > FRAC_PI_2 {
            cs[1] -= 3.0 * FRAC_PI_2;
            k1l *= IPX;
            k1r *= IPX;
            global_phase += FRAC_PI_2;
        }
        let mut conjs = 0;
        if cs[0] > FRAC_PI_4 {
            cs[0] = FRAC_PI_2 - cs[0];
            k1l *= IPY;
            k2r = IPY * k2r;
            conjs += 1;
            global_phase -= FRAC_PI_2;
        }
        if cs[1] > FRAC_PI_4 {
            cs[1] = FRAC_PI_2 - cs[1];
            k1l *= IPX;
            k2r = IPX * k2r;
            conjs += 1;
            global_phase += FRAC_PI_2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if cs[2] > FRAC_PI_2 {
            cs[2] -= 3.0 * FRAC_PI_2;
            k1l *= IPZ;
            k1r *= IPZ;
            global_phase += FRAC_PI_2;
            if conjs == 1 {
                global_phase -= PI;
            }
        }
        if conjs == 1 {
            cs[2] = FRAC_PI_2 - cs[2];
            k1l *= IPZ;
            k2r = IPZ * k2r;
            global_phase += FRAC_PI_2;
        }
        if cs[2] > FRAC_PI_4 {
            cs[2] -= FRAC_PI_2;
            k1l *= IPZ;
            k1r *= IPZ;
            global_phase -= FRAC_PI_2;
        }
        let [a, b, c] = [cs[1], cs[0], cs[2]];
        Ok(Self {
            a,
            b,
            c,
            phase: global_phase,
            k1l,
            k1r,
            k2l,
            k2r,
        })
    }
}

pub(crate) struct CxBasis {
    basis: Weyl,
    u0l: M2,
    u0r: M2,
    u1l: M2,
    u1ra: M2,
    u1rb: M2,
    u2la: M2,
    u2lb: M2,
    u2ra: M2,
    u2rb: M2,
    u3l: M2,
    u3r: M2,
    q0l: M2,
    q0r: M2,
    q1la: M2,
    q1lb: M2,
    q1ra: M2,
    q1rb: M2,
    q2l: M2,
    q2r: M2,
}

impl CxBasis {
    fn new() -> Result<Self> {
        // CX with the least-significant bit as control, matching local-factor order.
        let cx = M4::from_fn(|i, j| {
            if i == (if j & 1 == 1 { j ^ 2 } else { j }) {
                ONE
            } else {
                ZERO
            }
        });
        let basis = Weyl::new(cx)?;
        let b = basis.b;
        let temp = c64(0.5, -0.5);
        let k11l = Matrix2::new(
            temp * (M_IM * c64(0., -b).exp()),
            temp * c64(0., -b).exp(),
            temp * (M_IM * c64(0., b).exp()),
            temp * -(c64(0., b).exp()),
        );
        let k11r = Matrix2::new(
            FRAC_1_SQRT_2 * (IM * c64(0., -b).exp()),
            FRAC_1_SQRT_2 * -c64(0., -b).exp(),
            FRAC_1_SQRT_2 * c64(0., b).exp(),
            FRAC_1_SQRT_2 * (M_IM * c64(0., b).exp()),
        );
        let k32l_k21l = Matrix2::new(
            FRAC_1_SQRT_2 * c64(1., (2. * b).cos()),
            FRAC_1_SQRT_2 * (IM * (2. * b).sin()),
            FRAC_1_SQRT_2 * (IM * (2. * b).sin()),
            FRAC_1_SQRT_2 * c64(1., -(2. * b).cos()),
        );
        let temp = c64(0.5, 0.5);
        let k21r = Matrix2::new(
            temp * (M_IM * c64(0., -2. * b).exp()),
            temp * c64(0., -2. * b).exp(),
            temp * (IM * c64(0., 2. * b).exp()),
            temp * c64(0., 2. * b).exp(),
        );
        let k31l = Matrix2::new(
            FRAC_1_SQRT_2 * c64(0., -b).exp(),
            FRAC_1_SQRT_2 * c64(0., -b).exp(),
            FRAC_1_SQRT_2 * -c64(0., b).exp(),
            FRAC_1_SQRT_2 * c64(0., b).exp(),
        );
        let k31r = Matrix2::new(IM * c64(0., b).exp(), ZERO, ZERO, M_IM * c64(0., -b).exp());
        let temp = c64(0.5, 0.5);
        let k32r = Matrix2::new(
            temp * c64(0., b).exp(),
            temp * -c64(0., -b).exp(),
            temp * (M_IM * c64(0., b).exp()),
            temp * (M_IM * c64(0., -b).exp()),
        );
        let k1ld = basis.k1l.adjoint();
        let k1rd = basis.k1r.adjoint();
        let k2ld = basis.k2l.adjoint();
        let k2rd = basis.k2r.adjoint();
        // Pre-build the fixed parts of the matrices used in 3-part decomposition
        let u0l = k31l * k1ld;
        let u0r = k31r * k1rd;
        let u1l = k2ld * k32l_k21l * k1ld;
        let u1ra = k2rd * k32r;
        let u1rb = k21r * k1rd;
        let u2la = k2ld * K22L;
        let u2lb = k11l * k1ld;
        let u2ra = k2rd * K22R;
        let u2rb = k11r * k1rd;
        let u3l = k2ld * K12L;
        let u3r = k2rd * K12R;
        // Pre-build the fixed parts of the matrices used in the 2-part decomposition
        let q0l = K12L_DG * k1ld;
        let q0r = K12R_DG * IPZ * k1rd;
        let q1la = k2ld * k11l.adjoint();
        let q1lb = k11l * k1ld;
        let q1ra = k2rd * IPZ * k11r.adjoint();
        let q1rb = k11r * k1rd;
        let q2l = k2ld * K12L;
        let q2r = k2rd * K12R;
        Ok(Self {
            basis,
            u0l,
            u0r,
            u1l,
            u1ra,
            u1rb,
            u2la,
            u2lb,
            u2ra,
            u2rb,
            u3l,
            u3r,
            q0l,
            q0r,
            q1la,
            q1lb,
            q1ra,
            q1rb,
            q2l,
            q2r,
        })
    }

    pub(crate) fn get() -> Result<&'static Self> {
        static BASIS: OnceLock<Result<CxBasis>> = OnceLock::new();
        BASIS.get_or_init(Self::new).as_ref().map_err(Clone::clone)
    }

    pub(crate) fn factors(&self, u: MatRef<'_, C>, atol: f64) -> Result<(Vec<M2>, f64)> {
        let u = linalg::polar(u)?;
        let t = Weyl::new(M4::from_fn(|i, j| u[(i, j)]))?;
        let tol = atol / 4.0;
        let count = if t.a.abs() + t.b.abs() + t.c.abs() <= tol {
            0
        } else if (t.a - FRAC_PI_4).abs() + t.b.abs() + t.c.abs() <= tol {
            1
        } else if t.c.abs() <= tol {
            2
        } else {
            3
        };
        let matrices = match count {
            0 => vec![t.k1r * t.k2r, t.k1l * t.k2l],
            1 => vec![
                self.basis.k2r.adjoint() * t.k2r,
                self.basis.k2l.adjoint() * t.k2l,
                t.k1r * self.basis.k1r.adjoint(),
                t.k1l * self.basis.k1l.adjoint(),
            ],
            2 => vec![
                self.q2r * t.k2r,
                self.q2l * t.k2l,
                self.q1ra * rz(2.0 * t.b) * self.q1rb,
                self.q1la * rz(-2.0 * t.a) * self.q1lb,
                t.k1r * self.q0r,
                t.k1l * self.q0l,
            ],
            _ => vec![
                self.u3r * t.k2r,
                self.u3l * t.k2l,
                self.u2ra * rz(2.0 * t.b) * self.u2rb,
                self.u2la * rz(-2.0 * t.a) * self.u2lb,
                self.u1ra * rz(-2.0 * t.c) * self.u1rb,
                self.u1l,
                t.k1r * self.u0r,
                t.k1l * self.u0l,
            ],
        };
        Ok((
            matrices,
            t.phase - count as f64 * self.basis.phase + if count == 2 { PI } else { 0.0 },
        ))
    }
}

pub(crate) fn left_diagonal(u: MatRef<'_, C>, atol: f64) -> [C; 4] {
    if (0..4).all(|i| (0..4).all(|j| i == j || u[(i, j)].norm() <= atol)) {
        return std::array::from_fn(|i| u[(i, i)].conj());
    }
    let mut special = M4::from_fn(|i, j| u[(i, j)]);
    special *= C::from_polar(1.0, -special.determinant().arg() / 4.0);
    let yy = M4::new(
        ZERO, ZERO, ZERO, -ONE, ZERO, ZERO, ONE, ZERO, ZERO, ONE, ZERO, ZERO, -ONE, ZERO, ZERO,
        ZERO,
    );
    let t = (special * yy * special.transpose() * yy).diagonal();
    let numerator = t.sum().im;
    let denominator = (t[0] - t[1] - t[2] + t[3]).re;
    let psi = if numerator.abs() <= atol && denominator.abs() <= atol {
        0.0
    } else {
        numerator.atan2(denominator)
    };
    [-1.0, 1.0, 1.0, -1.0].map(|sign| C::from_polar(1.0, sign * psi / 2.0))
}
