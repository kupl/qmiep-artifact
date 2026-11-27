// QSD numerical construction adapted from Qiskit 2.5.1 qsd.rs.
// (C) Copyright IBM 2025, 2026
// Apache License, Version 2.0; see ../LICENSE-QISKIT and ../NOTICE.
// Modified for QMIEP pairing, inline PD corrections and native instructions.
use crate::fragment::CircuitFragment;
use crate::{
    C, Instruction, Kind, Mat, MatRef, Method, MultiplexedDPMultiplier, MultiplexedPermutation,
    Options, QMIEPSolver, Result, SynthesisResult,
};
use crate::{euler, kak, linalg};
use qmiep_solver::analytic::solve_two_qubit_spectrum;

struct Synthesizer {
    basis: &'static kak::CxBasis,
    atol: f64,
    method: Method,
    solver: QMIEPSolver,
}

#[derive(PartialEq)]
enum Boundary {
    Left,
    Right,
    Closed,
}

// Absorb the chronological tail P -> D -> V into V' = V D P.
// P|j> = |p[j]> gives (V D P)[i,j] = V[i,p[j]] * d[p[j]].
fn absorb(u: MatRef<'_, C>, images: impl Iterator<Item = usize>, diagonal: &[C]) -> Mat<C> {
    let mut product = Mat::zeros(u.nrows(), u.ncols());
    for (j, p) in images.enumerate() {
        let phase = diagonal[p];
        for i in 0..u.nrows() {
            product[(i, j)] = u[(i, p)] * phase;
        }
    }
    product
}

/// Decompose the UCG-Rz block for D ⊕ D† with the specified boundary CX.
fn synthesize_ucg_rz(diagonal: &[C], qubits: &[usize], side: Boundary) -> CircuitFragment {
    let n = diagonal.len();
    let mut angles: Vec<_> = diagonal.iter().map(|z| -2.0 * z.arg() / n as f64).collect();
    let mut width = 1;
    while width < n {
        for start in (0..n).step_by(2 * width) {
            for j in start..start + width {
                let (a, b) = (angles[j], angles[j + width]);
                angles[j] = a + b;
                angles[j + width] = a - b;
            }
        }
        width *= 2;
    }
    let mut instructions = Vec::with_capacity(2 * n);
    for i in 0..n {
        let gray = i ^ (i >> 1);
        instructions.push(Instruction::Rz {
            qubit: qubits[0],
            angle: angles[gray],
        });
        if i + 1 < n || side == Boundary::Closed {
            let next = (i + 1) % n;
            let bit = (gray ^ (next ^ (next >> 1))).trailing_zeros() as usize;
            instructions.push(Instruction::Cx {
                control: qubits[qubits.len() - bit - 1],
                target: qubits[0],
            });
        }
    }
    if side == Boundary::Right {
        instructions.reverse();
    }
    CircuitFragment::Block {
        instructions,
        global_phases: Vec::new(),
    }
}

fn synthesize_cz(control: usize, target: usize) -> CircuitFragment {
    CircuitFragment::Block {
        instructions: vec![
            Instruction::H { qubit: target },
            Instruction::Cx { control, target },
            Instruction::H { qubit: target },
        ],
        global_phases: Vec::new(),
    }
}

impl Synthesizer {
    // Two-qubit base case, with circuit time running left to right:
    // Exact:       U => W             (P = D = I, at most 3 CX).
    // Pd/Diagonal: U => W -> D        (P = I, at most 2 CX in W).
    // delta * U is synthesized as W; the returned D is delta^dagger.
    fn synthesize_two_qubit(
        &self,
        u: MatRef<'_, C>,
        qubits: &[usize],
        kind: Kind,
    ) -> Result<(CircuitFragment, MultiplexedDPMultiplier)> {
        let delta = if kind.is_exact() {
            [C::new(1.0, 0.0); 4]
        } else {
            kak::left_diagonal(u, self.atol)
        };
        let residual = Mat::from_fn(4, 4, |i, j| delta[i] * u[(i, j)]);
        let (factors, phase) = self.basis.factors(residual.as_ref(), self.atol)?;
        let mut instructions = Vec::with_capacity(factors.len() + factors.len() / 2 - 1);
        let mut global_phases = Vec::with_capacity(factors.len() + 1);
        for (index, pair) in factors.as_chunks::<2>().0.iter().enumerate() {
            if index > 0 {
                instructions.push(Instruction::Cx {
                    control: qubits[1],
                    target: qubits[0],
                });
            }
            for (q, matrix) in [qubits[1], qubits[0]].into_iter().zip(pair) {
                let (gate, phase) = euler::synthesize(
                    std::array::from_fn(|i| std::array::from_fn(|j| matrix[(i, j)])),
                    q,
                );
                instructions.push(gate);
                global_phases.push(phase);
            }
        }
        global_phases.push(phase);
        let multiplier = MultiplexedDPMultiplier {
            permutation: MultiplexedPermutation {
                zero: vec![0, 1],
                one: vec![0, 1],
            },
            diagonal: delta.map(|z| z.conj()).to_vec(),
        };
        Ok((
            CircuitFragment::Block {
                instructions,
                global_phases,
            },
            multiplier,
        ))
    }

    // Return the replacement circuit W and its left multiplier D P
    // satisfying U Q = D P W, with Q = I when there is no incoming correction.
    // Circuit order is W -> P -> D; Exact returns I, I.
    //
    // Four-qubit Exact example; each child acts on T = (q1, q2, q3):
    // U4: Exact, paired=false
    // +-- L3:  Pd    -> 2q children (L, M, N, O): (Pd, Pd, Diagonal, Pd)
    // +-- M3': Pd    -> 2q children (L, M, N, O): (Pd, Pd, Diagonal, Pd)
    // +-- N3': Pd    -> 2q children (L, M, N, O): (Pd, Pd, Diagonal, Pd)
    // `-- O3': Exact -> 2q children (L, M, N, O): (Pd, Pd, Pd, Exact)
    // M' = M D_L P_L, N' = N D_M P_M, O' = O D_N P_N.
    // These are depth-first calls: finish L and absorb its correction before M.
    fn synthesize(
        &self,
        u: MatRef<'_, C>,
        qubits: &[usize],
        kind: Kind,
        incoming: Option<&MultiplexedDPMultiplier>,
    ) -> Result<(CircuitFragment, MultiplexedDPMultiplier)> {
        if qubits.len() == 2 {
            return match incoming {
                Some(q) => self.synthesize_two_qubit(
                    absorb(u, q.permutation.images(), &q.diagonal).as_ref(),
                    qubits,
                    kind,
                ),
                None => self.synthesize_two_qubit(u, qubits, kind),
            };
        }
        self.synthesize_node(u, qubits, kind, incoming)
    }

    // One factorization and at most one QMIEP solve per node. Improved PD/D
    // children differ only in whether their output may expose a permutation.
    fn synthesize_node(
        &self,
        u: MatRef<'_, C>,
        qubits: &[usize],
        kind: Kind,
        incoming: Option<&MultiplexedDPMultiplier>,
    ) -> Result<(CircuitFragment, MultiplexedDPMultiplier)> {
        let n = qubits.len();
        let improved = matches!(
            kind,
            Kind::ExactFurtherDOptimized | Kind::DiagonalFurtherOptimal | Kind::PdFurtherDOptimized
        );
        let require_d = kind == Kind::DiagonalFurtherOptimal;
        let factors = linalg::block_zxz(u)?;
        // Both methods use Qiskit's source
        let a1 = linalg::polar(factors.a1.as_ref())?;
        let mut a2 = linalg::polar(factors.a2.as_ref())?;
        let m = a1.nrows();
        // Baseline keeps the same recursion and diagonal reduction, without QMIEP.
        let attempt_pairing = require_d
            || kind == Kind::PdFurtherDOptimized
            || (kind == Kind::Pd
                && match self.method {
                    Method::Analytic => n == 3,
                    Method::Heuristic => true,
                    Method::Baseline => false,
                });
        // Multiplier for Block-ZXZ's A1 ⊕ A2 block, last in circuit execution
        // (leftmost in the matrix product). With solver multiplier F, replace
        // A2 by A2' = F A2 and store C_A = F† = D_A P_A:
        // A1 ⊕ A2 = (I ⊕ C_A)(A1 ⊕ A2').
        let a_solver_multiplier = if attempt_pairing {
            let a = &a2 * a1.adjoint();
            let mut left = if n == 3 {
                solve_two_qubit_spectrum(a.as_ref(), 1e-10)?
            } else {
                self.solver.solve(a.as_ref())?.verified()?
            };
            if require_d && left.rows.iter().enumerate().any(|(i, &row)| i != row) {
                // Reuse the original ZXZ factors for the full right R block.
                None
            } else {
                // Improved D modes use numerical QMIEP even with the analytic label.
                if n == 3 && (self.method == Method::Heuristic || improved) {
                    let paired_a =
                        Mat::from_fn(m, m, |i, j| left.diagonal[i] * a[(left.rows[i], j)]);
                    let polish = self
                        .solver
                        .polish_two_qubit(paired_a.as_ref())?
                        .verified()?;
                    for (phase, multiplier) in left.diagonal.iter_mut().zip(polish.diagonal) {
                        *phase *= multiplier;
                    }
                }
                // F A2 A1† is antipodal; use the corrected A2 in the actual
                // demultiplex, whose product A1 (F A2)† is its adjoint.
                a2 = Mat::from_fn(m, m, |i, j| left.diagonal[i] * a2[(left.rows[i], j)]);
                Some(left)
            }
        } else {
            None
        };
        let paired = a_solver_multiplier.is_some();
        let identity = linalg::polar(Mat::<C>::identity(m, m).as_ref())?;
        let c = linalg::polar(factors.c.as_ref())?;
        // The preceding child's correction preserves the first selector bit:
        // Q = Q0 ⊕ Q1. Absorb it into the rightmost Block-ZXZ factor,
        // (I ⊕ C) Q = Q0 ⊕ C Q1. This keeps A2 A1† independent of Q even
        // in floating point, since Block-ZXZ is evaluated on the original U.
        let (c0, c1) = match incoming {
            Some(q) => {
                let (d0, d1) = q.diagonal.split_at(m);
                (
                    absorb(identity.as_ref(), q.permutation.zero.iter().copied(), d0),
                    absorb(c.as_ref(), q.permutation.one.iter().copied(), d1),
                )
            }
            None => (identity, c),
        };
        let left_factors = linalg::demultiplex(c0.as_ref(), c1.as_ref(), false, self.atol)?;
        let right_factors = linalg::demultiplex(a1.as_ref(), a2.as_ref(), paired, self.atol)?;
        let (sa, terminal, wa) = if paired {
            let short: Vec<_> = right_factors.diagonal.iter().step_by(2).copied().collect();
            // D² = S² Z_last, Z_last = diag(1,-1,...), and S repeats pairs:
            // Q (D ⊕ D†) W = (Q S†) (S ⊕ S†) CZ (D W).
            let terminal = Mat::from_fn(m, m, |i, j| right_factors.v[(i, j)] * short[j / 2].conj());
            let wa = Mat::from_fn(m, m, |i, j| {
                right_factors.diagonal[i] * right_factors.w[(i, j)]
            });
            (short, terminal, wa)
        } else {
            (right_factors.diagonal, right_factors.v, right_factors.w)
        };
        let b0 = &wa * &left_factors.v;
        let boundary_z = Mat::from_fn(m, m, |i, j| {
            C::new(
                if i != j {
                    0.0
                } else if i < m / 2 {
                    1.0
                } else {
                    -1.0
                },
                0.0,
            )
        });
        let b1 = &boundary_z * &wa * &factors.b * &left_factors.v * &boundary_z;
        let b0 = linalg::polar(b0.as_ref())?;
        let b1 = linalg::polar(b1.as_ref())?;
        let middle_factors = linalg::demultiplex(b0.as_ref(), b1.as_ref(), false, self.atol)?;
        let (left, sc) = (left_factors.w, left_factors.diagonal);
        let (middle, central, right) =
            (middle_factors.w, middle_factors.diagonal, middle_factors.v);

        // The numerical factors above encode this unpaired-node circuit.
        // For U4/Exact, L=left, M=middle, N=right, O=terminal are 3q gates.
        // T is a bundled register; [s] selects a rotation acting on q0.
        // R_L/R_R are open Rz-CX cascades; F_B is the closed UCRz cascade.
        // [L:0], [L:1], [L:2] are the three wires of ONE gate L; likewise M,N,O.
        // q0: -----------[R_L]-----[H]--------------[F_B]--------------[H]-----[R_R]-----------
        //                  |                          |                          |
        // q1: --[L:0]-----[s]--------------[M:0]-----[s]-----[N:0]--------------[s]-----[O:0]--
        //         |        |                 |        |        |                 |        |
        // q2: --[L:1]-----[s]--------------[M:1]-----[s]-----[N:1]--------------[s]-----[O:1]--
        //         |        |                 |        |        |                 |        |
        // q3: --[L:2]-----[s]--------------[M:2]-----[s]-----[N:2]--------------[s]-----[O:2]--
        // A paired 3q/Pd child instead has a shorter R_R, then CZ, O,
        // controlled-P_A, controlled-D_A (see the paired branch below).
        let selectors = &qubits[1..];
        let migration_kind = if improved {
            Kind::PdFurtherDOptimized
        } else {
            Kind::Pd
        };
        // A full R propagates N's permutation into O. The short R and CZ require D.
        let right_kind = if !paired {
            migration_kind
        } else if improved {
            Kind::DiagonalFurtherOptimal
        } else {
            Kind::Diagonal
        };
        let (left_block, left_multiplier) =
            self.synthesize(left.as_ref(), selectors, migration_kind, None)?;
        // L has emitted W_L and returned C_L = D_L P_L. Move that tail:
        // q0: -----------------------------[R_L]-----[H]------------
        //                                    |
        // T : ==[W_L]====[P_L]====[D_L]=====[s]===============[M]===
        //                         becomes
        // q0: ----------[R_L']-----[H]------------
        //                  |
        // T : ==[W_L]=====[s]==============[M']===
        // R_L' = (I x P_L^dagger) R_L (I x P_L); M' = M C_L.
        // The returned P_L preserves the first selector bit, so the open
        // cascade's boundary CX is unchanged; only rotation sectors move.
        let left_rotation = synthesize_ucg_rz(
            &left_multiplier
                .permutation
                .images()
                .map(|i| sc[i])
                .collect::<Vec<_>>(),
            qubits,
            Boundary::Left,
        );
        // Absorb C_L into M, then synthesize M' modulo PD. Its NEW correction
        // C_M belongs to M', not the original M, and moves into N:
        // q0: -----------------------------[F_B]-----------
        //                                    |
        // T : ==[W_M]====[P_M]====[D_M]=====[s]======[N]===
        //                         becomes
        // q0: ----------[F_B']-----------
        //                  |
        // T : ==[W_M]=====[s]=====[N']===
        // F_B' = (I x P_M^dagger) F_B (I x P_M); N' = N D_M P_M.
        let (middle_block, middle_multiplier) = self.synthesize(
            middle.as_ref(),
            selectors,
            migration_kind,
            Some(&left_multiplier),
        )?;
        let central_rotation = synthesize_ucg_rz(
            &middle_multiplier
                .permutation
                .images()
                .map(|i| central[i])
                .collect::<Vec<_>>(),
            qubits,
            Boundary::Closed,
        );
        // At U4/Exact, paired=false: N' is another 3q Pd call.
        // Inside a 3q/Pd child, paired=true: N' is a 2q Diagonal call.
        // That restriction leaves only D_N to pass the extra CZ below.
        let (right_block, right_multiplier) = self.synthesize(
            right.as_ref(),
            selectors,
            right_kind,
            Some(&middle_multiplier),
        )?;
        let right_rotation = if paired {
            // Local wires of a 3q Pd child (q1, q2, q3 at the U4 parent):
            // q1: ------------[H]-----[R_R]------@-------------
            //                           |        |
            // q2: -[W_N:0]-------------[s]-------|-----[O':0]--
            //         |                          |        |
            // q3: -[W_N:1]-----------------------@-----[O':1]--
            // W_N and O' each span q2,q3; @--@ is CZ(q1,q3).
            // N' = D_N W_N. D_N commutes with H, R_R and CZ, so O'=O D_N.
            // O' is synthesized as Pd, leaving the child's output correction.
            CircuitFragment::Sequence(vec![
                synthesize_ucg_rz(&sa, &qubits[..n - 1], Boundary::Right),
                synthesize_cz(qubits[0], qubits[n - 1]),
            ])
        } else {
            // Unpaired node, including U4/Exact: move the full C_N to O.
            // q0: ------------------------------[H]-----[R_R]-----------
            //                                             |
            // T : ==[W_N]====[P_N]====[D_N]==============[s]======[O]===
            //                         becomes
            // q0: ------------[H]----[R_R']-----------
            //                           |
            // T : ==[W_N]==============[s]=====[O']===
            // R_R' = (I x P_N^dagger) R_R (I x P_N); O' = O D_N P_N.
            synthesize_ucg_rz(
                &right_multiplier
                    .permutation
                    .images()
                    .map(|i| sa[i])
                    .collect::<Vec<_>>(),
                qubits,
                Boundary::Right,
            )
        };
        // The terminal inherits THIS node's kind. For U4/Exact:
        // T: ==[O']==  ->  ==[W_O]==, with C_O = I.
        // O3'/Exact repeats the unpaired recursion; its last O2/Exact leaf
        // implements the remaining correction as part of that input unitary.
        let (terminal_block, terminal_multiplier) =
            self.synthesize(terminal.as_ref(), selectors, kind, Some(&right_multiplier))?;
        let h = Instruction::H { qubit: qubits[0] };
        let circuit = CircuitFragment::Sequence(vec![
            left_block,
            left_rotation,
            CircuitFragment::Gate(h.clone()),
            middle_block,
            central_rotation,
            right_block,
            CircuitFragment::Gate(h),
            right_rotation,
            terminal_block,
        ]);
        let zero: Vec<_> = terminal_multiplier.permutation.images().collect();
        let mut d = terminal_multiplier.diagonal;
        let one = if let Some(left) = a_solver_multiplier {
            // Combine the A-block correction C_A with the separate correction
            // C_O from recursive synthesis of O': C_out = (I ⊕ C_A)(I_2 ⊗ C_O).
            // Paired-child tail, in circuit order:
            // W -> (I x P_O) -> (I x D_O) -> ctrl(P_A) -> ctrl(D_A).
            // Return one combined D_hat P_hat; do not emit this tail.
            // P_hat = P_O (+) P_A P_O,
            // D_hat = D_O (+) D_A P_A D_O P_A^dagger, where (+) is direct sum.
            // At larger paired nodes P_O can itself be nontrivial.
            // F selects rows. In F†, column j maps to row left.rows[j]
            // with phase conj(left.diagonal[j]).
            d.resize(2 * m, C::new(0.0, 0.0));
            for (j, &p) in left.rows.iter().enumerate() {
                d[m + p] = left.diagonal[j].conj() * d[j];
            }
            zero.iter().map(|&j| left.rows[j]).collect()
        } else {
            // Unpaired output is I_2 x C_O. For Exact, C_O=I recursively,
            // so the root returns identity corrections and its W equals U.
            d.extend_from_within(..m);
            zero.clone()
        };
        let multiplier = MultiplexedDPMultiplier {
            permutation: MultiplexedPermutation { zero, one },
            diagonal: d,
        };
        Ok((circuit, multiplier))
    }
}

pub(crate) fn synthesize(u: MatRef<'_, C>, options: Options) -> Result<SynthesisResult> {
    let n = u.nrows().ilog2() as usize;
    let synthesizer = Synthesizer {
        basis: kak::CxBasis::get()?,
        atol: options.atol,
        method: options.method,
        solver: options.solver,
    };
    let (circuit, multiplier) =
        synthesizer.synthesize(u, &(0..n).collect::<Vec<_>>(), options.kind, None)?;
    Ok(SynthesisResult {
        circuit: circuit.into_circuit(n),
        multiplier: (!options.kind.is_exact()).then_some(multiplier),
    })
}
