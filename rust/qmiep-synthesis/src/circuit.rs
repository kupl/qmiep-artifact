use crate::{C, Error, Mat, Result};
use serde::{Serialize, Serializer};
use std::f64::consts::FRAC_1_SQRT_2;

/// Primitive instructions in execution order. Qubit 0 is the most significant bit.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "gate", rename_all = "snake_case")]
pub enum Instruction {
    U {
        qubit: usize,
        theta: f64,
        phi: f64,
        lambda: f64,
    },
    Rz {
        qubit: usize,
        angle: f64,
    },
    H {
        qubit: usize,
    },
    Cx {
        control: usize,
        target: usize,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct Circuit {
    pub num_qubits: usize,
    pub instructions: Vec<Instruction>,
    pub global_phase: f64,
}

impl Circuit {
    /// Count CX gates after checking each primitive's qubit operands.
    pub fn cnot_count(&self) -> Result<usize> {
        let mut count = 0;
        for instruction in &self.instructions {
            match *instruction {
                Instruction::U { qubit, .. }
                | Instruction::Rz { qubit, .. }
                | Instruction::H { qubit } => {
                    if qubit >= self.num_qubits {
                        return Err(Error(format!("qubit {qubit} is out of range")));
                    }
                }
                Instruction::Cx { control, target } => {
                    if control >= self.num_qubits || target >= self.num_qubits || control == target
                    {
                        return Err(Error(format!("invalid CX operands ({control}, {target})")));
                    }
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Apply W, including global phase, without constructing a dense operator.
    /// TODO : delete later if not necessary
    pub fn apply(&self, state: &mut [C]) -> Result<()> {
        if self.num_qubits >= usize::BITS as usize || state.len() != 1usize << self.num_qubits {
            return Err(Error("state dimension does not match circuit".into()));
        }
        let bit = |q: usize| -> Result<usize> {
            if q >= self.num_qubits {
                return Err(Error("qubit is out of range".into()));
            }
            Ok(1 << (self.num_qubits - 1 - q))
        };
        let zero = C::new(0.0, 0.0);
        for op in &self.instructions {
            if let &Instruction::Cx { control, target } = op {
                let (c, t) = (bit(control)?, bit(target)?);
                if c == t {
                    return Err(Error("CX control and target must differ".into()));
                }
                for i in 0..state.len() {
                    if i & c != 0 && i & t == 0 {
                        state.swap(i, i | t);
                    }
                }
                continue;
            }
            let (q, m) = match *op {
                Instruction::U {
                    qubit,
                    theta,
                    phi,
                    lambda,
                } => (qubit, crate::euler::matrix(theta, phi, lambda)),
                Instruction::Rz { qubit, angle } => (
                    qubit,
                    [
                        [C::from_polar(1.0, -angle / 2.0), zero],
                        [zero, C::from_polar(1.0, angle / 2.0)],
                    ],
                ),
                Instruction::H { qubit } => {
                    let s = C::new(FRAC_1_SQRT_2, 0.0);
                    (qubit, [[s, s], [s, -s]])
                }
                Instruction::Cx { .. } => unreachable!(),
            };
            let mask = bit(q)?;
            for i in 0..state.len() {
                if i & mask == 0 {
                    let (a, b) = (state[i], state[i | mask]);
                    state[i] = m[0][0] * a + m[0][1] * b;
                    state[i | mask] = m[1][0] * a + m[1][1] * b;
                }
            }
        }
        let phase = C::from_polar(1.0, self.global_phase);
        for z in state {
            *z *= phase;
        }
        Ok(())
    }
    /// TODO : delete later if not necessary
    pub fn unitary(&self) -> Result<Mat<C>> {
        let n = 1usize
            .checked_shl(self.num_qubits as u32)
            .ok_or_else(|| Error("too many qubits to reconstruct".into()))?;
        let mut u = Mat::zeros(n, n);
        for j in 0..n {
            let mut state = vec![C::new(0.0, 0.0); n];
            state[j] = C::new(1.0, 0.0);
            self.apply(&mut state)?;
            for i in 0..n {
                u[(i, j)] = state[i];
            }
        }
        Ok(u)
    }
}

/// P = P₁ ⊕ P₂. The most significant qubit selects a permutation of the remaining qubits' basis states.
#[derive(Clone, Debug)]
pub struct MultiplexedPermutation {
    /// Images under P₁ in the selector-zero sector, indexed locally from zero.
    pub zero: Vec<usize>,
    /// Images under P₂ in the selector-one sector, indexed locally from zero.
    pub one: Vec<usize>,
}

impl MultiplexedPermutation {
    /// Full-register images p, with P|j⟩ = |p[j]⟩.
    pub fn images(&self) -> impl Iterator<Item = usize> + '_ {
        let m = self.zero.len();
        self.zero
            .iter()
            .copied()
            .chain(self.one.iter().map(move |&j| m + j))
    }
}

impl Serialize for MultiplexedPermutation {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_seq(self.images())
    }
}

/// The synthesis output satisfies U = D (P₁ ⊕ P₂) W, where W is the circuit
/// unitary and D is an unrestricted diagonal unitary on the full register.
#[derive(Clone, Debug, Serialize)]
pub struct MultiplexedDPMultiplier {
    pub permutation: MultiplexedPermutation,
    pub diagonal: Vec<C>,
}

impl MultiplexedDPMultiplier {
    fn apply(&self, state: &[C]) -> Result<Vec<C>> {
        let n = state.len();
        let m = n / 2;
        if !n.is_multiple_of(2)
            || self.permutation.zero.len() != m
            || self.permutation.one.len() != m
            || self.diagonal.len() != n
        {
            return Err(Error("multiplier dimension mismatch".into()));
        }
        let mut seen = vec![false; n];
        let mut result = vec![C::new(0.0, 0.0); n];
        for (sector, block) in [&self.permutation.zero, &self.permutation.one]
            .into_iter()
            .enumerate()
        {
            let offset = sector * m;
            for (j, &p) in block.iter().enumerate() {
                if p >= m || seen[offset + p] {
                    return Err(Error("invalid multiplier permutation".into()));
                }
                seen[offset + p] = true;
                result[offset + p] = self.diagonal[offset + p] * state[offset + j];
            }
        }
        Ok(result)
    }
}

/// Result of synthesizing a target U as U = D P W.
#[derive(Clone, Debug, Serialize)]
pub struct SynthesisResult {
    /// Emitted circuit implementing W, including its global phase.
    pub circuit: Circuit,
    /// Left multiplier for reconstructing U from W.
    /// - Exact synthesis returns `None` because W = U.
    /// - D-equivalent synthesis returns `Some` with P = I, so U = D W.
    /// - PD-equivalent synthesis returns `Some` with U = D P W.
    #[serde(rename = "correction")]
    pub multiplier: Option<MultiplexedDPMultiplier>,
}

impl SynthesisResult {
    /// Apply the synthesized U = D P W to a state.
    pub fn apply(&self, state: &[C]) -> Result<Vec<C>> {
        let mut state = state.to_vec();
        self.circuit.apply(&mut state)?;
        match &self.multiplier {
            Some(multiplier) => multiplier.apply(&state),
            None => Ok(state),
        }
    }

    /// Reconstruct U explicitly; this is never part of synthesis timing.
    pub fn reconstruct(&self) -> Result<Mat<C>> {
        let w = self.circuit.unitary()?;
        let Some(multiplier) = &self.multiplier else {
            return Ok(w);
        };
        let n = w.nrows();
        let mut u = Mat::zeros(n, n);
        for j in 0..n {
            let state: Vec<_> = (0..n).map(|i| w[(i, j)]).collect();
            let corrected = multiplier.apply(&state)?;
            for i in 0..n {
                u[(i, j)] = corrected[i];
            }
        }
        Ok(u)
    }
}
