use crate::{Circuit, Instruction};

/// A replacement for a unitary on the original circuit's qubit indices.
pub(crate) enum CircuitFragment {
    Gate(Instruction),
    Block {
        instructions: Vec<Instruction>,
        global_phases: Vec<f64>,
    },
    Sequence(Vec<Self>),
}

impl CircuitFragment {
    /// Flatten the replacements once, in circuit order, including each leaf's global phases.
    pub(crate) fn into_circuit(self, num_qubits: usize) -> Circuit {
        let mut circuit = Circuit {
            num_qubits,
            instructions: Vec::with_capacity(self.instruction_count()),
            global_phase: 0.0,
        };
        let mut compensation = 0.0;
        self.emit(&mut circuit, &mut compensation);
        circuit
    }

    fn instruction_count(&self) -> usize {
        match self {
            Self::Gate(_) => 1,
            Self::Block { instructions, .. } => instructions.len(),
            Self::Sequence(parts) => parts.iter().map(Self::instruction_count).sum(),
        }
    }

    fn emit(self, circuit: &mut Circuit, compensation: &mut f64) {
        match self {
            Self::Gate(gate) => circuit.instructions.push(gate),
            Self::Block {
                instructions,
                global_phases,
            } => {
                circuit.instructions.extend(instructions);
                // Keep compensated summation in leaf order, including across block boundaries.
                for phase in global_phases {
                    let y = phase - *compensation;
                    let total = circuit.global_phase + y;
                    *compensation = (total - circuit.global_phase) - y;
                    circuit.global_phase = total;
                }
            }
            Self::Sequence(parts) => {
                for part in parts {
                    part.emit(circuit, compensation);
                }
            }
        }
    }
}
