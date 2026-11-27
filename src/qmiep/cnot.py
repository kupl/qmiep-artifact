"""Count CNOTs in circuits over one-qubit unitaries and CX."""

from collections.abc import Iterable

import cirq
from qiskit.circuit import Gate, QuantumCircuit  # type: ignore[import-untyped]
from qiskit.circuit.library import (  # type: ignore[import-untyped]
    CXGate,
    GlobalPhaseGate,
)

from qmiep.gates import NQubitUnitary
from qmiep.rust import _count_native_cnots


def count_cirq_cnots(operations: Iterable[cirq.Operation]) -> int:
    """Check each emitted operation, ignoring explicit global phases."""
    count = 0
    for op in operations:
        if op.gate == cirq.CX:
            count += 1
        elif isinstance(op.gate, cirq.GlobalPhaseGate):
            continue
        elif not (
            len(op.qubits) == 1
            and op.qubits[0].dimension == 2
            and op.gate is not None
            and cirq.has_unitary(op)
        ):
            raise ValueError(f"Expected a one-qubit unitary or CX, got {op!r}")
    return count


def count_decomposed_cnots(operation: cirq.Operation) -> int:
    """Count the emitted primitives of a synthesis, including native instruction blocks."""
    gate = operation.gate
    if isinstance(gate, NQubitUnitary) and gate._native_circuit is not None:
        return _count_native_cnots(
            gate._native_circuit.instructions, len(operation.qubits)
        )
    if len(operation.qubits) <= 1 or gate == cirq.CX:
        return count_cirq_cnots((operation,))
    return sum(
        count_decomposed_cnots(child) for child in cirq.decompose_once(operation)
    )


def count_qiskit_cnots(circuit: QuantumCircuit) -> int:
    """Check the emitted gates, ignoring the circuit's global phase."""
    count = 0
    for instruction in circuit.data:
        gate = instruction.operation
        if isinstance(gate, CXGate) and gate.ctrl_state == 1:
            count += 1
        elif isinstance(gate, GlobalPhaseGate):
            continue
        elif not (isinstance(gate, Gate) and gate.num_qubits == 1):
            raise ValueError(f"Expected a one-qubit unitary or CX, got {gate.name!r}")
    return count
