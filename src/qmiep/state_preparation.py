from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cirq
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig

from qmiep._validation import _validation_enabled, _validation_scope
from qmiep.gates import Isometry, UnresolvedNQubitUnitary
from qmiep.rust import Algorithm
from qmiep.synthesizer import (
    IsometrySynthesize,
    NQubitUnitarySynthesisModuloDiagonal,
    NQubitUnitarySynthesisModuloPermutationDiagonal,
    SynthesisBackends,
)
from qmiep.synthesizer._backends import SynthesisMode, _algorithm_scope, _backend_scope


@dataclass(frozen=True)
class StatePreparationResult:
    circuit: cirq.FrozenCircuit
    state_vector: NDArray[np.complex128] | None
    schmidt_coefficients: NDArray[np.float64] | None = None
    e_basis: NDArray[np.complex128] | None = None
    f_basis: NDArray[np.complex128] | None = None
    label_map: NDArray[np.complex128] | None = None
    e_circuit: cirq.FrozenCircuit | None = None
    e_state_vector: NDArray[np.complex128] | None = None
    f_unitary: NDArray[np.complex128] | None = None
    f_diagonal: NDArray[np.complex128] | None = None
    f_circuit: cirq.FrozenCircuit | None = None
    f_state_vector: NDArray[np.complex128] | None = None
    cnot_circuit: cirq.FrozenCircuit | None = None
    reduction_circuit: cirq.FrozenCircuit | None = None
    reduced_state_vector: NDArray[np.complex128] | None = None
    recursive_result: StatePreparationResult | None = None


def state_preparation(
    state_vector: NDArray[np.complex128],
    *,
    atol: float = 1e-10,
    backends: SynthesisBackends | None = None,
    algorithm: Algorithm | None = None,
    mode: Literal["exact"] | SynthesisMode = "exact",
    seed: int | None = None,
    validate: bool | None = None,
) -> StatePreparationResult:
    """Prepare the specified complex amplitudes, including global phase.

    The selected algorithm (default heuristic) is used throughout the Schmidt
    recursion, with PD-equiv synthesis on the E register and D-equiv on F.
    Legacy mode="analytic" and mode="heuristic" select the algorithm.
    Validation defaults to on. With ``validate=False``, recursive reconstruction
    checks are skipped and the simulated state-vector fields are ``None``.
    """
    if mode != "exact":
        if mode not in ("analytic", "heuristic") or algorithm is not None:
            raise ValueError(
                "state preparation uses mode='exact'; select the synthesis algorithm with algorithm"
            )
        algorithm = mode
    backends = SynthesisBackends() if backends is None else backends
    with (
        _backend_scope(backends),
        _algorithm_scope("heuristic" if algorithm is None else algorithm, seed),
        _validation_scope(validate),
    ):
        return _state_preparation(state_vector, atol=atol)


def _state_preparation(
    state_vector: NDArray[np.complex128], *, atol: float
) -> StatePreparationResult:
    state = np.asarray(state_vector, dtype=np.complex128)
    if state.ndim != 1 or state.size < 2 or state.size & (state.size - 1):
        raise ValueError("state_vector must be a 1-D vector of length 2**n, n >= 1")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    if not np.all(np.isfinite(state)) or not np.isclose(
        np.linalg.norm(state), 1, rtol=0, atol=atol
    ):
        raise ValueError("state_vector must be finite and normalized")

    num_qubits = state.size.bit_length() - 1
    qubits = tuple(cirq.LineQubit.range(num_qubits))
    if num_qubits <= 3:
        if num_qubits == 3:
            circuit = three_qubit_state_preparation(state, qubits, atol=atol)
        elif num_qubits == 2:
            circuit = two_qubit_state_preparation(state, qubits, atol=atol)
        else:
            a, b = state / np.linalg.norm(state)
            circuit = cirq.FrozenCircuit(
                cirq.MatrixGate(np.array([[a, -b.conjugate()], [b, a.conjugate()]])).on(
                    qubits[0]
                )
            )
        return StatePreparationResult(
            circuit=circuit,
            state_vector=(
                cirq.final_state_vector(
                    circuit, qubit_order=qubits, dtype=np.complex128
                )
                if _validation_enabled()
                else None
            ),
        )

    e_qubits = qubits[: (num_qubits + 1) // 2]
    f_dimension = 2 ** (num_qubits // 2)

    e_basis, coefficients, vh = np.linalg.svd(
        state.reshape(2 ** len(e_qubits), f_dimension), full_matrices=False
    )
    order = np.argsort(-coefficients, kind="stable")
    coefficients = coefficients[order]
    e_basis = e_basis[:, order]
    f_basis = vh[order].T
    inverse = False
    if num_qubits % 2 == 0:
        target = UnresolvedNQubitUnitary(e_basis.conj().T, label="E†", atol=atol).on(
            *e_qubits
        )
        synthesized = NQubitUnitarySynthesisModuloPermutationDiagonal().run(
            cirq.Circuit(target), target=target
        )
        residual, first, second = synthesized.all_operations()
        label_map = (cirq.unitary(second) @ cirq.unitary(first)).conj().T
    else:
        target = Isometry(e_basis, label="E", atol=atol).on(*e_qubits)
        synthesized = IsometrySynthesize(mode="PD-equiv").run(
            cirq.Circuit(target), target=target
        )
        p, d, residual = tuple(synthesized.all_operations())
        label_map = np.diag(cirq.unitary(d))[:, None] * cirq.unitary(p)
        inverse = True

    e_circuit = cirq.FrozenCircuit(
        cirq.decompose(
            residual, keep=lambda op: len(op.qubits) <= 1 or op.gate == cirq.CX
        )
    )
    if inverse:
        e_circuit = cirq.inverse(e_circuit)

    permutation = np.argmax(np.abs(label_map), axis=0)
    f_unitary = f_basis[:, np.argsort(permutation)]
    f_qubits = qubits[len(e_qubits) :]
    f_operation = UnresolvedNQubitUnitary(
        f_unitary.conj().T, label="U_F†", atol=atol
    ).on(*f_qubits)
    f_synthesized = NQubitUnitarySynthesisModuloDiagonal().run(
        cirq.Circuit(f_operation), target=f_operation
    )
    f_residual, f_correction = tuple(f_synthesized.all_operations())
    phases = np.diag(cirq.unitary(f_correction))
    f_diagonal = np.tile(phases, f_dimension // len(phases))
    f_circuit = cirq.FrozenCircuit(
        cirq.decompose(
            f_residual, keep=lambda op: len(op.qubits) <= 1 or op.gate == cirq.CX
        )
    )

    cnot_circuit = cirq.FrozenCircuit(
        cirq.Moment(cirq.CX(upper, lower))
        for upper, lower in zip(e_qubits[num_qubits % 2 :], f_qubits, strict=True)
    )
    reduction = e_circuit + f_circuit + cnot_circuit

    recursive_result = _state_preparation(
        f_diagonal.conj() * (label_map @ coefficients), atol=atol
    )
    active_qubits = e_qubits[num_qubits % 2 :]
    recursive_circuit = cirq.FrozenCircuit(
        cirq.Circuit(recursive_result.circuit).transform_qubits(
            dict(
                zip(
                    cirq.LineQubit.range(len(active_qubits)), active_qubits, strict=True
                )
            )
        )
    )
    circuit = recursive_circuit + cirq.inverse(reduction)
    e_state = f_state = reduced_state = output = None
    if _validation_enabled():
        e_state = cirq.final_state_vector(
            e_circuit, initial_state=state, qubit_order=qubits, dtype=np.complex128
        )
        f_state = cirq.final_state_vector(
            f_circuit, initial_state=e_state, qubit_order=qubits, dtype=np.complex128
        )
        reduced_state = cirq.final_state_vector(
            cnot_circuit, initial_state=f_state, qubit_order=qubits, dtype=np.complex128
        )
        output = cirq.final_state_vector(
            circuit, qubit_order=qubits, dtype=np.complex128
        )
    return StatePreparationResult(
        schmidt_coefficients=coefficients,
        e_basis=e_basis,
        f_basis=f_basis,
        label_map=label_map,
        e_circuit=e_circuit,
        e_state_vector=e_state,
        f_unitary=f_unitary,
        f_diagonal=f_diagonal,
        f_circuit=f_circuit,
        f_state_vector=f_state,
        cnot_circuit=cnot_circuit,
        reduction_circuit=reduction,
        reduced_state_vector=reduced_state,
        recursive_result=recursive_result,
        circuit=circuit,
        state_vector=output,
    )


def two_qubit_state_preparation(
    state_vector: NDArray[np.complex128],
    qubits: Sequence[cirq.Qid] | None = None,
    *,
    atol: float = 1e-10,
) -> cirq.FrozenCircuit:
    state = np.asarray(state_vector, dtype=np.complex128)
    if state.shape != (4,):
        raise ValueError("state_vector must be a 1-D vector of length 4")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    if not np.all(np.isfinite(state)) or not np.isclose(
        np.linalg.norm(state), 1, rtol=0, atol=atol
    ):
        raise ValueError("state_vector must be finite and normalized")
    qubits = tuple(cirq.LineQubit.range(2) if qubits is None else qubits)
    if (
        len(qubits) != 2
        or any(not isinstance(q, cirq.Qid) or q.dimension != 2 for q in qubits)
        or len(set(qubits)) != 2
    ):
        raise ValueError("qubits must contain two distinct dimension-2 qids")

    left, weights, right = np.linalg.svd(state.reshape(2, 2))
    return cirq.FrozenCircuit(
        cirq.ry(2 * np.arctan2(weights[1], weights[0])).on(qubits[0]),
        cirq.CX(*qubits),
        cirq.MatrixGate(left).on(qubits[0]),
        cirq.MatrixGate(right.T).on(qubits[1]),
    )


def three_qubit_state_preparation(
    state_vector: NDArray[np.complex128],
    qubits: Sequence[cirq.Qid] | None = None,
    *,
    atol: float = 1e-10,
) -> cirq.FrozenCircuit:
    state = np.asarray(state_vector, dtype=np.complex128)
    if state.shape != (8,):
        raise ValueError("state_vector must be a 1-D vector of length 8")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    if not np.all(np.isfinite(state)) or not np.isclose(
        np.linalg.norm(state), 1, rtol=0, atol=atol
    ):
        raise ValueError("state_vector must be finite and normalized")
    qubits = tuple(cirq.LineQubit.range(3) if qubits is None else qubits)
    if (
        len(qubits) != 3
        or len(set(qubits)) != 3
        or any(not isinstance(q, cirq.Qid) or q.dimension != 2 for q in qubits)
    ):
        raise ValueError("qubits must contain three distinct dimension-2 qids")

    reduction = cirq.Circuit()

    def apply(*operations: cirq.Operation) -> None:
        nonlocal state
        fragment = cirq.Circuit(operations)
        state = cirq.final_state_vector(
            fragment, initial_state=state, qubit_order=qubits, dtype=np.complex128
        )
        reduction.append(fragment)

    def basis(ket: NDArray[np.complex128]) -> NDArray[np.complex128]:
        a, b = ket / np.linalg.norm(ket)
        return np.array([[a, -b.conjugate()], [b, a.conjugate()]])

    def align_pair(
        first: NDArray[np.complex128], second: NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        norms = np.linalg.norm(first), np.linalg.norm(second)
        if min(norms) == 0:
            return np.eye(2, dtype=np.complex128)
        first, second = first / norms[0], second / norms[1]
        phase = np.exp(-1j * np.angle(np.vdot(first, second)))
        return cirq.unitary(cirq.H) @ basis(first + phase * second).conj().T

    slices = state.reshape(2, 2, 2)
    roots = eig(-slices[0], slices[1], right=False, homogeneous_eigvals=True)
    candidates = [np.array([1, 0]), np.array([0, 1])]
    candidates.extend(
        root[::-1] / np.linalg.norm(root)
        for root in roots.T
        if np.linalg.norm(root) > 0
    )
    x, y = min(
        candidates,
        key=lambda v: np.linalg.svd(
            v[0] * slices[0] + v[1] * slices[1], compute_uv=False
        )[-1],
    )
    apply(
        cirq.MatrixGate(np.array([[x, y], [-y.conjugate(), x.conjugate()]])).on(
            qubits[0]
        )
    )
    left, _, _ = np.linalg.svd(state.reshape(2, 2, 2)[0])
    apply(cirq.MatrixGate(left.conj().T).on(qubits[1]))

    first, second = state.reshape(2, 2, 2)[1]
    apply(
        cirq.MatrixGate(align_pair(first, second)).on(qubits[2]),
        cirq.CX(qubits[1], qubits[2]),
    )

    factors = [np.linalg.svd(part)[2][0] for part in state.reshape(2, 2, 2)]
    apply(
        cirq.MatrixGate(align_pair(*factors)).on(qubits[2]),
        cirq.CX(qubits[0], qubits[2]),
    )
    _, _, right = np.linalg.svd(state.reshape(4, 2), full_matrices=False)
    apply(cirq.MatrixGate(basis(right[0]).conj().T).on(qubits[2]))

    left, weights, right = np.linalg.svd(state[::2].reshape(2, 2))
    apply(
        cirq.MatrixGate(left.conj().T).on(qubits[0]),
        cirq.MatrixGate(right.conj()).on(qubits[1]),
    )
    apply(
        cirq.CX(qubits[0], qubits[1]),
        cirq.MatrixGate(basis(weights).conj().T).on(qubits[0]),
    )
    return cirq.FrozenCircuit(cirq.inverse(reduction))


__all__ = [
    "StatePreparationResult",
    "state_preparation",
    "two_qubit_state_preparation",
    "three_qubit_state_preparation",
]
