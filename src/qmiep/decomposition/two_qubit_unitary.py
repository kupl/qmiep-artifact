"""Two-qubit synthesis with an optional left diagonal correction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import overload

import cirq
import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit.circuit.library import CXGate
from qiskit.synthesis import TwoQubitBasisDecomposer, TwoQubitWeylDecomposition
from scipy.linalg import polar

from qmiep.gates import DiagonalGate, NQubitUnitaryModuloDiagonal
from qmiep.matrices import DiagonalMatrix
from qmiep.utils import _unitary

__all__ = ["decompose_two_qubit_unitary"]

_YY = np.kron(cirq.unitary(cirq.Y), cirq.unitary(cirq.Y))
_DELTA_SIGNS = np.array([-1.0, 1.0, 1.0, -1.0])
_CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())
_CX_PHASE = TwoQubitWeylDecomposition(CXGate().to_matrix()).global_phase

GatePair = tuple[NQubitUnitaryModuloDiagonal, DiagonalGate]
OperationPair = tuple[cirq.Operation, cirq.Operation]


def _left_diagonal(
    unitary: NDArray[np.complex128], *, atol: float
) -> NDArray[np.complex128]:
    r"""Return Appendix A's :math:`\Delta` such that :math:`\Delta U` is 2-CNOT.

    Remove the global phase from :math:`U`, denote the resulting member of
    :math:`SU(4)` by :math:`\bar U`, and define

    .. math::

        \gamma(\bar U)=\bar U(Y\otimes Y)\bar U^T(Y\otimes Y),
        \qquad
        \operatorname{diag}(\gamma(\bar U))=(t_1,t_2,t_3,t_4).

    Appendix A, Step 1 chooses

    .. math::

        \psi=\operatorname{atan2}\!\left(
          \Im\sum_j t_j,\ \Re(t_1-t_2-t_3+t_4)
        \right),

    and :math:`\Delta=\operatorname{diag}(e^{-i\psi/2},e^{i\psi/2},
    e^{i\psi/2},e^{-i\psi/2})`.  Consequently
    :math:`\operatorname{tr}(\gamma(\Delta\bar U))` is real, which is the
    two-CNOT criterion used by the magic-basis/KAK synthesis.
    """
    determinant_phase = np.angle(np.linalg.det(unitary))
    special = np.exp(-0.25j * determinant_phase) * unitary
    gamma = special @ _YY @ special.T @ _YY
    t1, t2, t3, t4 = np.diag(gamma)
    numerator = float(np.imag(t1 + t2 + t3 + t4))
    denominator = float(np.real(t1 - t2 - t3 + t4))
    psi = (
        0.0
        if abs(numerator) <= atol and abs(denominator) <= atol
        else float(np.arctan2(numerator, denominator))
    )
    return np.exp(0.5j * psi * _DELTA_SIGNS)


def _residual_label(label: str) -> str:
    r"""Return the :math:`\widehat X` name of a modulo-diagonal residual."""
    base = label.rstrip("'")
    return f"{base}\N{COMBINING CIRCUMFLEX ACCENT}{label[len(base):]}"


def _kak_factors(
    residual: NDArray[np.complex128], *, atol: float
) -> tuple[tuple[NDArray[np.complex128], ...], float]:
    """Return alternating local factors (little endian) and the global phase."""
    # Project recursive roundoff, but validate against the original residual.
    # fidelity=None prevents Qiskit from rounding to a nearby special gate.
    target = TwoQubitWeylDecomposition(polar(residual)[0], fidelity=None)
    tolerance = atol / 4
    if abs(target.a) + abs(target.b) + abs(target.c) <= tolerance:
        cx_count, decompose = 0, _CX_DECOMPOSER.decomp0
    elif abs(target.a - np.pi / 4) + abs(target.b) + abs(target.c) <= tolerance:
        cx_count, decompose = 1, _CX_DECOMPOSER.decomp1
    elif abs(target.c) <= tolerance:
        cx_count, decompose = 2, _CX_DECOMPOSER.decomp2_supercontrolled
    else:
        cx_count, decompose = 3, _CX_DECOMPOSER.decomp3_supercontrolled

    # Qiskit's basis-factor methods consume the native Weyl decomposition.
    matrices = decompose(target._inner_decomposition)
    phase = target.global_phase - cx_count * _CX_PHASE
    if cx_count == 2:
        phase += np.pi
    return tuple(matrices), phase


def _synthesize(
    residual: NDArray[np.complex128], *, label: str, atol: float
) -> NQubitUnitaryModuloDiagonal:
    """Emit Cirq gates from exact KAK factors, including their global phase."""
    q0, q1 = cirq.LineQubit(0), cirq.LineQubit(1)
    qubits = (q0, q1)
    matrices, phase = _kak_factors(residual, atol=atol)
    cx_count = len(matrices) // 2 - 1
    operations: list[cirq.Operation] = []
    for index in range(cx_count + 1):
        # Qiskit's q0 is the least significant qubit, hence Cirq's q1.
        operations.extend(
            cirq.MatrixGate(matrix, unitary_check=False).on(qubit)
            for matrix, qubit in zip(
                matrices[2 * index : 2 * index + 2], (q1, q0), strict=True
            )
        )
        if index < cx_count:
            operations.append(cirq.CX(q1, q0))
    operations.append(cirq.global_phase_operation(np.exp(1j * phase)))
    return NQubitUnitaryModuloDiagonal(
        residual,
        cirq.Circuit(operations),
        qubit_order=qubits,
        label=label,
        atol=atol,
    )


def _validate_qubits(qubits: Sequence[cirq.Qid]) -> tuple[cirq.Qid, cirq.Qid]:
    ordered = tuple(qubits)
    if len(ordered) != 2:
        raise ValueError("qubits must contain exactly two qids")
    if not all(isinstance(qubit, cirq.Qid) for qubit in ordered):
        raise TypeError("qubits must contain only cirq.Qid instances")
    if ordered[0] == ordered[1]:
        raise ValueError("qubits must not contain duplicate qids")
    if any(qubit.dimension != 2 for qubit in ordered):
        raise ValueError("qubits must contain only two-dimensional qids")
    return ordered


@overload
def decompose_two_qubit_unitary(
    matrix: ArrayLike,
    qubits: None = None,
    *,
    upto_diagonal: bool = True,
    label: str = "U",
    atol: float = 1e-10,
) -> GatePair: ...


@overload
def decompose_two_qubit_unitary(
    matrix: ArrayLike,
    qubits: Sequence[cirq.Qid],
    *,
    upto_diagonal: bool = True,
    label: str = "U",
    atol: float = 1e-10,
) -> OperationPair: ...


def decompose_two_qubit_unitary(
    matrix: ArrayLike,
    qubits: Sequence[cirq.Qid] | None = None,
    *,
    upto_diagonal: bool = True,
    label: str = "U",
    atol: float = 1e-10,
) -> GatePair | OperationPair:
    r"""Decompose a two-qubit unitary in Cirq circuit-time order.

    With ``upto_diagonal=True``, the result is a residual gate :math:`W`
    containing at most two CNOTs followed by a diagonal correction :math:`D`:

    .. math::

        W=\Delta U,\qquad D=\Delta^\dagger,\qquad U=DW.

    Thus the returned order is ``(W, D)``: putting the two results directly in
    a :class:`cirq.Circuit` reconstructs :math:`U`.  Without ``qubits``, both
    results are :class:`cirq.Gate` objects.  Supplying two qubits returns the
    corresponding :class:`cirq.Operation` objects on those qubits.

    ``label`` names the decomposed :math:`U`; the residual is labelled
    :math:`\widehat U` so modulo-diagonal residuals are visually distinct from
    modulo-permutation-diagonal residuals :math:`\widetilde U`.

    With ``upto_diagonal=False``, :math:`D=I` and the same stable two-result
    interface is retained; the residual may then contain three CNOTs.
    """
    unitary = _unitary(matrix, atol=atol)
    if unitary.shape != (4, 4):
        raise ValueError("expected a two-qubit unitary with shape (4, 4)")
    ordered_qubits = None if qubits is None else _validate_qubits(qubits)

    if upto_diagonal and cirq.is_diagonal(unitary, atol=atol):
        delta = np.diag(unitary).conj()
    elif upto_diagonal:
        delta = _left_diagonal(unitary, atol=atol)
    else:
        delta = np.ones(4, dtype=np.complex128)
    residual = delta[:, np.newaxis] * unitary
    residual_gate = _synthesize(residual, label=_residual_label(label), atol=atol)
    diagonal_gate = DiagonalGate(
        DiagonalMatrix.from_diagonal(delta.conj(), atol=atol), label="D"
    )
    gates = residual_gate, diagonal_gate
    if ordered_qubits is None:
        return gates

    return gates[0].on(*ordered_qubits), gates[1].on(*ordered_qubits)
