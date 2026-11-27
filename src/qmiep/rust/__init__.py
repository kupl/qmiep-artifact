"""Rust spectrum solving and synthesis through PyO3, NumPy and Cirq."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, overload

import cirq
import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmiep.matrices import DiagonalMatrix, PermutationMatrix

if TYPE_CHECKING:
    from qmiep.rust._rust import Instruction

__all__ = [
    "solve_spectrum",
    "synthesize_unitary",
    "synthesize_unitary_diagonal",
    "synthesize_unitary_pd",
]

type Algorithm = Literal["analytic", "heuristic", "heuristic-opt"]
type OutputMode = Literal["exact", "PD-equiv", "D-equiv"]
ALGORITHMS: tuple[Algorithm, ...] = ("analytic", "heuristic", "heuristic-opt")
OUTPUT_MODES: tuple[OutputMode, ...] = ("exact", "PD-equiv", "D-equiv")

type Method = Literal["analytic", "heuristic", "baseline"]
type ExactMode = Literal["exact", "exact_further_D_optimized"]
type DiagonalMode = Literal["D_naive", "D_equivalent_further_optimal"]
type PDMode = Literal["pd", "pd_further_D_optimized"]


def solve_spectrum(
    unitary: ArrayLike, *, atol: float = 1e-10, seed: int | None = None
) -> tuple[PermutationMatrix, DiagonalMatrix]:
    """Return left multipliers P,D from Rust; all solving and verification is native."""
    from qmiep.rust import _rust

    rows, phases, success, error, *_ = _rust.solve_spectrum(
        np.asarray(unitary, dtype=np.complex128), atol=atol, seed=seed
    )
    if not success:
        raise RuntimeError(
            f"No verified antipodal solution within the search budget (error {error:.3g}). "
            "This does not establish nonexistence."
        )
    return PermutationMatrix.from_permutation(rows), DiagonalMatrix(phases)


@overload
def synthesize_unitary(
    unitary: ArrayLike,
    *,
    qubits: Sequence[cirq.Qid] | None = None,
    algorithm: Algorithm | None = None,
    mode: ExactMode = "exact",
    method: Method | None = None,
    seed: int | None = None,
    atol: float = 1e-10,
    check_input: bool = True,
) -> cirq.Circuit: ...


@overload
def synthesize_unitary(
    unitary: ArrayLike,
    *,
    qubits: Sequence[cirq.Qid] | None = None,
    algorithm: Algorithm | None = None,
    mode: Literal["PD-equiv"],
    seed: int | None = None,
    atol: float = 1e-10,
    check_input: bool = True,
) -> tuple[cirq.Circuit, NDArray[np.uintp], NDArray[np.complex128]]: ...


@overload
def synthesize_unitary(
    unitary: ArrayLike,
    *,
    qubits: Sequence[cirq.Qid] | None = None,
    algorithm: Algorithm | None = None,
    mode: Literal["D-equiv"],
    seed: int | None = None,
    atol: float = 1e-10,
    check_input: bool = True,
) -> tuple[cirq.Circuit, NDArray[np.complex128]]: ...


def synthesize_unitary(
    unitary: ArrayLike,
    *,
    qubits: Sequence[cirq.Qid] | None = None,
    algorithm: Algorithm | None = None,
    mode: ExactMode | OutputMode = "exact",
    method: Method | None = None,
    seed: int | None = None,
    atol: float = 1e-10,
    check_input: bool = True,
) -> (
    cirq.Circuit
    | tuple[cirq.Circuit, NDArray[np.complex128]]
    | tuple[cirq.Circuit, NDArray[np.uintp], NDArray[np.complex128]]
):
    """Synthesize a unitary with Algorithms 1–3, including global phase.

    Select analytic (default), heuristic, or heuristic-opt with `algorithm`.
    `mode="exact"` returns the circuit W implementing U. `"PD-equiv"`
    returns (W, p, d) with U = D P W and P|j⟩ = |p[j]⟩. `"D-equiv"`
    returns (W, d) with U = D W. In both cases D = diag(d).
    Qubits are ordered most significant first. The seed controls the QMIEP
    search. Legacy exact calls using method or exact_further_D_optimized
    are also accepted.
    """
    if method is not None or mode == "exact_further_D_optimized":
        if algorithm is not None or mode not in ("exact", "exact_further_D_optimized"):
            raise ValueError(
                "use algorithm/mode or the legacy exact mode/method separately"
            )
        circuit, _, _ = _synthesize(
            unitary,
            kind=mode,
            method=method,
            qubits=qubits,
            seed=seed,
            atol=atol,
            check_input=check_input,
        )
        return circuit
    circuit, p, d = _synthesize(
        unitary,
        algorithm=algorithm,
        mode=mode,
        qubits=qubits,
        seed=seed,
        atol=atol,
        check_input=check_input,
    )
    if mode == "PD-equiv":
        return circuit, p, d
    if mode == "D-equiv":
        return circuit, d
    return circuit


def synthesize_unitary_diagonal(
    unitary: ArrayLike,
    *,
    qubits: Sequence[cirq.Qid] | None = None,
    mode: DiagonalMode = "D_naive",
    algorithm: Algorithm | None = None,
    method: Method | None = None,
    seed: int | None = None,
    atol: float = 1e-10,
    check_input: bool = True,
) -> tuple[cirq.Circuit, NDArray[np.complex128]]:
    r"""Return ``(circuit, d)`` with the left correction :math:`U=DW`.

    ``W`` includes global phase and ``D = np.diag(d)`` has length ``2**n``.
    Thus ``U = d[:, None] * W``; no permutation correction is required.
    Options and qubit order are the same as in :func:`synthesize_unitary`.
    ``mode="D_naive"`` retains terminal-only D recursion.
    ``mode="D_equivalent_further_optimal"`` tries identity-permutation QMIEP
    pairing in a single pass. L/M permit PD and O requires D. Identity pairing
    uses the short R and CZ with a D-equivalent N. A nonidentity permutation
    selects the full R and allows N's PD correction to migrate into O.
    It does not guarantee minimum CX cost. It performs general QMIEP solves
    even with ``method="analytic"``; ``method="baseline"`` is incompatible.
    """
    if algorithm is not None:
        if mode != "D_naive" or method is not None:
            raise ValueError(
                "use algorithm or the legacy diagonal mode/method separately"
            )
        return synthesize_unitary(
            unitary,
            algorithm=algorithm,
            mode="D-equiv",
            qubits=qubits,
            seed=seed,
            atol=atol,
            check_input=check_input,
        )
    if mode not in ("D_naive", "D_equivalent_further_optimal"):
        raise ValueError("mode must be 'D_naive' or 'D_equivalent_further_optimal'")
    circuit, _, diagonal = _synthesize(
        unitary,
        kind="diagonal" if mode == "D_naive" else mode,
        qubits=qubits,
        method=method,
        seed=seed,
        atol=atol,
        check_input=check_input,
    )
    return circuit, diagonal


def synthesize_unitary_pd(
    unitary: ArrayLike,
    *,
    qubits: Sequence[cirq.Qid] | None = None,
    algorithm: Algorithm | None = None,
    method: Method | None = None,
    seed: int | None = None,
    atol: float = 1e-10,
    check_input: bool = True,
) -> tuple[cirq.Circuit, NDArray[np.uintp], NDArray[np.complex128]]:
    r"""Return ``(circuit, p, d)`` with the left correction :math:`U=DPW`.

    ``W`` is the circuit's matrix, including global phase. Both corrections
    are length-``2**n`` NumPy arrays: ``p`` contains permutation images,
    :math:`P|j\rangle=|p[j]\rangle`, and ``d`` contains the diagonal entries
    of :math:`D`. Thus ``P = np.eye(len(p))[:, p]``, ``D = np.diag(d)``, and
    ``U = d[:, None] * W[np.argsort(p), :]``. These are reconstruction
    factors, not the left multipliers returned by the spectrum solver.

    Options and qubit order are the same as in :func:`synthesize_unitary`.
    One-qubit synthesis returns identity corrections. For two qubits, and
    for ``method="baseline"``, the permutation is identity.
    """
    if algorithm is not None:
        if method is not None:
            raise ValueError("use algorithm or the legacy method separately")
        return synthesize_unitary(
            unitary,
            algorithm=algorithm,
            mode="PD-equiv",
            qubits=qubits,
            seed=seed,
            atol=atol,
            check_input=check_input,
        )
    return _synthesize(
        unitary,
        kind="pd",
        qubits=qubits,
        method=method,
        seed=seed,
        atol=atol,
        check_input=check_input,
    )


def _synthesize(
    unitary: ArrayLike,
    *,
    algorithm: Algorithm | None = None,
    mode: OutputMode | None = None,
    kind: (
        Literal[
            "exact",
            "exact_further_D_optimized",
            "diagonal",
            "pd",
            "pd_further_D_optimized",
            "D_equivalent_further_optimal",
        ]
        | None
    ) = None,
    qubits: Sequence[cirq.Qid] | None,
    method: Method | None = None,
    seed: int | None,
    atol: float,
    check_input: bool,
) -> tuple[cirq.Circuit, NDArray[np.uintp], NDArray[np.complex128]]:
    from qmiep.rust import _rust

    matrix = np.asarray(unitary, dtype=np.complex128)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
        or matrix.shape[0] & (matrix.shape[0] - 1)
    ):
        raise ValueError("unitary must be a square 2**n matrix with n >= 1")
    n = matrix.shape[0].bit_length() - 1
    qubits = tuple(cirq.LineQubit.range(n) if qubits is None else qubits)
    if (
        len(qubits) != n
        or any(not isinstance(q, cirq.Qid) or q.dimension != 2 for q in qubits)
        or len(set(qubits)) != n
    ):
        raise ValueError(f"qubits must contain {n} distinct dimension-2 Cirq Qids")
    instructions, phase, p, d = _rust.synthesize(
        matrix,
        algorithm=algorithm,
        mode=mode,
        kind=kind,
        method=method,
        seed=seed,
        atol=atol,
        check_input=check_input,
    )
    return _circuit_from_instructions(instructions, phase, qubits), p, d


def _count_native_cnots(instructions: Sequence["Instruction"], num_qubits: int) -> int:
    """Validate and count exported gates without constructing a Cirq circuit."""
    count = 0
    for instruction in instructions:
        match instruction:
            case ("cx", control, target) if control != target and (
                0 <= control < num_qubits and 0 <= target < num_qubits
            ):
                count += 1
            case ("h", q) | ("rz", q, _) | ("u", q, _, _, _) if 0 <= q < num_qubits:
                continue
            case _:
                raise ValueError(
                    f"Expected a one-qubit unitary or CX, got {instruction!r}"
                )
    return count


@dataclass(frozen=True, slots=True)
class _NativeCircuit:
    """Compiled gates in logical qubit order, with their global phase."""

    num_qubits: int
    instructions: tuple["Instruction", ...]
    phase: float
    transposed: bool = False

    def transpose(self) -> "_NativeCircuit":
        return replace(self, transposed=not self.transposed)

    def to_cirq(self, qubits: Sequence[cirq.Qid]) -> cirq.FrozenCircuit:
        return _circuit_from_instructions(
            self.instructions, self.phase, qubits, transposed=self.transposed
        ).freeze()


def _circuit_from_instructions(
    instructions: Sequence["Instruction"],
    phase: float,
    qubits: Sequence[cirq.Qid],
    *,
    transposed: bool = False,
) -> cirq.Circuit:
    """Emit native instructions as a Cirq circuit, including global phase."""
    operations = []
    # H, Rz and CX are symmetric, and transposing preserves the global phase.
    ordered = reversed(instructions) if transposed else instructions
    for instruction in ordered:
        match instruction:
            case ("cx", control, target):
                operations.append(cirq.CX(qubits[control], qubits[target]))
            case ("h", qubit):
                operations.append(cirq.H(qubits[qubit]))
            case ("rz", qubit, angle):
                operations.append(cirq.rz(angle)(qubits[qubit]))
            case ("u", qubit, theta, phi, lam):
                cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
                gate_matrix = np.array(
                    [
                        [cosine, -np.exp(1j * lam) * sine],
                        [np.exp(1j * phi) * sine, np.exp(1j * (phi + lam)) * cosine],
                    ]
                )
                gate = cirq.MatrixGate(gate_matrix.T if transposed else gate_matrix)
                operations.append(gate(qubits[qubit]))
            case _:
                raise RuntimeError(f"unknown Rust instruction: {instruction!r}")
    operations.append(cirq.global_phase_operation(np.exp(1j * phase)))
    return cirq.Circuit(operations)
