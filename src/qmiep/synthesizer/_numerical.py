"""Numerical unitary recursion followed by a single Cirq emission pass.

The recursion stores permutation images, diagonal entries and primitive
instructions, never Circuit/Moment/atomic-gate objects. A correction (p, d)
always means U = diag(d) P W, with P|j> = |p[j]> and W the emitted prefix.
"""

from collections.abc import Hashable
from dataclasses import dataclass, field
from math import fsum
from typing import Literal

import cirq
import numpy as np
from numpy.typing import NDArray

from qmiep.decomposition import BlockZXZDecomposer
from qmiep.decomposition.two_qubit_unitary import _kak_factors, _left_diagonal
from qmiep.gates.atomic import _fast_walsh_hadamard_transform
from qmiep.rust import solve_spectrum
from qmiep.synthesizer._backends import _synthesis_seed

type Matrix = NDArray[np.complex128]
type Correction = tuple[NDArray[np.intp], Matrix]
type Kind = Literal[
    "exact",
    "exact_further_D_optimized",
    "diagonal",
    "pd",
    "pd_further_D_optimized",
    "D_equivalent_further_optimal",
]
type Mode = Literal["analytic", "heuristic"]
type Instruction = tuple[str, tuple[int, ...], Matrix | float | None]


def _eigenbasis(matrix: Matrix, *, paired: bool, atol: float) -> tuple[Matrix, Matrix]:
    """Use the same eigenbasis and antipodal ordering as the spectral rewrite."""
    spectrum, basis = cirq.linalg.unitary_eig(matrix, atol=atol)
    if paired:
        angles = np.mod(np.angle(spectrum), 2 * np.pi)
        order = np.argsort(angles)
        gaps = np.diff(np.r_[angles[order], angles[order[0]] + 2 * np.pi])
        order = np.roll(order, -int(np.argmax(gaps)) - 1)
        half = len(order) // 2
        if not np.allclose(
            spectrum[order[:half]], -spectrum[order[half:]], rtol=0, atol=atol
        ):
            raise ValueError("the spectrum is not antipodal")
        order = np.column_stack((order[:half], order[half:])).ravel()
        basis = basis[:, order]
        spectrum = np.kron(spectrum[order[::2]], [1, -1])
    return spectrum, basis


def _absorb(matrix: Matrix, correction: Correction) -> Matrix:
    """Compute V D P by scaling columns and selecting their permutation images."""
    permutation, diagonal = correction
    return (matrix * diagonal)[:, permutation]


@dataclass(slots=True)
class _Program:
    instructions: list[Instruction] = field(default_factory=list)
    phases: list[float] = field(default_factory=list)
    native_pd: bool = False
    native_diagonal: bool = False

    def rotation(
        self,
        diagonal: Matrix,
        qubits: tuple[int, ...],
        *,
        side: Literal["left", "right", "closed"],
    ) -> None:
        """Emit an open/reversed/closed Gray cascade from its sector phases."""
        angles = _fast_walsh_hadamard_transform(-2 * np.angle(diagonal)) / len(diagonal)
        fragment: list[Instruction] = []
        for index in range(len(angles)):
            gray = index ^ (index >> 1)
            fragment.append(("rz", qubits[:1], float(angles[gray])))
            next_index = (index + 1) % len(angles)
            next_gray = next_index ^ (next_index >> 1)
            bit = (gray ^ next_gray).bit_length() - 1
            fragment.append(("cx", (qubits[-bit - 1], qubits[0]), None))
        if side != "closed":
            fragment.pop()
        if side == "right":
            fragment.reverse()
        self.instructions.extend(fragment)

    def leaf(
        self, matrix: Matrix, qubits: tuple[int, ...], kind: Kind, atol: float
    ) -> Correction:
        if kind in ("exact", "exact_further_D_optimized"):
            delta = np.ones(4, dtype=complex)
        elif cirq.is_diagonal(matrix, atol=atol):
            delta = np.diag(matrix).conj()
        else:
            delta = _left_diagonal(matrix, atol=atol)
        factors, phase = _kak_factors(delta[:, None] * matrix, atol=atol)
        for index in range(0, len(factors), 2):
            self.instructions.append(("matrix", qubits[1:], factors[index]))
            self.instructions.append(("matrix", qubits[:1], factors[index + 1]))
            if index + 2 < len(factors):
                self.instructions.append(("cx", qubits[::-1], None))
        self.phases.append(phase)
        return np.arange(4), delta.conj()

    def solve(
        self,
        matrix: Matrix,
        qubits: tuple[int, ...],
        kind: Kind,
        mode: Mode,
        atol: float,
    ) -> Correction:
        """Append W's instructions and return its left PD correction."""
        if kind == "exact_further_D_optimized" and mode != "heuristic":
            raise ValueError("exact_further_D_optimized only supports mode='heuristic'")
        if (kind in ("pd", "pd_further_D_optimized") and self.native_pd) or (
            kind in ("diagonal", "D_equivalent_further_optimal")
            and self.native_diagonal
        ):
            from qmiep.rust import _synthesize

            circuit, p, d = _synthesize(
                matrix,
                kind=kind,
                qubits=tuple(cirq.LineQubit(q) for q in qubits),
                method=mode,
                seed=_synthesis_seed(),
                atol=atol,
                check_input=False,
            )
            for op in circuit.all_operations():
                if not op.qubits:
                    self.phases.append(float(np.angle(cirq.unitary(op).item())))
                else:
                    self.instructions.append(
                        (
                            "cx" if op.gate == cirq.CX else "matrix",
                            tuple(q.x for q in op.qubits),
                            None if op.gate == cirq.CX else cirq.unitary(op),
                        )
                    )
            return p, d
        return self._solve_node(matrix, qubits, kind, mode, atol)

    def _solve_node(
        self,
        matrix: Matrix,
        qubits: tuple[int, ...],
        kind: Kind,
        mode: Mode,
        atol: float,
    ) -> Correction:
        """Build one node, propagating PD-permitted or D-required constraints."""
        n = len(qubits)
        if n == 2:
            return self.leaf(matrix, qubits, kind, atol)

        factors = BlockZXZDecomposer().solve(matrix, atol=atol)
        a = factors.a2 @ factors.a1.conj().T
        improved = kind in (
            "exact_further_D_optimized",
            "pd_further_D_optimized",
            "D_equivalent_further_optimal",
        )
        require_d = kind == "D_equivalent_further_optimal"
        use_qmiep = (
            require_d
            or kind == "pd_further_D_optimized"
            or (kind == "pd" and (n == 3 or mode == "heuristic"))
        )
        if use_qmiep:
            left_p, left_d = solve_spectrum(
                a, atol=min(1e-10, atol / len(a)), seed=_synthesis_seed()
            )
            use_qmiep = not require_d or np.array_equal(left_p.array, np.eye(len(a)))
            if use_qmiep:
                # A_PD = D_left P_left A; reconstruction A = D_A P_A A_PD.
                p_a = np.argmax(left_p.array, axis=1)
                d_left = np.diag(left_d.array)
                a = d_left[:, None] * a[p_a]
                d_a = d_left.conj()[np.argsort(p_a)]
            # D-required with P != I uses the original A and the same ZXZ
            # factors. Its children still use improved synthesis below.

        spectrum_c, q_c = _eigenbasis(factors.c, paired=False, atol=atol)
        spectrum_a, q_a = _eigenbasis(a, paired=use_qmiep, atol=atol)
        s_c = np.exp(0.5j * np.angle(spectrum_c))
        s_a = np.exp(0.5j * np.angle(spectrum_a[::2] if use_qmiep else spectrum_a))
        left = s_c[:, None] * q_c.conj().T
        terminal = q_a * (np.repeat(s_a, 2) if use_qmiep else s_a)

        # Absorb the boundary CZs and basis changes into the central multiplexor.
        b0 = q_a.conj().T @ factors.a1 @ q_c
        b1 = q_a.conj().T @ factors.a1 @ factors.b @ q_c
        half = len(b1) // 2
        b1[half:, :] *= -1
        b1[:, half:] *= -1
        spectrum_b, right = _eigenbasis(b0 @ b1.conj().T, paired=False, atol=atol)
        central = np.exp(0.5j * np.angle(spectrum_b))
        middle = central[:, None] * (right.conj().T @ b1)

        selectors = qubits[1:]
        migration_kind: Kind = "pd_further_D_optimized" if improved else "pd"
        if not use_qmiep:
            right_kind: Kind = migration_kind
        elif improved:
            right_kind = "D_equivalent_further_optimal"
        else:
            right_kind = "diagonal"

        correction = self.solve(left, selectors, migration_kind, mode, atol)
        self.rotation(s_c.conj()[correction[0]], qubits, side="left")
        self.instructions.append(("h", qubits[:1], None))
        correction = self.solve(
            _absorb(middle, correction), selectors, migration_kind, mode, atol
        )
        self.rotation(central[correction[0]], qubits, side="closed")
        right = _absorb(right, correction)
        correction = self.solve(right, selectors, right_kind, mode, atol)
        self.instructions.append(("h", qubits[:1], None))
        self.rotation(
            s_a.conj() if use_qmiep else s_a.conj()[correction[0]],
            qubits[:-1] if use_qmiep else qubits,
            side="right",
        )
        if use_qmiep:
            # The paired spectrum exposes CZ(q0, q_last), stored as H-CX-H.
            self.instructions.extend(
                [
                    ("h", qubits[-1:], None),
                    ("cx", (qubits[0], qubits[-1]), None),
                    ("h", qubits[-1:], None),
                ]
            )
        p_o, d_o = self.solve(
            _absorb(terminal, correction), selectors, kind, mode, atol
        )

        if use_qmiep:
            # (I ⊕ D_A P_A)(I ⊗ D_O P_O), with diagonals kept AFTER permutations.
            p = np.r_[p_o, len(p_o) + p_a[p_o]]
            d = np.r_[d_o, d_a * d_o[np.argsort(p_a)]]
            return p, d
        return np.r_[p_o, len(p_o) + p_o], np.tile(d_o, 2)

    def circuit(
        self, num_qubits: int, *, tags: tuple[Hashable, ...] = ()
    ) -> cirq.FrozenCircuit:
        """Materialize Cirq objects only after the complete numerical recursion."""
        qubits = tuple(cirq.LineQubit.range(num_qubits))
        operations = []
        for name, indices, value in self.instructions:
            if name == "matrix":
                gate = cirq.MatrixGate(value, unitary_check=False)
            elif name == "rz":
                gate = cirq.rz(value)
            else:
                gate = cirq.CX if name == "cx" else cirq.H
            operations.append(gate.on(*(qubits[index] for index in indices)))
        operations.append(cirq.global_phase_operation(np.exp(1j * fsum(self.phases))))
        if tags:
            operations = [operation.with_tags(*tags) for operation in operations]
        return cirq.FrozenCircuit(operations)


def _synthesize_numerical(
    matrix: Matrix,
    *,
    mode: Mode,
    atol: float,
    tags: tuple[Hashable, ...] = (),
    kind: Literal["exact", "exact_further_D_optimized"] = "exact",
    native_pd: bool = False,
    native_diagonal: bool = False,
) -> cirq.FrozenCircuit:
    """Synthesize exactly; public input and output validation live at the boundary."""
    n = len(matrix).bit_length() - 1
    program = _Program(native_pd=native_pd, native_diagonal=native_diagonal)
    program.solve(matrix, tuple(range(n)), kind, mode, atol)
    return program.circuit(n, tags=tags)
