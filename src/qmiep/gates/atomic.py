"""Atomic gates with intrinsic Cirq decompositions."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Literal

import cirq
import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmiep._validation import _validation_enabled
from qmiep.gates._base import (
    _RESOLVED_MARK,
    _SELECTOR_SYMBOL,
    _block_diagonal,
    _MatrixGate,
    _qubit_count,
    _symbols,
)
from qmiep.gates.instructions import DiagonalGate, _validate_isometry
from qmiep.matrices import DiagonalMatrix
from qmiep.rust import _NativeCircuit
from qmiep.utils import _readonly, _unitary

__all__ = [
    "RGate",
    "ControlledDiagonal",
    "NQubitUnitary",
    "NQubitUnitaryModuloDiagonal",
    "NQubitUnitaryModuloPermutationDiagonal",
    "IsometryModuloPermutationDiagonal",
    "UniformlyControlledRz",
    "UniformlyControlledRy",
]


def _fast_walsh_hadamard_transform(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the unnormalized Walsh-Hadamard transform of ``values``."""
    transformed = values.copy()
    half_block = 1
    while half_block < len(transformed):
        blocks = transformed.reshape(-1, 2, half_block)
        left = blocks[:, 0].copy()
        right = blocks[:, 1].copy()
        blocks[:, 0] = left + right
        blocks[:, 1] = left - right
        half_block *= 2
    return transformed


def _checked_qubit_order(
    qubit_order: Sequence[cirq.Qid], num_qubits: int
) -> tuple[cirq.Qid, ...]:
    ordered_qubits = tuple(qubit_order)
    if len(ordered_qubits) != num_qubits:
        raise ValueError("qubit_order length must match the matrix qubit count")
    if not all(isinstance(qubit, cirq.Qid) for qubit in ordered_qubits):
        raise TypeError("qubit_order must contain only cirq.Qid instances")
    if len(set(ordered_qubits)) != len(ordered_qubits):
        raise ValueError("qubit_order must not contain duplicate qids")
    if any(qubit.dimension != 2 for qubit in ordered_qubits):
        raise ValueError("qubit_order must contain only two-dimensional qids")
    return ordered_qubits


def _canonical_decomposition(
    decomposition: cirq.AbstractCircuit,
    ordered_qubits: tuple[cirq.Qid, ...],
    canonical_qubits: tuple[cirq.Qid, ...],
) -> cirq.FrozenCircuit:
    if not isinstance(decomposition, cirq.AbstractCircuit):
        raise TypeError("decomposition must be a cirq.AbstractCircuit")

    frozen = decomposition.freeze()
    if not frozen.all_qubits().issubset(ordered_qubits):
        raise ValueError("decomposition uses qids outside qubit_order")
    if cirq.is_measurement(frozen):
        raise ValueError("decomposition must not contain measurements")
    if cirq.is_parameterized(frozen):
        raise ValueError("decomposition must not contain unresolved parameters")
    if any(
        operation.gate is None
        or (
            type(operation.gate).__module__ != __name__
            and len(operation.qubits) > 1
            and operation.gate != cirq.CX
        )
        for operation in frozen.all_operations()
    ):
        raise ValueError(
            "decomposition must contain only gates defined in "
            "qmiep.gates.atomic, zero- or one-qubit gates, or CX gates"
        )
    if ordered_qubits == canonical_qubits:
        return frozen
    qubit_map = dict(zip(ordered_qubits, canonical_qubits, strict=True))
    return frozen.unfreeze().transform_qubits(qubit_map).freeze()


class NQubitUnitary(_MatrixGate):
    r"""An exactly synthesized :math:`U` backed by a concrete circuit.

    ``decomposition`` exposes an immutable circuit on canonical line qubits.
    A native instruction stream is expanded when Cirq primitives are needed.
    """

    def __init__(
        self,
        matrix: ArrayLike,
        decomposition: cirq.AbstractCircuit | _NativeCircuit,
        *,
        qubit_order: Sequence[cirq.Qid],
        label: str = "U",
        atol: float = 1e-10,
    ) -> None:
        unitary = _unitary(matrix, atol=atol)
        num_qubits = unitary.shape[0].bit_length() - 1
        self._canonical_qubits = tuple(cirq.LineQubit.range(num_qubits))
        ordered_qubits = _checked_qubit_order(qubit_order, num_qubits)
        self._native_circuit: _NativeCircuit | None = None
        self._decomposition: cirq.FrozenCircuit | None = None
        if isinstance(decomposition, _NativeCircuit):
            if decomposition.num_qubits != num_qubits:
                raise ValueError(
                    "native circuit width must match the matrix qubit count"
                )
            self._native_circuit = decomposition
        else:
            self._decomposition = _canonical_decomposition(
                decomposition, ordered_qubits, self._canonical_qubits
            )
        if _validation_enabled():
            try:
                decomposed_matrix = self.decomposition.unitary(
                    qubit_order=self._canonical_qubits,
                    qubits_that_should_be_present=self._canonical_qubits,
                    ignore_terminal_measurements=False,
                    dtype=np.complex128,
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "decomposition must contain only unitary operations"
                ) from error

            if decomposed_matrix.shape != unitary.shape or not np.allclose(
                decomposed_matrix, unitary, rtol=0, atol=atol
            ):
                raise ValueError("decomposition unitary must match matrix exactly")

        self._label = label
        self._atol = atol
        super().__init__(unitary, _symbols(label, num_qubits, mark=_RESOLVED_MARK))

    @property
    def decomposition(self) -> cirq.FrozenCircuit:
        """The immutable decomposition on canonical line qubits."""
        if self._decomposition is None:
            assert self._native_circuit is not None
            self._decomposition = self._native_circuit.to_cirq(self._canonical_qubits)
        return self._decomposition

    @property
    def label(self) -> str:
        """The label used in circuit diagrams."""
        return self._label

    @property
    def atol(self) -> float:
        """The absolute tolerance used to validate this gate."""
        return self._atol

    def _value_equality_values_(self) -> tuple[object, ...]:
        return *super()._value_equality_values_(), self.decomposition

    def _has_unitary_(self) -> bool:
        return True

    def _decompose_(self, qubits: tuple[cirq.Qid, ...]) -> tuple[cirq.Operation, ...]:
        if qubits == self._canonical_qubits:
            return tuple(self.decomposition.all_operations())
        if self._decomposition is None and self._native_circuit is not None:
            return tuple(self._native_circuit.to_cirq(qubits).all_operations())
        qubit_map: dict[cirq.Qid, cirq.Qid] = dict(
            zip(self._canonical_qubits, qubits, strict=True)
        )
        return tuple(
            operation.transform_qubits(qubit_map)
            for operation in self.decomposition.all_operations()
        )

    def __str__(self) -> str:
        return f"{self.label}{_RESOLVED_MARK}"

    def __repr__(self) -> str:
        return (
            f"NQubitUnitary(matrix={self.matrix!r}, "
            f"decomposition={self.decomposition!r}, "
            f"qubit_order={self._canonical_qubits!r}, label={self.label!r})"
        )


class IsometryModuloPermutationDiagonal(_MatrixGate):
    r"""A resolved rectangular isometry with input-side PD corrections removed.

    Stores only :math:`V_{\rm PD}\in\mathbb C^{2d\times d}` and its concrete
    implementation. The first qid enters in :math:`|0\rangle`. Cirq's square
    unitary protocol evaluates that implementation; it never completes the
    rectangular matrix. External input corrections obey :math:`V=V_{\rm PD}DP`.
    """

    def __init__(
        self,
        matrix: ArrayLike,
        decomposition: cirq.AbstractCircuit,
        *,
        qubit_order: Sequence[cirq.Qid],
        label: str = "V_PD",
        atol: float = 1e-10,
    ) -> None:
        isometry = _validate_isometry(matrix, atol=atol)
        num_qubits = len(isometry).bit_length() - 1
        self._canonical_qubits = tuple(cirq.LineQubit.range(num_qubits))
        self._decomposition = _canonical_decomposition(
            decomposition,
            _checked_qubit_order(qubit_order, num_qubits),
            self._canonical_qubits,
        )
        if _validation_enabled():
            unitary = _unitary(self._unitary_(), atol=atol)
            if not np.allclose(
                unitary[:, : isometry.shape[1]], isometry, rtol=0, atol=atol
            ):
                raise ValueError("decomposition must reconstruct the input isometry")
        self._label = label
        self._atol = atol
        super().__init__(isometry, _symbols(label, num_qubits, mark=_RESOLVED_MARK))

    @property
    def isometry(self) -> NDArray[np.complex128]:
        return self.matrix

    @property
    def decomposition(self) -> cirq.FrozenCircuit:
        """Concrete implementation on canonical line qubits."""
        return self._decomposition

    @property
    def label(self) -> str:
        return self._label

    @property
    def atol(self) -> float:
        return self._atol

    def _unitary_(self) -> NDArray[np.complex128]:
        return self.decomposition.unitary(
            qubit_order=self._canonical_qubits,
            qubits_that_should_be_present=self._canonical_qubits,
            ignore_terminal_measurements=False,
        )

    def _decompose_(self, qubits: tuple[cirq.Qid, ...]) -> tuple[cirq.Operation, ...]:
        mapping = dict(zip(self._canonical_qubits, qubits, strict=True))
        return tuple(
            op.transform_qubits(mapping) for op in self.decomposition.all_operations()
        )

    def _value_equality_values_(self) -> tuple[object, ...]:
        return *super()._value_equality_values_(), self.decomposition


@cirq.value_equality
class RGate(cirq.Gate):
    r"""The open UCG-Rz cascade :math:`R` on target-first ordered qids.

    For selector bits :math:`x=(x_1,\ldots,x_k)`,

    .. math::

        R=\sum_x X^{x_1}R_z(\theta_x)\otimes|x\rangle\langle x|.

    Thus :math:`C R=D\oplus D^\dagger`, where
    :math:`C=\mathrm{CX}(q_1,q_0)` and
    :math:`D=\operatorname{diag}(e^{-i\theta_x/2})`.
    The angle can depend on all selectors.
    Construction always checks the alternating Rz/CX Gray-code cascade with
    the final (left) or initial (right) CX removed. Validation is purely
    syntactic; the matrix is computed lazily only when requested.
    ``side`` retains the synthesizer's left/right boundary role.
    """

    def __init__(
        self,
        decomposition: cirq.AbstractCircuit,
        *,
        qubit_order: Sequence[cirq.Qid],
        side: Literal["left", "right"] = "left",
        atol: float = 1e-10,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        ordered_qubits = tuple(qubit_order)
        if len(ordered_qubits) < 2:
            raise ValueError("RGate requires a target and at least one selector")
        if not all(isinstance(qubit, cirq.Qid) for qubit in ordered_qubits):
            raise TypeError("qubit_order must contain only cirq.Qid instances")
        if len(set(ordered_qubits)) != len(ordered_qubits):
            raise ValueError("qubit_order must not contain duplicate qids")
        if any(qubit.dimension != 2 for qubit in ordered_qubits):
            raise ValueError("qubit_order must contain only two-dimensional qids")
        if not isinstance(decomposition, cirq.AbstractCircuit):
            raise TypeError("decomposition must be a cirq.AbstractCircuit")
        frozen = decomposition.freeze()
        self._validate_syntax(frozen, ordered_qubits, side)
        self._canonical_qubits = tuple(cirq.LineQubit.range(len(ordered_qubits)))
        qubit_map = dict(zip(ordered_qubits, self._canonical_qubits, strict=True))
        self._decomposition = frozen.unfreeze().transform_qubits(qubit_map).freeze()
        self._side = side
        self._atol = atol

    @property
    def decomposition(self) -> cirq.FrozenCircuit:
        """The immutable decomposition on canonical line qubits."""
        return self._decomposition

    @property
    def label(self) -> str:
        return "R_L" if self.side == "left" else "R_R"

    @property
    def atol(self) -> float:
        """The tolerance used by downstream matrix-based rewrites."""
        return self._atol

    @cached_property
    def matrix(self) -> NDArray[np.complex128]:
        """Compute and cache the matrix on demand, never for validation."""
        return _readonly(self.decomposition.unitary(qubit_order=self._canonical_qubits))

    def _num_qubits_(self) -> int:
        return len(self._canonical_qubits)

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> NDArray[np.complex128]:
        return self.matrix

    def _value_equality_values_(self) -> tuple[object, ...]:
        return self._canonical_qubits, self.side, self.decomposition

    def _decompose_(self, qubits: tuple[cirq.Qid, ...]) -> tuple[cirq.Operation, ...]:
        qubit_map = dict(zip(self._canonical_qubits, qubits, strict=True))
        return tuple(
            op.transform_qubits(qubit_map) for op in self.decomposition.all_operations()
        )

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=_symbols(
                self.label, len(self._canonical_qubits), mark=_RESOLVED_MARK
            )
        )

    def __str__(self) -> str:
        return f"{self.label}{_RESOLVED_MARK}"

    @staticmethod
    def _validate_syntax(
        decomposition: cirq.AbstractCircuit,
        qubits: tuple[cirq.Qid, ...],
        side: Literal["left", "right"],
    ) -> None:
        """Check the open Gray cascade without constructing its matrix."""
        operations = tuple(decomposition.all_operations())
        if side == "right":
            operations = operations[::-1]
        num_selectors = len(qubits) - 1
        if len(operations) != 2 ** (num_selectors + 1) - 1:
            raise ValueError("RGate syntax requires 2**k Rz and 2**k - 1 CX gates")
        for position, operation in enumerate(operations):
            if position % 2 == 0:
                if type(operation.gate) is not cirq.Rz or operation.qubits != (
                    qubits[0],
                ):
                    raise ValueError("RGate syntax requires alternating Rz(q0) and CX")
                if cirq.is_parameterized(operation) or not np.isfinite(
                    operation.gate.exponent
                ):
                    raise ValueError("RGate syntax requires finite, resolved Rz angles")
            else:
                # Transition j-1 -> j flips the least significant set bit of j.
                transition = (position + 1) // 2
                bit = (transition & -transition).bit_length() - 1
                expected = cirq.CX(qubits[num_selectors - bit], qubits[0])
                if operation.untagged != expected:
                    raise ValueError("RGate syntax requires the Gray-code CX order")

    @property
    def side(self) -> Literal["left", "right"]:
        """The boundary side represented by this rotation fragment."""
        return self._side

    def __repr__(self) -> str:
        return (
            f"RGate(decomposition={self.decomposition!r}, "
            f"qubit_order={self._canonical_qubits!r}, side={self.side!r})"
        )


class NQubitUnitaryModuloPermutationDiagonal(NQubitUnitary):
    r"""A resolved residual :math:`W=D^\dagger P^\dagger U` with a circuit.

    External permutation and diagonal corrections remain available to a later
    rewrite.  Circuit-time order ``W, D, P`` reconstructs :math:`U=PDW`.
    """

    def __init__(
        self,
        residual_matrix: ArrayLike,
        decomposition: cirq.AbstractCircuit | _NativeCircuit,
        *,
        qubit_order: Sequence[cirq.Qid],
        label: str = "Ũ",
        atol: float = 1e-10,
    ) -> None:
        super().__init__(
            residual_matrix,
            decomposition,
            qubit_order=qubit_order,
            label=label,
            atol=atol,
        )

    @property
    def residual_matrix(self) -> NDArray[np.complex128]:
        """The immutable representative :math:`W` implemented by this gate."""
        return self.matrix

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"residual_matrix={self.residual_matrix!r}, "
            f"decomposition={self.decomposition!r}, "
            f"qubit_order={self._canonical_qubits!r}, label={self.label!r})"
        )


class NQubitUnitaryModuloDiagonal(NQubitUnitaryModuloPermutationDiagonal):
    r"""The :math:`P=I` specialization :math:`W=D^\dagger U`."""

    def __init__(
        self,
        residual_matrix: ArrayLike,
        decomposition: cirq.AbstractCircuit | _NativeCircuit,
        *,
        qubit_order: Sequence[cirq.Qid],
        label: str = "Û",
        atol: float = 1e-10,
    ) -> None:
        super().__init__(
            residual_matrix,
            decomposition,
            qubit_order=qubit_order,
            label=label,
            atol=atol,
        )


class ControlledDiagonal(_MatrixGate):
    r"""A one-control atomic diagonal unitary :math:`I\oplus D`.

    The first qid is the control and all remaining qids are targets of the
    unit-modulus diagonal matrix :math:`D`.
    """

    def __init__(
        self,
        diagonal: ArrayLike,
        *,
        label: str = "D",
        cx_boundary: Literal["start", "end"] = "end",
        atol: float = 1e-10,
    ) -> None:
        if cx_boundary not in ("start", "end"):
            raise ValueError("cx_boundary must be 'start' or 'end'")

        self._diagonal = DiagonalMatrix(diagonal, atol=atol)
        self._label = label
        self._cx_boundary = cx_boundary
        self._atol = atol
        target = np.asarray(self._diagonal, dtype=np.complex128)
        dimension = target.shape[0]
        num_targets = _qubit_count(dimension, matrix_name="diagonal")

        self._target_matrix = _readonly(target)
        super().__init__(
            _block_diagonal(
                np.eye(dimension, dtype=np.complex128), self._target_matrix
            ),
            ("@", *_symbols(label, num_targets, mark=_RESOLVED_MARK)),
        )

    @property
    def diagonal(self) -> DiagonalMatrix:
        """The immutable target diagonal-unitary value."""
        return self._diagonal

    @property
    def target_matrix(self) -> NDArray[np.complex128]:
        """The immutable target diagonal matrix :math:`D`."""
        return self._target_matrix

    @property
    def label(self) -> str:
        """The target label used in circuit diagrams."""
        return self._label

    @property
    def cx_boundary(self) -> Literal["start", "end"]:
        """The boundary at which the Gray-code cascade exposes a CX gate."""
        return self._cx_boundary

    @property
    def atol(self) -> float:
        """The absolute tolerance used to validate this gate."""
        return self._atol

    def _value_equality_values_(self) -> tuple[object, ...]:
        return *super()._value_equality_values_(), self.cx_boundary

    def _decompose_(self, qubits: tuple[cirq.Qid, ...]) -> list[cirq.Operation]:
        r"""Decompose into a target diagonal and an interleaved UCRz cascade.

        Write :math:`D=\operatorname{diag}(e^{i\phi_x})` and
        :math:`S=\operatorname{diag}(e^{i\phi_x/2})`.  Then

        .. math::

            I\oplus D
            =(I_2\otimes S)
              \sum_x R_z(\phi_x)\otimes |x\rangle\!\langle x|.

        The controlled block is the :math:`I\oplus D` factor in Eq. (2) of
        De Vos and De Baerdemacker, Phys. Rev. A 94, 052317 (2016).  Its
        uniformly controlled rotation is synthesized by the binary-reflected
        Gray-code construction of Möttönen et al., Quant. Inf. Comput. 5, 467
        (2005).

        For :math:`k` target qids and :math:`N=2^k`, let
        :math:`g_j=j\mathbin{\mathtt{xor}}(j\mathbin{\mathtt{>>}}1)`.  The
        interleaved rotation angles are

        .. math::

            \theta_j=\frac1N\sum_x(-1)^{g_j\cdot x}\phi_x.

        Closing the Gray-code cycle produces exactly :math:`N=2^{m-1}` CX
        gates for an :math:`m`-qid controlled diagonal.  The default order
        ends in CX.  Reversing the full sequence gives the same diagonal
        unitary and exposes CX at the beginning instead.
        """
        control, *targets = qubits
        phases = np.angle(np.diag(self.target_matrix))
        dimension = len(phases)
        transformed = _fast_walsh_hadamard_transform(phases)
        square_root = DiagonalMatrix.from_diagonal(np.exp(0.5j * phases))
        operations = [DiagonalGate(square_root).on(*targets)]

        for index in range(dimension):
            gray = index ^ (index >> 1)
            operations.append(cirq.rz(transformed[gray] / dimension).on(control))

            next_index = (index + 1) % dimension
            next_gray = next_index ^ (next_index >> 1)
            flipped_bit = (gray ^ next_gray).bit_length() - 1
            operations.append(cirq.CX(targets[-flipped_bit - 1], control))

        return operations if self.cx_boundary == "end" else operations[::-1]

    def __str__(self) -> str:
        return f"C{self.label}{_RESOLVED_MARK}"

    def __repr__(self) -> str:
        return (
            f"ControlledDiagonal(diagonal={np.diag(self.target_matrix)!r}, "
            f"label={self.label!r}, cx_boundary={self.cx_boundary!r})"
        )


class UniformlyControlledRy(_MatrixGate):
    r"""Apply :math:`R_y(\alpha_x)` to the first qid, selected by the rest.

    ``angles[x]`` uses the binary basis order of the selector qids. For
    :math:`C=\operatorname{diag}(\cos(\alpha_x/2))` and
    :math:`S=\operatorname{diag}(\sin(\alpha_x/2))`, the matrix is
    :math:`\left(\begin{smallmatrix}C&-S\\S&C\end{smallmatrix}\right)`.
    A closed Gray-code cascade implements it with :math:`2^k` Ry and CX
    gates for :math:`k\geq1` selectors.
    """

    def __init__(
        self,
        angles: ArrayLike,
        *,
        label: str = "F(Ry)",
        atol: float = 1e-10,
    ) -> None:
        if not np.isfinite(atol) or atol < 0:
            raise ValueError("atol must be finite and non-negative")
        values = np.asarray(angles)
        if values.ndim != 1 or np.iscomplexobj(values):
            raise ValueError("angles must be a one-dimensional real array")
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("angles must be finite")
        num_selectors = _qubit_count(len(values), matrix_name="angles")
        self._angles = _readonly(values)
        self._label = label
        self._atol = atol
        cosine = np.diag(np.cos(values / 2))
        sine = np.diag(np.sin(values / 2))
        super().__init__(
            _readonly(
                np.block([[cosine, -sine], [sine, cosine]]).astype(np.complex128)
            ),
            (
                *_symbols(label, 1, mark=_RESOLVED_MARK),
                *(_SELECTOR_SYMBOL for _ in range(num_selectors)),
            ),
        )

    @property
    def angles(self) -> NDArray[np.float64]:
        """The immutable Ry angle table in selector basis order."""
        return self._angles

    @property
    def label(self) -> str:
        """The label shown on the rotated qid."""
        return self._label

    @property
    def atol(self) -> float:
        """The absolute tolerance associated with this gate."""
        return self._atol

    def _decompose_(self, qubits: tuple[cirq.Qid, ...]) -> list[cirq.Operation]:
        r"""Use :math:`\beta_j=2^{-k}\sum_x(-1)^{x\cdot g_j}\alpha_x`."""
        target, *selectors = qubits
        dimension = len(self.angles)
        transformed = _fast_walsh_hadamard_transform(self.angles) / dimension
        operations = []
        for index in range(dimension):
            gray = index ^ (index >> 1)
            operations.append(cirq.ry(transformed[gray]).on(target))
            next_index = (index + 1) % dimension
            next_gray = next_index ^ (next_index >> 1)
            flipped_bit = (gray ^ next_gray).bit_length() - 1
            operations.append(cirq.CX(selectors[-flipped_bit - 1], target))
        return operations

    def __str__(self) -> str:
        return f"{self.label}{_RESOLVED_MARK}"


class UniformlyControlledRz(_MatrixGate):
    r"""A symmetric diagonal multiplexor :math:`D\oplus D^\dagger`.

    The first qid is the uniformly rotated qid and the remaining qids select
    an entry of :math:`D`.  If

    .. math::

        D=\operatorname{diag}(e^{i\phi_x}),

    then selector state :math:`|x\rangle` applies
    :math:`R_z(-2\phi_x)` to the first qid.  The closed Gray-code cascade uses
    exactly :math:`2^k` CX gates for :math:`k` selector qids and needs no
    additional target-diagonal compensation.
    """

    def __init__(
        self,
        diagonal: ArrayLike,
        *,
        label: str = "F(Rz)",
        cx_boundary: Literal["start", "end"] = "end",
        atol: float = 1e-10,
    ) -> None:
        if cx_boundary not in ("start", "end"):
            raise ValueError("cx_boundary must be 'start' or 'end'")

        self._cx_boundary = cx_boundary
        self._diagonal = DiagonalMatrix(diagonal, atol=atol)
        self._label = label
        self._atol = atol
        branch = np.asarray(self._diagonal, dtype=np.complex128)
        num_selectors = _qubit_count(branch.shape[0], matrix_name="diagonal")
        self._branch_matrix = _readonly(branch)
        super().__init__(
            _block_diagonal(self._branch_matrix, self._branch_matrix.conj().T),
            (
                *_symbols(label, 1, mark=_RESOLVED_MARK),
                *(_SELECTOR_SYMBOL for _ in range(num_selectors)),
            ),
        )

    @property
    def diagonal(self) -> DiagonalMatrix:
        r"""The immutable :math:`D` in :math:`D\oplus D^\dagger`."""
        return self._diagonal

    @property
    def branch_matrix(self) -> NDArray[np.complex128]:
        r"""The immutable matrix :math:`D` on the zero branch."""
        return self._branch_matrix

    @property
    def label(self) -> str:
        """The label shown on the uniformly rotated qid."""
        return self._label

    @property
    def atol(self) -> float:
        """The absolute tolerance used to validate this gate."""
        return self._atol

    @property
    def cx_boundary(self) -> Literal["start", "end"]:
        """The boundary exposing CX from the first selector to the target."""
        return self._cx_boundary

    def _value_equality_values_(self) -> tuple[object, ...]:
        return *super()._value_equality_values_(), self.cx_boundary

    def _decompose_(self, qubits: tuple[cirq.Qid, ...]) -> list[cirq.Operation]:
        r"""Decompose by the cyclic Gray-code construction of Figure 2.

        For :math:`D=\operatorname{diag}(e^{i\phi_x})`, the desired rotation
        in selector sector :math:`x` is :math:`R_z(-2\phi_x)`.  If
        :math:`g_j=j\mathbin{\mathtt{xor}}(j\mathbin{\mathtt{>>}}1)` is the
        binary-reflected Gray code and :math:`N=2^k`, the interleaved angles
        are

        .. math::

            \theta_j=\frac1N\sum_x(-1)^{x\cdot g_j}(-2\phi_x).

        The CX following :math:`R_z(\theta_j)` is controlled by the selector
        bit flipped between :math:`g_j` and :math:`g_{j+1\bmod N}`.  The final
        edge closes the Gray cycle, so the circuit has exactly :math:`N`
        rotations and :math:`N` CX gates. The open cascade is returned as
        one :class:`RGate`, followed by CX(q1, q0). For ``cx_boundary="start"``,
        the reversed open cascade follows that CX instead; its q1=1 sector
        is :math:`R_z(\theta)X=XR_z(-\theta)`.
        """
        rotation, *selectors = qubits
        phases = -2 * np.angle(np.diag(self.branch_matrix))
        dimension = len(phases)
        transformed = _fast_walsh_hadamard_transform(phases)
        operations: list[cirq.Operation] = []

        for index in range(dimension):
            gray = index ^ (index >> 1)
            operations.append(cirq.rz(transformed[gray] / dimension).on(rotation))

            next_index = (index + 1) % dimension
            next_gray = next_index ^ (next_index >> 1)
            flipped_bit = (gray ^ next_gray).bit_length() - 1
            operations.append(cirq.CX(selectors[-flipped_bit - 1], rotation))

        # The closing Gray-code edge uses the first selector.
        assert operations[-1] == cirq.CX(qubits[1], qubits[0])

        # Each Rz/CX matrix is symmetric, so reversal gives U.T = U.
        fragment = operations[:-1]
        if self.cx_boundary == "start":
            fragment = fragment[::-1]
        rotation_gate = RGate(
            cirq.FrozenCircuit(cirq.Moment(op) for op in fragment),
            qubit_order=qubits,
            side="left" if self.cx_boundary == "end" else "right",
            atol=self.atol,
        ).on(*qubits)
        boundary = operations[-1]
        return (
            [rotation_gate, boundary]
            if self.cx_boundary == "end"
            else [boundary, rotation_gate]
        )

    def __str__(self) -> str:
        return f"{self.label}{_RESOLVED_MARK}"

    def __repr__(self) -> str:
        return (
            f"UniformlyControlledRz(diagonal={np.diag(self.branch_matrix)!r}, "
            f"label={self.label!r}, cx_boundary={self.cx_boundary!r})"
        )
