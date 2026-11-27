"""Non-decomposable semantic instructions used by circuit rewriting."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmiep.gates._base import (
    _SELECTOR_SYMBOL,
    _UNRESOLVED_MARK,
    _block_diagonal,
    _MatrixGate,
    _qubit_count,
    _symbols,
)
from qmiep.matrices import DiagonalMatrix, PermutationMatrix
from qmiep.utils import _readonly, _unitary

__all__ = [
    "CentralMultiplexor",
    "ControlledPermutation",
    "ControlledRelativeUnitary",
    "ControlledSeparatedUnitary",
    "ControlledSpectralUnitary",
    "ControlledUnitary",
    "DiagonalGate",
    "GaugeUnitary",
    "Isometry",
    "MultiplexedDiagonal",
    "MultiplexedPermutation",
    "PermutationGate",
    "UnresolvedNQubitUnitary",
]

_DECOMPOSITION_PROTOCOLS = ("_decompose_", "_decompose_with_context_")


def _has_antipodal_spectrum(matrix: NDArray[np.complex128], *, atol: float) -> bool:
    r"""Return whether the eigenvalue multiset is invariant under :math:`z\mapsto-z`."""
    eigenvalues = np.linalg.eigvals(matrix)
    adjacency = np.isclose(
        eigenvalues[:, np.newaxis],
        -eigenvalues[np.newaxis, :],
        rtol=0,
        atol=atol,
    )
    matched_left = np.full(len(eigenvalues), -1, dtype=np.intp)

    def augment(left: int, seen: NDArray[np.bool_]) -> bool:
        for right in np.flatnonzero(adjacency[left]):
            if seen[right]:
                continue
            seen[right] = True
            if matched_left[right] < 0 or augment(matched_left[right], seen):
                matched_left[right] = left
                return True
        return False

    return all(
        augment(left, np.zeros(len(eigenvalues), dtype=np.bool_))
        for left in range(len(eigenvalues))
    )


class _Instruction(_MatrixGate):
    """Base class that forbids Cirq decomposition on semantic instructions."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        forbidden = tuple(
            protocol
            for protocol in _DECOMPOSITION_PROTOCOLS
            if any(protocol in base.__dict__ for base in cls.__mro__)
        )
        if forbidden:
            raise TypeError(
                f"{cls.__name__} is a semantic instruction and cannot define "
                f"Cirq decomposition protocols: {', '.join(forbidden)}"
            )


def _validate_isometry(matrix: ArrayLike, *, atol: float) -> NDArray[np.complex128]:
    """Validate and return an (n-1)-to-n-qubit isometry matrix for n >= 2."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    isometry = np.asarray(matrix, dtype=np.complex128)
    if isometry.ndim != 2:
        raise ValueError("an isometry must be a two-dimensional matrix")
    rows, columns = isometry.shape
    if columns < 2 or columns & (columns - 1) or rows != 2 * columns:
        raise ValueError("expected shape (2**n, 2**(n-1)) with n >= 2")
    if not np.all(np.isfinite(isometry)):
        raise ValueError("isometry entries must be finite")
    if not np.allclose(
        isometry.conj().T @ isometry, np.eye(columns), rtol=0, atol=atol
    ):
        raise ValueError("isometry columns must be orthonormal")
    return _readonly(isometry)


class UnresolvedNQubitUnitary(_Instruction):
    r"""An unresolved semantic :math:`U\in U(2^n)`, for :math:`n\geq1`.

    A rewrite rule must replace this instruction with a concrete synthesis.
    """

    def __init__(
        self, matrix: ArrayLike, *, label: str = "U", atol: float = 1e-10
    ) -> None:
        normalized = _unitary(matrix, atol=atol)
        self._label = label
        self._atol = atol
        num_qubits = normalized.shape[0].bit_length() - 1
        super().__init__(normalized, _symbols(label, num_qubits, mark=_UNRESOLVED_MARK))

    def __str__(self) -> str:
        return f"{self._label}{_UNRESOLVED_MARK}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(matrix={self.matrix!r}, label={self.label!r})"

    @property
    def label(self) -> str:
        """The label used in circuit diagrams."""
        return self._label

    @property
    def atol(self) -> float:
        """The absolute tolerance used to validate this instruction."""
        return self._atol


class Isometry(_Instruction):
    r"""An unresolved (n-1)-to-n isometry on n qids, with n >= 2.

    ``matrix`` and ``isometry`` expose the same immutable
    :math:`2^n\times2^{n-1}` input V; no unitary completion is stored.
    The first qid is an ancilla initially in :math:`|0\rangle`, followed by
    the input register in binary basis order. The synthesizer constructs a
    completion when needed and preserves its ancilla-zero columns, including
    phase. This instruction has no Cirq unitary or decomposition.
    """

    def __init__(
        self, matrix: ArrayLike, *, label: str = "V", atol: float = 1e-10
    ) -> None:
        isometry = _validate_isometry(matrix, atol=atol)
        self._label = label
        self._atol = atol
        super().__init__(
            isometry,
            _symbols(label, isometry.shape[0].bit_length() - 1, mark=_UNRESOLVED_MARK),
        )

    @property
    def isometry(self) -> NDArray[np.complex128]:
        """The rectangular input map, including its original global phase."""
        return self.matrix

    def _unitary_(self) -> None:
        """A rectangular isometry does not specify an n-qubit unitary."""
        return None

    @property
    def label(self) -> str:
        """The label used in circuit diagrams."""
        return self._label

    @property
    def atol(self) -> float:
        """The tolerance used to validate the isometry."""
        return self._atol

    def __str__(self) -> str:
        return f"{self.label}{_UNRESOLVED_MARK}"

    def __repr__(self) -> str:
        return f"Isometry(matrix={self.isometry!r}, label={self.label!r})"


class GaugeUnitary(UnresolvedNQubitUnitary):
    r"""One of the four target-register gauge blocks :math:`L,M,N,O`."""

    _ROLES = frozenset(("L", "M", "N", "O"))

    def __init__(
        self,
        matrix: ArrayLike,
        *,
        role: Literal["L", "M", "N", "O"],
        atol: float = 1e-10,
    ) -> None:
        if role not in self._ROLES:
            raise ValueError("role must be one of 'L', 'M', 'N', or 'O'")
        self._role = role
        super().__init__(matrix, label=role, atol=atol)

    @property
    def role(self) -> Literal["L", "M", "N", "O"]:
        """The block's position in the left-to-right gauge chain."""
        return self._role

    def __repr__(self) -> str:
        return f"GaugeUnitary(matrix={self.matrix!r}, role={self.role!r})"


class PermutationGate(_Instruction):
    """A semantic instruction backed by a :class:`PermutationMatrix`."""

    def __init__(self, permutation: PermutationMatrix, *, label: str = "P") -> None:
        if not isinstance(permutation, PermutationMatrix):
            raise TypeError("permutation must be a PermutationMatrix")

        dimension = permutation.array.shape[0]
        num_qubits = _qubit_count(dimension, matrix_name="permutation")
        self._permutation = permutation
        self._label = label
        matrix = _readonly(np.asarray(permutation, dtype=np.complex128))
        super().__init__(matrix, _symbols(label, num_qubits, mark=_UNRESOLVED_MARK))

    @property
    def permutation(self) -> PermutationMatrix:
        """The immutable permutation-matrix value represented here."""
        return self._permutation

    @property
    def label(self) -> str:
        """The label used in circuit diagrams."""
        return self._label

    def __str__(self) -> str:
        return f"{self._label}{_UNRESOLVED_MARK}"

    def __repr__(self) -> str:
        return (
            f"PermutationGate(permutation={self.permutation!r}, "
            f"label={self.label!r})"
        )


class MultiplexedPermutation(PermutationGate):
    r"""A one-selector permutation :math:`P_0\oplus P_1`.

    In the basis ``|selector>|target>``, the first qid selects
    :attr:`zero_permutation` or :attr:`one_permutation` on the remaining qids.
    :attr:`permutation` stores the full direct sum. Passing only one
    permutation retains the controlled special case :math:`I\oplus P`.
    """

    def __init__(
        self,
        zero_permutation: PermutationMatrix,
        one_permutation: PermutationMatrix | None = None,
        *,
        label: str = "P",
    ) -> None:
        if not isinstance(zero_permutation, PermutationMatrix):
            raise TypeError("zero_permutation must be a PermutationMatrix")
        if one_permutation is None:
            one_permutation = zero_permutation
            zero_permutation = PermutationMatrix(
                np.eye(one_permutation.array.shape[0], dtype=np.uint8)
            )
        if not isinstance(one_permutation, PermutationMatrix):
            raise TypeError("one_permutation must be a PermutationMatrix")
        if zero_permutation.array.shape != one_permutation.array.shape:
            raise ValueError("the two permutation blocks must have equal size")

        dimension = zero_permutation.array.shape[0]
        num_targets = _qubit_count(dimension, matrix_name="target permutation")
        lifted = np.zeros((2 * dimension, 2 * dimension), dtype=np.uint8)
        lifted[:dimension, :dimension] = zero_permutation.array
        lifted[dimension:, dimension:] = one_permutation.array
        self._zero_permutation = zero_permutation
        self._one_permutation = one_permutation
        super().__init__(PermutationMatrix(lifted), label=label)
        identity = np.eye(dimension, dtype=np.uint8)
        if np.array_equal(zero_permutation.array, identity):
            selector, display_label = "@", label
        elif np.array_equal(one_permutation.array, identity):
            selector, display_label = "(0)", label
        else:
            selector, display_label = _SELECTOR_SYMBOL, f"diag({label}_0, {label}_1)"
        self._wire_symbols = (
            selector,
            *_symbols(display_label, num_targets, mark=_UNRESOLVED_MARK),
        )

    @property
    def zero_permutation(self) -> PermutationMatrix:
        """The immutable permutation selected by control zero."""
        return self._zero_permutation

    @property
    def one_permutation(self) -> PermutationMatrix:
        """The immutable permutation selected by control one."""
        return self._one_permutation

    @property
    def target_permutation(self) -> PermutationMatrix:
        """Backward-compatible alias for :attr:`one_permutation`."""
        return self.one_permutation

    def __repr__(self) -> str:
        return (
            f"MultiplexedPermutation(zero_permutation={self.zero_permutation!r}, "
            f"one_permutation={self.one_permutation!r}, label={self.label!r})"
        )


ControlledPermutation = MultiplexedPermutation


class DiagonalGate(_Instruction):
    """A semantic instruction backed by a :class:`DiagonalMatrix`."""

    def __init__(self, diagonal: DiagonalMatrix, *, label: str = "D") -> None:
        if not isinstance(diagonal, DiagonalMatrix):
            raise TypeError("diagonal must be a DiagonalMatrix")

        dimension = diagonal.array.shape[0]
        num_qubits = _qubit_count(dimension, matrix_name="diagonal")
        self._diagonal = diagonal
        self._label = label
        matrix = _readonly(np.asarray(diagonal, dtype=np.complex128))
        super().__init__(matrix, _symbols(label, num_qubits, mark=_UNRESOLVED_MARK))

    @property
    def diagonal(self) -> DiagonalMatrix:
        """The immutable diagonal-matrix value represented here."""
        return self._diagonal

    @property
    def label(self) -> str:
        """The label used in circuit diagrams."""
        return self._label

    def __str__(self) -> str:
        return f"{self._label}{_UNRESOLVED_MARK}"

    def __repr__(self) -> str:
        return f"DiagonalGate(diagonal={self.diagonal!r}, label={self.label!r})"


class MultiplexedDiagonal(DiagonalGate):
    r"""A semantic diagonal :math:`D_0\oplus D_1` selected by the first qid."""

    def __init__(
        self,
        zero_diagonal: DiagonalMatrix,
        one_diagonal: DiagonalMatrix,
        *,
        label: str = "D",
    ) -> None:
        if not isinstance(zero_diagonal, DiagonalMatrix) or not isinstance(
            one_diagonal, DiagonalMatrix
        ):
            raise TypeError("both diagonal blocks must be DiagonalMatrix values")
        if zero_diagonal.array.shape != one_diagonal.array.shape:
            raise ValueError("the two diagonal blocks must have equal size")
        self._zero_diagonal = zero_diagonal
        self._one_diagonal = one_diagonal
        super().__init__(
            DiagonalMatrix(_block_diagonal(zero_diagonal.array, one_diagonal.array)),
            label=label,
        )
        num_targets = _qubit_count(zero_diagonal.array.shape[0], matrix_name="diagonal")
        self._wire_symbols = (
            _SELECTOR_SYMBOL,
            *_symbols(
                f"diag({label}_0, {label}_1)", num_targets, mark=_UNRESOLVED_MARK
            ),
        )

    @property
    def zero_diagonal(self) -> DiagonalMatrix:
        """The diagonal selected by control zero."""
        return self._zero_diagonal

    @property
    def one_diagonal(self) -> DiagonalMatrix:
        """The diagonal selected by control one."""
        return self._one_diagonal


class ControlledUnitary(_Instruction):
    r"""A semantic one-control block-diagonal unitary :math:`U_1\oplus U_2`.

    With the first qid as the control, this instruction represents

    .. math::

        |0\rangle\!\langle0|\otimes U_1
        +|1\rangle\!\langle1|\otimes U_2.
    """

    def __init__(
        self,
        zero_matrix: ArrayLike,
        one_matrix: ArrayLike,
        *,
        zero_label: str = "U_1",
        one_label: str = "U_2",
        atol: float = 1e-10,
    ) -> None:
        zero = _unitary(zero_matrix, atol=atol)
        one = _unitary(one_matrix, atol=atol)
        if zero.shape != one.shape:
            raise ValueError("the two controlled unitary blocks must have equal size")

        self._zero_matrix = zero
        self._one_matrix = one
        self._zero_label = zero_label
        self._one_label = one_label
        self._atol = atol
        num_targets = zero.shape[0].bit_length() - 1
        identity = np.eye(zero.shape[0])
        if np.allclose(zero, identity, rtol=0, atol=atol):
            display_label = (
                f"{one_label}*"
                if _has_antipodal_spectrum(one, atol=atol)
                else one_label
            )
            wire_symbols = (
                "@",
                *_symbols(display_label, num_targets, mark=_UNRESOLVED_MARK),
            )
        elif np.allclose(one, identity, rtol=0, atol=atol):
            display_label = (
                f"{zero_label}*"
                if _has_antipodal_spectrum(zero, atol=atol)
                else zero_label
            )
            wire_symbols = (
                "(0)",
                *_symbols(display_label, num_targets, mark=_UNRESOLVED_MARK),
            )
        else:
            wire_symbols = (
                _SELECTOR_SYMBOL,
                *_symbols(
                    f"diag({zero_label}, {one_label})",
                    num_targets,
                    mark=_UNRESOLVED_MARK,
                ),
            )
        super().__init__(_block_diagonal(zero, one), wire_symbols)

    @property
    def zero_matrix(self) -> NDArray[np.complex128]:
        r"""The immutable unitary :math:`U_1` selected by :math:`|0\rangle`."""
        return self._zero_matrix

    @property
    def one_matrix(self) -> NDArray[np.complex128]:
        r"""The immutable unitary :math:`U_2` selected by :math:`|1\rangle`."""
        return self._one_matrix

    @property
    def zero_label(self) -> str:
        r"""The label of the :math:`|0\rangle` block."""
        return self._zero_label

    @property
    def one_label(self) -> str:
        r"""The label of the :math:`|1\rangle` block."""
        return self._one_label

    @property
    def atol(self) -> float:
        """The absolute tolerance used to validate this instruction."""
        return self._atol

    def __str__(self) -> str:
        return f"diag({self._zero_label}, {self._one_label}){_UNRESOLVED_MARK}"

    def __repr__(self) -> str:
        return (
            f"ControlledUnitary(zero_matrix={self.zero_matrix!r}, "
            f"one_matrix={self.one_matrix!r}, zero_label={self._zero_label!r}, "
            f"one_label={self._one_label!r})"
        )


class CentralMultiplexor(ControlledUnitary):
    r"""The semantic central multiplexor :math:`B_1\oplus B_2`."""

    def __init__(
        self, b1_matrix: ArrayLike, b2_matrix: ArrayLike, *, atol: float = 1e-10
    ) -> None:
        super().__init__(
            b1_matrix,
            b2_matrix,
            zero_label="B_1",
            one_label="B_2",
            atol=atol,
        )

    @property
    def b1_matrix(self) -> NDArray[np.complex128]:
        r"""The target unitary selected by control state :math:`|0\rangle`."""
        return self.zero_matrix

    @property
    def b2_matrix(self) -> NDArray[np.complex128]:
        r"""The target unitary selected by control state :math:`|1\rangle`."""
        return self.one_matrix

    def __repr__(self) -> str:
        return (
            f"CentralMultiplexor(b1_matrix={self.b1_matrix!r}, "
            f"b2_matrix={self.b2_matrix!r})"
        )


class _IdentityControlledUnitary(ControlledUnitary):
    """Shared implementation for semantic instructions of the form ``I⊕U``."""

    def __init__(
        self,
        target_matrix: ArrayLike,
        *,
        label: str,
        atol: float,
    ) -> None:
        target = _unitary(target_matrix, atol=atol)
        self._validate_target(target, atol=atol)
        super().__init__(
            np.eye(target.shape[0], dtype=np.complex128),
            target,
            zero_label="I",
            one_label=label,
            atol=atol,
        )

    @staticmethod
    def _validate_target(target: NDArray[np.complex128], *, atol: float) -> None:
        del target, atol

    @property
    def target_matrix(self) -> NDArray[np.complex128]:
        """The immutable target unitary."""
        return self.one_matrix

    @property
    def label(self) -> str:
        """The target label used in circuit diagrams."""
        return self.one_label

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(target_matrix={self.target_matrix!r}, "
            f"label={self.label!r})"
        )


class ControlledSpectralUnitary(_IdentityControlledUnitary):
    r"""A semantic :math:`I\oplus U` marked for spectral decomposition."""

    def __init__(
        self,
        target_matrix: ArrayLike,
        *,
        label: str = "U",
        atol: float = 1e-10,
    ) -> None:
        super().__init__(target_matrix, label=label, atol=atol)


class ControlledRelativeUnitary(_IdentityControlledUnitary):
    r"""A semantic :math:`I\oplus V` produced by multiplexor factoring."""

    def __init__(
        self,
        target_matrix: ArrayLike,
        *,
        label: str = "V",
        atol: float = 1e-10,
    ) -> None:
        super().__init__(target_matrix, label=label, atol=atol)


class ControlledSeparatedUnitary(_IdentityControlledUnitary):
    r"""A controlled two-qubit :math:`V^*` with antipodally paired spectrum."""

    def __init__(
        self,
        target_matrix: ArrayLike,
        *,
        label: str = "V",
        atol: float = 1e-10,
    ) -> None:
        super().__init__(target_matrix, label=label, atol=atol)

    @staticmethod
    def _validate_target(target: NDArray[np.complex128], *, atol: float) -> None:
        if target.shape != (4, 4):
            raise ValueError("a separated target unitary must have shape (4, 4)")
        if not np.isclose(np.trace(target), 0, rtol=0, atol=atol):
            raise ValueError("a separated target unitary must have zero trace")
