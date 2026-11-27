"""Shared implementation for immutable matrix-valued Cirq gates."""

from __future__ import annotations

import cirq
import numpy as np
from numpy.typing import NDArray

from qmiep.utils import _readonly

_UNRESOLVED_MARK = "?"
_RESOLVED_MARK = "!"
_SELECTOR_SYMBOL = "□"  # Multiplexor selector, distinct from a binary control.


def _symbols(label: str, num_qubits: int, *, mark: str = "") -> tuple[str, ...]:
    """Return the per-qid diagram symbols for a gate labelled ``label``.

    ``mark`` is :data:`_UNRESOLVED_MARK` for a semantic instruction awaiting a
    rewrite and :data:`_RESOLVED_MARK` for a gate carrying a concrete
    decomposition, so a diagram separates the two even under one label.
    """
    # Combining marks have zero terminal width but confuse Cirq's text padding.
    label = label.replace("\N{COMBINING TILDE}", "~").replace(
        "\N{COMBINING CIRCUMFLEX ACCENT}", "^"
    )
    label = f"{label}{mark}"
    if num_qubits == 1:
        return (label,)
    return tuple(f"{label}[{index}]" for index in range(1, num_qubits + 1))


def _block_diagonal(
    zero_block: NDArray[np.complex128], one_block: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    dimension = zero_block.shape[0]
    matrix = np.zeros((2 * dimension, 2 * dimension), dtype=np.complex128)
    matrix[:dimension, :dimension] = zero_block
    matrix[dimension:, dimension:] = one_block
    return _readonly(matrix)


def _qubit_count(dimension: int, *, matrix_name: str) -> int:
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError(f"the {matrix_name} dimension must be a positive power of two")
    return dimension.bit_length() - 1


@cirq.value_equality(distinct_child_types=True)
class _MatrixGate(cirq.Gate):
    """Common Cirq protocols for immutable matrix-valued gates."""

    def __init__(
        self, matrix: NDArray[np.complex128], wire_symbols: tuple[str, ...]
    ) -> None:
        self._matrix = matrix
        self._wire_symbols = wire_symbols

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """The immutable matrix represented by this gate."""
        return self._matrix

    def _num_qubits_(self) -> int:
        return len(self._wire_symbols)

    def _unitary_(self) -> NDArray[np.complex128] | None:
        return self._matrix

    def _circuit_diagram_info_(
        self, args: cirq.CircuitDiagramInfoArgs
    ) -> cirq.CircuitDiagramInfo:
        del args
        return cirq.CircuitDiagramInfo(wire_symbols=self._wire_symbols)

    def _value_equality_values_(self) -> tuple[object, ...]:
        return self._matrix.shape, self._matrix.tobytes(), self._wire_symbols
