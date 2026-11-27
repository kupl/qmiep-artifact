"""Merge the Section 5.2 central block into one quantum multiplexor."""

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import cirq
import numpy as np

from qmiep.gates import CentralMultiplexor, ControlledUnitary


@dataclass(frozen=True, slots=True)
class _Match:
    first_h: int
    left_cz: int
    central: int
    right_cz: int
    last_h: int


def _is_hadamard(operation: cirq.Operation, qubit: cirq.Qid) -> bool:
    return operation.gate == cirq.H and operation.qubits == (qubit,)


def _cz_partner(operation: cirq.Operation, qubit: cirq.Qid) -> cirq.Qid | None:
    if operation.gate != cirq.CZ or qubit not in operation.qubits:
        return None
    return operation.qubits[0] if operation.qubits[1] == qubit else operation.qubits[1]


def _find_matches(
    operations: Sequence[cirq.Operation], tags_to_ignore: Sequence[Hashable]
) -> tuple[_Match, ...]:
    """Find non-overlapping ``H-CZ-multiplexor-CZ-H`` causal patterns."""
    positions_by_qubit: dict[cirq.Qid, list[int]] = defaultdict(list)
    for index, operation in enumerate(operations):
        for qubit in operation.qubits:
            positions_by_qubit[qubit].append(index)

    ignored = set(tags_to_ignore)
    candidates: list[_Match] = []
    for control, positions in positions_by_qubit.items():
        for offset in range(len(positions) - 4):
            indices = positions[offset : offset + 5]
            first_h, left_cz, central, right_cz, last_h = (
                operations[index] for index in indices
            )
            central_gate = central.gate
            if (
                not _is_hadamard(first_h, control)
                or not _is_hadamard(last_h, control)
                or type(central_gate) is not ControlledUnitary
                or central.qubits[0] != control
            ):
                continue

            targets = set(central.qubits[1:])
            if (
                _cz_partner(left_cz, control) not in targets
                or _cz_partner(right_cz, control) not in targets
            ):
                continue
            if any(
                not ignored.isdisjoint(operation.tags)
                for operation in (first_h, left_cz, central, right_cz, last_h)
            ):
                continue
            candidates.append(_Match(*indices))

    matches: list[_Match] = []
    previous_last_h = -1
    for match in sorted(candidates, key=lambda candidate: candidate.first_h):
        if match.first_h > previous_last_h:
            matches.append(match)
            previous_last_h = match.last_h
    return tuple(matches)


def _replacement(
    operations: Sequence[cirq.Operation],
    match: _Match,
    tags_to_ignore: Sequence[Hashable],
) -> tuple[cirq.Operation, ...] | None:
    central = operations[match.central]
    gate = central.gate
    if not isinstance(gate, ControlledUnitary):
        return None

    scope = frozenset(central.qubits)
    ignored = set(tags_to_ignore)
    absorbed: list[cirq.Operation] = []
    disjoint: list[cirq.Operation] = []
    for operation in operations[match.left_cz : match.right_cz + 1]:
        operation_qubits = frozenset(operation.qubits)
        if operation_qubits.issubset(scope):
            if not ignored.isdisjoint(operation.tags) or not cirq.has_unitary(
                operation
            ):
                return None
            absorbed.append(operation)
        elif operation_qubits.isdisjoint(scope):
            disjoint.append(operation)
        else:
            return None

    matrix = cirq.Circuit(absorbed).unitary(qubit_order=central.qubits)
    dimension = gate.zero_matrix.shape[0]
    if not np.allclose(matrix[:dimension, dimension:], 0, rtol=0, atol=gate.atol):
        return None
    if not np.allclose(matrix[dimension:, :dimension], 0, rtol=0, atol=gate.atol):
        return None

    tags: list[Hashable] = []
    for operation in absorbed:
        tags.extend(tag for tag in operation.tags if tag not in tags)
    merged = CentralMultiplexor(
        matrix[:dimension, :dimension],
        matrix[dimension:, dimension:],
        atol=gate.atol,
    ).on(*central.qubits)
    return (*disjoint, merged.with_tags(*tags))


@cirq.transformer(add_deep_support=True)
def merge_central_multiplexor_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
) -> cirq.Circuit:
    r"""Merge both boundary CZs and their central block into :math:`B_1\oplus B_2`.

    For the three-qubit layout currently produced by the synthesizer,

    .. math::

        B_1 &= Q_V^{\dagger}A_1Q_C,\\
        B_2 &= (I\otimes Z)Q_V^{\dagger}
               A_1BQ_C(Z\otimes I).

    The implementation computes these blocks from the matched subcircuit, so it
    does not depend on diagram labels or fixed moment indices.
    """
    transformer_context = context or cirq.TransformerContext()
    operations = tuple(circuit.all_operations())
    matches = _find_matches(operations, transformer_context.tags_to_ignore)

    rewritten: list[cirq.Operation] = []
    cursor = 0
    changed = False
    for match in matches:
        replacement = _replacement(
            operations, match, transformer_context.tags_to_ignore
        )
        if replacement is None:
            continue
        rewritten.extend(operations[cursor : match.left_cz])
        rewritten.extend(replacement)
        cursor = match.right_cz + 1
        changed = True
    if not changed:
        return cirq.Circuit(circuit)

    rewritten.extend(operations[cursor:])
    return cirq.Circuit(cirq.Moment(operation) for operation in rewritten)
