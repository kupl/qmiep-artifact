"""Move Hadamards through boundary CX gates, turning them into CZ gates."""

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Literal

import cirq

_Direction = Literal["cx_h", "h_cx"]
_Match = tuple[int, int, _Direction]


def _is_hadamard(operation: cirq.Operation) -> bool:
    return operation.gate == cirq.H


def _is_cx_targeting(operation: cirq.Operation, target: cirq.Qid) -> bool:
    return operation.gate == cirq.CX and operation.qubits[1] == target


def _find_matches(
    operations: Sequence[cirq.Operation], tags_to_ignore: Sequence[Hashable]
) -> tuple[_Match, ...]:
    """Find non-overlapping CX/H pairs adjacent in the target's causal order."""
    positions_by_qubit: dict[cirq.Qid, list[int]] = defaultdict(list)
    for index, operation in enumerate(operations):
        for qubit in operation.qubits:
            positions_by_qubit[qubit].append(index)

    ignored = set(tags_to_ignore)
    candidates: list[_Match] = []
    for target, positions in positions_by_qubit.items():
        for left, right in zip(positions, positions[1:], strict=False):
            before = operations[left]
            after = operations[right]
            if not ignored.isdisjoint(before.tags) or not ignored.isdisjoint(
                after.tags
            ):
                continue
            if _is_cx_targeting(before, target) and _is_hadamard(after):
                candidates.append((left, right, "cx_h"))
            elif _is_hadamard(before) and _is_cx_targeting(after, target):
                candidates.append((left, right, "h_cx"))

    matches: list[_Match] = []
    previous_right = -1
    for match in sorted(candidates):
        if match[0] > previous_right:
            matches.append(match)
            previous_right = match[1]
    return tuple(matches)


def _cz_from(cx: cirq.Operation) -> cirq.Operation:
    return cirq.CZ.on(*cx.qubits).with_tags(*cx.tags)


@cirq.transformer(add_deep_support=True)
def hadamard_cx_to_cz_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
) -> cirq.Circuit:
    r"""Apply the two boundary identities from Section 5.2.

    .. math::

        \operatorname{CX}_{i\to0};H_0
        &=H_0;\operatorname{CZ}_{i0},\\
        H_0;\operatorname{CX}_{i\to0}
        &=\operatorname{CZ}_{i0};H_0.

    Matching uses causal adjacency on the Hadamard/CX target. Operations between
    the pair therefore do not touch that target, so moving the Hadamard across
    them is unitary-preserving.
    """
    transformer_context = context or cirq.TransformerContext()
    operations = tuple(circuit.all_operations())
    matches = _find_matches(operations, transformer_context.tags_to_ignore)
    if not matches:
        return cirq.Circuit(circuit)

    rewritten: list[cirq.Operation] = []
    cursor = 0
    for left, right, direction in matches:
        rewritten.extend(operations[cursor:left])
        between = operations[left + 1 : right]
        if direction == "cx_h":
            cx, hadamard = operations[left], operations[right]
            rewritten.extend((hadamard, _cz_from(cx), *between))
        else:
            hadamard, cx = operations[left], operations[right]
            rewritten.extend((*between, _cz_from(cx), hadamard))
        cursor = right + 1
    rewritten.extend(operations[cursor:])
    return cirq.Circuit(cirq.Moment(operation) for operation in rewritten)
