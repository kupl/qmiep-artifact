"""Relabel the four unresolved target-register blocks as a semantic gauge chain."""

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Literal

import cirq

from qmiep.gates import (
    GaugeUnitary,
    UniformlyControlledRz,
    UnresolvedNQubitUnitary,
)

_Role = Literal["L", "M", "N", "O"]
_ROLES: tuple[_Role, ...] = ("L", "M", "N", "O")
_SOURCE_ROLES: dict[str, _Role] = {
    "Q_C†_S": "L",
    "B_L": "M",
    "B_R": "N",
    "Q_A_PD_S": "O",
    "Q_A_S": "O",
}


def _source_role(operation: cirq.Operation) -> _Role | None:
    gate = operation.gate
    if type(gate) is not UnresolvedNQubitUnitary or len(operation.qubits) < 2:
        return None
    return _SOURCE_ROLES.get(gate.label)


def _has_central_spectrum(
    operations: Sequence[cirq.Operation],
    left: int,
    right: int,
    targets: tuple[cirq.Qid, ...],
) -> bool:
    return any(
        isinstance(operation.gate, UniformlyControlledRz)
        and operation.qubits[1:] == targets
        for operation in operations[left + 1 : right]
    )


def _find_chains(
    operations: Sequence[cirq.Operation], tags_to_ignore: Sequence[Hashable]
) -> tuple[tuple[int, int, int, int], ...]:
    positions: dict[tuple[cirq.Qid, ...], list[tuple[int, _Role]]] = defaultdict(list)
    for index, operation in enumerate(operations):
        role = _source_role(operation)
        if role is not None:
            positions[operation.qubits].append((index, role))

    ignored = set(tags_to_ignore)
    candidates: list[tuple[int, int, int, int]] = []
    for targets, entries in positions.items():
        for offset in range(len(entries) - len(_ROLES) + 1):
            window = entries[offset : offset + len(_ROLES)]
            raw_indices = tuple(index for index, _ in window)
            indices = (
                raw_indices[0],
                raw_indices[1],
                raw_indices[2],
                raw_indices[3],
            )
            roles = tuple(role for _, role in window)
            if roles != _ROLES:
                continue
            if any(not ignored.isdisjoint(operations[index].tags) for index in indices):
                continue
            if not _has_central_spectrum(operations, indices[1], indices[2], targets):
                continue
            candidates.append(indices)

    matches: list[tuple[int, int, int, int]] = []
    previous_end = -1
    for candidate in sorted(candidates):
        if candidate[0] > previous_end:
            matches.append(candidate)
            previous_end = candidate[-1]
    return tuple(matches)


@cirq.transformer(add_deep_support=True)
def relabel_gauge_unitaries_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
) -> cirq.Circuit:
    r"""Relabel one recognized four-block chain as ``L,M,N,O``.

    Recognize ``Q_C†_S, B_L, B_R, Q_A_S`` for exact/D-modulus synthesis
    and ``Q_C†_S, B_L, B_R, Q_A_PD_S`` for PD-modulus synthesis. The four
    replacements are transactional and become :class:`GaugeUnitary`
    instructions, so subsequent synthesis matches semantic roles rather than
    circuit-diagram text.
    """
    transformer_context = context or cirq.TransformerContext()
    operations = tuple(circuit.all_operations())
    chains = _find_chains(operations, transformer_context.tags_to_ignore)
    if not chains:
        return cirq.Circuit(circuit)

    roles_by_index = {
        index: role
        for chain in chains
        for index, role in zip(chain, _ROLES, strict=True)
    }
    rewritten: list[cirq.Operation] = []
    for index, operation in enumerate(operations):
        role = roles_by_index.get(index)
        if role is None:
            rewritten.append(operation)
            continue
        gate = operation.gate
        assert isinstance(gate, UnresolvedNQubitUnitary)
        replacement = GaugeUnitary(gate.matrix, role=role, atol=gate.atol).on(
            *operation.qubits
        )
        rewritten.append(replacement.with_tags(*operation.tags))
    return cirq.Circuit(cirq.Moment(operation) for operation in rewritten)
