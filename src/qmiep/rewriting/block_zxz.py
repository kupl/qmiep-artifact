"""Block-ZXZ circuit rewrite rule."""

import cirq
import numpy as np

from qmiep.decomposition import BlockZXZDecomposer
from qmiep.gates import (
    ControlledSpectralUnitary,
    ControlledUnitary,
    UnresolvedNQubitUnitary,
)

__all__ = ["block_zxz_operations", "block_zxz_rule"]


def block_zxz_operations(target: cirq.Operation) -> tuple[cirq.Operation, ...]:
    r"""Return one block-ZXZ level of a single unresolved multi-qubit target.

    The circuit-time replacement of :math:`U` is

    .. math::

        (I\oplus C),\ H,\ (B_1\oplus B_2),\ H,\ (I\oplus A),

    In the basis :math:`|q_0\rangle\otimes|t\rangle`, set
    :math:`A=A_2A_1^\dagger`, :math:`B_1=A_1`, and :math:`B_2=A_1B`.
    Indeed,

    .. math::

        A_1\oplus A_2=(I\oplus A)(I\otimes A_1),
        \qquad
        (I\otimes A_1)H=H(I\otimes A_1).

    Thus the common target factor is absorbed into the central multiplexor.
    This exact factorization needs no eigensolve.  Every replacement carries
    the target's tags.
    """
    gate = target.gate
    if not isinstance(gate, UnresolvedNQubitUnitary):
        raise TypeError("target gate must be an UnresolvedNQubitUnitary")
    if len(target.qubits) < 2:
        raise ValueError("block-ZXZ needs a target of at least two qubits")

    factors = BlockZXZDecomposer().solve(gate.matrix, atol=gate.atol)
    identity = np.eye(factors.c.shape[0])
    qubits = target.qubits
    replacements = (
        ControlledSpectralUnitary(
            factors.c,
            label="C",
            atol=gate.atol,
        ).on(*qubits),
        cirq.H.on(qubits[0]),
        ControlledUnitary(
            factors.a1,
            factors.a1 @ factors.b,
            zero_label="B_1",
            one_label="B_2",
            atol=gate.atol,
        ).on(*qubits),
        cirq.H.on(qubits[0]),
        ControlledUnitary(
            identity,
            factors.a2 @ factors.a1.conj().T,
            zero_label="I",
            one_label="A",
            atol=gate.atol,
        ).on(*qubits),
    )
    if not target.tags:
        return replacements
    return tuple(replacement.with_tags(*target.tags) for replacement in replacements)


def _selected_operation(
    circuit: cirq.AbstractCircuit,
    target: cirq.Operation,
    context: cirq.TransformerContext,
) -> cirq.Operation:
    """Validate that ``target`` names one rewritable operation of ``circuit``."""
    if not isinstance(target, cirq.Operation):
        raise TypeError("target must be a cirq.Operation")
    occurrences = sum(operation is target for operation in circuit.all_operations())
    if occurrences != 1:
        raise ValueError(
            "target must directly reference exactly one operation in circuit; "
            f"found {occurrences}"
        )
    if any(tag in context.tags_to_ignore for tag in target.tags):
        raise ValueError("target carries a tag ignored by the context")
    return target


def _expand_selected(
    circuit: cirq.AbstractCircuit, target: cirq.Operation
) -> cirq.Circuit:
    """Expand ``target`` in place, leaving every other operation identical.

    The first replacement takes the target's slot.  A correction joins the next
    moment only when that moment is free on *every* qubit of the target, and
    takes a fresh moment otherwise.  Requiring the whole target register to be
    free keeps the replacement atomic with respect to anything already
    scheduled on those qubits, while still letting independent blocks expanded
    one at a time land exactly where one untargeted pass would put them.
    """
    replacements = block_zxz_operations(target)
    moments = list(circuit)
    index = next(
        position
        for position, moment in enumerate(moments)
        if any(operation is target for operation in moment.operations)
    )
    moments[index] = cirq.Moment(
        replacements[0] if operation is target else operation
        for operation in moments[index].operations
    )

    for offset, correction in enumerate(replacements[1:], start=index + 1):
        if offset < len(moments) and not moments[offset].operates_on(target.qubits):
            moments[offset] = moments[offset].with_operation(correction)
        else:
            moments.insert(offset, cirq.Moment(correction))
    return cirq.Circuit(moments)


@cirq.transformer
def block_zxz_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    target: cirq.Operation | None = None,
) -> cirq.Circuit:
    r"""Replace an unresolved multi-qubit unitary by one block-ZXZ level.

    Without ``target`` every eligible :class:`UnresolvedNQubitUnitary` in the
    circuit is expanded at once.  Passing ``target`` expands only that one
    operation, selected by object identity, so a circuit holding several
    unresolved unitaries -- as block-ZXZ recursion produces -- can be driven
    one block at a time.  Targeting matches top-level operations, so it ignores
    ``context.deep``; every other operation is left untouched by identity.
    """
    transformer_context = context or cirq.TransformerContext()
    if target is not None:
        frozen = cirq.FrozenCircuit(circuit)
        selected = _selected_operation(frozen, target, transformer_context)
        return _expand_selected(frozen, selected)

    def map_operation(operation: cirq.Operation, _: int) -> cirq.OP_TREE:
        gate = operation.gate
        if not isinstance(gate, UnresolvedNQubitUnitary) or len(operation.qubits) == 1:
            return operation
        return block_zxz_operations(operation)

    mapped = cirq.map_operations_and_unroll(
        circuit,
        map_operation,
        deep=transformer_context.deep,
        tags_to_ignore=transformer_context.tags_to_ignore,
    )
    return cirq.Circuit(mapped)
