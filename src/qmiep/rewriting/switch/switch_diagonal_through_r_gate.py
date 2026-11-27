"""Commute a selector diagonal through an open R-gate cascade."""

import cirq

from qmiep.gates import DiagonalGate, RGate


@cirq.transformer
def switch_diagonal_through_r_gate_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
) -> cirq.Circuit:
    r"""Exchange ``D -> R`` and ``R -> D`` without changing either gate.

    In target-first order |q0>|x>, write R = sum_x R_x tensor |x><x|
    and I tensor D = sum_x d_x I tensor |x><x|. Both products equal
    sum_x d_x R_x tensor |x><x|, for either side of the open cascade.
    D may act on any subset of R's selectors and additional spectator
    qubits, in any qubit order, but must not act on the target q0.
    Including spectators y, both products act within each (x, y) sector
    as d_(x,y) R_x, so neither operation changes.

    Exactly two top-level operations are required. The input is unchanged,
    tags are preserved, and ignored tags are rejected. No matrices or
    decompositions are constructed.
    """
    operations = tuple(circuit.all_operations())
    if len(operations) != 2:
        raise ValueError("circuit must contain exactly one DiagonalGate and one RGate")
    diagonal, rotation = (
        operations[::-1] if type(operations[0].gate) is RGate else operations
    )
    if not isinstance(diagonal.gate, DiagonalGate) or type(rotation.gate) is not RGate:
        raise TypeError("circuit must contain exactly one DiagonalGate and one RGate")
    if rotation.qubits[0] in diagonal.qubits:
        raise ValueError("diagonal must not act on the RGate target (q0)")
    ignored = set((context or cirq.TransformerContext()).tags_to_ignore)
    if any(not ignored.isdisjoint(op.tags) for op in operations):
        raise ValueError("the input pair carries an ignored tag")
    return cirq.Circuit(cirq.Moment(op) for op in reversed(operations))


__all__ = ["switch_diagonal_through_r_gate_rule"]
