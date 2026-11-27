"""Exchange a multiplexed permutation and an R gate in a two-operation circuit."""

import cirq
import numpy as np

from qmiep.gates import MultiplexedPermutation, RGate
from qmiep.gates.atomic import _fast_walsh_hadamard_transform


def _switched_rotation(
    multiplexed_permutation: cirq.Operation,
    rotation: cirq.Operation,
    *,
    rotation_first: bool,
) -> cirq.Operation:
    r"""Relabel sector angles, then restore the same open Gray cascade.

    For chronological Rz angles a_j and Gray words g_j, moving all Xs left
    gives theta_x = sum_j (-1)**(x dot g_j) a_j. Thus a Walsh-Hadamard
    transform converts between circuit angles and sector angles. If
    C|x> = |pi(x)>, C -> R uses theta'_x = theta_{pi(x)}, whereas
    R -> C uses theta'_x = theta_{pi^{-1}(x)}.
    The same angle conversion applies to left and right fragments.
    """
    gate = rotation.gate
    permutation_gate = multiplexed_permutation.gate
    assert type(gate) is RGate
    assert type(permutation_gate) is MultiplexedPermutation
    operations = tuple(cirq.decompose_once(rotation))
    angles = np.asarray([op.gate.exponent * np.pi for op in operations[::2]])
    indices = np.arange(len(angles))
    gray = indices ^ (indices >> 1)
    coefficients = np.empty_like(angles)
    coefficients[gray] = angles
    sector_angles = _fast_walsh_hadamard_transform(coefficients)

    # Columns give pi(x); rows give pi^{-1}(x).
    images = np.argmax(
        permutation_gate.permutation.array, axis=1 if rotation_first else 0
    )
    sector_angles = sector_angles[images]
    angles = _fast_walsh_hadamard_transform(sector_angles)[gray] / len(angles)
    fragment = []
    for index, operation in enumerate(operations):
        if index % 2 == 0:
            operation = operation.untagged.with_gate(
                cirq.rz(angles[index // 2])
            ).with_tags(*operation.tags)
        tags = tuple(tag for tag in rotation.tags if tag not in operation.tags)
        fragment.append(operation.with_tags(*tags) if tags else operation)
    return (
        RGate(
            cirq.FrozenCircuit(cirq.Moment(op) for op in fragment),
            qubit_order=rotation.qubits,
            side=gate.side,
            atol=gate.atol,
        )
        .on(*rotation.qubits)
        .with_tags(*rotation.tags)
    )


@cirq.transformer
def switch_multiplexed_permutation_through_r_gate_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
) -> cirq.Circuit:
    r"""Switch ``C_P -> R`` or ``R -> C_P`` in a two-operation subcircuit.

    C_P = P_0 ⊕ P_1 acts on exactly (q1, ..., qk), with q1 selecting
    P_b, and R acts on (q0, ..., qk). For P_b|y> = |pi_b(y)>, C_P -> R
    uses R' = C_P^dagger R C_P and theta'[b, y] = theta[b, pi_b(y)].
    R -> C_P uses R' = C_P R C_P^dagger and
    theta'[b, y] = theta[b, pi_b^{-1}(y)]. Both preserve R's side.
    Since C_P preserves q1, it commutes with the residual CX(q1, q0)
    of R's open Gray cascade.

    The input must contain exactly these two top-level operations in either
    order. Tags are preserved; ignored operations are rejected. The input
    circuit is not modified. No unitary matrices are constructed.
    """
    operations = tuple(circuit.all_operations())
    if len(operations) != 2:
        raise ValueError(
            "circuit must contain exactly one MultiplexedPermutation and one RGate"
        )
    rotation_first = type(operations[0].gate) is RGate
    multiplexed_permutation, rotation = (
        operations[::-1] if rotation_first else operations
    )
    if (
        type(multiplexed_permutation.gate) is not MultiplexedPermutation
        or type(rotation.gate) is not RGate
    ):
        raise TypeError(
            "circuit must contain exactly one MultiplexedPermutation and one RGate"
        )
    if multiplexed_permutation.qubits != rotation.qubits[1:]:
        raise ValueError("multiplexed_permutation must act on exactly (q1, ..., qk)")
    ignored = set((context or cirq.TransformerContext()).tags_to_ignore)
    if any(not ignored.isdisjoint(op.tags) for op in operations):
        raise ValueError("the input pair carries an ignored tag")
    replacement = _switched_rotation(
        multiplexed_permutation, rotation, rotation_first=rotation_first
    )
    return cirq.Circuit(
        (multiplexed_permutation, replacement)
        if rotation_first
        else (replacement, multiplexed_permutation)
    )


__all__ = ["switch_multiplexed_permutation_through_r_gate_rule"]
