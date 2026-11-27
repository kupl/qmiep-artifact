"""Exchange a multiplexed permutation and UCG Rz by relabeling its sectors."""

import cirq
import numpy as np

from qmiep.gates import MultiplexedPermutation, UniformlyControlledRz


@cirq.transformer
def switch_multiplexed_permutation_through_ucg_rz_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
) -> cirq.Circuit:
    r"""Switch ``P -> U`` or ``U -> P`` in a two-operation subcircuit.

    In the basis |q0>|x>, U = D ⊕ D† targets q0 and P = P0 ⊕ P1 acts
    on exactly (q1, ..., qk). Write P|x> = |pi(x)> and D = diag(d_x).
    With C = I ⊗ P, chronological P -> U becomes U' -> P, where
    U' = C† U C and d'_x = d_{pi(x)}. Conversely U -> P becomes P -> U',
    where U' = C U C† and d'_x = d_{pi^{-1}(x)}.

    Only diagonal entries are reordered; no decomposition or angle
    extraction is needed. The permutation operation, UCG label, tolerance,
    CX boundary, and tags are preserved. The input is not modified.
    Exactly two top-level operations are required; ignored tags are rejected.
    """
    operations = tuple(circuit.all_operations())
    if len(operations) != 2:
        raise ValueError(
            "circuit must contain exactly one MultiplexedPermutation "
            "and one UniformlyControlledRz"
        )
    rotation_first = type(operations[0].gate) is UniformlyControlledRz
    permutation, rotation = operations[::-1] if rotation_first else operations
    if (
        type(permutation.gate) is not MultiplexedPermutation
        or type(rotation.gate) is not UniformlyControlledRz
    ):
        raise TypeError(
            "circuit must contain exactly one MultiplexedPermutation "
            "and one UniformlyControlledRz"
        )
    if permutation.qubits != rotation.qubits[1:]:
        raise ValueError("multiplexed_permutation must act on exactly (q1, ..., qk)")
    ignored = set((context or cirq.TransformerContext()).tags_to_ignore)
    if any(not ignored.isdisjoint(op.tags) for op in operations):
        raise ValueError("the input pair carries an ignored tag")

    gate = rotation.gate
    # Columns give pi(x); rows give pi^{-1}(x).
    images = np.argmax(
        permutation.gate.permutation.array, axis=1 if rotation_first else 0
    )
    replacement = (
        UniformlyControlledRz(
            np.diag(gate.branch_matrix)[images],
            label=gate.label,
            cx_boundary=gate.cx_boundary,
            atol=gate.atol,
        )
        .on(*rotation.qubits)
        .with_tags(*rotation.tags)
    )
    return cirq.Circuit(
        (permutation, replacement) if rotation_first else (replacement, permutation)
    )


__all__ = ["switch_multiplexed_permutation_through_ucg_rz_rule"]
