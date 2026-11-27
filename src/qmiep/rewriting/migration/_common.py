"""Shared PD synthesis, correction migration, and destination merging."""

from typing import Literal

import cirq
import numpy as np

from qmiep._matrix import _operation_matrix
from qmiep.gates import (
    DiagonalGate,
    MultiplexedPermutation,
    NQubitUnitaryModuloPermutationDiagonal,
    RGate,
    UniformlyControlledRz,
    UnresolvedNQubitUnitary,
)
from qmiep.matrices import DiagonalMatrix, PermutationMatrix
from qmiep.rewriting.switch import (
    switch_diagonal_through_r_gate_rule,
    switch_multiplexed_permutation_through_r_gate_rule,
    switch_multiplexed_permutation_through_ucg_rz_rule,
)
from qmiep.rust import PDMode, _NativeCircuit


def _transpose_decomposition(
    operation: cirq.Operation,
) -> cirq.FrozenCircuit | _NativeCircuit:
    """Transpose a resolved circuit, preserving phase rather than conjugating it."""
    gate = operation.gate
    if (
        isinstance(gate, NQubitUnitaryModuloPermutationDiagonal)
        and gate._native_circuit is not None
    ):
        return gate._native_circuit.transpose()
    primitives = tuple(
        cirq.decompose(
            operation,
            keep=lambda op: len(op.qubits) <= 1 or op.gate == cirq.CX,
        )
    )
    return cirq.FrozenCircuit(
        cirq.Moment(
            op
            if not op.qubits or op.gate == cirq.CX
            else cirq.MatrixGate(cirq.unitary(op).T).on(*op.qubits).with_tags(*op.tags)
        )
        for op in reversed(primitives)
    )


def _pd_migration(
    circuit: cirq.AbstractCircuit,
    *,
    rotation_type: type[RGate] | type[UniformlyControlledRz],
    context: cirq.TransformerContext | None = None,
    direction: Literal["left", "right"] = "right",
    mode: Literal["analytic", "heuristic"] = "heuristic",
    kind: PDMode = "pd",
) -> cirq.Circuit:
    """Migrate corrections for one target-first R or UCG(Rz) pattern."""
    if direction not in ("left", "right"):
        raise ValueError("direction must be 'left' or 'right'")
    if mode not in ("analytic", "heuristic"):
        raise ValueError("mode must be 'analytic' or 'heuristic'")
    operations = tuple(circuit.all_operations())
    if len(operations) != 3:
        raise ValueError("PD migration requires exactly three operations: U -> F -> V")
    left, rotation, right = operations
    if type(rotation.gate) is not rotation_type:
        raise TypeError(
            f"the middle operation must be a {rotation_type.__name__} with target q0"
        )
    selectors = set(rotation.qubits[1:])
    for operation in (left, right):
        if set(operation.qubits) != selectors:
            raise ValueError(
                "both unitaries must act on all rotation selectors (q1, ..., qk)"
            )
        if not cirq.has_unitary(operation):
            raise TypeError("both outer operations must be unitary")
    ignored = set((context or cirq.TransformerContext()).tags_to_ignore)
    if any(not ignored.isdisjoint(operation.tags) for operation in operations):
        raise ValueError("the input pattern carries an ignored tag")

    # Import locally: the synthesis synthesizer itself imports rewriting rules.
    from qmiep.synthesizer import NQubitUnitarySynthesisModuloPermutationDiagonal

    synthesizer = NQubitUnitarySynthesisModuloPermutationDiagonal(mode=mode, kind=kind)
    ordered_selectors = rotation.qubits[1:]
    atol = max(getattr(op.gate, "atol", 1e-10) for op in operations)
    source_op, destination = (right, left) if direction == "left" else (left, right)
    label = getattr(source_op.gate, "label", "U")
    matrix = _operation_matrix(source_op, ordered_selectors)
    source = (
        UnresolvedNQubitUnitary(
            matrix.T if direction == "left" else matrix,
            label=label,
            atol=atol,
        )
        .on(*ordered_selectors)
        .with_tags(*source_op.tags)
    )
    # A single selector needs no entanglers or PD correction: P = D = I.
    if len(ordered_selectors) == 1:
        residual = (
            NQubitUnitaryModuloPermutationDiagonal(
                matrix,
                cirq.FrozenCircuit(cirq.MatrixGate(matrix).on(*ordered_selectors)),
                qubit_order=ordered_selectors,
                label=f"{label}_PD",
                atol=atol,
            )
            .on(*ordered_selectors)
            .with_tags(*source_op.tags)
        )
        merged = (
            UnresolvedNQubitUnitary(
                _operation_matrix(destination, ordered_selectors),
                label=getattr(destination.gate, "label", "V"),
                atol=atol,
            )
            .on(*ordered_selectors)
            .with_tags(*dict.fromkeys((*source_op.tags, *destination.tags)))
        )
        result = (
            (merged, rotation, residual)
            if direction == "left"
            else (residual, rotation, merged)
        )
        return cirq.Circuit(cirq.Moment(op) for op in result)

    synthesized = synthesizer.run(cirq.Circuit(source), target=source, context=context)
    residual, first, second = tuple(synthesized.all_operations())
    permutation_first = isinstance(first.gate, MultiplexedPermutation)
    permutation, diagonal = (first, second) if permutation_first else (second, first)
    residual_gate = residual.gate
    assert isinstance(residual_gate, NQubitUnitaryModuloPermutationDiagonal)
    atol = max(atol, residual_gate.atol)
    residual = (
        NQubitUnitaryModuloPermutationDiagonal(
            (
                residual_gate.residual_matrix.T
                if direction == "left"
                else residual_gate.residual_matrix
            ),
            (
                _transpose_decomposition(residual)
                if direction == "left"
                else (residual_gate._native_circuit or residual_gate.decomposition)
            ),
            qubit_order=(
                ordered_selectors
                if direction == "left"
                else cirq.LineQubit.range(len(ordered_selectors))
            ),
            label=f"{label}_PD",
            atol=atol,
        )
        .on(*ordered_selectors)
        .with_tags(*residual.tags)
    )

    # Recursive synthesis may expose corrections on only the final three qids.
    p = _operation_matrix(permutation, ordered_selectors)
    if permutation.qubits != ordered_selectors:
        half = p.shape[0] // 2
        if np.any(p[:half, half:]) or np.any(p[half:, :half]):
            raise ValueError("the residual permutation must preserve selector q1")
        permutation = (
            MultiplexedPermutation(
                PermutationMatrix(p[:half, :half]),
                PermutationMatrix(p[half:, half:]),
                label=permutation.gate.label,
            )
            .on(*ordered_selectors)
            .with_tags(*permutation.tags)
        )
    d = _operation_matrix(diagonal, ordered_selectors)
    # The 2q base returns D -> P; Step 13 already returns P -> D_after.
    diagonal_values = np.diag(d)
    if not permutation_first:
        diagonal_values = diagonal_values[np.argmax(p, axis=1)]
    diagonal = (
        DiagonalGate(
            DiagonalMatrix.from_diagonal(diagonal_values, atol=atol),
            label=f"{diagonal.gate.label}'",
        )
        .on(*ordered_selectors)
        .with_tags(*diagonal.tags)
    )

    # Synthesis of V.T gives V.T = D' P W, hence V = W.T P.T D'.
    # Circuit order on the left of W.T is D' -> P.T, not its adjoint.
    if direction == "left":
        permutation = (
            MultiplexedPermutation(
                PermutationMatrix(permutation.gate.zero_permutation.array.T),
                PermutationMatrix(permutation.gate.one_permutation.array.T),
                label=permutation.gate.label,
            )
            .on(*ordered_selectors)
            .with_tags(*permutation.tags)
        )

    if rotation_type is RGate:
        switched = tuple(
            switch_diagonal_through_r_gate_rule(
                (
                    cirq.Circuit(rotation, diagonal)
                    if direction == "left"
                    else cirq.Circuit(diagonal, rotation)
                ),
                context=context,
            ).all_operations()
        )
        diagonal, rotation = switched if direction == "left" else switched[::-1]
        switch_permutation = switch_multiplexed_permutation_through_r_gate_rule
    else:
        # UCG(Rz) and I tensor D' are both diagonal: D' -> F becomes F -> D'
        # with both operations unchanged, before switching P through F.
        switch_permutation = switch_multiplexed_permutation_through_ucg_rz_rule
    switched = tuple(
        switch_permutation(
            (
                cirq.Circuit(rotation, permutation)
                if direction == "left"
                else cirq.Circuit(permutation, rotation)
            ),
            context=context,
        ).all_operations()
    )
    permutation, rotation = switched if direction == "left" else switched[::-1]
    merged_tags = tuple(
        dict.fromkeys(
            (*source_op.tags, *permutation.tags, *diagonal.tags, *destination.tags)
        )
    )
    destination_matrix = _operation_matrix(destination, ordered_selectors)
    p = cirq.unitary(permutation)
    d = np.diag(cirq.unitary(diagonal))
    # Chronological V,D,P means P D V; P,D,V means V D P.
    merged_matrix = (
        (d[:, None] * destination_matrix)[np.argmax(p, axis=1)]
        if direction == "left"
        else (destination_matrix * d)[:, np.argmax(p, axis=0)]
    )
    merged = (
        UnresolvedNQubitUnitary(
            merged_matrix,
            label=getattr(destination.gate, "label", "V"),
            atol=atol,
        )
        .on(*ordered_selectors)
        .with_tags(*merged_tags)
    )
    result = (
        (merged, rotation, residual)
        if direction == "left"
        else (residual, rotation, merged)
    )
    return cirq.Circuit(cirq.Moment(op) for op in result)
