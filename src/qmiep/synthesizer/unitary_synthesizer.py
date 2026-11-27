"""Targeted N-qubit unitary-synthesis synthesizers."""

from typing import Literal

import cirq
import numpy as np
from cirq.transformers.transformer_api import NoOpTransformerLogger
from numpy.typing import NDArray

from qmiep._matrix import _operation_matrix, _remove_left_correction
from qmiep._validation import _validation_enabled
from qmiep.decomposition import decompose_two_qubit_unitary
from qmiep.gates import (
    ControlledSpectralUnitary,
    DiagonalGate,
    MultiplexedDiagonal,
    MultiplexedPermutation,
    NQubitUnitary,
    NQubitUnitaryModuloDiagonal,
    NQubitUnitaryModuloPermutationDiagonal,
    RGate,
    UniformlyControlledRz,
    UnresolvedNQubitUnitary,
)
from qmiep.matrices import DiagonalMatrix, PermutationMatrix
from qmiep.rewriting import (
    block_zxz_rule,
    hadamard_cx_to_cz_rule,
    merge_central_multiplexor_rule,
    multiplexed_unitary_decomposition_by_spectrum_rule,
    pd_migration_through_r_rule,
    pd_migration_through_ucg_rule,
    relabel_gauge_unitaries_rule,
    switch_diagonal_through_r_gate_rule,
)
from qmiep.rust import DiagonalMode, ExactMode, PDMode, _NativeCircuit, solve_spectrum
from qmiep.synthesizer._backends import (
    _synthesis_algorithm,
    _synthesis_backends,
    _synthesis_mode,
    _synthesis_seed,
)
from qmiep.synthesizer._numerical import Kind, _Program, _synthesize_numerical
from qmiep.synthesizer.base import (
    _replaced,
    _TargetedRewriteSynthesizer,
)


def _on_target(gate: cirq.Gate, target: cirq.Operation) -> cirq.Operation:
    operation = gate.on(*target.qubits)
    return operation.with_tags(*target.tags) if target.tags else operation


def _rust_replacement(
    target: cirq.Operation,
    kind: Kind,
    *,
    mode: Literal["analytic", "heuristic"] = "heuristic",
) -> tuple[cirq.Operation, ...]:
    """Adapt native U = D P W to the existing atomic replacement layouts."""
    from qmiep.rust import _rust

    source = target.gate
    assert isinstance(source, UnresolvedNQubitUnitary)
    instructions, phase, p, d = _rust.synthesize(
        source.matrix,
        kind=kind,
        method=mode,
        seed=_synthesis_seed(),
        atol=source.atol,
        check_input=_validation_enabled(),
    )
    circuit = _NativeCircuit(len(target.qubits), tuple(instructions), phase)
    return _numerical_replacement(target, kind, circuit, target.qubits, p, d)


def _numerical_replacement(
    target: cirq.Operation,
    kind: Kind,
    circuit: cirq.AbstractCircuit | _NativeCircuit,
    qubit_order: tuple[cirq.Qid, ...],
    p: NDArray[np.intp] | NDArray[np.uintp],
    d: NDArray[np.complex128],
) -> tuple[cirq.Operation, ...]:
    """Wrap a numerical decomposition U = D P W in the rewrite gate layout."""
    source = target.gate
    assert isinstance(source, UnresolvedNQubitUnitary)
    gate_type = {
        "exact": NQubitUnitary,
        "exact_further_D_optimized": NQubitUnitary,
        "diagonal": NQubitUnitaryModuloDiagonal,
        "D_equivalent_further_optimal": NQubitUnitaryModuloDiagonal,
        "pd": NQubitUnitaryModuloPermutationDiagonal,
        "pd_further_D_optimized": NQubitUnitaryModuloPermutationDiagonal,
    }[kind]
    label = source.label
    if gate_type is not NQubitUnitary:
        suffix = (
            "PD"
            if gate_type is NQubitUnitaryModuloPermutationDiagonal
            else kind.upper()
        )
        label = f"{label}_{suffix}"
    residual = _on_target(
        gate_type(
            _remove_left_correction(source.matrix, p, d),
            circuit,
            qubit_order=qubit_order,
            label=label,
            atol=source.atol,
        ),
        target,
    )
    if gate_type is NQubitUnitary:
        return (residual,)
    half = len(d) // 2
    if kind in ("diagonal", "D_equivalent_further_optimal"):
        # Only D_naive repeats its correction on both halves for n >= 3.
        qubits = (
            target.qubits
            if len(target.qubits) == 2 or kind == "D_equivalent_further_optimal"
            else target.qubits[1:]
        )
        diagonal = DiagonalGate(
            DiagonalMatrix.from_diagonal(d[: 2 ** len(qubits)], atol=source.atol)
        ).on(*qubits)
        return residual, diagonal.with_tags(*target.tags)
    if len(target.qubits) == 2:
        return (
            residual,
            _on_target(
                DiagonalGate(DiagonalMatrix.from_diagonal(d, atol=source.atol)), target
            ),
            _on_target(MultiplexedPermutation(PermutationMatrix(np.eye(half))), target),
        )
    # PD permutations preserve the first bit, matching multiplexed P.
    permutation = np.eye(len(p))[:, p]
    return (
        residual,
        _on_target(
            MultiplexedPermutation(
                PermutationMatrix(permutation[:half, :half]),
                PermutationMatrix(permutation[half:, half:]),
            ),
            target,
        ),
        _on_target(
            MultiplexedDiagonal(
                DiagonalMatrix.from_diagonal(d[:half], atol=source.atol),
                DiagonalMatrix.from_diagonal(d[half:], atol=source.atol),
            ),
            target,
        ),
    )


def _decompose_controlled_diagonal_with_basis(
    expanded: cirq.AbstractCircuit,
    controlled_d: cirq.Operation,
    neighbor: cirq.Operation,
    *,
    cx_boundary: Literal["start", "end"],
    atol: float,
) -> cirq.Circuit:
    """Expose a boundary CX and absorb sqrt(D) into a neighboring basis gate.

    Use the following basis for ``start`` and the preceding basis for ``end``.
    """
    diagonal = controlled_d.gate.sub_gate.diagonal.array
    square_root = np.exp(0.5j * np.angle(np.diag(diagonal)))
    ucg = UniformlyControlledRz(
        square_root.conj(), cx_boundary=cx_boundary, atol=atol
    ).on(*controlled_d.qubits)
    rotation_operations = tuple(cirq.decompose_once(ucg))
    cx = cirq.CX(controlled_d.qubits[1], controlled_d.qubits[0])
    assert rotation_operations[0 if cx_boundary == "start" else -1] == cx
    assert isinstance(
        rotation_operations[1 if cx_boundary == "start" else 0].gate, RGate
    )

    compensation = DiagonalGate(DiagonalMatrix.from_diagonal(square_root)).on(
        *controlled_d.qubits[1:]
    )
    embedded_s = np.diag(_operation_matrix(compensation, neighbor.qubits))
    combined = (
        neighbor.gate.matrix * embedded_s
        if cx_boundary == "start"
        else embedded_s[:, None] * neighbor.gate.matrix
    )
    adjusted_basis = _on_target(
        UnresolvedNQubitUnitary(combined, label=f"{neighbor.gate.label}_S", atol=atol),
        neighbor,
    )
    rotation_operations = tuple(
        op.with_tags(*controlled_d.tags) if controlled_d.tags else op
        for op in rotation_operations
    )
    expanded = cirq.Circuit(
        _replaced(cirq.FrozenCircuit(expanded), controlled_d, rotation_operations)
    )
    expanded = cirq.Circuit(
        _replaced(cirq.FrozenCircuit(expanded), neighbor, (adjusted_basis,))
    )
    return expanded


def _pack_exact_unitary(
    source: UnresolvedNQubitUnitary,
    target: cirq.Operation,
    operations: tuple[cirq.Operation, ...],
) -> cirq.Operation:
    """Wrap an exact operation sequence as one decomposable atomic gate."""
    decomposition = cirq.FrozenCircuit(
        cirq.Moment(operation) for operation in operations
    )
    exact = NQubitUnitary(
        source.matrix,
        decomposition,
        qubit_order=target.qubits,
        label=source.label,
        atol=source.atol,
    )
    return _on_target(exact, target)


def _cz_to_cx(operations: tuple[cirq.Operation, ...]) -> tuple[cirq.Operation, ...]:
    """Return ``operations`` with every CZ expressed as ``H, CX, H``."""
    converted: list[cirq.Operation] = []
    for operation in operations:
        if operation.gate != cirq.CZ:
            converted.append(operation)
            continue
        control, target = operation.qubits
        replacement = (
            cirq.H(target),
            cirq.CX(control, target),
            cirq.H(target),
        )
        converted.extend(
            child.with_tags(*operation.tags) if operation.tags else child
            for child in replacement
        )
    return tuple(converted)


def _synthesize_through_step_10(
    target: cirq.Operation,
    *,
    context: cirq.TransformerContext,
    pd_mode: Literal["analytic", "heuristic"] = "heuristic",
) -> cirq.Circuit:
    """Build the shared synthesis prefix, ending in unresolved O.

    Steps 1-10 synthesize L, M, and N modulo PD and migrate their corrections
    into O using pd_mode recursively. The caller chooses exact, D-modulus,
    or PD-modulus synthesis for that terminal.
    """
    source = target.gate
    assert type(source) is UnresolvedNQubitUnitary

    # STEP 1. : Apply block zxz
    # Three-qubit diagram (0, 1, 2 denote target.qubits in order):
    # 0: ───@───────H───□────────────────────H───@───────
    #       │           │                        │
    # 1: ───C?[1]───────diag(B_1, B_2)?[1]───────A?[1]───
    #       │           │                        │
    # 2: ───C?[2]───────diag(B_1, B_2)?[2]───────A?[2]───
    expanded = block_zxz_rule(cirq.Circuit(target), target=target, context=context)

    # STEP 2. : Spectrally decompose controlled-C and controlled-A.
    # 0: ──────────────@───────────────────H───□────────────────────H──────────────@───────────────────
    #                  │                       │                                   │
    # 1: ───Q_C†?[1]───D_C?[1]───Q_C?[1]───────diag(B_1, B_2)?[1]───────Q_A†?[1]───D_A?[1]───Q_A?[1]───
    #       │          │         │             │                        │          │         │
    # 2: ───Q_C†?[2]───D_C?[2]───Q_C?[2]───────diag(B_1, B_2)?[2]───────Q_A†?[2]───D_A?[2]───Q_A?[2]───
    # X = Q_X D_X Q_X†: circuit order Q_X† -> ctrl(D_X) -> Q_X.
    operations = tuple(expanded.all_operations())
    controlled_c, controlled_a = operations[0], operations[-1]
    for operation, label in ((controlled_c, "C"), (controlled_a, "A")):
        spectral = multiplexed_unitary_decomposition_by_spectrum_rule(
            cirq.Circuit(operation),
            context=context,
            antipodal=False,
            label=label,
            atol=source.atol,
        )
        expanded = cirq.Circuit(
            _replaced(
                cirq.FrozenCircuit(expanded),
                operation,
                tuple(spectral.all_operations()),
            )
        )

    # STEP 3. : Expose CX on the right of D_C and on the left of D_A.
    # 0: ────────────────R_L![1]───X─────────────H───□────────────────────H──────────────X───R_R![1]───────────────
    #                    │         │                 │                                   │   │
    # 1: ───Q_C†_S?[1]───R_L![2]───@───Q_C?[1]───────diag(B_1, B_2)?[1]───────Q_A†?[1]───@───R_R![2]───Q_A_S?[1]───
    #       │            │             │             │                        │              │         │
    # 2: ───Q_C†_S?[2]───R_L![3]───────Q_C?[2]───────diag(B_1, B_2)?[2]───────Q_A†?[2]───────R_R![3]───Q_A_S?[2]───
    # For S_X = sqrt(D_X), ctrl(D_X) = (I ⊗ S_X) UCRz(S_X†).
    # Absorb S_C into Q_C† on the left and S_A into Q_A on the right.
    operations = tuple(expanded.all_operations())
    q_c_dagger, controlled_d_c = operations[:2]
    controlled_d_a, q_a = operations[-2:]
    expanded = _decompose_controlled_diagonal_with_basis(
        expanded,
        controlled_d_c,
        q_c_dagger,
        cx_boundary="end",
        atol=source.atol,
    )
    expanded = _decompose_controlled_diagonal_with_basis(
        expanded, controlled_d_a, q_a, cx_boundary="start", atol=source.atol
    )

    # STEP 4. : Move the boundary Hadamards through CX, turning CX into CZ.
    # 0: ────────────────R_L![1]───H───@─────────────□───────────────────────────────@───H───R_R![1]───────────────
    #                    │             │             │                               │       │
    # 1: ───Q_C†_S?[1]───R_L![2]───────@───Q_C?[1]───diag(B_1, B_2)?[1]───Q_A†?[1]───@───────R_R![2]───Q_A_S?[1]───
    #       │            │                 │         │                    │                  │         │
    # 2: ───Q_C†_S?[2]───R_L![3]───────────Q_C?[2]───diag(B_1, B_2)?[2]───Q_A†?[2]───────────R_R![3]───Q_A_S?[2]───
    expanded = hadamard_cx_to_cz_rule(expanded, context=context)

    # STEP 5. : Merge both boundary CZs and everything between them into B.
    # 0: ────────────────R_L![1]───H───□────────────────────H───R_R![1]───────────────
    #                    │             │                        │
    # 1: ───Q_C†_S?[1]───R_L![2]───────diag(B_1, B_2)?[1]───────R_R![2]───Q_A_S?[1]───
    #       │            │             │                        │         │
    # 2: ───Q_C†_S?[2]───R_L![3]───────diag(B_1, B_2)?[2]───────R_R![3]───Q_A_S?[2]───
    expanded = merge_central_multiplexor_rule(expanded, context=context)

    # STEP 6. : Spectrally decompose B into B_L -> UCG Rz -> B_R.
    # 0: ────────────────R_L![1]───H─────────────F_B(Rz)!─────────────H───R_R![1]───────────────
    #                    │                       │                        │
    # 1: ───Q_C†_S?[1]───R_L![2]───────B_L?[1]───□──────────B_R?[1]───────R_R![2]───Q_A_S?[1]───
    #       │            │             │         │          │             │         │
    # 2: ───Q_C†_S?[2]───R_L![3]───────B_L?[2]───□──────────B_R?[2]───────R_R![3]───Q_A_S?[2]───
    expanded = multiplexed_unitary_decomposition_by_spectrum_rule(
        expanded,
        target=expanded[3].operations[0],
        context=context,
        atol=source.atol,
    )

    # STEP 7. : Relabel the target-register unitaries as L, M, N, O.
    # 0: ───────────R_L![1]───H───────────F_B(Rz)!───────────H───R_R![1]───────────
    #               │                     │                      │
    # 1: ───L?[1]───R_L![2]───────M?[1]───□──────────N?[1]───────R_R![2]───O?[1]───
    #       │       │             │       │          │           │         │
    # 2: ───L?[2]───R_L![3]───────M?[2]───□──────────N?[2]───────R_R![3]───O?[2]───
    expanded = relabel_gauge_unitaries_rule(expanded, context=context)

    # STEP 8. : Migrate L's PD corrections through R_L and merge into M.
    # 0: ──────────────R_L![1]───H───────────F_B(Rz)!───────────H───R_R![1]───────────
    #                  │                     │                      │
    # 1: ───L_PD![1]───R_L![2]───────M?[1]───□──────────N?[1]───────R_R![2]───O?[1]───
    #       │          │             │       │          │           │         │
    # 2: ───L_PD![2]───R_L![3]───────M?[2]───□──────────N?[2]───────R_R![3]───O?[2]───
    # Selector-only corrections commute with H(q0); keep H in place.
    indices = (0, 1, 3)  # L, R_L, M in the fixed Step 7 layout.
    migrated = tuple(
        pd_migration_through_r_rule(
            cirq.Circuit(expanded[index] for index in indices),
            context=context,
            mode=pd_mode,
        ).all_operations()
    )
    for index, operation in zip(indices, migrated, strict=True):
        expanded[index] = cirq.Moment(operation)

    # STEP 9. : Migrate M's PD corrections through F_B(Rz) into N.
    # 0: ──────────────R_L![1]───H──────────────F_B(Rz)!───────────H───R_R![1]───────────
    #                  │                        │                      │
    # 1: ───L_PD![1]───R_L![2]───────M_PD![1]───□──────────N?[1]───────R_R![2]───O?[1]───
    #       │          │             │          │          │           │         │
    # 2: ───L_PD![2]───R_L![3]───────M_PD![2]───□──────────N?[2]───────R_R![3]───O?[2]───
    indices = (3, 4, 5)  # M, F_B(Rz), N.
    migrated = tuple(
        pd_migration_through_ucg_rule(
            cirq.Circuit(expanded[index] for index in indices),
            context=context,
            mode=pd_mode,
        ).all_operations()
    )
    for index, operation in zip(indices, migrated, strict=True):
        expanded[index] = cirq.Moment(operation)

    # STEP 10. : Synthesize N modulo PD and migrate its corrections into O.
    # 0: ──────────────R_L![1]───H──────────────F_B(Rz)!──────────────H───R_R![1]───────────
    #                  │                        │                         │
    # 1: ───L_PD![1]───R_L![2]───────M_PD![1]───□──────────N_PD![1]───────R_R![2]───O?[1]───
    #       │          │             │          │          │              │         │
    # 2: ───L_PD![2]───R_L![3]───────M_PD![2]───□──────────N_PD![2]───────R_R![3]───O?[2]───
    # H(q0) commutes with the selector correction; leave O unresolved.
    indices = (5, 7, 8)  # N, R_R, O.
    migrated = tuple(
        pd_migration_through_r_rule(
            cirq.Circuit(expanded[index] for index in indices),
            context=context,
            mode=pd_mode,
        ).all_operations()
    )
    for index, operation in zip(indices, migrated, strict=True):
        expanded[index] = cirq.Moment(operation)
    return expanded


class NQubitUnitarySynthesis(_TargetedRewriteSynthesizer[UnresolvedNQubitUnitary]):
    r"""Synthesize :math:`U` as one atomic gate with a concrete decomposition.

    A two-qubit target becomes one exact atomic gate with at most three CX
    gates. For three or more qubits, Steps 1-10 are shared with D-modulus
    synthesis. Step 11 recursively synthesizes the terminal O exactly and
    packages the entire circuit, leaving no exposed corrections. Pass a
    :class:`cirq.TransformerLogger` through ``context`` and call ``logger.show``
    to print every internal rewrite stage.

    ``mode`` selects the PD synthesis strategy throughout the recursion.
    The default ``backend="numerical"`` computes factors and primitive
    instructions first, then creates a flat Cirq decomposition. Use
    ``backend="rewrite"`` for the original hierarchical, staged circuit.
    Supplying a transformer logger also selects the staged backend.
    ``run(backends=SynthesisBackends(unitary="rust"))`` selects native
    synthesis instead. Selecting Rust for nested D/PD calls uses the staged
    Python path when exact synthesis itself remains Python.

    ``kind="exact_further_D_optimized"`` recursively improves L, M and N
    with PD corrections permitted, then recurses with the same exact kind
    on O. In the recursive calls, N permits PD before the full R and
    requires D before the short R and CZ. O inherits its parent's mode.
    Each node constructs only one circuit layout.
    Its CNOT count is not guaranteed to be below ``kind="exact"``.
    This kind uses numerical Python orchestration regardless of ``backend``
    or logger, honoring the selected D/PD child backends. It only supports
    ``mode="heuristic"`` and uses heuristic PD recursion even inside an
    enclosing analytic scope; all improved non-exact nodes attempt QMIEP.
    """

    __slots__ = ("mode", "backend", "kind")
    source_gate_type = UnresolvedNQubitUnitary
    mode: Literal["analytic", "heuristic"]
    backend: Literal["numerical", "rewrite"]
    kind: ExactMode

    def __init__(
        self,
        *,
        mode: Literal["analytic", "heuristic"] = "heuristic",
        backend: Literal["numerical", "rewrite"] = "numerical",
        kind: ExactMode = "exact",
    ) -> None:
        if mode not in ("analytic", "heuristic"):
            raise ValueError("mode must be 'analytic' or 'heuristic'")
        if backend not in ("numerical", "rewrite"):
            raise ValueError("backend must be 'numerical' or 'rewrite'")
        if kind not in ("exact", "exact_further_D_optimized"):
            raise ValueError("kind must be 'exact' or 'exact_further_D_optimized'")
        if kind == "exact_further_D_optimized" and mode != "heuristic":
            raise ValueError("exact_further_D_optimized only supports mode='heuristic'")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "kind", kind)

    @property
    def replacement_gate_types(self) -> tuple[type[cirq.Gate], ...]:
        return (NQubitUnitary,)

    def _rewrite(
        self,
        target: cirq.Operation,
        *,
        context: cirq.TransformerContext,
    ) -> tuple[cirq.Operation, ...]:
        source = self._source_gate(target)

        backends = _synthesis_backends()
        kind: ExactMode = (
            "exact_further_D_optimized"
            if _synthesis_algorithm() == "heuristic-opt"
            else self.kind
        )
        mode = (
            "heuristic"
            if kind == "exact_further_D_optimized"
            else _synthesis_mode(self.mode)
        )
        if backends.unitary == "rust" and len(target.qubits) >= 2:
            return _rust_replacement(target, kind, mode=mode)

        if len(target.qubits) >= 2 and (
            kind == "exact_further_D_optimized"
            or (
                self.backend == "numerical"
                and backends.diagonal == backends.pd == "python"
                and type(context.logger) is NoOpTransformerLogger
            )
        ):
            decomposition = _synthesize_numerical(
                source.matrix,
                mode=mode,
                atol=source.atol,
                tags=target.tags,
                kind=kind,
                native_pd=backends.pd == "rust",
                native_diagonal=backends.diagonal == "rust",
            )
            exact = NQubitUnitary(
                source.matrix,
                decomposition,
                qubit_order=cirq.LineQubit.range(len(target.qubits)),
                label=source.label,
                atol=source.atol,
            )
            return (_on_target(exact, target),)

        if len(target.qubits) == 2:
            # An exact two-qubit synthesis: keep the residual's circuit but
            # present it as U itself, since no correction is factored out.
            residual, _ = decompose_two_qubit_unitary(
                source.matrix,
                upto_diagonal=False,
                atol=source.atol,
            )
            exact = NQubitUnitary(
                source.matrix,
                residual.decomposition,
                qubit_order=tuple(cirq.LineQubit.range(2)),
                label=source.label,
                atol=source.atol,
            )
            return (_on_target(exact, target),)

        elif len(target.qubits) >= 3:
            expanded = _synthesize_through_step_10(
                target, context=context, pd_mode=mode
            )

            # STEP 11. : Exactly synthesize O and package the complete circuit.
            # 0: ──────────────R_L![1]───H──────────────F_B(Rz)!──────────────H───R_R![1]───────────
            #                  │                        │                         │
            # 1: ───L_PD![1]───R_L![2]───────M_PD![1]───□──────────N_PD![1]───────R_R![2]───O![1]───
            #       │          │             │          │          │              │         │
            # 2: ───L_PD![2]───R_L![3]───────M_PD![2]───□──────────N_PD![2]───────R_R![3]───O![2]───
            o = tuple(expanded.all_operations())[-1]
            exact = self.run(expanded, target=o, context=context)
            return (_pack_exact_unitary(source, target, tuple(exact.all_operations())),)

        raise self._unsupported(target)


class NQubitUnitarySynthesisModuloDiagonal(
    _TargetedRewriteSynthesizer[UnresolvedNQubitUnitary]
):
    r"""Return circuit-time :math:`(W,D)` with a concrete diagonal-modulus residual.

    A two-qubit target returns :math:`(W,D)` satisfying :math:`U=DW`, with
    at most two CX gates in :math:`W`.

    In the default ``mode="D_naive"``, for :math:`n\geq3`, Steps 1-10
    synthesize :math:`L,M,N` modulo PD,
    migrating their corrections through the rotations into the next block.
    Step 11 recursively writes :math:`O=D_O O_D` and packages the prefix
    through :math:`O_D` as :class:`NQubitUnitaryModuloDiagonal`.
    The remaining :class:`DiagonalGate` acts on the target register, so in
    the basis :math:`|q_0\rangle\otimes|t\rangle`,

    .. math::

        U=(I_2\otimes D_O)W.

    ``mode="D_equivalent_further_optimal"`` attempts QMIEP on controlled-A.
    With P=I it uses the separable spectrum; otherwise it uses the unpaired
    layout without repeating the factorization. L/M permit PD and O requires D.
    N requires D before the short R and CZ. The full R permits a PD correction
    from N and propagates it into O. The exposed D acts on the entire input register.
    Python uses numerical recursion; selected Rust PD children are honored.
    """

    __slots__ = ("mode",)
    source_gate_type = UnresolvedNQubitUnitary
    mode: DiagonalMode

    def __init__(self, *, mode: DiagonalMode = "D_naive") -> None:
        if mode not in ("D_naive", "D_equivalent_further_optimal"):
            raise ValueError("mode must be 'D_naive' or 'D_equivalent_further_optimal'")
        object.__setattr__(self, "mode", mode)

    @property
    def replacement_gate_types(self) -> tuple[type[cirq.Gate], ...]:
        return (NQubitUnitaryModuloDiagonal, DiagonalGate)

    @property
    def _optimized(self) -> bool:
        return (
            self.mode == "D_equivalent_further_optimal"
            or _synthesis_algorithm() == "heuristic-opt"
        )

    def _replacement_layout(
        self, source: cirq.Operation
    ) -> tuple[tuple[type[cirq.Gate], tuple[cirq.Qid, ...]], ...]:
        if len(source.qubits) < 3 or self._optimized:
            return super()._replacement_layout(source)
        return (
            (NQubitUnitaryModuloDiagonal, source.qubits),
            (DiagonalGate, source.qubits[1:]),
        )

    def _rewrite(
        self,
        target: cirq.Operation,
        *,
        context: cirq.TransformerContext,
    ) -> tuple[cirq.Operation, ...]:
        source = self._source_gate(target)

        if self._optimized and len(target.qubits) >= 2:
            if _synthesis_backends().diagonal == "rust":
                return _rust_replacement(
                    target,
                    "D_equivalent_further_optimal",
                    mode=_synthesis_mode("heuristic"),
                )
            program = _Program(native_pd=_synthesis_backends().pd == "rust")
            _, d = program.solve(
                source.matrix,
                tuple(range(len(target.qubits))),
                "D_equivalent_further_optimal",
                _synthesis_mode("heuristic"),
                source.atol,
            )
            residual = NQubitUnitaryModuloDiagonal(
                d.conj()[:, None] * source.matrix,
                program.circuit(len(target.qubits), tags=target.tags),
                qubit_order=cirq.LineQubit.range(len(target.qubits)),
                label=f"{source.label}_D",
                atol=source.atol,
            )
            return (
                _on_target(residual, target),
                _on_target(
                    DiagonalGate(DiagonalMatrix.from_diagonal(d, atol=source.atol)),
                    target,
                ),
            )

        if _synthesis_backends().diagonal == "rust" and len(target.qubits) >= 2:
            return _rust_replacement(
                target, "diagonal", mode=_synthesis_mode("heuristic")
            )

        if len(target.qubits) == 2:
            # decompose_two_qubit_unitary already returns (W, D) with U = DW.
            residual, diagonal = decompose_two_qubit_unitary(
                source.matrix,
                upto_diagonal=True,
                label=source.label,
                atol=source.atol,
            )
            return _on_target(residual, target), diagonal.on(*target.qubits)
        elif len(target.qubits) >= 3:
            expanded = _synthesize_through_step_10(
                target, context=context, pd_mode=_synthesis_mode("heuristic")
            )

            # STEP 11. : Synthesize O modulo D and package the prefix through O_D.
            # 0: ──────────────R_L![1]───H──────────────F_B(Rz)!──────────────H───R_R![1]─────────────────────
            #                  │                        │                         │
            # 1: ───L_PD![1]───R_L![2]───────M_PD![1]───□──────────N_PD![1]───────R_R![2]───O_D![1]───D?[1]───
            #       │          │             │          │          │              │         │         │
            # 2: ───L_PD![2]───R_L![3]───────M_PD![2]───□──────────N_PD![2]───────R_R![3]───O_D![2]───D?[2]───
            operations = tuple(expanded.all_operations())
            o = operations[-1]
            o_residual, diagonal = tuple(
                self.run(cirq.Circuit(o), target=o, context=context).all_operations()
            )
            o_gate = o_residual.gate
            o_d = _on_target(
                NQubitUnitaryModuloDiagonal(
                    o_gate.residual_matrix,
                    o_gate.decomposition,
                    qubit_order=cirq.LineQubit.range(len(o.qubits)),
                    label="O_D",
                    atol=o_gate.atol,
                ),
                o_residual,
            )
            prefix = (*operations[:-1], o_d)
            diagonal_values = np.diag(_operation_matrix(diagonal, o.qubits))
            residual = NQubitUnitaryModuloDiagonal(
                np.tile(diagonal_values.conj(), 2)[:, None] * source.matrix,
                cirq.FrozenCircuit(cirq.Moment(op) for op in prefix),
                qubit_order=target.qubits,
                label=f"{source.label}_D",
                atol=source.atol,
            )
            # Recursive O synthesis can expose D on a suffix of O's register.
            # Lift it with identities so the returned D acts on all O qubits.
            diagonal_gate = DiagonalGate(
                DiagonalMatrix.from_diagonal(diagonal_values, atol=source.atol),
                label=diagonal.gate.label,
            )
            return _on_target(residual, target), _on_target(diagonal_gate, o)
        raise self._unsupported(target)


class NQubitUnitarySynthesisModuloPermutationDiagonal(
    _TargetedRewriteSynthesizer[UnresolvedNQubitUnitary]
):
    r"""Synthesize a residual and exposed permutation/diagonal corrections.

    Two-qubit inputs keep circuit order ``W, D, P`` with ``P = I``.
    For three or more qubits, return ``W, P_hat, D_hat``, so
    :math:`U=\widehat D\widehat P W`. All three operations act on the full
    input register, and W packages the prefix through O_PD.

    Both modes use the two-qubit QMIEP solver for the three-qubit base case.
    The default ``mode="heuristic"`` retains Steps 1-13 and uses the general
    QMIEP solver above three qubits. For ``mode="analytic"`` and
    :math:`n\geq4`, the shared Steps 1-10 synthesize L, M, and N in the same
    mode, followed by recursive PD synthesis of O. Thus
    :math:`O=D_O P_O O_{\rm PD}` gives :math:`\widehat P=I_2\otimes P_O`
    and :math:`\widehat D=I_2\otimes D_O`, without a general QMIEP solve.
    With ``kind="pd_further_D_optimized"``, use improved recursion, permitting
    PD corrections at this node and using improved D synthesis where required.
    """

    __slots__ = ("mode", "kind")
    source_gate_type = UnresolvedNQubitUnitary
    mode: Literal["analytic", "heuristic"]
    kind: PDMode

    def __init__(
        self,
        *,
        mode: Literal["analytic", "heuristic"] = "heuristic",
        kind: PDMode = "pd",
    ) -> None:
        if mode not in ("analytic", "heuristic"):
            raise ValueError("mode must be 'analytic' or 'heuristic'")
        if kind not in ("pd", "pd_further_D_optimized"):
            raise ValueError("kind must be 'pd' or 'pd_further_D_optimized'")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "kind", kind)

    @property
    def replacement_gate_types(self) -> tuple[type[cirq.Gate], ...]:
        return (
            NQubitUnitaryModuloPermutationDiagonal,
            DiagonalGate,
            MultiplexedPermutation,
        )

    def _replacement_layout(
        self, source: cirq.Operation
    ) -> tuple[tuple[type[cirq.Gate], tuple[cirq.Qid, ...]], ...]:
        if len(source.qubits) == 2:
            return super()._replacement_layout(source)
        return (
            (NQubitUnitaryModuloPermutationDiagonal, source.qubits),
            (MultiplexedPermutation, source.qubits),
            (MultiplexedDiagonal, source.qubits),
        )

    def _rewrite(
        self,
        target: cirq.Operation,
        *,
        context: cirq.TransformerContext,
    ) -> tuple[cirq.Operation, ...]:
        source = self._source_gate(target)

        mode = _synthesis_mode(self.mode)
        backends = _synthesis_backends()
        n = len(target.qubits)
        kind: PDMode = (
            "pd_further_D_optimized"
            if _synthesis_algorithm() == "heuristic-opt"
            else self.kind
        )
        if backends.pd == "rust" and n >= 2:
            return _rust_replacement(target, kind, mode=mode)
        if kind == "pd_further_D_optimized" and n >= 2:
            program = _Program(native_diagonal=backends.diagonal == "rust")
            p, d = program.solve(
                source.matrix, tuple(range(n)), kind, mode, source.atol
            )
            return _numerical_replacement(
                target,
                kind,
                program.circuit(n, tags=target.tags),
                tuple(cirq.LineQubit.range(n)),
                p,
                d,
            )

        if len(target.qubits) == 2:
            # Two-qubit inputs need no row exchange, so P = I in the
            # common controlled-permutation correction I⊕P.
            modulo_diagonal, diagonal = decompose_two_qubit_unitary(
                source.matrix,
                upto_diagonal=True,
                label=source.label,
                atol=source.atol,
            )
            residual = NQubitUnitaryModuloPermutationDiagonal(
                modulo_diagonal.residual_matrix,
                modulo_diagonal.decomposition,
                qubit_order=tuple(cirq.LineQubit.range(2)),
                label=f"{source.label}\N{COMBINING TILDE}",
                atol=modulo_diagonal.atol,
            )
            identity = MultiplexedPermutation(
                PermutationMatrix(np.eye(2, dtype=np.uint8))
            )
            return (
                _on_target(residual, target),
                diagonal.on(*target.qubits),
                identity.on(*target.qubits),
            )
        elif len(target.qubits) >= 4 and mode == "analytic":
            return self._rewrite_analytic(target, context=context)
        elif len(target.qubits) >= 3:
            return self._rewrite_multi_qubit(target, context=context)
        raise self._unsupported(target)

    def _rewrite_analytic(
        self,
        target: cirq.Operation,
        *,
        context: cirq.TransformerContext,
    ) -> tuple[cirq.Operation, ...]:
        """Recurse through the shared prefix to the three-qubit QMIEP base case."""
        source = self._source_gate(target)
        expanded = _synthesize_through_step_10(
            target, context=context, pd_mode=_synthesis_mode(self.mode)
        )

        # STEP 11. : Synthesize O = D_O P_O O_PD recursively and expose P_O, D_O.
        # Four-qubit diagram (0, 1, 2, 3 denote target.qubits in order):
        # 0: ──────────────R_L![1]───H──────────────F_B(Rz)!──────────────H───R_R![1]──────────────────────────────────
        #                  │                        │                         │
        # 1: ───L_PD![1]───R_L![2]───────M_PD![1]───□──────────N_PD![1]───────R_R![2]───O_PD![1]───P_O?[1]───D_O?[1]───
        #       │          │             │          │          │              │         │          │         │
        # 2: ───L_PD![2]───R_L![3]───────M_PD![2]───□──────────N_PD![2]───────R_R![3]───O_PD![2]───P_O?[2]───D_O?[2]───
        #       │          │             │          │          │              │         │          │         │
        # 3: ───L_PD![3]───R_L![4]───────M_PD![3]───□──────────N_PD![3]───────R_R![4]───O_PD![3]───P_O?[3]───D_O?[3]───
        operations = tuple(expanded.all_operations())
        o = operations[-1]
        o_residual, permutation, diagonal = tuple(
            self.run(cirq.Circuit(o), target=o, context=context).all_operations()
        )
        o_gate = o_residual.gate
        assert isinstance(o_gate, NQubitUnitaryModuloPermutationDiagonal)
        o_pd = _on_target(
            NQubitUnitaryModuloPermutationDiagonal(
                o_gate.residual_matrix,
                o_gate.decomposition,
                qubit_order=cirq.LineQubit.range(len(o.qubits)),
                label="O_PD",
                atol=o_gate.atol,
            ),
            o_residual,
        )
        prefix = (*operations[:-1], o_pd)
        residual = NQubitUnitaryModuloPermutationDiagonal(
            _remove_left_correction(
                source.matrix,
                np.argmax(permutation.gate.matrix, axis=0),
                np.diag(diagonal.gate.matrix),
            ),
            cirq.FrozenCircuit(cirq.Moment(op) for op in prefix),
            qubit_order=target.qubits,
            label=f"{source.label}\N{COMBINING TILDE}",
            atol=source.atol,
        )
        # U = (I_2 tensor D_O)(I_2 tensor P_O) W; both branches share O's factors.
        assert isinstance(permutation.gate, MultiplexedPermutation)
        assert isinstance(diagonal.gate, MultiplexedDiagonal)
        p_o = permutation.gate.permutation
        d_o = diagonal.gate.diagonal
        return (
            _on_target(residual, target),
            _on_target(MultiplexedPermutation(p_o, p_o), target),
            _on_target(MultiplexedDiagonal(d_o, d_o), target),
        )

    def _rewrite_multi_qubit(
        self,
        target: cirq.Operation,
        *,
        context: cirq.TransformerContext,
    ) -> tuple[cirq.Operation, ...]:
        """Build Steps 1-13 for n >= 3; diagrams show the n = 3 example."""
        source = self._source_gate(target)

        # STEP 1. : Apply block zxz
        # Text diagram of expanded (0, 1, 2 denote target.qubits in order):
        # 0: ───@───────H───□────────────────────H───@───────
        #       │           │                        │
        # 1: ───C?[1]───────diag(B_1, B_2)?[1]───────A?[1]───
        #       │           │                        │
        # 2: ───C?[2]───────diag(B_1, B_2)?[2]───────A?[2]───
        expanded = block_zxz_rule(cirq.Circuit(target), target=target, context=context)

        # STEP 2. : Apply QMIEP to A in the last controlled-A.
        # Text diagram (0, 1, 2 denote target.qubits in order):
        # 0: ───@───────H───□────────────────────H───@───────────@───────@────────
        #       │           │                        │           │       │
        # 1: ───C?[1]───────diag(B_1, B_2)?[1]───────A_PD*?[1]───P?[1]───D'?[1]───
        #       │           │                        │           │       │
        # 2: ───C?[2]───────diag(B_1, B_2)?[2]───────A_PD*?[2]───P?[2]───D'?[2]───

        controlled_a = tuple(expanded.all_operations())[-1]
        a = controlled_a.gate.one_matrix
        left_p, left_d = solve_spectrum(
            a, atol=min(1e-10, source.atol / len(a)), seed=_synthesis_seed()
        )
        # The solver returns left multipliers: A_PD = D_left P_left A.
        a_pd = left_d.array @ left_p.array @ a
        # Reconstruction factors for the circuit below satisfy A = P D A_PD.
        p = PermutationMatrix(left_p.array.conj().T)
        d = DiagonalMatrix(left_d.array.conj().T)
        # Step 3 explicitly verifies every antipodal pair before decomposition.

        # Circuit order A_PD -> P -> D' requires D' = P D P^dagger:
        # D' P A_PD = P D A_PD = A.
        d_after_p = DiagonalMatrix(p.array @ d.array @ p.array.conj().T)
        replacements = (
            _on_target(
                ControlledSpectralUnitary(a_pd, label="A_PD", atol=source.atol),
                controlled_a,
            ),
            _on_target(MultiplexedPermutation(p), controlled_a),
            _on_target(DiagonalGate(d_after_p, label="D'").controlled(), controlled_a),
        )
        assert not _validation_enabled() or np.allclose(
            cirq.Circuit(replacements).unitary(qubit_order=target.qubits),
            cirq.unitary(controlled_a),
            rtol=0,
            atol=source.atol,
        ), "Step 2 must preserve controlled-A"

        expanded = cirq.Circuit(
            _replaced(cirq.FrozenCircuit(expanded), controlled_a, replacements)
        )

        # STEP 3. : Spectrally decompose controlled-C and controlled-A_PD.
        # 0: ──────────────@───────────────────H───□────────────────────H─────────────────@───────@────────────────@───────@────────
        #                  │                       │                                      │       │                │       │
        # 1: ───Q_C†?[1]───D_C?[1]───Q_C?[1]───────diag(B_1, B_2)?[1]───────Q_A_PD†?[1]───D'_A?───┼───Q_A_PD?[1]───P?[1]───D'?[1]───
        #       │          │         │             │                        │                     │   │            │       │
        # 2: ───Q_C†?[2]───D_C?[2]───Q_C?[2]───────diag(B_1, B_2)?[2]───────Q_A_PD†?[2]───────────@───Q_A_PD?[2]───P?[2]───D'?[2]───
        # Wires follow target.qubits; D'_A is distinct from Step 2's D'.
        controlled_c = tuple(expanded.all_operations())[0]
        controlled_a_pd = replacements[0]
        for operation, label in ((controlled_c, "C"), (controlled_a_pd, "A_PD")):
            spectral = multiplexed_unitary_decomposition_by_spectrum_rule(
                cirq.Circuit(operation),
                context=context,
                antipodal=operation is controlled_a_pd,
                label=label,
                diagonal_label="D'_A" if operation is controlled_a_pd else "D_C",
                atol=source.atol,
            )
            expanded = cirq.Circuit(
                _replaced(
                    cirq.FrozenCircuit(expanded),
                    operation,
                    tuple(spectral.all_operations()),
                )
            )
        assert not _validation_enabled() or np.allclose(
            expanded.unitary(qubit_order=target.qubits),
            source.matrix,
            rtol=0,
            atol=source.atol,
        ), "Steps 1-3 must preserve the original unitary"

        # STEP 4. : Expose CX on the right of D_C and on the left of D'_A.
        # 0: ────────────────R_L![1]───X─────────────H───□────────────────────H─────────────────X───R_R![1]───@──────────────────@───────@────────
        #                    │         │                 │                                      │   │         │                  │       │
        # 1: ───Q_C†_S?[1]───R_L![2]───@───Q_C?[1]───────diag(B_1, B_2)?[1]───────Q_A_PD†?[1]───@───R_R![2]───┼───Q_A_PD_S?[1]───P?[1]───D'?[1]───
        #       │            │             │             │                        │                           │   │              │       │
        # 2: ───Q_C†_S?[2]───R_L![3]───────Q_C?[2]───────diag(B_1, B_2)?[2]───────Q_A_PD†?[2]─────────────────@───Q_A_PD_S?[2]───P?[2]───D'?[2]───
        # For S = sqrt(D), ctrl(D) = (I ⊗ S) UCRz(S^dagger).
        # Absorb S into Q_C^dagger on the left, or into Q_A_PD on the right.
        # Fixed Step 3 layout; capture operations before inserting new moments.
        controlled_d_c, q_c_dagger = (
            expanded[1].operations[0],
            expanded[0].operations[0],
        )
        controlled_d_a, q_a = expanded[7].operations[0], expanded[9].operations[0]
        expanded = _decompose_controlled_diagonal_with_basis(
            expanded, controlled_d_c, q_c_dagger, cx_boundary="end", atol=source.atol
        )
        expanded = _decompose_controlled_diagonal_with_basis(
            expanded, controlled_d_a, q_a, cx_boundary="start", atol=source.atol
        )
        assert not _validation_enabled() or np.allclose(
            expanded.unitary(qubit_order=target.qubits),
            source.matrix,
            rtol=0,
            atol=source.atol,
        ), "Steps 1-4 must preserve the original unitary"

        # STEP 5. : Move the boundary Hadamards through CX, turning CX into CZ.
        # 0: ────────────────R_L![1]───H───@─────────────□──────────────────────────────────@───H───R_R![1]───@──────────────────@───────@────────
        #                    │             │             │                                  │       │         │                  │       │
        # 1: ───Q_C†_S?[1]───R_L![2]───────@───Q_C?[1]───diag(B_1, B_2)?[1]───Q_A_PD†?[1]───@───────R_R![2]───┼───Q_A_PD_S?[1]───P?[1]───D'?[1]───
        #       │            │                 │         │                    │                               │   │              │       │
        # 2: ───Q_C†_S?[2]───R_L![3]───────────Q_C?[2]───diag(B_1, B_2)?[2]───Q_A_PD†?[2]─────────────────────@───Q_A_PD_S?[2]───P?[2]───D'?[2]───
        expanded = hadamard_cx_to_cz_rule(expanded, context=context)

        # STEP 6. : Merge both boundary CZs and everything between them into B.
        # 0: ────────────────R_L![1]───H───□────────────────────H───R_R![1]───@──────────────────@───────@────────
        #                    │             │                        │         │                  │       │
        # 1: ───Q_C†_S?[1]───R_L![2]───────diag(B_1, B_2)?[1]───────R_R![2]───┼───Q_A_PD_S?[1]───P?[1]───D'?[1]───
        #       │            │             │                                  │   │              │       │
        # 2: ───Q_C†_S?[2]───R_L![3]───────diag(B_1, B_2)?[2]─────────────────@───Q_A_PD_S?[2]───P?[2]───D'?[2]───
        expanded = merge_central_multiplexor_rule(expanded, context=context)

        # STEP 7. : Spectrally decompose B into B_L -> UCG Rz -> B_R.
        # 0: ────────────────R_L![1]───H─────────────F_B(Rz)!─────────────H───R_R![1]───@──────────────────@───────@────────
        #                    │                       │                        │         │                  │       │
        # 1: ───Q_C†_S?[1]───R_L![2]───────B_L?[1]───□──────────B_R?[1]───────R_R![2]───┼───Q_A_PD_S?[1]───P?[1]───D'?[1]───
        #       │            │             │         │          │                       │   │              │       │
        # 2: ───Q_C†_S?[2]───R_L![3]───────B_L?[2]───□──────────B_R?[2]─────────────────@───Q_A_PD_S?[2]───P?[2]───D'?[2]───
        expanded = multiplexed_unitary_decomposition_by_spectrum_rule(
            expanded, target=expanded[3].operations[0], context=context
        )
        assert not _validation_enabled() or np.allclose(
            expanded.unitary(qubit_order=target.qubits),
            source.matrix,
            rtol=0,
            atol=source.atol,
        ), "Steps 1-7 must preserve the original unitary"

        # STEP 8. : Relabel the target-register unitaries as L, M, N, O.
        # 0: ───────────R_L![1]───H───────────F_B(Rz)!───────────H───R_R![1]───@───────────@───────@────────
        #               │                     │                      │         │           │       │
        # 1: ───L?[1]───R_L![2]───────M?[1]───□──────────N?[1]───────R_R![2]───┼───O?[1]───P?[1]───D'?[1]───
        #       │       │             │       │          │                     │   │       │       │
        # 2: ───L?[2]───R_L![3]───────M?[2]───□──────────N?[2]─────────────────@───O?[2]───P?[2]───D'?[2]───
        expanded = relabel_gauge_unitaries_rule(expanded, context=context)

        # STEP 9. : Migrate L's PD corrections through R_L and merge into M.
        # 0: ──────────────R_L![1]───H────────────F_B(Rz)!───────────H───R_R![1]───@───────────@───────@────────
        #                  │                      │                      │         │           │       │
        # 1: ───L_PD![1]───R_L![2]───────M'?[1]───□──────────N?[1]───────R_R![2]───┼───O?[1]───P?[1]───D'?[1]───
        #       │          │             │        │          │                     │   │       │       │
        # 2: ───L_PD![2]───R_L![3]───────M'?[2]───□──────────N?[2]─────────────────@───O?[2]───P?[2]───D'?[2]───
        # H(q0) commutes with M on the target register; keep its moment.
        indices = (0, 1, 3)  # L, R_L, M in the fixed Step 8 layout.
        migrated = tuple(
            pd_migration_through_r_rule(
                cirq.Circuit(expanded[index] for index in indices),
                context=context,
                mode=_synthesis_mode(self.mode),
            ).all_operations()
        )
        for index, operation in zip(indices, migrated, strict=True):
            expanded[index] = cirq.Moment(operation)
        merged_m = migrated[-1]
        expanded[3] = cirq.Moment(
            _on_target(
                UnresolvedNQubitUnitary(
                    merged_m.gate.matrix, label="M'", atol=merged_m.gate.atol
                ),
                merged_m,
            )
        )
        assert not _validation_enabled() or np.allclose(
            expanded.unitary(qubit_order=target.qubits),
            source.matrix,
            rtol=0,
            atol=source.atol,
        ), "Steps 1-9 must preserve the original unitary"

        # STEP 10. : Migrate PD corrections from M' through F_B(Rz) into N.
        # 0: ──────────────R_L![1]───H───────────────F_B(Rz)!───────────H───R_R![1]───@───────────@───────@────────
        #                  │                         │                      │         │           │       │
        # 1: ───L_PD![1]───R_L![2]───────M'_PD![1]───□──────────N?[1]───────R_R![2]───┼───O?[1]───P?[1]───D'?[1]───
        #       │          │             │           │          │                     │   │       │       │
        # 2: ───L_PD![2]───R_L![3]───────M'_PD![2]───□──────────N?[2]─────────────────@───O?[2]───P?[2]───D'?[2]───
        indices = (3, 4, 5)  # M', F_B(Rz), N in the fixed Step 9 layout.
        migrated = tuple(
            pd_migration_through_ucg_rule(
                cirq.Circuit(expanded[index] for index in indices),
                context=context,
                mode=_synthesis_mode(self.mode),
            ).all_operations()
        )
        for index, operation in zip(indices, migrated, strict=True):
            expanded[index] = cirq.Moment(operation)
        assert not _validation_enabled() or np.allclose(
            expanded.unitary(qubit_order=target.qubits),
            source.matrix,
            rtol=0,
            atol=source.atol,
        ), "Steps 1-10 must preserve the original unitary"

        # STEP 11. : Synthesize N = D N_D and commute D through R_R and CZ into O.
        # 0: ──────────────R_L![1]───H───────────────F_B(Rz)!─────────────H───R_R![1]───@───────────@───────@────────
        #                  │                         │                        │         │           │       │
        # 1: ───L_PD![1]───R_L![2]───────M'_PD![1]───□──────────N_D![1]───────R_R![2]───┼───O?[1]───P?[1]───D'?[1]───
        #       │          │             │           │          │                       │   │       │       │
        # 2: ───L_PD![2]───R_L![3]───────M'_PD![2]───□──────────N_D![2]─────────────────@───O?[2]───P?[2]───D'?[2]───
        # O keeps its label after absorbing D: O_new = O D.
        n, right_rotation, cz, o = (
            expanded[index].operations[0] for index in (5, 7, 8, 9)
        )
        residual, diagonal = tuple(
            NQubitUnitarySynthesisModuloDiagonal()
            .run(cirq.Circuit(n), target=n, context=context)
            .all_operations()
        )
        residual_gate = residual.gate
        expanded[5] = cirq.Moment(
            _on_target(
                NQubitUnitaryModuloDiagonal(
                    residual_gate.residual_matrix,
                    residual_gate.decomposition,
                    qubit_order=cirq.LineQubit.range(len(n.qubits)),
                    label="N_D",
                    atol=residual_gate.atol,
                ),
                residual,
            )
        )
        # D acts on the target register, so it commutes with H(q0).
        commuted_rotation, diagonal = tuple(
            switch_diagonal_through_r_gate_rule(
                cirq.Circuit(diagonal, right_rotation), context=context
            ).all_operations()
        )
        assert commuted_rotation is right_rotation
        # CZ and I tensor D are diagonal; keep the original CZ unchanged.
        assert cz.gate == cirq.CZ
        assert set(diagonal.qubits) <= set(o.qubits)
        expanded[9] = cirq.Moment(
            UnresolvedNQubitUnitary(
                o.gate.matrix * np.diag(_operation_matrix(diagonal, o.qubits)),
                label=o.gate.label,
                atol=max(o.gate.atol, residual_gate.atol),
            )
            .on(*o.qubits)
            .with_tags(*dict.fromkeys((*diagonal.tags, *o.tags)))
        )
        assert not _validation_enabled() or np.allclose(
            expanded.unitary(qubit_order=target.qubits),
            source.matrix,
            rtol=0,
            atol=source.atol,
        ), "Steps 1-11 must preserve the original unitary"

        # STEP 12. : Synthesize O = P D O_PD and leave both corrections exposed.
        # 0: ──────────────R_L![1]───H───────────────F_B(Rz)!─────────────H───R_R![1]───@───────────────────────────@───────@────────
        #                  │                         │                        │         │                           │       │
        # 1: ───L_PD![1]───R_L![2]───────M'_PD![1]───□──────────N_D![1]───────R_R![2]───┼───O_PD![1]───D?[1]───@────P?[1]───D'?[1]───
        #       │          │             │           │          │                       │   │          │       │    │       │
        # 2: ───L_PD![2]───R_L![3]───────M'_PD![2]───□──────────N_D![2]─────────────────@───O_PD![2]───D?[2]───P?───P?[2]───D'?[2]───
        # Circuit order is O_PD -> D -> P, before the existing controlled P, D'.
        o = expanded[9].operations[0]
        residual, first, second = tuple(
            self.run(cirq.Circuit(o), target=o, context=context).all_operations()
        )
        # Normalize recursive W -> P -> D_after to W -> D -> P for Step 13.
        if isinstance(first.gate, MultiplexedPermutation):
            permutation = first
            p_o = permutation.gate.matrix
            diagonal = _on_target(
                DiagonalGate(
                    DiagonalMatrix(
                        np.diag(second.gate.matrix)[np.argmax(p_o, axis=0)],
                        atol=source.atol,
                    ),
                    label=second.gate.label,
                ),
                second,
            )
        else:
            diagonal, permutation = first, second
        residual_gate = residual.gate
        residual = _on_target(
            NQubitUnitaryModuloPermutationDiagonal(
                residual_gate.residual_matrix,
                residual_gate.decomposition,
                qubit_order=cirq.LineQubit.range(len(o.qubits)),
                label="O_PD",
                atol=residual_gate.atol,
            ),
            residual,
        )
        expanded = cirq.Circuit(
            _replaced(
                cirq.FrozenCircuit(expanded), o, (residual, diagonal, permutation)
            )
        )
        assert not _validation_enabled() or np.allclose(
            expanded.unitary(qubit_order=target.qubits),
            source.matrix,
            rtol=0,
            atol=source.atol,
        ), "Steps 1-12 must preserve the original unitary"

        # STEP 13. : Merge the tail into q0-multiplexed P_hat -> D_hat.
        # General multiplexed form; an identity branch can use a control symbol.
        # 0: ──────────────R_L![1]───H───────────────F_B(Rz)!─────────────H───R_R![1]───@──────────────□────────────────────□────────────────────
        #                  │                         │                        │         │              │                    │
        # 1: ───L_PD![1]───R_L![2]───────M'_PD![1]───□──────────N_D![1]───────R_R![2]───┼───O_PD![1]───diag(P_0, P_1)?[1]───diag(D_0, D_1)?[1]───
        #       │          │             │           │          │                       │   │          │                    │
        # 2: ───L_PD![2]───R_L![3]───────M'_PD![2]───□──────────N_D![2]─────────────────@───O_PD![2]───diag(P_0, P_1)?[2]───diag(D_0, D_1)?[2]───
        # P_hat = (I ⊕ P_A)(I ⊗ P_O) = P_O ⊕ (P_A P_O).
        # Thus P_0 = P_O and P_1 = P_A P_O; P_0 need not be the identity.
        # D_hat = (I ⊕ D'_A) P_hat (I ⊗ D_O) P_hat^dagger.
        # Thus U = D_hat P_hat W, with W ending at O_PD.
        operations = tuple(expanded.all_operations())
        d_o, p_o, controlled_p, controlled_d = operations[-4:]
        p_o_matrix = p_o.gate.permutation.array
        merged_p = MultiplexedPermutation(
            PermutationMatrix(controlled_p.gate.zero_permutation.array @ p_o_matrix),
            PermutationMatrix(controlled_p.gate.one_permutation.array @ p_o_matrix),
        )
        d_values = (
            np.diag(cirq.unitary(controlled_d))
            * np.tile(np.diag(d_o.gate.matrix), 2)[np.argmax(merged_p.matrix, axis=1)]
        )
        half = len(d_values) // 2
        merged_d = MultiplexedDiagonal(
            DiagonalMatrix.from_diagonal(d_values[:half], atol=source.atol),
            DiagonalMatrix.from_diagonal(d_values[half:], atol=source.atol),
        )
        # The atomic residual stores CZ as H-CX-H and retains the other blocks.
        prefix = _cz_to_cx(operations[:-4])
        residual_gate = NQubitUnitaryModuloPermutationDiagonal(
            _remove_left_correction(
                source.matrix, np.argmax(merged_p.matrix, axis=0), d_values
            ),
            cirq.FrozenCircuit(cirq.Moment(operation) for operation in prefix),
            qubit_order=target.qubits,
            label=f"{source.label}\N{COMBINING TILDE}",
            atol=source.atol,
        )
        return (
            _on_target(residual_gate, target),
            _on_target(merged_p, target),
            _on_target(merged_d, target),
        )


__all__ = [
    "NQubitUnitarySynthesis",
    "NQubitUnitarySynthesisModuloDiagonal",
    "NQubitUnitarySynthesisModuloPermutationDiagonal",
]
