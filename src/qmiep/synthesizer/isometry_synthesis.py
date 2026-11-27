from typing import Literal

import cirq
import numpy as np

from qmiep.decomposition import BlockZXZDecomposer, BlockZXZFactors
from qmiep.gates import (
    CentralMultiplexor,
    ControlledUnitary,
    DiagonalGate,
    Isometry,
    IsometryModuloPermutationDiagonal,
    MultiplexedPermutation,
    NQubitUnitary,
    NQubitUnitaryModuloPermutationDiagonal,
    PermutationGate,
    UniformlyControlledRz,
    UnresolvedNQubitUnitary,
)
from qmiep.matrices import DiagonalMatrix, PermutationMatrix
from qmiep.rewriting import (
    hadamard_cx_to_cz_rule,
    multiplexed_unitary_decomposition_by_spectrum_rule,
    pd_migration_through_r_rule,
    pd_migration_through_ucg_rule,
)
from qmiep.rewriting.migration._common import _transpose_decomposition
from qmiep.rust import ALGORITHMS, Algorithm, _NativeCircuit
from qmiep.synthesizer._backends import (
    _algorithm_scope,
    _synthesis_algorithm,
    _synthesis_seed,
)
from qmiep.synthesizer.base import (
    _reconstruction,
    _TargetedRewriteSynthesizer,
    _tolerance,
)
from qmiep.synthesizer.unitary_synthesizer import (
    NQubitUnitarySynthesis,
    NQubitUnitarySynthesisModuloPermutationDiagonal,
)

type IsometryMode = Literal["exact", "PD-equiv"]
ISOMETRY_MODES: tuple[IsometryMode, ...] = ("exact", "PD-equiv")


def _isometry_blocks(
    target: cirq.Operation, *, context: cirq.TransformerContext
) -> tuple[tuple[cirq.Operation, ...], BlockZXZFactors]:
    source = target.gate
    assert isinstance(source, Isometry)
    qubits = target.qubits

    # STEP 1. Block-ZXZ on the ancilla-zero subspace (Eq. 36, C = I).
    # q0 |0⟩ ─────H───────●────H──────□──────
    #                     │           │
    # q1     ──┌─────┐──┌─┴─┐─────┌───┴───┐──
    #          │ C=I │  │ B │     │ A₁,A₂ │
    # q[2:]  ──└─────┘──└───┘─────└───────┘──
    isometry = source.isometry
    basis, _ = np.linalg.qr(isometry, mode="complete")
    completion = np.column_stack((isometry, basis[:, isometry.shape[1] :]))
    factors = BlockZXZDecomposer().solve(completion, atol=source.atol)
    step_1 = (
        cirq.H(qubits[0]),
        ControlledUnitary(
            np.eye(factors.b.shape[0]),
            factors.b,
            zero_label="I",
            one_label="B",
            atol=source.atol,
        ).on(*qubits),
        cirq.H(qubits[0]),
        ControlledUnitary(
            factors.a1,
            factors.a2,
            zero_label="A_1",
            one_label="A_2",
            atol=source.atol,
        ).on(*qubits),
    )

    # STEP 2. Decompose A₁ ⊕ A₂.
    # q0 |0⟩ ─────H───────●──────H────Rz─────────
    #                     │           │
    # q1     ──┌─────┐──┌─┴─┐──┌───┐──□───┌───┐──
    #          │ C=I │  │ B │  │ V │  │   │ N │
    # q[2:]  ──└─────┘──└───┘──└───┘──□───└───┘──
    multiplexor = CentralMultiplexor(factors.a1, factors.a2, atol=source.atol).on(
        *qubits
    )
    spectral = multiplexed_unitary_decomposition_by_spectrum_rule(
        cirq.Circuit(multiplexor), context=context, atol=source.atol
    )
    v, diagonal, n_block = tuple(spectral.all_operations())
    v = UnresolvedNQubitUnitary(v.gate.matrix, label="V", atol=source.atol).on(
        *qubits[1:]
    )
    ucg = UniformlyControlledRz(
        diagonal.gate.branch_matrix,
        label="Rz",
        cx_boundary="start",
        atol=source.atol,
    ).on(*qubits)
    n_block = UnresolvedNQubitUnitary(
        n_block.gate.matrix, label="N", atol=source.atol
    ).on(*qubits[1:])
    step_2 = (*step_1[:3], v, ucg, n_block)

    # STEP 3. Expose the boundary CX of the uniformly controlled Rz.
    # q0 |0⟩ ─────H───────●──────H────⊕──┌───┐─────────
    #                     │           │  │   │
    # q1     ──┌─────┐──┌─┴─┐──┌───┐──●──│ R │──┌───┐──
    #          │ C=I │  │ B │  │ V │     │   │  │ N │
    # q[2:]  ──└─────┘──└───┘──└───┘─────└───┘──└───┘──
    step_3 = (*step_2[:4], *cirq.decompose_once(ucg), n_block)

    # STEP 4. Move H through CX to form CZ.
    # q0 |0⟩ ─────H───────●───────────●──H──┌───┐─────────
    #                     │           │     │   │
    # q1     ──┌─────┐──┌─┴─┐──┌───┐──●─────│ R │──┌───┐──
    #          │ C=I │  │ B │  │ V │        │   │  │ N │
    # q[2:]  ──└─────┘──└───┘──└───┘────────└───┘──└───┘──
    step_4 = tuple(
        hadamard_cx_to_cz_rule(
            cirq.Circuit(cirq.Moment(op) for op in step_3), context=context
        ).all_operations()
    )

    # STEP 5. Merge controlled B, V and CZ into a multiplexor.
    # q0 |0⟩ ─────H──────────□───────H──┌───┐─────────
    #                        │          │   │
    # q1     ──┌─────┐──┌────┴────┐─────│ R │──┌───┐──
    #          │ C=I │  │ V,Z₁VB  │     │   │  │ N │
    # q[2:]  ──└─────┘──└─────────┘─────└───┘──└───┘──
    controlled_b, v, _ = step_4[1:4]
    z_target = np.kron(cirq.unitary(cirq.Z), np.eye(isometry.shape[1] // 2))
    v_matrix = v.gate.matrix
    merged = CentralMultiplexor(
        v_matrix, z_target @ v_matrix @ controlled_b.gate.one_matrix, atol=source.atol
    ).on(*qubits)
    step_5 = (step_4[0], merged, *step_4[4:])

    # STEP 6. Decompose the merged multiplexor.
    # q0 |0⟩ ──H──Rz──H──┌───┐─────
    #             │      │ R │
    # q[1:]  ──L──□───M──└───┘──N──
    spectral_merged = multiplexed_unitary_decomposition_by_spectrum_rule(
        cirq.Circuit(merged), context=context, atol=source.atol
    )
    step_6 = (step_5[0], *spectral_merged.all_operations(), *step_5[2:])

    # STEP 7. Use L, M and N as in the proof of Proposition 4.2.
    # q0 |0⟩ ──H──Rz──H──┌───┐─────
    #             │      │ R │
    # q[1:]  ──L──□───M──└───┘──N──
    relabelled = list(step_6)
    for index, label in ((1, "L"), (3, "M")):
        operation = step_6[index]
        relabelled[index] = UnresolvedNQubitUnitary(
            operation.gate.matrix, label=label, atol=operation.gate.atol
        ).on(*operation.qubits)
    rz = relabelled[2]
    relabelled[2] = UniformlyControlledRz(
        rz.gate.branch_matrix,
        label="Rz",
        cx_boundary=rz.gate.cx_boundary,
        atol=rz.gate.atol,
    ).on(*rz.qubits)
    step_7 = tuple(relabelled)

    # STEP 8. Migrate N's input correction through R into M.
    # q0 |0⟩ ──H──Rz──H───┌─────┐────────
    #             │       │ R′  │
    # q[1:]  ──L──□───M′──└─────┘──N_PD──
    m_prime, r_prime, n_pd = tuple(
        pd_migration_through_r_rule(
            cirq.Circuit(step_7[3], step_7[5], step_7[6]),
            context=context,
            direction="left",
        ).all_operations()
    )
    step_8 = (*step_7[:3], m_prime, step_7[4], r_prime, n_pd)

    # STEP 9. Absorb the input correction of M′ into L.
    # q0 |0⟩ ──H──Rz───H────┌───┐────────
    #             │         │ R │
    # q[1:]  ──L──□───M_PD──└───┘──N_PD──
    l_prime, f_prime, m_pd = tuple(
        pd_migration_through_ucg_rule(
            cirq.Circuit(*step_8[1:4]), context=context, direction="left"
        ).all_operations()
    )
    return (step_8[0], l_prime, f_prime, m_pd, *step_8[4:]), factors


class IsometrySynthesize(_TargetedRewriteSynthesizer[Isometry]):
    r"""Synthesize an isometry exactly or up to an input PD correction.

    In PD-equiv mode the circuit applies the input permutation and diagonal
    before the residual isometry, giving :math:`J=J_{\rm PD} D P`.
    All data unitaries use the selected algorithm. An omitted algorithm or
    seed is inherited from an enclosing synthesis. Standalone calls default
    to the heuristic algorithm.
    """

    __slots__ = ("algorithm", "mode", "seed")
    source_gate_type = Isometry
    algorithm: Algorithm | None
    mode: IsometryMode
    seed: int | None

    def __init__(
        self,
        *,
        algorithm: Algorithm | None = None,
        mode: IsometryMode | None = None,
        seed: int | None = None,
        pd: bool | None = None,
    ) -> None:
        if pd is not None:
            if not isinstance(pd, bool):
                raise TypeError("pd must be a bool")
            legacy_mode = "PD-equiv" if pd else "exact"
            if mode is not None and mode != legacy_mode:
                raise ValueError("pd and mode select different isometry outputs")
            mode = legacy_mode
        if mode is not None and mode not in ISOMETRY_MODES:
            raise ValueError("isometry mode must be 'exact' or 'PD-equiv'")
        if algorithm is not None and algorithm not in ALGORITHMS:
            raise ValueError(
                "algorithm must be 'analytic', 'heuristic', or 'heuristic-opt'"
            )
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "mode", mode or "exact")
        object.__setattr__(self, "seed", seed)

    @property
    def pd(self) -> bool:
        return self.mode == "PD-equiv"

    @property
    def replacement_gate_types(self) -> tuple[type[cirq.Gate], ...]:
        if self.pd:
            return (PermutationGate, DiagonalGate, IsometryModuloPermutationDiagonal)
        return (NQubitUnitary,)

    def _replacement_layout(
        self, source: cirq.Operation
    ) -> tuple[tuple[type[cirq.Gate], tuple[cirq.Qid, ...]], ...]:
        qubits = source.qubits
        *corrections, residual = self.replacement_gate_types
        return (*((gate, qubits[1:]) for gate in corrections), (residual, qubits))

    def _rewrite(
        self, target: cirq.Operation, *, context: cirq.TransformerContext
    ) -> tuple[cirq.Operation, ...]:
        algorithm = _synthesis_algorithm() if self.algorithm is None else self.algorithm
        seed = _synthesis_seed() if self.seed is None else self.seed
        with _algorithm_scope(algorithm, seed):
            return self._synthesize(target, context)

    def _synthesize(
        self, target: cirq.Operation, context: cirq.TransformerContext
    ) -> tuple[cirq.Operation, ...]:
        source = self._source_gate(target)
        step_9, factors = _isometry_blocks(target, context=context)
        h, l_block, *tail = step_9
        if self.mode == "exact":
            # STEP 10 (exact). Synthesize L in full (Proposition 4.2).
            # q0 |0⟩ ──H──Rz───H────┌───┐────────
            #             │         │ R │
            # q[1:]  ──L──□───M_PD──└───┘──N_PD──
            if len(l_block.qubits) == 1:
                l_exact = cirq.MatrixGate(l_block.gate.matrix).on(*l_block.qubits)
            else:
                synthesized = NQubitUnitarySynthesis().run(
                    cirq.Circuit(l_block), target=l_block, context=context
                )
                (l_exact,) = synthesized.all_operations()
            implementation = cirq.FrozenCircuit(h, l_exact, *tail)
            # With C = I, the two block columns in Step 1 sum to [A₁; A₂].
            j = source.isometry
            complement = np.vstack((factors.a1, factors.a2)) - j
            gate = NQubitUnitary(
                np.column_stack((j, complement)),
                implementation,
                qubit_order=target.qubits,
                label="U_Iso",
                atol=_tolerance(source, (l_exact, *tail)),
            )
            return (gate.on(*target.qubits).with_tags(*target.tags),)

        # STEP 10 (PD). Transposing the decomposition of L.T gives L = L_PD D P.
        # q0 |0⟩ ─────────H────Rz───H────┌───┐────────
        #                      │         │ R │
        # q[1:]  ──P──D──L_PD──□───M_PD──└───┘──N_PD──
        qubits = target.qubits
        selectors = qubits[1:]
        atol = _tolerance(source, step_9)
        l_circuit: cirq.FrozenCircuit | _NativeCircuit
        if len(selectors) == 1:
            columns = np.arange(2)
            phases = np.ones(2)
            l_matrix = l_block.gate.matrix
            l_circuit = cirq.FrozenCircuit(cirq.MatrixGate(l_matrix).on(*selectors))
        else:
            transposed = UnresolvedNQubitUnitary(
                l_block.gate.matrix.T, label="L", atol=atol
            ).on(*selectors)
            synthesized = NQubitUnitarySynthesisModuloPermutationDiagonal().run(
                cirq.Circuit(transposed), target=transposed, context=context
            )
            residual, first, second = synthesized.all_operations()
            atol = max(
                atol, *(getattr(op.gate, "atol", 0) for op in (residual, first, second))
            )
            permutation_first = isinstance(first.gate, MultiplexedPermutation)
            p, d = (first, second) if permutation_first else (second, first)
            columns = np.argmax(cirq.unitary(p), axis=0)
            phases = np.diag(cirq.unitary(d))
            if permutation_first:
                # (D P)ᵀ has entry d[p[i]] in row i, column p[i].
                phases = phases[columns]
            l_matrix = residual.gate.residual_matrix.T
            l_circuit = _transpose_decomposition(residual)
        l_residual = NQubitUnitaryModuloPermutationDiagonal(
            l_matrix, l_circuit, qubit_order=selectors, label="L_PD", atol=atol
        ).on(*selectors)
        implementation = cirq.FrozenCircuit(
            cirq.Moment(op) for op in (h, l_residual, *tail)
        )
        # C[i, columns[i]] = d[i], so J C† is J[:, columns] diag(conj(d)).
        iso = IsometryModuloPermutationDiagonal(
            source.isometry[:, columns] * phases.conj(),
            implementation,
            label="U_Iso_PD",
            qubit_order=qubits,
            atol=atol,
        ).on(*qubits)
        diagonal = DiagonalGate(
            DiagonalMatrix.from_diagonal(phases, atol=atol), label="D"
        ).on(*selectors)
        permutation = PermutationGate(
            PermutationMatrix.from_permutation(columns), label="P"
        ).on(*selectors)
        return tuple(op.with_tags(*target.tags) for op in (permutation, diagonal, iso))

    def _validate_reconstruction(
        self,
        source: cirq.Operation,
        source_gate: Isometry,
        operations: tuple[cirq.Operation, ...],
    ) -> None:
        half = source_gate.isometry.shape[1]
        reconstructed = _reconstruction(operations, source.qubits)[:, :half]
        if not np.allclose(
            reconstructed,
            source_gate.isometry,
            rtol=0,
            atol=_tolerance(source_gate, operations),
        ):
            raise ValueError(
                "replacement operations must reconstruct the input isometry"
            )


__all__ = ["IsometrySynthesize"]
