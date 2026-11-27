"""Shared orchestration and validation for targeted circuit synthesis."""

from abc import ABC, abstractmethod
from typing import cast

import cirq
import numpy as np
from numpy.typing import NDArray

from qmiep._validation import _validation_enabled, _validation_scope
from qmiep.gates import Isometry, UnresolvedNQubitUnitary
from qmiep.synthesizer._backends import SynthesisBackends, _backend_scope


def _target_operation(
    circuit: cirq.AbstractCircuit, target: cirq.Operation
) -> cirq.Operation:
    if not isinstance(target, cirq.Operation):
        raise TypeError("target must be a cirq.Operation")
    occurrences = sum(operation is target for operation in circuit.all_operations())
    if occurrences != 1:
        raise ValueError(
            "target must directly reference exactly one operation in circuit; "
            f"found {occurrences}"
        )
    return target


def _replaced(
    circuit: cirq.FrozenCircuit,
    target: cirq.Operation,
    operations: tuple[cirq.Operation, ...],
) -> cirq.FrozenCircuit:
    """Return ``circuit`` with ``target`` expanded in place into ``operations``.

    The first replacement operation keeps the target's moment so unrelated
    operations stay where they are; each correction follows in its own moment.
    """
    moments: list[cirq.Moment] = []
    for moment in circuit:
        if any(operation is target for operation in moment.operations):
            moments.append(
                cirq.Moment(
                    operations[0] if operation is target else operation
                    for operation in moment.operations
                )
            )
            moments.extend(cirq.Moment(operation) for operation in operations[1:])
        else:
            moments.append(moment)
    return cirq.FrozenCircuit(moments)


def _reconstruction(
    operations: tuple[cirq.Operation, ...], qubits: tuple[cirq.Qid, ...]
) -> NDArray[np.complex128]:
    """Return the unitary implemented by ``operations`` in circuit-time order."""
    return cirq.Circuit(operations).unitary(
        qubit_order=qubits,
        qubits_that_should_be_present=qubits,
        dtype=np.complex128,
    )


def _tolerance(
    source: UnresolvedNQubitUnitary | Isometry, operations: tuple[cirq.Operation, ...]
) -> float:
    """Return the loosest tolerance declared by the target and its replacement."""
    tolerances = [source.atol]
    for operation in operations:
        tolerance = getattr(operation.gate, "atol", None)
        if isinstance(tolerance, float):
            tolerances.append(tolerance)
    return max(tolerances)


class _TargetedRewriteSynthesizer[SourceGate: (UnresolvedNQubitUnitary, Isometry)](ABC):
    """Replace one selected unresolved operation by concrete operations.

    :meth:`run` owns everything the synthesizers share: selecting the target,
    honoring the transformer context, checking the declared replacement, and
    splicing it into the circuit.  A subclass writes its own rewriting logic in
    :meth:`_rewrite` and declares the layout that logic produces.
    """

    __slots__ = ()

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("targeted rewrite synthesizers are immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError("targeted rewrite synthesizers are immutable")

    @property
    def name(self) -> str:
        """The public name under which this synthesizer is exported."""
        return type(self).__name__.removeprefix("_")

    @property
    @abstractmethod
    def source_gate_type(self) -> type[SourceGate]:
        """The exact semantic instruction type accepted by this synthesizer."""

    @property
    @abstractmethod
    def replacement_gate_types(self) -> tuple[type[cirq.Gate], ...]:
        """The default atomic replacement gate types, in circuit-time order."""

    def _replacement_layout(
        self, source: cirq.Operation
    ) -> tuple[tuple[type[cirq.Gate], tuple[cirq.Qid, ...]], ...]:
        """Return each replacement's exact gate type and ordered qids."""
        return tuple(
            (gate_type, source.qubits) for gate_type in self.replacement_gate_types
        )

    @abstractmethod
    def _rewrite(
        self,
        target: cirq.Operation,
        *,
        context: cirq.TransformerContext,
    ) -> tuple[cirq.Operation, ...]:
        """Return the circuit-time replacement operations for ``target``."""

    def run(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        target: cirq.Operation,
        context: cirq.TransformerContext | None = None,
        validate: bool | None = None,
        backends: SynthesisBackends | None = None,
    ) -> cirq.FrozenCircuit:
        """Return ``circuit`` with only ``target`` replaced by its synthesis.

        ``validate=False`` skips unitary reconstruction checks in this call,
        including recursive synthesis, atomic gates, and intermediate rewrites.
        Input and structural checks, and matrices needed by synthesis, remain.
        ``None`` inherits the enclosing call's policy; validation is on by default.
        ``backends`` selects Python or Rust separately for exact, diagonal-,
        and PD-modulus synthesis. ``None`` inherits the enclosing selection,
        defaulting to Python. The selection also applies inside isometries.
        """
        initial = cirq.FrozenCircuit(circuit)
        source = _target_operation(initial, target)
        source_gate = self._source_gate(source)
        transformer_context = cirq.TransformerContext() if context is None else context
        if any(tag in transformer_context.tags_to_ignore for tag in source.tags):
            raise ValueError("target carries a tag ignored by the context")

        with _backend_scope(backends), _validation_scope(validate):
            operations = self._rewrite(source, context=transformer_context)

            self._validate_replacement(source, source_gate, operations)
            return _replaced(initial, source, operations)

    def _validate_replacement(
        self,
        source: cirq.Operation,
        source_gate: SourceGate,
        operations: tuple[cirq.Operation, ...],
    ) -> None:
        """Reject a replacement that is not a faithful in-place synthesis."""
        layout = self._replacement_layout(source)
        if not isinstance(operations, tuple):
            raise TypeError("a targeted rewrite must return an operation tuple")
        if not operations:
            raise ValueError("a targeted rewrite must not return an empty tuple")
        if len(operations) != len(layout):
            raise ValueError(
                f"a targeted rewrite must return exactly {len(layout)} "
                "replacement operations"
            )

        for index, (operation, (gate_type, qubits)) in enumerate(
            zip(operations, layout, strict=True)
        ):
            if not isinstance(operation, cirq.Operation):
                raise TypeError(f"operations[{index}] must be a cirq.Operation")
            if type(operation.gate) is not gate_type:
                raise TypeError(
                    f"operations[{index}] gate must be exactly {gate_type.__name__}"
                )
            if operation.qubits != qubits:
                raise ValueError(
                    f"operations[{index}] must act on its expected ordered qubits"
                )

        if any(tag not in operations[0].tags for tag in source.tags):
            raise ValueError("the first replacement must preserve the target's tags")

        if _validation_enabled():
            self._validate_reconstruction(source, source_gate, operations)

    def _validate_reconstruction(
        self,
        source: cirq.Operation,
        source_gate: SourceGate,
        operations: tuple[cirq.Operation, ...],
    ) -> None:
        """Check the full unitary; an isometry may restrict the input subspace."""
        reconstructed = _reconstruction(operations, source.qubits)
        if reconstructed.shape != source_gate.matrix.shape or not np.allclose(
            reconstructed,
            source_gate.matrix,
            rtol=0,
            atol=_tolerance(source_gate, operations),
        ):
            raise ValueError("replacement operations must reconstruct the target")

    def _source_gate(self, target: cirq.Operation) -> SourceGate:
        """Validate and return the semantic gate selected for rewriting."""
        gate = target.gate
        if type(gate) is not self.source_gate_type:
            raise TypeError(
                f"target gate must be exactly {self.source_gate_type.__name__}"
            )
        return cast(SourceGate, gate)

    def _unsupported(self, target: cirq.Operation) -> NotImplementedError:
        """Return the error for a target size this synthesizer cannot rewrite yet."""
        return NotImplementedError(
            f"{self.name} has no rewrite rule for a "
            f"{len(target.qubits)}-qubit target"
        )
