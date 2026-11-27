"""Eigenbasis decomposition of a single controlled unitary."""

import cirq
import numpy as np

from qmiep._validation import _validation_enabled
from qmiep.gates import (
    CentralMultiplexor,
    ControlledUnitary,
    DiagonalGate,
    UniformlyControlledRz,
    UnresolvedNQubitUnitary,
)
from qmiep.matrices import DiagonalMatrix


@cirq.transformer
def multiplexed_unitary_decomposition_by_spectrum_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    target: cirq.Operation | None = None,
    antipodal: bool | None = None,
    label: str = "U",
    diagonal_label: str | None = None,
    atol: float = 1e-10,
) -> cirq.Circuit:
    r"""Expand one closed, single-qubit controlled unitary through its eigenbasis.

    For :math:`U=Q\Lambda Q^\dagger`, circuit order is
    :math:`Q^\dagger,\operatorname{ctrl}(\Lambda),Q`. The first wire is the
    control; basis changes act only on the remaining wires in their input order.

    By default, detect antipodal spectra automatically. Order paired eigenvalues
    and eigenvectors as :math:`(a,-a,b,-b,\ldots)` to obtain
    :math:`\Lambda=D\otimes Z`, with Z on the LAST target wire.
    Emit :math:`Q^\dagger,\operatorname{ctrl}(D),CZ,Q` in this case.
    D acts on all targets except the last; for one target it is a scalar,
    implemented as a phase on the control.
    ``antipodal=True`` requires this structure; ``False`` disables pairing.
    For a multiplexor, diagonalize :math:`B_1B_2^\dagger=QD^2Q^\dagger`.
    With :math:`W=DQ^\dagger B_2`, emit :math:`W`, UCG-Rz, :math:`Q`, since
    :math:`B_1\oplus B_2=(I\otimes Q)(D\oplus D^\dagger)(I\otimes W)`.
    Antipodal pairing applies only to identity-controlled unitaries.

    Without ``target``, exactly one top-level operation is required. Otherwise
    replace only the identity-selected operation in a larger circuit.
    Tags and ignored operations
    are preserved. Both QMIEP and native Cirq controlled gates are supported.
    """
    operations = tuple(circuit.all_operations())
    if target is not None:
        if sum(op is target for op in operations) != 1:
            raise ValueError("target must reference exactly one top-level operation")
        replacement = multiplexed_unitary_decomposition_by_spectrum_rule(
            cirq.Circuit(target),
            context=context,
            antipodal=antipodal,
            label=label,
            diagonal_label=diagonal_label,
            atol=atol,
        )
        return cirq.Circuit(
            cirq.map_operations_and_unroll(
                circuit,
                lambda op, _: (
                    tuple(replacement.all_operations()) if op is target else op
                ),
                deep=False,
                tags_to_ignore=context.tags_to_ignore if context else (),
            )
        )
    if len(operations) != 1:
        raise ValueError("expected exactly one controlled-unitary operation")
    operation = operations[0]
    if context and any(tag in context.tags_to_ignore for tag in operation.tags):
        return cirq.Circuit(circuit)
    gate = operation.gate
    bare = operation.untagged
    if isinstance(gate, ControlledUnitary) and (
        isinstance(gate, CentralMultiplexor)
        or not np.allclose(
            gate.zero_matrix, np.eye(len(gate.zero_matrix)), rtol=0, atol=atol
        )
    ):
        if antipodal:
            raise ValueError(
                "antipodal pairing requires an identity-controlled unitary"
            )
        squared_spectrum, basis = cirq.linalg.unitary_eig(
            gate.zero_matrix @ gate.one_matrix.conj().T, atol=atol
        )
        diagonal = np.diag(np.exp(0.5j * np.angle(squared_spectrum)))
        left = diagonal @ basis.conj().T @ gate.one_matrix
        replacements = (
            UnresolvedNQubitUnitary(left, label="B_L", atol=atol).on(
                *operation.qubits[1:]
            ),
            UniformlyControlledRz(diagonal, label="F_B(Rz)", atol=atol).on(
                *operation.qubits
            ),
            UnresolvedNQubitUnitary(basis, label="B_R", atol=atol).on(
                *operation.qubits[1:]
            ),
        )
        result = cirq.Circuit(
            cirq.Moment(op.with_tags(*operation.tags)) for op in replacements
        )
        assert not _validation_enabled() or np.allclose(
            result.unitary(qubit_order=operation.qubits),
            cirq.unitary(operation),
            rtol=0,
            atol=atol,
        ), "spectral decomposition must preserve the multiplexor"
        return result
    if isinstance(gate, ControlledUnitary):
        matrix = gate.one_matrix
    else:
        if isinstance(bare, cirq.ControlledOperation):
            controls, values = len(bare.controls), bare.control_values
            sub = bare.sub_operation
        elif isinstance(gate, cirq.ControlledGate):
            controls, values = gate.num_controls(), gate.control_values
            sub = gate.sub_gate
        else:
            raise TypeError("expected a controlled-unitary operation")
        if controls != 1 or values != cirq.ProductOfSums(((1,),)):
            raise ValueError("expected one control activated on |1>")
        matrix = cirq.unitary(sub)
    if len(operation.qubits) < 2 or any(q.dimension != 2 for q in operation.qubits):
        raise ValueError("expected a qubit control and at least one target qubit")

    spectrum, basis = cirq.linalg.unitary_eig(matrix, atol=atol)
    paired = False
    if antipodal is not False:
        # Cut the eigenvalue circle inside its largest gap, away from clustered
        # or repeated eigenvalues. Opposite half-circles then have matching order.
        angles = np.mod(np.angle(spectrum), 2 * np.pi)
        order = np.argsort(angles)
        gaps = np.diff(np.r_[angles[order], angles[order[0]] + 2 * np.pi])
        order = np.roll(order, -int(np.argmax(gaps)) - 1)
        half = len(order) // 2
        paired = np.allclose(
            spectrum[order[:half]], -spectrum[order[half:]], rtol=0, atol=atol
        )
        if paired:
            order = np.column_stack((order[:half], order[half:])).ravel()
            basis = basis[:, order]
            spectrum = np.kron(spectrum[order[::2]], [1, -1])
        elif antipodal:
            raise ValueError("the spectrum is not antipodal")
    diagonal_label = diagonal_label or f"D_{label}"
    targets = operation.qubits[1:]
    control = operation.qubits[0]
    if paired:
        if len(targets) == 1:
            controlled_d = DiagonalGate(
                DiagonalMatrix(np.diag([1, spectrum[0]])), label=diagonal_label
            ).on(control)
        else:
            controlled_d = (
                DiagonalGate(
                    DiagonalMatrix(np.diag(spectrum[::2])), label=diagonal_label
                )
                .controlled()
                .on(control, *targets[:-1])
            )
        central = (controlled_d, cirq.CZ(control, targets[-1]))
    else:
        central = (
            DiagonalGate(DiagonalMatrix(np.diag(spectrum)), label=diagonal_label)
            .controlled()
            .on(*operation.qubits),
        )
    replacements = (
        UnresolvedNQubitUnitary(basis.conj().T, label=f"Q_{label}†", atol=atol).on(
            *targets
        ),
        *central,
        UnresolvedNQubitUnitary(basis, label=f"Q_{label}", atol=atol).on(*targets),
    )
    result = cirq.Circuit(
        cirq.Moment(op.with_tags(*operation.tags) if operation.tags else op)
        for op in replacements
    )
    assert not _validation_enabled() or np.allclose(
        result.unitary(qubit_order=operation.qubits),
        cirq.unitary(operation),
        rtol=0,
        atol=atol,
    ), "spectral decomposition must preserve the controlled unitary"
    return result
