"""Matrix operations used at the circuit/numerical synthesis boundary."""

import cirq
import numpy as np
from numpy.typing import NDArray


def _operation_matrix(
    operation: cirq.Operation, qubits: tuple[cirq.Qid, ...]
) -> NDArray[np.complex128]:
    """Embed an operation in the requested qubit order without simulating a circuit."""
    matrix = cirq.unitary(operation)
    if operation.qubits == qubits:
        return matrix
    remaining = tuple(q for q in qubits if q not in operation.qubits)
    order = (*operation.qubits, *remaining)
    axes = tuple(order.index(q) for q in qubits)
    n = len(qubits)
    return (
        np.kron(matrix, np.eye(2 ** len(remaining)))
        .reshape((2,) * (2 * n))
        .transpose((*axes, *(axis + n for axis in axes)))
        .reshape(2**n, 2**n)
    )


def _remove_left_correction(
    matrix: NDArray[np.complex128],
    permutation: NDArray[np.intp],
    diagonal: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    r"""Return W=P†D†U for U=DPW, lifting suffix corrections with identities.

    ``permutation[j]`` is pi(j) for P|j> = |pi(j)>; hence
    W[j, :] = conj(diagonal[pi(j)]) * U[pi(j), :].
    """
    rows = np.arange(len(matrix)).reshape(-1, len(permutation))[:, permutation].ravel()
    phases = np.tile(diagonal, len(matrix) // len(diagonal))
    return phases[rows, None].conj() * matrix[rows]
