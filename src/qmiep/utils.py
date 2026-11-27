"""Shared internal utilities for QMIEP."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    """Return an immutable, bytes-backed contiguous copy."""
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _unitary(values: ArrayLike, *, atol: float) -> NDArray[np.complex128]:
    """Validate and return an immutable positive-qubit unitary matrix."""
    if atol < 0:
        raise ValueError("atol must be non-negative")

    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("a unitary matrix must be square")

    dimension = matrix.shape[0]
    if dimension < 2 or dimension & (dimension - 1):
        raise ValueError("the matrix dimension must be a positive power of two")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("unitary matrix entries must be finite")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(dimension), rtol=0, atol=atol):
        raise ValueError("the matrix must be unitary")
    return _readonly(matrix)
