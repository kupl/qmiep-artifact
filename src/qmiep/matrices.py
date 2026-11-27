"""Validated matrix value objects used by QMIEP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmiep.utils import _readonly


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PermutationMatrix:
    r"""A matrix :math:`P\in\{0,1\}^{n\times n}` with one 1 per row and column."""

    array: NDArray[np.uint8]

    def __init__(self, values: ArrayLike) -> None:
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("a permutation matrix must be square")
        if not np.all((array == 0) | (array == 1)):
            raise ValueError("every entry must be exactly 0 or 1")
        if not np.all(array.sum(axis=0) == 1) or not np.all(array.sum(axis=1) == 1):
            raise ValueError("each row and column must contain exactly one 1")

        normalized = np.asarray(array.real, dtype=np.uint8)
        object.__setattr__(self, "array", _readonly(normalized))

    def __array__(
        self, dtype: np.dtype[Any] | None = None, copy: bool | None = None
    ) -> NDArray[Any]:
        return (
            np.array(self.array, dtype=dtype, copy=True)
            if copy
            else np.asarray(self.array, dtype=dtype)
        )

    @classmethod
    def from_permutation(cls, permutation: ArrayLike) -> Self:
        r"""Construct :math:`P` with :math:`P_{i,p_i}=1`."""
        indices = np.asarray(permutation)
        if indices.ndim != 1:
            raise ValueError("permutation indices must be a 1-D array")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("permutation indices must be integers")
        if not np.array_equal(np.sort(indices), np.arange(len(indices))):
            raise ValueError("indices must be a permutation of 0, ..., n - 1")
        return cls(np.eye(len(indices), dtype=np.uint8)[indices])


@dataclass(frozen=True, slots=True, init=False, eq=False)
class DiagonalMatrix:
    r"""A diagonal matrix :math:`D` whose entries satisfy :math:`|D_{ii}|=1`."""

    array: NDArray[np.complex128]

    def __init__(self, values: ArrayLike, *, atol: float = 1e-10) -> None:
        if atol < 0:
            raise ValueError("atol must be non-negative")

        array = np.asarray(values)
        if array.ndim == 1:
            diagonal = array
        elif array.ndim == 2 and array.shape[0] == array.shape[1]:
            diagonal = np.diag(array)
            if not np.allclose(array, np.diag(diagonal), rtol=0, atol=atol):
                raise ValueError("all off-diagonal entries must be zero")
        else:
            raise ValueError("expected a 1-D diagonal or a square diagonal matrix")

        diagonal = np.asarray(diagonal, dtype=np.complex128)
        if not np.all(np.isfinite(diagonal)):
            raise ValueError("diagonal entries must be finite")
        if not np.allclose(np.abs(diagonal), 1, rtol=0, atol=atol):
            raise ValueError("every diagonal entry must have complex modulus 1")

        object.__setattr__(self, "array", _readonly(np.diag(diagonal)))

    def __array__(
        self, dtype: np.dtype[Any] | None = None, copy: bool | None = None
    ) -> NDArray[Any]:
        return (
            np.array(self.array, dtype=dtype, copy=True)
            if copy
            else np.asarray(self.array, dtype=dtype)
        )

    @classmethod
    def from_diagonal(cls, diagonal: ArrayLike, *, atol: float = 1e-10) -> Self:
        """Construct from diagonal entries."""
        return cls(diagonal, atol=atol)
