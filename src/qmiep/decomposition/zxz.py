"""One-level block-ZXZ decomposition of arbitrary unitary matrices."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qmiep.utils import _readonly, _unitary


@dataclass(frozen=True, slots=True)
class BlockZXZFactors:
    r"""Factors satisfying the normalized-Hadamard block-ZXZ identity.

    .. math::

        U=(A_1\oplus A_2)(H\otimes I)(I\oplus B)
          (H\otimes I)(I\oplus C).
    """

    a1: NDArray[np.complex128]
    a2: NDArray[np.complex128]
    b: NDArray[np.complex128]
    c: NDArray[np.complex128]


class BlockZXZDecomposer:
    """Compute one level of the block-ZXZ decomposition."""

    def solve(self, matrix: ArrayLike, *, atol: float = 1e-10) -> BlockZXZFactors:
        r"""Return :math:`A_1,A_2,B,C` for an arbitrary unitary :math:`U`."""
        unitary = _unitary(matrix, atol=atol)
        block_size = unitary.shape[0] // 2
        x = unitary[:block_size, :block_size]
        y = unitary[:block_size, block_size:]
        u21 = unitary[block_size:, :block_size]
        u22 = unitary[block_size:, block_size:]

        sx, ux = self._left_polar(x)
        sy, uy = self._left_polar(y)
        c_dagger = 1j * uy.conj().T @ ux
        c = c_dagger.conj().T
        a1 = (sx + 1j * sy) @ ux
        a2 = u21 + u22 @ c_dagger
        b = 2 * a1.conj().T @ x - np.eye(block_size)

        return BlockZXZFactors(*map(_readonly, (a1, a2, b, c)))

    @staticmethod
    def _left_polar(
        matrix: NDArray[np.complex128],
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        r"""Return :math:`S,U` such that the input equals :math:`SU`."""
        left, singular_values, right_dagger = np.linalg.svd(matrix, full_matrices=True)
        positive = (left * singular_values) @ left.conj().T
        unitary = left @ right_dagger
        return positive, unitary
