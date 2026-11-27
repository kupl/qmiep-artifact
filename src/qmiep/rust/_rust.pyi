from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type Instruction = (
    tuple[Literal["u"], int, float, float, float]
    | tuple[Literal["rz"], int, float]
    | tuple[Literal["h"], int]
    | tuple[Literal["cx"], int, int]
)

def synthesize(
    unitary: NDArray[np.complex128],
    *,
    algorithm: Literal["analytic", "heuristic", "heuristic-opt"] | None = None,
    mode: Literal["exact", "PD-equiv", "D-equiv"] | None = None,
    kind: (
        Literal[
            "exact",
            "exact_further_D_optimized",
            "diagonal",
            "D_naive",
            "D_equivalent_further_optimal",
            "pd",
            "pd_further_D_optimized",
        ]
        | None
    ) = None,
    method: Literal["analytic", "heuristic", "baseline"] | None = None,
    seed: int | None = None,
    atol: float = 1e-10,
    check_input: bool = True,
) -> tuple[list[Instruction], float, NDArray[np.uintp], NDArray[np.complex128]]: ...
def solve_spectrum(
    unitary: NDArray[np.complex128],
    *,
    candidates: Iterable[Sequence[int]] | None = None,
    max_permutations: int | None = None,
    restarts: int = 3,
    maxiter: int = 500,
    atol: float = 1e-10,
    seed: int | None = None,
) -> tuple[NDArray[np.uintp], NDArray[np.complex128], bool, float, int, int, int]: ...
