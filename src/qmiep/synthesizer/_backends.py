"""Per-kind backend selection scoped to a synthesis call and its children."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from qmiep.rust import ALGORITHMS, Algorithm

type SynthesisBackend = Literal["python", "rust"]
type SynthesisMode = Literal["analytic", "heuristic"]


@dataclass(frozen=True, slots=True)
class SynthesisBackends:
    """Select the implementation of exact U, U_D, and U_PD synthesis.

    Pass this configuration as ``backends`` to ``state_preparation`` or a
    synthesizer's ``run`` method. Python orchestration passes the selection
    to nested synthesis calls, including isometry synthesis. Each Rust call
    uses the direct PyO3 binding and executes its entire recursion in Rust;
    selections for other kinds do not override that native recursion.
    """

    unitary: SynthesisBackend = "rust"
    diagonal: SynthesisBackend = "rust"
    pd: SynthesisBackend = "rust"

    def __post_init__(self) -> None:
        for name in ("unitary", "diagonal", "pd"):
            if getattr(self, name) not in ("python", "rust"):
                raise ValueError(f"{name} backend must be 'python' or 'rust'")


# Standalone rewrite synthesizers retain their Python implementation unless
# given a configuration. State preparation supplies the all-Rust defaults.
_PYTHON_BACKENDS = SynthesisBackends(unitary="python", diagonal="python", pd="python")
_BACKENDS = ContextVar("qmiep_synthesis_backends", default=_PYTHON_BACKENDS)
_ALGORITHM: ContextVar[Algorithm | None] = ContextVar(
    "qmiep_synthesis_algorithm", default=None
)
_SEED: ContextVar[int | None] = ContextVar("qmiep_synthesis_seed", default=None)


def _synthesis_backends() -> SynthesisBackends:
    return _BACKENDS.get()


def _synthesis_algorithm(default: Algorithm = "heuristic") -> Algorithm:
    return _ALGORITHM.get() or default


def _synthesis_mode(default: SynthesisMode) -> SynthesisMode:
    algorithm = _synthesis_algorithm(default)
    return "heuristic" if algorithm == "heuristic-opt" else algorithm


def _synthesis_seed() -> int | None:
    return _SEED.get()


@contextmanager
def _algorithm_scope(algorithm: Algorithm, seed: int | None) -> Iterator[None]:
    if algorithm not in ALGORITHMS:
        raise ValueError(
            "algorithm must be 'analytic', 'heuristic', or 'heuristic-opt'"
        )
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
    ):
        raise ValueError("seed must be an integer in [0, 2**64) or None")
    algorithm_token = _ALGORITHM.set(algorithm)
    seed_token = _SEED.set(seed)
    try:
        yield
    finally:
        _SEED.reset(seed_token)
        _ALGORITHM.reset(algorithm_token)


@contextmanager
def _mode_scope(mode: SynthesisMode, seed: int | None) -> Iterator[None]:
    if mode not in ("analytic", "heuristic"):
        raise ValueError("mode must be 'analytic' or 'heuristic'")
    with _algorithm_scope(mode, seed):
        yield


@contextmanager
def _backend_scope(backends: SynthesisBackends | None) -> Iterator[None]:
    if backends is not None and not isinstance(backends, SynthesisBackends):
        raise TypeError("backends must be a SynthesisBackends instance or None")
    token = _BACKENDS.set(_BACKENDS.get() if backends is None else backends)
    try:
        yield
    finally:
        _BACKENDS.reset(token)
