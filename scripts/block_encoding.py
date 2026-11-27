"""Benchmark full-rank block encoding with one ancilla (Section 4.1).

--qubits counts the ancilla and the data qubits. Each sample uses a complex
Gaussian matrix A and alpha = ||A||_2. The comparison uses the Rust backend's
Qiskit QSD baseline with D-migration from V into W.

python scripts/block_encoding.py --algorithm all --qubits 4 5 6
python scripts/block_encoding.py --algorithm heuristic-opt --qubits 5
python scripts/block_encoding.py --qubits 3 4 --samples 5 2 --compare-with-qiskit
"""

from collections.abc import Sequence
from time import perf_counter

import cirq
import numpy as np

from qmiep import BlockEncodingSynthesize
from qmiep._synthesis_benchmark import (
    RUST_BACKENDS,
    Arguments,
    Case,
    parse_benchmark_args,
    run_benchmark,
)
from qmiep._validation import _validation_scope
from qmiep.cnot import count_cirq_cnots

BASELINE = "Qiskit QSD + D (Rust)"
type Sample = tuple[np.ndarray, float]


def parse_args(argv: Sequence[str] | None = None) -> Arguments:
    return parse_benchmark_args(
        argv,
        description=__doc__,
        modes=("exact",),
        qiskit_name=BASELINE,
        default_max=6,
    )


def spectral_norm(matrix: np.ndarray) -> float:
    """Return ||A||_2 for a full-rank matrix in the domain of Section 4.1."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A must be square")
    d = matrix.shape[0]
    if d == 0 or d & (d - 1) or not np.isfinite(matrix).all():
        raise ValueError("A must have finite entries and power-of-two dimension")
    sigma = np.linalg.svd(matrix, compute_uv=False)
    if sigma[-1] <= d * np.finfo(float).eps * sigma[0]:
        raise ValueError("A must have full numerical rank")
    return float(sigma[0])


def sample_matrix(n: int, seed: int) -> Sample:
    d = 2 ** (n - 1)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    return matrix, spectral_norm(matrix)


def verify_block_encoding(
    unitary: np.ndarray, matrix: np.ndarray, alpha: float, *, atol: float = 1e-10
) -> float:
    """Check unitarity and (⟨0| ⊗ I) U (|0⟩ ⊗ I) = A/alpha in operator norm."""
    norm = spectral_norm(matrix)
    d = len(matrix)
    if (
        not np.isrealobj(alpha)
        or not np.isfinite(alpha)
        or alpha <= 0
        or norm - alpha > d * np.finfo(float).eps * norm
    ):
        raise ValueError("alpha must be real, finite, positive and at least ||A||_2")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    if unitary.shape != (2 * d, 2 * d) or not np.isfinite(unitary).all():
        raise RuntimeError(f"Expected a finite {2 * d} by {2 * d} circuit matrix")
    # Ancilla q(0) is the most significant bit on both input and output.
    block_error = float(np.linalg.norm(unitary[:d, :d] - matrix / alpha, ord=2))
    if not np.isfinite(block_error) or block_error > atol:
        raise RuntimeError(
            f"Ancilla-zero block error {block_error:.3e} exceeds {atol:g}"
        )
    unitarity_error = np.linalg.norm(unitary.conj().T @ unitary - np.eye(2 * d), ord=2)
    if not np.isfinite(unitarity_error) or unitarity_error > atol:
        raise RuntimeError(f"Unitarity error {unitarity_error:.3e} exceeds {atol:g}")
    return block_error


def measure(
    name: Case, problem: Sample, seed: int, validate: bool
) -> tuple[float, int]:
    matrix, alpha = problem
    algorithm, mode = (None, "baseline") if name == BASELINE else name
    with _validation_scope(False):
        start = perf_counter()
        result = BlockEncodingSynthesize().run(
            matrix,
            alpha=alpha,
            algorithm=algorithm,
            mode=mode,
            seed=seed,
            backends=RUST_BACKENDS,
        )
        seconds = perf_counter() - start
    gates = cirq.decompose(
        result,
        keep=lambda op: op.gate is not None
        and (len(op.qubits) <= 1 or op.gate == cirq.CX),
    )
    count = count_cirq_cnots(gates)
    if validate:
        qubits = cirq.LineQubit.range(len(matrix).bit_length())
        unitary = cirq.Circuit(gates).unitary(qubit_order=qubits)
        verify_block_encoding(unitary, matrix, alpha)
    return seconds, count


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    warmup = sample_matrix(3, args.seed)
    for name in args.cases:
        measure(name, warmup, args.seed, False)
    if args.include_qiskit:
        measure(BASELINE, warmup, args.seed, False)
    run_benchmark(
        args,
        sample=sample_matrix,
        measure=measure,
        qiskit_name=BASELINE,
        validation="unitarity and ancilla-zero block",
        timing="Times cover the SVD and block-encoding circuit construction, including "
        "Rust unitary synthesis. QMIEP uses PD-migration and the QSD baseline uses "
        "D-migration. Sampling, warm-up, CNOT counting and verification are excluded.",
    )


if __name__ == "__main__":
    main()
