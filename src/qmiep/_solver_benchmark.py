"""Shared sampling, workers and terminal summaries for the QMIEP solver benchmarks."""

import argparse
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.pool import Pool
from time import monotonic

import cirq
import numpy as np
from numpy.typing import NDArray
from rich import box
from rich.live import Live
from rich.table import Table

from qmiep._synthesis_benchmark import qubit_sample_schedule


def haar_unitary(qubits: int, sample: int, seed: int) -> NDArray[np.complex128]:
    stream = np.random.SeedSequence([seed, qubits, sample, 0])
    rng = np.random.RandomState(stream.generate_state(4))
    return cirq.testing.random_unitary(2**qubits, random_state=rng)


def solver_seeds(qubits: int, sample: int, seed: int, count: int) -> list[int]:
    stream = np.random.SeedSequence([seed, qubits, sample, 1])
    return [int(s) for s in stream.generate_state(count, dtype=np.uint64)]


def validate_qmeip_solution(
    matrix: NDArray[np.complex128],
    rows: NDArray[np.uintp],
    diagonal: NDArray[np.complex128],
    *,
    atol: float,
) -> None:
    r"""Check the permutation P, the diagonal unitary D and the spectrum of DPU.

    For the sorted eigenphases \(\alpha_0 \le \cdots \le \alpha_{2m-1}\) of DPU,
    require \(\max_{0 \le j < m} |\alpha_{j+m} - \alpha_j - \pi| < \mathrm{atol}\).
    """
    n = len(matrix)
    if not np.issubdtype(rows.dtype, np.integer) or not np.array_equal(
        np.sort(rows), np.arange(n)
    ):
        raise RuntimeError("QMIEP rows must be a permutation of the input rows")
    if diagonal.shape != (n,) or not np.allclose(
        np.abs(diagonal), 1, rtol=0, atol=atol
    ):
        raise RuntimeError("QMIEP diagonal must contain one unit-modulus entry per row")

    corrected = diagonal[:, None] * matrix[rows]
    phases = np.sort(np.angle(np.linalg.eigvals(corrected)))
    half = n // 2
    error = np.max(np.abs(phases[half:] - phases[:half] - np.pi))
    if not np.isfinite(error) or error >= atol:
        raise RuntimeError(
            f"QMIEP antipodal error {error:.6g} rad must be below {atol:.6g} rad"
        )


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_benchmark_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    low: int,
    high: int,
    *,
    minimum: int | None = None,
    samples: int | None = None,
    sample_groups: Sequence[tuple[int, int, int]] = (),
) -> argparse.Namespace:
    add = parser.add_argument
    add(
        "--qubits",
        type=int,
        nargs="+",
        help="Explicit qubit counts, in execution order",
    )
    add(
        "--samples",
        type=positive_int,
        nargs="+",
        default=None if samples is None else [samples],
        help="One count for all selected sizes, or one count per size in order",
    )
    for start, end, default in sample_groups:
        sizes = str(start) if start == end else f"{start}-{end}"
        add(
            f"--samples-{sizes}",
            dest=f"samples_{start}_{end}",
            type=positive_int,
            default=default,
            help=f"Samples per size for {sizes} qubits (default: {default})",
        )
    add("--min-qubits", type=int, help=f"Inclusive range start (default: {low})")
    add("--max-qubits", type=int, help=f"Inclusive range end (default: {high})")
    add("--seed", type=int, default=0)
    add("--workers", type=positive_int, default=1, help="Concurrent samples")
    add(
        "--threads",
        type=positive_int,
        default=4,
        help="Threads per worker for Rust linear algebra (RAYON_NUM_THREADS) "
        "and BLAS/OpenMP (default: %(default)s)",
    )
    add(
        "--skip-validation",
        action="store_true",
        help="Skip the independent NumPy check of the returned D, P and the spectrum of DPU",
    )
    add(
        "--quiet",
        "--table-only",
        action="store_true",
        help="Print only the summary table",
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    defaults = {
        q: getattr(args, f"samples_{start}_{end}")
        for start, end, _ in sample_groups
        for q in range(start, end + 1)
    }
    args.samples_by_qubits = qubit_sample_schedule(
        parser,
        args,
        minimum=low if minimum is None else minimum,
        default_min=low,
        default_max=high,
        maximum=high,
        default_counts=defaults,
    )
    if args.qubits is None:
        args.min_qubits, args.max_qubits = min(args.samples_by_qubits), max(
            args.samples_by_qubits
        )
    if args.samples is not None and len(args.samples) == 1:
        args.samples = args.samples[0]
    return args


def thread_limits(threads: int) -> dict[str, str]:
    return dict.fromkeys(
        (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "RAYON_NUM_THREADS",
        ),
        str(threads),
    )


def result_table(*headers: str) -> Table:
    table = Table(box=box.ROUNDED, header_style="bold")
    for header in headers:
        table.add_column(header, justify="right", no_wrap=True)
    return table


@contextmanager
def workers(args: argparse.Namespace, table: Table) -> Iterator[tuple[Pool, Live]]:
    # Spawn even a single worker so numerical libraries inherit the thread limits.
    os.environ.update(thread_limits(args.threads))
    with (
        Live(table, auto_refresh=False) as live,
        get_context("spawn").Pool(args.workers) as pool,
    ):
        yield pool, live


@dataclass
class SolverSummary:
    samples: int = 0
    total_seconds: float = 0.0
    nontrivial: int = 0
    nontrivial_seconds: float = 0.0

    def add(self, seconds: float, nontrivial: bool) -> None:
        self.samples += 1
        self.total_seconds += seconds
        if nontrivial:
            self.nontrivial += 1
            self.nontrivial_seconds += seconds

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.samples

    @property
    def nontrivial_percent(self) -> float:
        return 100 * self.nontrivial / self.samples

    @property
    def nontrivial_mean_seconds(self) -> float | None:
        return self.nontrivial_seconds / self.nontrivial if self.nontrivial else None


def sample_summaries[T](
    pool: Pool,
    solve: Callable[[tuple[int, int, T]], tuple[float, bool]],
    qubits: int,
    count: int,
    settings: T,
    *,
    quiet: bool,
) -> Iterator[SolverSummary]:
    specs = ((qubits, i, settings) for i in range(count))
    summary = SolverSummary()
    last_report = monotonic()
    for seconds, nontrivial in pool.imap_unordered(solve, specs):
        summary.add(seconds, nontrivial)
        if summary.samples == count or monotonic() - last_report >= 30:
            if not quiet:
                print(
                    f"{qubits}q: {summary.samples}/{count}, P != I: {summary.nontrivial}",
                    file=sys.stderr,
                    flush=True,
                )
            yield summary
            last_report = monotonic()
