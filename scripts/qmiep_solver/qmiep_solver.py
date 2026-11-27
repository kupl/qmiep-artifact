r"""Sample Haar unitaries and benchmark the Rust QMIEP solver.

    python scripts/qmiep_solver/qmiep_solver.py --samples 50 --table-only
    conda run --no-capture-output -n qmiep python scripts/qmiep_solver/qmiep_solver.py \
        --qubits  2       3       4       5       6      7     8     9    10 \
        --samples 2000000 1000000 1000000 1000000 100000 10000 10000 1000 100

The default --threads 4 sets RAYON_NUM_THREADS=4 in each worker. Rust's faer
library uses this pool for parallel linear algebra within a single solve,
including eigendecomposition. The same option sets the OpenBLAS, OpenMP and
Accelerate thread limits. Increase --workers to solve multiple samples at once.

The default range is 3--10 qubits. Select 2 qubits to include the analytic solver.
Default samples per size: 2000000 for 2q, 100000 for 3--5q, 10000 for 6--8q, 1000 for 9--10q.
--samples overrides all group counts. A failed solve aborts the run.
"""

import argparse
import sys
from functools import partial
from time import perf_counter

import numpy as np

from qmiep import _solver_benchmark as bench
from qmiep._solver_benchmark import haar_unitary
from qmiep.rust import _rust

SOLVER_OPTIONS = dict(max_permutations=None, restarts=3, maxiter=500, atol=1e-10)
SAMPLE_GROUPS = ((2, 2, 2_000_000), (3, 5, 100_000), (6, 8, 10_000), (9, 10, 1_000))
TIMING = (
    "Wall time of the native PyO3 solve_spectrum call, including matrix "
    "copy, input validation, search and fresh spectral verification; "
    "excludes sampling, imports, process startup, independent NumPy validation and result logging. "
    "With workers > 1, calls share CPU resources."
)


def solve_sample(spec, *, validate=True):
    qubits, sample, seed = spec
    matrix = haar_unitary(qubits, sample, seed)
    (solver_seed,) = bench.solver_seeds(qubits, sample, seed, 1)
    started = perf_counter()
    native = _rust.solve_spectrum(matrix, seed=solver_seed, **SOLVER_OPTIONS)
    elapsed = perf_counter() - started
    rows, diagonal, success, error, *_ = native
    if not success:
        raise RuntimeError(
            f"QMIEP failed: {qubits}q, sample={sample}, seed={seed}, "
            f"solver_seed={solver_seed}: antipodal_error={error:.6g}"
        )
    nontrivial = bool(np.any(rows != np.arange(len(matrix))))
    if validate:
        # For the 2m sorted eigenphases of DPU, require max_{0≤j<m} |α_{j+m} − α_j − π| < atol.
        bench.validate_qmeip_solution(
            matrix, rows, diagonal, atol=SOLVER_OPTIONS["atol"]
        )
        if qubits == 2:
            # For U(4), P=I is feasible exactly when the diagonal lengths close a polygon.
            lengths = np.abs(matrix.diagonal())
            longest = lengths.max()
            if nontrivial != (longest > lengths.sum() - longest):
                raise RuntimeError(
                    f"Two-qubit permutation disagrees with trace criterion: sample={sample}, seed={seed}"
                )
    return elapsed, nontrivial


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    return bench.parse_benchmark_args(
        parser, argv, 3, 10, minimum=2, sample_groups=SAMPLE_GROUPS
    )


def sample_counts(args):
    return args.samples_by_qubits


def add_summary(table, qubits, summary):
    subset = summary.nontrivial_mean_seconds
    table.add_row(
        str(qubits),
        str(summary.samples),
        f"{summary.mean_seconds:.6f}",
        str(summary.nontrivial),
        f"{summary.nontrivial_percent:.5f}",
        "-" if subset is None else f"{subset:.6f}",
    )


def main(argv=None):
    args = parse_args(argv)
    counts = sample_counts(args)
    solve = partial(solve_sample, validate=not args.skip_validation)
    if not args.quiet:
        print(
            f"Rust pairing, samples/size={counts}, seed={args.seed}, "
            f"workers={args.workers}, threads/worker={args.threads}, "
            f"NumPy validation={'off' if args.skip_validation else 'on'}",
            flush=True,
        )
        print(TIMING, file=sys.stderr, flush=True)
    table = bench.result_table(
        "Qubits",
        "Samples",
        "Mean solve\n(s)",
        "P ≠ I",
        "P ≠ I\n(%)",
        "Mean solve\nP ≠ I (s)",
    )
    with bench.workers(args, table) as (pool, live):
        for qubits, count in counts.items():
            for summary in bench.sample_summaries(
                pool, solve, qubits, count, args.seed, quiet=args.quiet
            ):
                if summary.samples == count:
                    add_summary(table, qubits, summary)
                    live.refresh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
