import argparse
import sys
from collections.abc import Sequence
from time import perf_counter

import cirq
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from rich.console import Console

from qmiep import _synthesis_benchmark as bench
from qmiep.cnot import count_cirq_cnots, count_qiskit_cnots
from qmiep.rust import ALGORITHMS
from qmiep.state_preparation import state_preparation

DEFAULT_ALGORITHMS = ("heuristic", "analytic")
QISKIT = "Qiskit StatePrep"
VERIFICATION_ATOL = 1e-9
TIMING = (
    "QMIEP times cover recursive circuit construction and Rust unitary synthesis. "
    "Qiskit times cover StatePreparation and translation to U/CX at optimization level 0. "
    "Sampling, CNOT counting and independent state-vector simulations are excluded. "
    "Means use successful samples."
)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    bench.add_algorithm_argument(parser, default=None)
    parser.add_argument(
        "--mode",
        default="exact",
        metavar="exact",
        choices=(
            "exact",
            "all",
            "ours-heuristic",
            "ours-analytic",
            "heuristic",
            "analytic",
            "both",
        ),
        help="Output mode for the prepared state (exact, including global phase)",
    )
    bench.add_sampling_arguments(parser, default_min=2, default_max=9)
    bench.add_qiskit_arguments(parser, QISKIT)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the comparison of the simulated and target states up to global phase",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Print configuration and table only"
    )
    args = parser.parse_args(argv)
    if args.mode not in ("exact", "all"):
        if args.algorithm is not None:
            parser.error("use --mode exact with --algorithm")
        args.algorithm = (
            DEFAULT_ALGORITHMS
            if args.mode == "both"
            else (args.mode.removeprefix("ours-"),)
        )
    args.mode = "exact"
    args.algorithm = bench.expand_choices(
        parser, "algorithm", args.algorithm or DEFAULT_ALGORITHMS, ALGORITHMS
    )
    counts, args.samples_by_algorithm = bench.sampling_schedule(
        parser,
        args,
        algorithms=args.algorithm,
        minimum=1,
        default_min=2,
        default_max=9,
        seed_bits=64,
    )
    if args.qiskit == "only":
        args.algorithm = (QISKIT,)
    elif args.qiskit:
        args.algorithm = (*args.algorithm, QISKIT)
    return args, counts


def sample_state(n, seed):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return state / np.linalg.norm(state)


def prepare(state, algorithm, seed):
    if algorithm == QISKIT:
        return transpile(
            StatePreparation(state).definition,
            basis_gates=["u", "cx"],
            optimization_level=0,
            seed_transpiler=seed,
        )
    return state_preparation(
        state,
        algorithm=algorithm,
        seed=seed,
        backends=bench.RUST_BACKENDS,
        mode="exact",
        validate=False,
    ).circuit


def measure(state, algorithm, seed, validate):
    started = perf_counter()
    circuit = prepare(state, algorithm, seed)
    seconds = perf_counter() - started
    count = (
        count_qiskit_cnots(circuit)
        if algorithm == QISKIT
        else count_cirq_cnots(circuit.all_operations())
    )
    if validate:
        # Simulate C|0…0⟩ independently and compare every amplitude after global-phase alignment.
        actual = (
            Statevector.from_instruction(circuit).data
            if algorithm == QISKIT
            else cirq.final_state_vector(
                circuit,
                qubit_order=cirq.LineQubit.range(len(state).bit_length() - 1),
                dtype=np.complex128,
            )
        )
        error = bench.max_error_up_to_global_phase(actual, state)
        if not np.isfinite(error) or error > VERIFICATION_ATOL:
            raise RuntimeError(f"state reconstruction failed: {error}")
    return seconds, count


def main(argv: Sequence[str] | None = None) -> int:
    args, counts = parse_args(argv)
    warmup = np.zeros(16, dtype=complex)
    warmup[0] = 1
    for algorithm in args.algorithm:
        prepare(warmup, algorithm, args.seed)

    console = Console(highlight=False)
    validate = not args.skip_validation
    console.print(
        f"algorithm={','.join(args.algorithm)}, mode={args.mode}, seed={args.seed}, "
        f"{bench.sampling_description(counts, args.samples_by_algorithm)}, "
        f"state-vector validation={'on' if validate else 'off'}",
        markup=False,
    )
    if not args.quiet:
        print(TIMING, file=sys.stderr, flush=True)
    failed = False
    cases = [(algorithm, args.mode) for algorithm in args.algorithm]
    for n, seed, (algorithm, _), state, summary in bench.benchmark_trials(
        cases,
        counts,
        args.samples_by_algorithm,
        seed=args.seed,
        sample=sample_state,
        console=console,
        redirect_stderr=sys.stderr.isatty(),
    ):
        label = f"{algorithm}, n={n}, seed={seed}"
        if not args.quiet:
            print(f"Starting {label}", file=sys.stderr, flush=True)
        try:
            seconds, count = measure(state, algorithm, seed, validate)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failed = True
            summary.failed += 1
            print(f"{label}: {exc}", file=sys.stderr, flush=True)
        else:
            summary.add(seconds, count)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
