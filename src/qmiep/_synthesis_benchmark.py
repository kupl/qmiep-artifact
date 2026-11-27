"""Shared CLI, sampling schedule and reporting for synthesis benchmarks."""

import argparse
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import product

import cirq
import numpy as np
from numpy.typing import NDArray
from rich import box
from rich.console import Console
from rich.live import Live
from rich.table import Table

from qmiep.rust import ALGORITHMS
from qmiep.synthesizer import SynthesisBackends

RUST_BACKENDS = SynthesisBackends(unitary="rust", diagonal="rust", pd="rust")

type Case = str | tuple[str, str]


def max_error_up_to_global_phase(
    actual: NDArray[np.complex128], expected: NDArray[np.complex128]
) -> float:
    """Return the max entrywise error using Cirq's global-phase alignment."""
    if actual.shape != expected.shape:
        return float("inf")
    actual, expected = cirq.match_global_phase(actual, expected)
    return float(np.max(np.abs(actual - expected)))


@dataclass(frozen=True)
class Arguments:
    algorithms: tuple[str, ...]
    mode: tuple[str, ...]
    modes: tuple[str, ...]
    samples_by_qubits: dict[int, int]
    seed: int
    include_qiskit: bool
    skip_validation: bool
    quiet: bool
    samples_by_algorithm: dict[str, int] = field(default_factory=dict)

    @property
    def cases(self) -> tuple[tuple[str, str], ...]:
        return tuple(product(self.algorithms, self.modes))


@dataclass
class Summary:
    samples: int = 0
    total_seconds: float = 0.0
    total_cnot: int = 0
    failed: int = 0

    def add(self, seconds: float, cnot: int) -> None:
        self.samples += 1
        self.total_seconds += seconds
        self.total_cnot += cnot


def add_algorithm_argument(
    parser: argparse.ArgumentParser, *, default: Sequence[str] | None = ("analytic",)
) -> None:
    parser.add_argument(
        "--algorithm",
        choices=(*ALGORITHMS, "all"),
        nargs="+",
        default=default,
        help="Algorithms to benchmark, in execution order"
        + (f" (default: {' '.join(default)})" if default else ""),
    )


def expand_choices(
    parser: argparse.ArgumentParser,
    flag: str,
    selected: Sequence[str],
    choices: Sequence[str],
) -> tuple[str, ...]:
    if len(set(selected)) != len(selected):
        parser.error(f"--{flag} must contain distinct choices")
    if "all" in selected:
        if len(selected) != 1:
            parser.error(f"--{flag} all cannot be combined with individual choices")
        return tuple(choices)
    return tuple(selected)


def add_qiskit_arguments(parser: argparse.ArgumentParser, name: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--compare-with-qiskit",
        dest="qiskit",
        action="store_const",
        const="compare",
        help=f"Also run {name} on the same samples",
    )
    group.add_argument(
        "--qiskit-only",
        dest="qiskit",
        action="store_const",
        const="only",
        help=f"Run only {name}, once per sample",
    )


def _algorithm_sample_count(value: str) -> tuple[str, int]:
    algorithm, _, raw_count = value.partition("=")
    try:
        count = int(raw_count)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected ALGORITHM=COUNT, got {value!r}"
        ) from None
    if algorithm not in ALGORITHMS or count < 1:
        raise argparse.ArgumentTypeError(
            f"expected a synthesis algorithm and a positive count, got {value!r}"
        )
    return algorithm, count


def add_sampling_arguments(
    parser: argparse.ArgumentParser, *, default_min: int, default_max: int
) -> None:
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        help="Explicit qubit counts, in execution order",
    )
    parser.add_argument(
        "--min-qubits", type=int, help=f"Inclusive range start (default: {default_min})"
    )
    parser.add_argument(
        "--max-qubits", type=int, help=f"Inclusive range end (default: {default_max})"
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[1],
        help="Default sample count for all sizes, or one count per selected size in order (default: 1)",
    )
    parser.add_argument(
        "--samples-by-algorithm",
        metavar="ALGORITHM=COUNT",
        type=_algorithm_sample_count,
        action="extend",
        nargs="+",
        default=[],
        help="Override --samples for a selected algorithm at every size and mode, e.g. heuristic-opt=100. "
        "Algorithms share the first samples of the same seed sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling and heuristic search seed (default: 0)",
    )


def qubit_sample_schedule(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    minimum: int,
    default_min: int,
    default_max: int,
    maximum: int | None = None,
    default_counts: dict[int, int] | None = None,
) -> dict[int, int]:
    qubits = args.qubits
    if qubits is not None:
        if args.min_qubits is not None or args.max_qubits is not None:
            parser.error(
                "--qubits cannot be combined with --min-qubits or --max-qubits"
            )
        if min(qubits) < minimum or len(set(qubits)) != len(qubits):
            parser.error(f"--qubits must contain distinct integers >= {minimum}")
        largest = max(qubits)
    else:
        start = default_min if args.min_qubits is None else args.min_qubits
        end = default_max if args.max_qubits is None else args.max_qubits
        if not minimum <= start <= end:
            parser.error(f"require {minimum} <= --min-qubits <= --max-qubits")
        qubits = range(start, end + 1)
        largest = end
    if maximum is not None and largest > maximum:
        parser.error(f"qubit counts must be <= {maximum}")
    samples = args.samples
    if samples is None and default_counts is not None:
        return {q: default_counts[q] for q in qubits}
    if min(samples) < 1:
        parser.error("--samples must contain positive integers")
    if len(samples) == 1:
        samples = samples * len(qubits)
    if len(samples) != len(qubits):
        parser.error("--samples needs one count or one count per selected qubit size")
    return dict(zip(qubits, samples, strict=True))


def sampling_schedule(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    algorithms: Sequence[str],
    minimum: int,
    default_min: int,
    default_max: int,
    seed_bits: int = 32,
) -> tuple[dict[int, int], dict[str, int]]:
    by_qubits = qubit_sample_schedule(
        parser, args, minimum=minimum, default_min=default_min, default_max=default_max
    )
    overrides = dict(args.samples_by_algorithm)
    if len(overrides) != len(args.samples_by_algorithm):
        parser.error("--samples-by-algorithm accepts one count per algorithm")
    for algorithm in overrides:
        if algorithm not in algorithms:
            parser.error(
                f"--samples-by-algorithm requires {algorithm!r} to be selected with --algorithm"
            )
    if args.qiskit == "only":
        overrides.clear()
    default_max_samples = max(by_qubits.values())
    max_samples = max(
        overrides.get(algorithm, default_max_samples) for algorithm in algorithms
    )
    if args.qiskit:
        max_samples = max(max_samples, default_max_samples)
    if not 0 <= args.seed <= 2**seed_bits - max_samples:
        parser.error(
            f"all sample seeds (seed + sample index) must be within [0, 2**{seed_bits})"
        )
    return by_qubits, overrides


def sample_batches[T: Case](
    cases: Sequence[T], default_samples: int, overrides: dict[str, int]
) -> Iterator[tuple[int, tuple[T, ...]]]:
    """Yield each shared input index and the cases scheduled for that input."""
    counts = {
        case: overrides.get(
            case[0] if isinstance(case, tuple) else case, default_samples
        )
        for case in cases
    }
    for i in range(max(counts.values())):
        yield i, tuple(case for case, count in counts.items() if i < count)


def sampling_description(by_qubits: dict[int, int], overrides: dict[str, int]) -> str:
    description = f"samples per qubit count={by_qubits}"
    if overrides:
        return f"default {description}, samples by algorithm (all sizes)={overrides}"
    return description


def parse_benchmark_args(
    argv: Sequence[str] | None,
    *,
    description: str | None,
    modes: Sequence[str],
    qiskit_name: str,
    minimum: int = 1,
    default_max: int = 9,
) -> Arguments:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        choices=(*modes, "all"),
        nargs="+",
        default=[modes[0]],
        help=f"One or more modes in execution order, or all (default: {modes[0]})",
    )
    add_algorithm_argument(parser)
    add_sampling_arguments(parser, default_min=3, default_max=default_max)
    add_qiskit_arguments(parser, qiskit_name)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip independent output verification for QMIEP and Qiskit",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print configuration and results table only",
    )
    args = parser.parse_args(argv)

    algorithms = expand_choices(parser, "algorithm", args.algorithm, ALGORITHMS)
    selected_modes = expand_choices(parser, "mode", args.mode, modes)
    by_qubits, by_algorithm = sampling_schedule(
        parser,
        args,
        algorithms=algorithms,
        minimum=minimum,
        default_min=3,
        default_max=default_max,
    )
    if args.qiskit == "only":
        algorithms = ()
        args.mode = selected_modes = ("exact",)
    return Arguments(
        mode=tuple(args.mode),
        modes=selected_modes,
        algorithms=algorithms,
        samples_by_qubits=by_qubits,
        samples_by_algorithm=by_algorithm,
        seed=args.seed,
        include_qiskit=bool(args.qiskit),
        skip_validation=args.skip_validation,
        quiet=args.quiet,
    )


def make_table(summaries: dict[int, dict[Case, Summary]]) -> Table:
    table = Table(box=box.ROUNDED, header_style="bold")
    for header in (
        "Qubits",
        "Algorithm",
        "Mode",
        "Samples",
        "Mean time (s)",
        "Mean CNOT",
    ):
        table.add_column(
            header,
            justify="left" if header in ("Algorithm", "Mode") else "right",
            no_wrap=True,
        )
    for num_qubits, group in summaries.items():
        for name, summary in group.items():
            if not (summary.samples or summary.failed):
                continue
            labels = name if isinstance(name, tuple) else (name, "exact")
            mean_time = mean_cnot = "-"
            if summary.samples:
                mean_time = f"{summary.total_seconds / summary.samples:.6f}"
                mean_cnot = (
                    str(summary.total_cnot // summary.samples)
                    if summary.total_cnot % summary.samples == 0
                    else f"{summary.total_cnot / summary.samples:.2f}"
                )
            samples = str(summary.samples)
            if summary.failed:
                samples = f"{summary.samples}/{summary.samples + summary.failed}"
                table.caption = (
                    "For groups with failures, Samples shows successful/attempted runs."
                )
            table.add_row(str(num_qubits), *labels, samples, mean_time, mean_cnot)
        table.add_section()
    return table


def benchmark_trials[Name: Case, Sample](
    cases: Sequence[Name],
    samples_by_qubits: dict[int, int],
    overrides: dict[str, int],
    *,
    seed: int,
    sample: Callable[[int, int], Sample],
    console: Console,
    redirect_stderr: bool = True,
) -> Iterator[tuple[int, int, Name, Sample, Summary]]:
    """Share one input across its scheduled cases and display each updated summary."""
    summaries: dict[int, dict[Case, Summary]] = {}
    with Live(
        make_table(summaries),
        console=console,
        auto_refresh=False,
        redirect_stderr=redirect_stderr,
    ) as live:
        for n, count in samples_by_qubits.items():
            group = summaries[n] = {case: Summary() for case in cases}
            for index, active in sample_batches(cases, count, overrides):
                sample_seed = seed + index
                problem = sample(n, sample_seed)
                for case in active:
                    yield n, sample_seed, case, problem, group[case]
                    live.update(make_table(summaries), refresh=True)


def run_benchmark[Sample](
    args: Arguments,
    *,
    sample: Callable[[int, int], Sample],
    measure: Callable[[Case, Sample, int, bool], tuple[float, int]],
    qiskit_name: str,
    timing: str,
    validation: str,
) -> None:
    console = Console(highlight=False)
    algorithms = (
        (*args.algorithms, qiskit_name) if args.include_qiskit else args.algorithms
    )
    console.print(
        f"algorithm={','.join(algorithms)}, mode={','.join(args.mode)}, seed={args.seed}, "
        f"{sampling_description(args.samples_by_qubits, args.samples_by_algorithm)}, "
        f"{validation} validation={'off' if args.skip_validation else 'on'}",
        markup=False,
    )
    if not args.quiet:
        console.print(timing, style="dim")
    names: tuple[Case, ...] = (
        (*args.cases, qiskit_name) if args.include_qiskit else args.cases
    )
    for n, seed, name, problem, summary in benchmark_trials(
        names,
        args.samples_by_qubits,
        args.samples_by_algorithm,
        seed=args.seed,
        sample=sample,
        console=console,
    ):
        if not args.quiet:
            label = " / ".join(name) if isinstance(name, tuple) else name
            console.print(f"Starting {label}, n={n}, seed={seed}", style="dim")
        summary.add(*measure(name, problem, seed, not args.skip_validation))
