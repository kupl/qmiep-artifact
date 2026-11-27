from collections.abc import Sequence
from time import perf_counter

import cirq
import cirq.testing
import numpy as np
import qiskit.synthesis
from qiskit.quantum_info import Operator

from qmiep import _synthesis_benchmark as bench
from qmiep._matrix import _remove_left_correction
from qmiep._synthesis_benchmark import Arguments, Case
from qmiep.cnot import count_cirq_cnots, count_qiskit_cnots
from qmiep.rust import (
    OUTPUT_MODES,
    _circuit_from_instructions,
    _count_native_cnots,
    _rust,
)

QISKIT = "Qiskit QSD"
TIMING = (
    "Times cover native PyO3 calls for QMIEP (including input transfer and "
    "instruction export) and QSD calls for Qiskit. Sampling, warm-up, Cirq "
    "emission, CNOT counting and validation are excluded. "
    "Sampled inputs skip the unitarity product check."
)


def parse_args(argv: Sequence[str] | None = None) -> Arguments:
    return bench.parse_benchmark_args(
        argv, description=__doc__, modes=OUTPUT_MODES, qiskit_name=QISKIT
    )


def verify_circuit(circuit, unitary):
    if isinstance(circuit, cirq.Circuit):
        qubits = cirq.LineQubit.range(len(unitary).bit_length() - 1)
        actual = circuit.unitary(qubit_order=qubits)
    else:
        actual = Operator(circuit).data
    # Align one global phase for the whole matrix, then check every entry at every size.
    error = bench.max_error_up_to_global_phase(actual, unitary)
    if not np.isfinite(error) or error > 1e-10:
        label = "QMIEP" if isinstance(circuit, cirq.Circuit) else "Qiskit"
        raise RuntimeError(f"{label} reconstruction failed: {error:.3e}")
    return error


def measure(
    name: Case, matrix: np.ndarray, seed: int, validate: bool
) -> tuple[float, int]:
    if name == QISKIT:
        started = perf_counter()
        circuit = qiskit.synthesis.qs_decomposition(matrix)
        seconds = perf_counter() - started
        count = count_qiskit_cnots(circuit)
        if validate:
            verify_circuit(circuit, matrix)
        return seconds, count

    algorithm, mode = name
    n = len(matrix).bit_length() - 1
    started = perf_counter()
    instructions, phase, p, d = _rust.synthesize(
        matrix, algorithm=algorithm, mode=mode, seed=seed, check_input=False
    )
    seconds = perf_counter() - started
    if not validate:
        return seconds, _count_native_cnots(instructions, n)

    circuit = _circuit_from_instructions(instructions, phase, cirq.LineQubit.range(n))
    count = count_cirq_cnots(circuit.all_operations())
    # U = D P W, so compare the emitted W with P† D† U. D-equiv also requires P = I.
    if mode == "D-equiv" and not np.array_equal(p, np.arange(len(matrix))):
        raise RuntimeError("D-equiv synthesis returned a permutation")
    target = matrix if mode == "exact" else _remove_left_correction(matrix, p, d)
    verify_circuit(circuit, target)
    return seconds, count


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    warmup = np.eye(4, dtype=complex)
    for algorithm, mode in args.cases:
        _rust.synthesize(
            warmup, algorithm=algorithm, mode=mode, seed=args.seed, check_input=False
        )
    if args.include_qiskit:
        qiskit.synthesis.qs_decomposition(warmup)
    bench.run_benchmark(
        args,
        sample=lambda n, seed: cirq.testing.random_unitary(2**n, random_state=seed),
        measure=measure,
        qiskit_name=QISKIT,
        validation="full-matrix",
        timing=TIMING,
    )


if __name__ == "__main__":
    main()
