from collections.abc import Sequence
from time import perf_counter

import cirq
import cirq.testing
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import Isometry as QiskitIsometry
from qiskit.quantum_info import Operator

from qmiep import Isometry, IsometrySynthesize
from qmiep import _synthesis_benchmark as bench
from qmiep._synthesis_benchmark import Arguments, Case
from qmiep.cnot import count_cirq_cnots, count_decomposed_cnots, count_qiskit_cnots
from qmiep.synthesizer.isometry_synthesis import ISOMETRY_MODES

QISKIT = "Qiskit Isometry"
TIMING = (
    "QMIEP times include isometry decomposition, Rust gate synthesis and circuit assembly. "
    "Qiskit times include isometry synthesis and translation to U/CX at optimization level 0. "
    "Sampling, warm-up, CNOT counting and independent verification are excluded. "
    "Equivalent-mode counts omit the input corrections."
)


def parse_args(argv: Sequence[str] | None = None) -> Arguments:
    return bench.parse_benchmark_args(
        argv,
        description=__doc__,
        modes=ISOMETRY_MODES,
        qiskit_name=QISKIT,
        minimum=2,
        default_max=6,
    )


def sample_isometry(n: int, seed: int) -> np.ndarray:
    return cirq.testing.random_unitary(2**n, random_state=seed)[:, : 2 ** (n - 1)]


def qiskit_isometry(matrix: np.ndarray, seed: int):
    return transpile(
        QiskitIsometry(matrix, 0, 0).definition,
        basis_gates=["u", "cx"],
        optimization_level=0,
        seed_transpiler=seed,
        qubits_initially_zero=False,
    )


def measure(
    name: Case, matrix: np.ndarray, seed: int, validate: bool
) -> tuple[float, int]:
    columns = matrix.shape[1]
    if name == QISKIT:
        start = perf_counter()
        circuit = qiskit_isometry(matrix, seed)
        seconds = perf_counter() - start
        count = count_qiskit_cnots(circuit)
        if not validate:
            return seconds, count
        actual = Operator(circuit).data[:, :columns]
    else:
        algorithm, mode = name
        qubits = cirq.LineQubit.range(len(matrix).bit_length() - 1)
        target = Isometry(matrix).on(*qubits)
        source = cirq.Circuit(target)
        synthesizer = IsometrySynthesize(algorithm=algorithm, mode=mode, seed=seed)
        start = perf_counter()
        result = synthesizer.run(
            source, target=target, backends=bench.RUST_BACKENDS, validate=False
        )
        seconds = perf_counter() - start
        *corrections, implementation = result.all_operations()
        if not validate:
            return seconds, count_decomposed_cnots(implementation)
        gates = cirq.decompose(
            implementation, keep=lambda op: len(op.qubits) <= 1 or op.gate == cirq.CX
        )
        count = count_cirq_cnots(gates)
        actual = cirq.Circuit(gates).unitary(qubit_order=qubits)[:, :columns]
        if mode == "PD-equiv":
            # J = J_PD D P: scale the columns by D, then select column p(j) for input j.
            permutation, diagonal = corrections
            phases = np.diag(cirq.unitary(diagonal))
            indices = np.argmax(cirq.unitary(permutation), axis=0)
            actual = (actual * phases)[:, indices]
    # Column j is the output for |0⟩⊗|j⟩. Use one global phase for all columns
    # together so that an incorrect relative phase between columns fails validation.
    if not cirq.allclose_up_to_global_phase(actual, matrix, rtol=0, atol=1e-10):
        error = bench.max_error_up_to_global_phase(actual, matrix)
        raise RuntimeError(f"{name} isometry reconstruction failed: {error:.3e}")
    return seconds, count


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    warmup = sample_isometry(3, args.seed)
    for name in args.cases:
        measure(name, warmup, args.seed, False)
    if args.include_qiskit:
        qiskit_isometry(warmup, args.seed)
    bench.run_benchmark(
        args,
        sample=sample_isometry,
        measure=measure,
        qiskit_name=QISKIT,
        validation="all-column",
        timing=TIMING,
    )


if __name__ == "__main__":
    main()
