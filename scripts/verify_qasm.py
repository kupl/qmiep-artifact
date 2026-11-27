"""Verify sampled synthesis results and count one- and two-qubit gates in the reloaded QASM.

Qubit counts include the ancilla for block encoding and isometry. Equivalent
modes export the residual circuit and verify it with the synthesis corrections.
Each target comparison allows one global phase for the entire matrix or vector.
The default is one three-qubit sample per target, using analytic/exact.
Use --save-qasm to keep the verified files in ./qasm_files.

python scripts/verify_qasm.py --target all --algorithm all --qubits 2 3 --samples 2 1
python scripts/verify_qasm.py --target unitary isometry --mode PD-equiv --qubits 4
"""

import argparse
from itertools import product
from pathlib import Path
from tempfile import TemporaryFile

import cirq
import cirq.testing
import numpy as np
from block_encoding import sample_matrix
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import circuit_from_qasm
from isometry import sample_isometry
from state_preparation import prepare, sample_state

from qmiep import BlockEncodingSynthesize, Isometry, IsometrySynthesize
from qmiep import _synthesis_benchmark as bench
from qmiep._matrix import _remove_left_correction
from qmiep.cnot import count_cirq_cnots
from qmiep.rust import ALGORITHMS, OUTPUT_MODES, synthesize_unitary
from qmiep.synthesizer.isometry_synthesis import ISOMETRY_MODES

ATOL = 1e-10


def qasm_gate_counts(imported):
    """Count parsed gates before decomposition and require every two-qubit gate to be a CNOT."""
    counts = {1: 0, 2: 0}
    for op in imported.all_operations():
        size = len(op.qubits)
        if size not in counts or not cirq.has_unitary(op):
            raise ValueError(
                f"Expected a one- or two-qubit unitary gate, found {op} ({size} qubits)"
            )
        if size == 2 and op.gate != cirq.CX:
            raise ValueError(f"Expected CNOT as the two-qubit gate, found {op}")
        counts[size] += 1
    return counts


def round_trip(circuit, qubits, path):
    """Write the synthesized gates, then parse the file into a Cirq circuit in logical qubit order."""
    gates = cirq.decompose(
        circuit,
        keep=lambda op: op.gate is not None
        and (len(op.qubits) <= 1 or op.gate == cirq.CX),
    )
    operations = []
    phase = 1.0
    for op in gates:
        if op.gate == cirq.CX:
            operations.append(op)
        elif not op.qubits:
            phase *= cirq.unitary(op).item()
        else:
            # U = phase * U3. The trace overlap of these 2-by-2 unitaries gives that phase.
            matrix = cirq.unitary(op)
            u3 = QasmUGate.from_matrix(matrix)
            phase *= np.vdot(cirq.unitary(u3), matrix) / 2
            operations.append(u3.on(*op.qubits))

    # QASM 2 has no global-phase instruction. P = diag(1, phase) gives X P X P = phase * I.
    shift = QasmUGate(0, 0, np.angle(phase) / np.pi).on(qubits[0])
    operations.extend([shift, cirq.X(qubits[0])] * 2)
    qasm = cirq.QasmOutput(operations, tuple(qubits), precision=17)
    with (
        path.open("w+", encoding="utf-8")
        if path
        else TemporaryFile("w+", encoding="utf-8")
    ) as file:
        file.write(str(qasm))
        file.seek(0)
        imported = circuit_from_qasm(file.read())
    counts = qasm_gate_counts(imported)
    expected_cnots = count_cirq_cnots(gates)
    if counts[2] != expected_cnots:
        raise RuntimeError(
            f"CNOT count differs: synthesis={expected_cnots}, QASM={counts[2]}"
        )
    print(f"1q={counts[1]}  2q={counts[2]} (all CNOT)  ", end="")
    return imported.transform_qubits(
        {cirq.NamedQubit(f"q_{i}"): q for i, q in enumerate(qubits)}
    )


def verify_equivalence(actual, expected):
    """Use Cirq's global-phase predicate for PASS and report the phase-aligned 2-norm."""
    if not cirq.equal_up_to_global_phase(actual, expected, atol=ATOL):
        raise RuntimeError("Reloaded QASM does not match the target up to global phase")
    actual, expected = cirq.match_global_phase(actual, expected)
    return np.linalg.norm(actual - expected, ord=2)


def check_unitary(matrix, qubits, algorithm, mode, seed, path):
    """Compare the full imported W with U, D†U, or P†D†U using the returned corrections."""
    result = synthesize_unitary(
        matrix, qubits=qubits, algorithm=algorithm, mode=mode, seed=seed
    )
    if mode == "exact":
        circuit, expected = result, matrix
    else:
        circuit, *corrections, diagonal = result
        permutation = corrections[0] if corrections else np.arange(len(diagonal))
        expected = _remove_left_correction(matrix, permutation, diagonal)
    actual = round_trip(circuit, qubits, path).unitary(qubit_order=qubits)
    return verify_equivalence(actual, expected)


def check_isometry(matrix, qubits, algorithm, mode, seed, path):
    """Check every ancilla-zero input column. In PD-equiv mode the input correction gives J = W[:, :k] D P."""
    target = Isometry(matrix).on(*qubits)
    result = IsometrySynthesize(algorithm=algorithm, mode=mode, seed=seed).run(
        cirq.Circuit(target),
        target=target,
        backends=bench.RUST_BACKENDS,
        validate=False,
    )
    *corrections, implementation = result.all_operations()
    imported = round_trip(implementation, qubits, path)
    actual = imported.unitary(qubit_order=qubits)[:, : matrix.shape[1]]
    if corrections:
        actual = actual @ cirq.Circuit(corrections).unitary(qubit_order=qubits[1:])
    return verify_equivalence(actual, matrix)


def check_block_encoding(sample, qubits, algorithm, mode, seed, path):
    """Check unitarity, then compare the entire ancilla-zero block with A/alpha up to one phase."""
    matrix, alpha = sample
    circuit = BlockEncodingSynthesize().run(
        matrix,
        alpha=alpha,
        algorithm=algorithm,
        mode=mode,
        seed=seed,
        backends=bench.RUST_BACKENDS,
    )
    actual = round_trip(circuit, qubits, path).unitary(qubit_order=qubits)
    d = len(matrix)
    unitarity_error = np.linalg.norm(actual.conj().T @ actual - np.eye(2 * d), ord=2)
    if not np.isfinite(unitarity_error) or unitarity_error > ATOL:
        raise RuntimeError(f"Unitarity error {unitarity_error:.3e} exceeds {ATOL:g}")
    return verify_equivalence(actual[:d, :d], matrix / alpha)


def check_state(state, qubits, algorithm, _mode, seed, path):
    """Simulate the imported circuit on |0...0> and compare all complex amplitudes with the sample."""
    circuit = round_trip(prepare(state, algorithm, seed), qubits, path)
    actual = cirq.final_state_vector(circuit, qubit_order=qubits, dtype=np.complex128)
    return verify_equivalence(actual, state)


TARGETS = {
    "unitary": (
        lambda n, seed: cirq.testing.random_unitary(2**n, random_state=seed),
        check_unitary,
        OUTPUT_MODES,
    ),
    "isometry": (sample_isometry, check_isometry, ISOMETRY_MODES),
    "block-encoding": (sample_matrix, check_block_encoding, ("exact",)),
    "state-prep": (sample_state, check_state, ("exact",)),
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--target", nargs="+", choices=(*TARGETS, "all"), default=["all"]
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=(*OUTPUT_MODES, "all"),
        default=["exact"],
        help="Output modes. 'all' runs each target's supported modes.",
    )
    parser.add_argument(
        "--save-qasm", action="store_true", help="Keep QASM files in ./qasm_files"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Keep QASM files in this directory instead"
    )
    bench.add_algorithm_argument(parser)
    bench.add_sampling_arguments(parser, default_min=3, default_max=3)
    parser.set_defaults(qiskit=None)
    args = parser.parse_args()
    targets = bench.expand_choices(parser, "target", args.target, tuple(TARGETS))
    algorithms = bench.expand_choices(parser, "algorithm", args.algorithm, ALGORITHMS)
    modes = bench.expand_choices(parser, "mode", args.mode, OUTPUT_MODES)
    counts, overrides = bench.sampling_schedule(
        parser,
        args,
        algorithms=algorithms,
        minimum=2 if "isometry" in targets else 1,
        default_min=3,
        default_max=3,
    )
    cases_by_target = {}
    for target in targets:
        _, _, supported = TARGETS[target]
        selected = supported if args.mode == ["all"] else modes
        if not set(selected).issubset(supported):
            parser.error(f"{target} supports --mode {' '.join(supported)}")
        cases_by_target[target] = tuple(product(algorithms, selected))

    output_dir = args.output_dir or (Path("qasm_files") if args.save_qasm else None)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for target, cases in cases_by_target.items():
        sample, check, _ = TARGETS[target]
        for n, count in counts.items():
            qubits = cirq.LineQubit.range(n)
            for index, active in bench.sample_batches(cases, count, overrides):
                seed = args.seed + index
                spec = sample(n, seed)
                for algorithm, mode in active:
                    name = f"{target}_{algorithm}_{mode}_{n}q_seed{seed}.qasm"
                    path = output_dir / name if output_dir is not None else None
                    print(f"{path or name}: ", end="", flush=True)
                    error = check(spec, qubits, algorithm, mode, seed, path)
                    print(f"PASS  phase-aligned 2-norm error={error:.3e}")


if __name__ == "__main__":
    main()
