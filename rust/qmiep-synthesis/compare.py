"""One Haar sample per size; benchmark the standalone CLI against Qiskit/Python.

All sampling, JSON I/O, process startup, verification and CNOT counting are
outside synthesis timing. Rust timings include first-use CX basis setup.
"""

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter

import cirq
import numpy as np
from qiskit.synthesis import qs_decomposition

from qmiep.cnot import count_cirq_cnots, count_qiskit_cnots


def check_cirq(record, unitary):
    """Independently interpret Rust gates and the left DP correction with Cirq."""
    result = record["synthesis"]
    qubits = cirq.LineQubit.range(record["num_qubits"])
    operations = []
    for op in result["circuit"]["instructions"]:
        gate = op["gate"]
        if gate == "cx":
            operations.append(cirq.CX(qubits[op["control"]], qubits[op["target"]]))
        elif gate == "h":
            operations.append(cirq.H(qubits[op["qubit"]]))
        elif gate == "rz":
            operations.append(cirq.rz(op["angle"])(qubits[op["qubit"]]))
        elif gate == "u":
            theta, phi, lam = op["theta"], op["phi"], op["lambda"]
            cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
            matrix = np.array(
                [
                    [cosine, -np.exp(1j * lam) * sine],
                    [np.exp(1j * phi) * sine, np.exp(1j * (phi + lam)) * cosine],
                ]
            )
            operations.append(cirq.MatrixGate(matrix)(qubits[op["qubit"]]))
        else:
            raise ValueError(f"unknown Rust instruction: {gate}")
    operations.append(
        cirq.global_phase_operation(np.exp(1j * result["circuit"]["global_phase"]))
    )
    w = cirq.Circuit(operations).unitary(qubit_order=qubits)
    reconstructed = w
    if (multiplier := result["correction"]) is not None:
        p = np.asarray(multiplier["permutation"])
        diagonal = np.asarray(multiplier["diagonal"])
        d = diagonal[:, 0] + 1j * diagonal[:, 1]
        reconstructed = d[:, None] * w[np.argsort(p)]
    error = float(np.max(np.abs(reconstructed - unitary)))
    if error > 1e-10:
        raise RuntimeError(f"independent Cirq reconstruction failed: {error}")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--min-qubits", type=int, choices=range(3, 10), default=3)
    parser.add_argument("--max-qubits", type=int, choices=range(3, 10), default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--method", choices=("analytic", "heuristic"), default="analytic"
    )
    parser.add_argument("--compare-python", action="store_true")
    args = parser.parse_args()
    if args.min_qubits > args.max_qubits or not 0 <= args.seed < 2**32:
        parser.error("invalid qubit range or seed")
    synthesizers = {}
    if args.compare_python:
        from qmiep.gates import UnresolvedNQubitUnitary
        from qmiep.synthesizer import NQubitUnitarySynthesis

        for mode in ("analytic", "heuristic"):
            synthesizers[mode] = NQubitUnitarySynthesis(mode=mode, backend="numerical")
            target = UnresolvedNQubitUnitary(np.eye(4)).on(*cirq.LineQubit.range(2))
            synthesizers[mode].run(cirq.Circuit(target), target=target, validate=False)
    qs_decomposition(np.eye(4))
    print("n | Rust s | Qiskit s | Rust CX | Qiskit CX | max residual", flush=True)
    for n in range(args.min_qubits, args.max_qubits + 1):
        u = cirq.testing.random_unitary(2**n, random_state=args.seed)
        payload = json.dumps({"matrix": np.stack((u.real, u.imag), axis=-1).tolist()})
        command = [
            str(args.rust_binary.resolve()),
            "--verify",
            "--skip-input-check",
            "--method",
            args.method,
            "--seed",
            str(args.seed),
        ]
        if n > 5:
            command.append("--summary")
        run = subprocess.run(
            command, input=payload, capture_output=True, text=True, check=True
        )
        native = json.loads(run.stdout)
        if n <= 5:
            check_cirq(native, u)
        started = perf_counter()
        reference = qs_decomposition(u)
        qiskit_seconds = perf_counter() - started
        qiskit_count = count_qiskit_cnots(reference)
        if synthesizers:
            target = UnresolvedNQubitUnitary(u).on(*cirq.LineQubit.range(n))
            circuit = cirq.Circuit(target)
            for mode, synthesizer in synthesizers.items():
                started = perf_counter()
                result = synthesizer.run(circuit, target=target, validate=False)
                seconds = perf_counter() - started
                # The numerical synthesizer wraps its emitted circuit in NQubitUnitary.
                (wrapped,) = result.all_operations()
                count = count_cirq_cnots(wrapped.gate.decomposition.all_operations())
                print(f"n={n} Python {mode}: {seconds:.6f} s, {count} CX", flush=True)
                if mode == args.method and count != native["cnot_count"]:
                    raise RuntimeError(f"{args.method} CNOT count differs at n={n}")
        print(
            f"{n} | {native['synthesis_seconds']:.6f} | {qiskit_seconds:.6f} | "
            f"{native['cnot_count']} | {qiskit_count} | "
            f"{native['verification_max_error']:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
