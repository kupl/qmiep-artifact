# Reproduction of Numerical Experiments

Run the commands from the repository root after [installation](installation.md).

## Python API (PyO3)

This example synthesizes and verifies a five-qubit unitary. Allow about
**3 seconds** on an Apple M4 Pro, including Conda startup and Python imports.

```python
import cirq
import cirq.testing
import numpy as np

from qmiep.cnot import count_cirq_cnots
from qmiep.rust import synthesize_unitary

qubits = cirq.LineQubit.range(5)
unitary = cirq.testing.random_unitary(2 ** len(qubits), random_state=1230231)
circuit = synthesize_unitary(
    unitary, qubits=qubits,
    algorithm="heuristic-opt", # or "analytic" or "heuristic"
    mode="exact",
    seed=10
)
actual = circuit.unitary(qubit_order=qubits)
np.testing.assert_allclose(actual, unitary, rtol=0, atol=1e-10)
print("Maximum reconstruction error:", np.max(np.abs(actual - unitary)))
print("CNOT count:", count_cirq_cnots(circuit.all_operations()))
```

Choose `analytic`, `heuristic`, or `heuristic-opt` for Algorithms 1, 2, or 3,
respectively.

## Reproduction Scripts

For a quicker run, use fewer qubits and fewer samples per qubit count.
For example, in the commands for Tables 6–8, lower `--max-qubits` and change
`--samples-by-algorithm heuristic-opt=100` to `--samples-by-algorithm heuristic-opt=1`.

The estimates give wall-clock time for each command on a Mac mini with a
12-core Apple M4 Pro and 24 GB of RAM. For Tables 6–8, `heuristic-opt` runs
100 times per size, and each other selected method runs once.

To use four Rayon threads for synthesis, run `export RAYON_NUM_THREADS=4` in
the same shell. The QMIEP solver accepts `--threads 4` to set the thread count
per worker.

### Table 5 and the numerical entries in Table 1 — Unitary synthesis

**Estimated run time** — about **5 hours** for all three QMIEP algorithms and
the Qiskit baseline. Synthesis alone accounts for about 4.5 hours.

```sh
conda run -n qmiep python scripts/unitary.py --algorithm all --mode exact --compare-with-qiskit \
  --qubits 4 5 6 7 8 9 10 11 12 --samples 100 100 100 100 100 100 100 10 10 --skip-validation
```

Table 1 uses the 4–8-qubit CNOT counts from Table 5.

### Tables 3 and 4 — QMIEP solver

**Estimated run time** — **2–2.5 hours** with the default single worker, four
threads, and independent validation.

One run reports mean solver time for 3–10 qubits (Table 3) and the count and
percentage of samples returning $P \ne I$ for 2–10 qubits (Table 4).

```sh
conda run -n qmiep python scripts/qmiep_solver/qmiep_solver.py --threads 4 \
  --qubits 2 3 4 5 6 7 8 9 10 --samples 2000000 1000000 1000000 1000000 100000 10000 10000 1000 100
```

### Table 6 — Block encoding

**Estimated run time** — about **2 hours**. A nine-qubit `heuristic-opt` sample
took about 68 seconds including the unitarity and ancilla-zero block checks.

```sh
conda run -n qmiep python scripts/block_encoding.py --algorithm all --compare-with-qiskit \
  --min-qubits 4 --max-qubits 9 --samples-by-algorithm heuristic-opt=100
```

### Table 7 — Isometry synthesis (Ours rows)

**Estimated run time** — about **3 hours**. A nine-qubit `heuristic-opt` sample
took about 102 seconds with all output columns verified.

```sh
conda run -n qmiep python scripts/isometry.py --algorithm all --mode exact \
  --min-qubits 4 --max-qubits 9 --samples-by-algorithm heuristic-opt=100
```

### Table 8 — State preparation (Ours rows)

**Estimated run time** — about **3 minutes**, including state-vector validation.

```sh
conda run -n qmiep python scripts/state_preparation.py --algorithm all \
  --min-qubits 6 --max-qubits 12 --samples-by-algorithm heuristic-opt=100
```
