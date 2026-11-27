# Artifact for the Paper "Quantum Circuit Synthesis for Unitaries via the Quantum Multiplicative Inverse Eigenvalue Problem"

> Caution: The final version of the artifact will be replaced by the Zendo repo upon publication.

This artifact contains the software implementation of the synthesis algorithms and a reproduction
of the numerical experiments reported in the paper.

- [Installation](docs/installation.md)
- [Reproduction of Numerical Experiments](docs/reproduction.md)

## Code Structure

| Path | Contents |
| --- | --- |
| [`src/qmiep/rust/`](src/qmiep/rust/) | Python interface to native spectrum solving and unitary synthesis. |
| [`rust/qmiep-solver/`](rust/qmiep-solver/) | Analytic and heuristic QMIEP solvers and the `qmiep-spectrum` program. |
| [`rust/qmiep-synthesis/`](rust/qmiep-synthesis/) | Native circuit synthesis and the `qmiep-synthesis` program. |
| [`rust/qmiep-python/`](rust/qmiep-python/) | PyO3 bindings connecting NumPy inputs to the Rust implementation. |
| [`src/qmiep/decomposition/`](src/qmiep/decomposition/), [`rewriting/`](src/qmiep/rewriting/), [`gates/`](src/qmiep/gates/) | Python decompositions, circuit rewrite rules, and gate representations. |
| [`src/qmiep/synthesizer/`](src/qmiep/synthesizer/) | Unitary, isometry, and block-encoding synthesis interfaces. |
| [`src/qmiep/state_preparation.py`](src/qmiep/state_preparation.py) | Recursive quantum state preparation. |
| [`scripts/`](scripts/) | Numerical experiment entry points. |


## Contact

Chan-Gu Kang (Research Associate, Korea University): changukang@korea.ac.kr
