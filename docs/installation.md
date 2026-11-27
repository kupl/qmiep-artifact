# Installation

Run the commands below from a terminal on macOS or Linux. We assume that
Git, Make, and internet access are available.

## Install Python and Rust

1. Install [Anaconda Distribution](https://www.anaconda.com/download) for your
   operating system and CPU architecture, following the official
   [installation guide](https://www.anaconda.com/docs/getting-started/installation).
   Miniconda also works: the build only requires the `conda` command. Reopen
   your terminal after installation and check that Conda is available:

   ```sh
   conda --version
   ```

2. Install the latest stable Rust toolchain, including Cargo, using
   [rustup](https://rust-lang.org/tools/install/):

   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   . "$HOME/.cargo/env"
   ```

   If Rust is already installed, update it with `rustup update stable`.
   (NOTE: This project uses the [Rust 2024 edition](https://doc.rust-lang.org/edition-guide/rust-2024/index.html);
   older toolchains such as Rust 1.80 cannot build it. Use the current stable
   toolchain to satisfy the dependencies as well.) 
   
   Upon installation, one can check the versions with the following commands:

   ```sh
   rustc --version
   cargo --version
   make --version
   ```

## Build with Make

Obtain the repository and then run its default Make target:

```sh
git clone https://github.com/kupl/qmiep-artifact
cd QMIEP
make
```

<!-- TODO Change the link -->

If you already have the artifact, run `make` from its root directory.
The default target, `make setup`, performs the following steps:

1. Creates the `qmiep` Conda environment with Python 3.12 if it does not exist.
   An existing environment is reused; it must already have Python 3.12 or later.
2. Installs the package and its Python, test, and lint dependencies from
   [`pyproject.toml`](../pyproject.toml). Maturin builds and installs the PyO3
   extension as part of this step.
3. Builds the `qmiep-spectrum` and `qmiep-synthesis` Rust command-line programs
   in release mode. Cargo reads the manifests under [`rust/`](../rust/Cargo.toml)
   and uses [`rust/Cargo.lock`](../rust/Cargo.lock) for dependency versions (optionally used for internal debugging).

Rust build artifacts are placed in `/tmp/qmiep-rust-target` by default.
The command-line programs are in its `release/` subdirectory. Python scripts
call the installed extension directly and do not launch these programs.

All commands below use the default environment name, `qmiep`, although no manual
activation is needed because they use `conda run` in this manual.

| Command | Purpose |
| --- | --- |
| `make` / `make setup` | Install the Python package and build the Rust programs. |
| `make env` | Create the Conda environment if needed. |
| `make install` | Install Python dependencies and rebuild the PyO3 extension. |
| `make rust-build` | Build the standalone Rust programs. |
| `make lint` | Run isort, Black, and Ruff on `src/`. |

## Check the installation

Check the native extension and run a small synthesis experiment:

```sh
conda run -n qmiep python -c "from qmiep.rust import _rust; print('QMIEP Rust extension loaded')"
conda run --no-capture-output -n qmiep python scripts/unitary.py \
  --algorithm analytic --mode exact --qubits 3 --samples 1 --seed 0 --quiet
```
The first command prints `QMIEP Rust extension loaded`. The second prints a
table with one successful sample, its synthesis time, and a CNOT count of 19.
Allow about **2–3 seconds** for the second command on an Apple M4 Pro,
including startup and circuit reconstruction. A validation failure raises
an error.
