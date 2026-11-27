.DEFAULT_GOAL := setup

CONDA ?= conda
CONDA_ENV ?= qmiep
PYTHON_VERSION ?= 3.12
CARGO ?= cargo
CARGO_TARGET_DIR ?= /tmp/qmiep-rust-target
export CARGO_TARGET_DIR

CONDA_RUN = $(CONDA) run --no-capture-output -n "$(CONDA_ENV)"

.PHONY: setup env install rust-build lint help check-conda check-rust

setup: install rust-build

check-conda:
	@command -v "$(CONDA)" >/dev/null || { echo "Install Conda and add it to PATH first." >&2; exit 1; }

check-rust:
	@command -v "$(CARGO)" >/dev/null && command -v rustc >/dev/null || { echo "Install Rust/Cargo and add them to PATH first." >&2; exit 1; }

env: check-conda
	@conda_envs="$$( $(CONDA) env list)" || exit $$?; \
	if printf '%s\n' "$$conda_envs" | awk '{print $$1}' | grep -Fxq "$(CONDA_ENV)"; then \
		echo "Using existing Conda environment: $(CONDA_ENV)"; \
	else \
		$(CONDA) create -n "$(CONDA_ENV)" "python=$(PYTHON_VERSION)" pip -y; \
	fi

# Maturin builds and installs the PyO3 extension with the Python package.
install: check-rust env
	$(CONDA_RUN) python -m pip install -e ".[test,lint]"

rust-build: check-rust
	$(CARGO) build --release --locked --manifest-path rust/Cargo.toml \
		-p qmiep-solver -p qmiep-synthesis --bins

lint: check-conda
	$(CONDA_RUN) isort src
	$(CONDA_RUN) black src
	$(CONDA_RUN) ruff check src

help:
	@printf '%s\n' \
		'make / make setup  Create the environment, install Python/PyO3, and build Rust CLIs.' \
		'make env           Create the Conda environment if missing.' \
		'make install       Install Python dependencies and rebuild the PyO3 extension.' \
		'make rust-build    Build the Rust CLIs in release mode.' \
		'make lint          Format and lint Python sources in the Conda environment.'
