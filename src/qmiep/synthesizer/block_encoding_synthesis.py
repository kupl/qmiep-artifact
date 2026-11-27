from contextlib import nullcontext
from typing import Literal

import cirq
import numpy as np
from numpy.typing import ArrayLike

from qmiep._validation import _validation_enabled
from qmiep.gates import UniformlyControlledRz, UnresolvedNQubitUnitary
from qmiep.rewriting import pd_migration_through_ucg_rule
from qmiep.rust import (
    ALGORITHMS,
    Algorithm,
    ExactMode,
    Method,
    PDMode,
    synthesize_unitary,
    synthesize_unitary_diagonal,
)
from qmiep.synthesizer._backends import (
    SynthesisBackends,
    _backend_scope,
    _mode_scope,
)
from qmiep.synthesizer.unitary_synthesizer import NQubitUnitarySynthesis


class BlockEncodingSynthesize:
    def run(
        self,
        matrix: ArrayLike,
        *,
        alpha: float,
        algorithm: Algorithm | None = None,
        mode: Method | Literal["exact", "exact_further_D_optimal"] = "exact",
        seed: int | None = None,
        backends: SynthesisBackends | None = None,
        atol: float = 1e-10,
    ) -> cirq.FrozenCircuit:
        """Encode a full-rank 2**k square matrix on k data qubits and ancilla q(0).

        The ancilla-zero block equals matrix / alpha for alpha >= ||matrix||_2.
        Numerical rank uses the SVD cutoff 2**k * eps * sigma_max.
        Both data blocks use the selected algorithm (default heuristic).
        V has PD-equiv output; the corrected W has exact output.
        The baseline uses the Rust implementation of Qiskit QSD with D-migration.
        """
        a = np.asarray(matrix, dtype=np.complex128)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("matrix must be square")
        d = a.shape[0]
        if d == 0 or d & (d - 1):
            raise ValueError("matrix dimension must be a positive power of two")
        if not np.all(np.isfinite(a)):
            raise ValueError("matrix entries must be finite")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not np.isfinite(atol) or atol <= 0:
            raise ValueError("atol must be finite and positive")

        # A = W Σ V in Proposition 4.1, with V the right factor returned by SVD.
        w, sigma, v = np.linalg.svd(a)
        rank_tol = (d * np.finfo(sigma.dtype).eps) * sigma[0]
        if not np.isfinite(sigma[0]) or sigma[0] - alpha > rank_tol:
            raise ValueError("alpha must be at least the matrix spectral norm")
        if sigma[-1] <= rank_tol:
            raise ValueError("matrix must have full rank at the SVD cutoff")
        phi = np.arccos(np.minimum(sigma / alpha, 1.0))
        ancilla, *data = cirq.LineQubit.range(d.bit_length())

        exact_kind: ExactMode = "exact"
        pd_kind: PDMode = "pd"
        if algorithm is not None and (algorithm not in ALGORITHMS or mode != "exact"):
            raise ValueError(
                "choose one valid algorithm or a legacy block-encoding mode"
            )
        if algorithm == "heuristic-opt" or mode == "exact_further_D_optimal":
            mode = "heuristic"
            exact_kind = "exact_further_D_optimized"
            pd_kind = "pd_further_D_optimized"
        elif mode == "exact":
            mode = algorithm or "heuristic"
        backends = SynthesisBackends() if backends is None else backends
        scope = nullcontext() if mode == "baseline" else _mode_scope(mode, seed)
        with _backend_scope(backends), scope:
            if not data:
                return cirq.FrozenCircuit(
                    cirq.global_phase_operation(w[0, 0] * v[0, 0]),
                    cirq.ry(-2 * phi[0]).on(ancilla),
                )

            # q(0)    ──S†───H────Rz────H───S──
            #                     │
            # data    ──────L_PD──□─────M──────
            rz = UniformlyControlledRz(np.exp(1j * phi), label="Rz", atol=atol).on(
                ancilla, *data
            )
            if mode == "baseline":
                # V = D V_D. The selector diagonal commutes through Rz into W D.
                check_input = _validation_enabled()
                v_d, phases = synthesize_unitary_diagonal(
                    v,
                    qubits=data,
                    method=mode,
                    seed=seed,
                    atol=atol,
                    check_input=check_input,
                )
                wd = synthesize_unitary(
                    w * phases,
                    qubits=data,
                    method=mode,
                    seed=seed,
                    atol=atol,
                    check_input=check_input,
                )
                left_block: cirq.Operation = v_d.freeze().to_op()
                m: cirq.Operation = wd.freeze().to_op()
            else:
                left = UnresolvedNQubitUnitary(v, label="L", atol=atol).on(*data)
                right = UnresolvedNQubitUnitary(w, label="M", atol=atol).on(*data)
                left_block, rz, m = pd_migration_through_ucg_rule(
                    cirq.Circuit(left, rz, right),
                    direction="right",
                    mode=mode,
                    kind=pd_kind,
                ).all_operations()
                if len(data) == 1:
                    m = cirq.MatrixGate(cirq.unitary(m), name="M").on(*data)
                else:
                    (m,) = (
                        NQubitUnitarySynthesis(mode=mode, kind=exact_kind)
                        .run(cirq.Circuit(m), target=m)
                        .all_operations()
                    )

        return cirq.FrozenCircuit(
            cirq.Moment(cirq.S(ancilla) ** -1),
            cirq.Moment(cirq.H(ancilla), left_block),
            cirq.Moment(rz),
            cirq.Moment(cirq.H(ancilla), m),
            cirq.Moment(cirq.S(ancilla)),
        )
