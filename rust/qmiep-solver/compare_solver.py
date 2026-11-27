"""Benchmark Rust pairing Newton on identical seeded Haar matrices.

Reported solve time excludes JSON I/O and process startup, and includes input
validation, optimization and fresh spectral verification. All policies use the
same final eigenphase tolerance; their internal stopping rules differ.
Dimensions 2 and 4 use analytic solvers. An optional reference binary receives
the same settings; legacy binaries therefore use their default residual policy.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--reference-binary", type=Path)
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=[8, 16, 32, 64, 128]
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.samples < 1 or any(n < 2 or n & (n - 1) for n in args.dimensions):
        parser.error("positive samples and power-of-two dimensions >= 2 required")
    if not np.isfinite(args.atol) or args.atol <= 0 or args.seed < 0:
        parser.error("positive finite atol and nonnegative seed required")
    binaries = {"current": args.rust_binary.resolve()}
    if args.reference_binary is not None:
        binaries["reference"] = args.reference_binary.resolve()
    failed = False
    for n in args.dimensions:
        records = {label: [] for label in binaries}
        methods = {}
        for sample in range(args.samples):
            seed = args.seed + 10000 * n + sample
            rng = np.random.default_rng(seed)
            q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
            u = q * (np.diag(r) / abs(np.diag(r))).conj()
            matrix = np.stack((u.real, u.imag), axis=-1).tolist()
            for implementation in binaries:
                solver = dict(
                    max_permutations=1,
                    restarts=0,
                    seed=seed + 1,
                    atol=args.atol,
                )
                run = subprocess.run(
                    [str(binaries[implementation])],
                    input=json.dumps(dict(matrix=matrix, solver=solver)),
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
                if not run.stdout:
                    raise RuntimeError(run.stderr)
                native = json.loads(run.stdout)
                result = native["result"]
                d = np.asarray(result["diagonal"])
                v = (d[:, 0] + 1j * d[:, 1])[:, None] * u[result["rows"]]
                angles = np.sort(np.angle(np.linalg.eigvals(v)))
                error = float(np.max(abs(angles[n // 2 :] - angles[: n // 2] - np.pi)))
                methods[implementation] = native.get("method", "heuristic")
                verified = bool(result["success"] and error <= args.atol)
                if not verified:
                    failed = True
                records[implementation].append(
                    (native["solve_seconds"], result["nfev"], verified)
                )
        for implementation, group in records.items():
            seconds, evaluations, verified = zip(*group, strict=True)
            detail = (
                f"newton, {np.median(evaluations):.0f} evaluations"
                if methods[implementation] == "heuristic"
                else methods[implementation]
            )
            print(
                f"n={n.bit_length() - 1} N={n} {implementation} {detail}: "
                f"{1000 * np.median(seconds):.3f} ms, "
                f"{sum(verified)}/{len(group)} verified",
                flush=True,
            )
    if failed:
        raise SystemExit("some runs failed independent verification")


if __name__ == "__main__":
    main()
