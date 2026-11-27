//! One native Haar sample per requested qubit count (default: 10).
//! Sampling and output verification are outside the reported solve times.
use qmiep_synthesis::{Complex64 as C, Mat, Method, Options, QMIEPSolver, synthesize};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use std::{env, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut sizes = env::args()
        .skip(1)
        .map(|arg| arg.parse::<u32>())
        .collect::<Result<Vec<_>, _>>()?;
    if sizes.is_empty() {
        sizes.push(10);
    }
    for qubits in sizes {
        if !(2..=12).contains(&qubits) {
            return Err("qubit counts must be between 2 and 12".into());
        }
        let n = 1 << qubits;
        let seed = 42 + n as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let gaussian = Mat::from_fn(n, n, |_, _| {
            C::new(
                StandardNormal.sample(&mut rng),
                StandardNormal.sample(&mut rng),
            )
        });
        let svd = gaussian
            .svd()
            .map_err(|error| format!("Haar sampling SVD: {error:?}"))?;
        let u = svd.U() * svd.V().adjoint();
        drop(svd);
        drop(gaussian);
        let solver = QMIEPSolver {
            seed: Some(7),
            ..Default::default()
        };
        let start = Instant::now();
        let solved = solver.solve(u.as_ref())?;
        let solve_seconds = start.elapsed().as_secs_f64();
        assert!(solved.success && solved.antipodal_error <= solver.atol);
        println!(
            "{}",
            serde_json::json!({
                "kind": "solver", "qubits": qubits, "input_seed": seed, "solver_seed": 7,
                "seconds": solve_seconds, "error": solved.antipodal_error, "nfev": solved.nfev,
                "threads": env::var("RAYON_NUM_THREADS").ok(), "samples": 1,
            })
        );
        let start = Instant::now();
        let result = synthesize(
            u.as_ref(),
            Options {
                method: Method::Heuristic,
                solver,
                check_input: false,
                ..Default::default()
            },
        )?;
        let synthesis_seconds = start.elapsed().as_secs_f64();
        let error = if qubits <= 5 {
            (result.reconstruct()? - &u).norm_max()
        } else {
            let mut error = 0.0_f64;
            for sample in 0..4 {
                let state: Vec<_> = (0..n)
                    .map(|i| match sample {
                        0 => C::new(f64::from(i == 0), 0.0),
                        1 => C::new(f64::from(i + 1 == n), 0.0),
                        _ => C::from_polar(
                            1.0 / (n as f64).sqrt(),
                            i as f64 * if sample == 2 { 0.271 } else { 0.618 },
                        ),
                    })
                    .collect();
                let actual = result.apply(&state)?;
                for i in 0..n {
                    let expected: C = (0..n).map(|j| u[(i, j)] * state[j]).sum();
                    let residual = (actual[i] - expected).norm();
                    assert!(residual.is_finite());
                    error = error.max(residual);
                }
            }
            error
        };
        assert!(error < 1e-10);
        let cnot = result.circuit.cnot_count()?;
        assert_eq!(
            cnot,
            (41 * 4usize.pow(qubits) - 108 * 2usize.pow(qubits) + 64) / 96
        );
        println!(
            "{}",
            serde_json::json!({
                "kind": "synthesis", "qubits": qubits, "input_seed": seed, "solver_seed": 7,
                "seconds": synthesis_seconds, "cnot": cnot, "error": error,
                "verification": if qubits <= 5 { "full_matrix" } else { "four_states" },
                "threads": env::var("RAYON_NUM_THREADS").ok(), "samples": 1,
            })
        );
    }
    Ok(())
}
