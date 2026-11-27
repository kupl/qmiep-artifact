//! Standalone solver: input {"matrix": [...], "solver": {optional settings}}.
use qmiep_solver::{
    Complex64, DPMultiplier, Mat, MatRef, QMIEPResult, QMIEPSolver, solve_one_qubit_spectrum,
    solve_two_qubit_spectrum,
};
use serde::{Deserialize, Serialize};
use std::{
    env,
    f64::consts::PI,
    fs,
    io::{self, Read},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    matrix: Vec<Vec<Complex64>>,
    #[serde(default)]
    solver: QMIEPSolver,
}

#[derive(Serialize)]
#[serde(untagged)]
enum SolverResult {
    Heuristic(QMIEPResult),
    Analytic {
        #[serde(flatten)]
        multiplier: DPMultiplier,
        success: bool,
        antipodal_error: f64,
        permutations_tried: Option<usize>,
        attempts: Option<usize>,
        nfev: Option<usize>,
    },
}

fn analytic(
    u: MatRef<'_, Complex64>,
    atol: f64,
) -> Result<SolverResult, Box<dyn std::error::Error>> {
    if !atol.is_finite() || atol <= 0.0 {
        return Err("solver atol must be finite and positive".into());
    }
    let multiplier = if u.nrows() == 2 {
        solve_one_qubit_spectrum(u, 1e-10)?
    } else {
        solve_two_qubit_spectrum(u, 1e-10)?
    };
    let v = Mat::from_fn(u.nrows(), u.ncols(), |i, j| {
        multiplier.diagonal[i] * u[(multiplier.rows[i], j)]
    });
    let values = v
        .eigenvalues()
        .map_err(|e| format!("spectral verification: {e:?}"))?;
    if values.iter().any(|z| !z.is_finite()) {
        return Err("nonfinite eigenvalue in spectral verification".into());
    }
    let mut angles: Vec<_> = values.iter().map(|z| z.arg()).collect();
    angles.sort_by(f64::total_cmp);
    let half = angles.len() / 2;
    let error = (0..half)
        .map(|i| (angles[i + half] - angles[i] - PI).abs())
        .fold(0.0, f64::max);
    Ok(SolverResult::Analytic {
        multiplier,
        success: error <= atol,
        antipodal_error: error,
        permutations_tried: None,
        attempts: None,
        nfev: None,
    })
}

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next();
    if args.next().is_some() {
        return Err("expected at most one JSON path (or - for stdin)".into());
    }
    if matches!(path.as_deref(), Some("--help" | "-h")) {
        println!(
            "qmiep-spectrum [matrix.json|-]\nInput: {{\"matrix\": [[[re,im], ...], ...], \"solver\": {{\"seed\": 42}}}}\nSolver fields: atol, max_permutations, restarts, maxiter, seed. Pairing is the only residual, with damped Gauss-Newton optimization.\nOne and two qubits use analytic solvers; optimization settings apply from three qubits.\nOutput: left row selection and diagonal for V = D P U, plus verified search diagnostics. Analytic search counts are null. Timing includes input validation and fresh spectral verification.\nAn exhausted search emits success=false and exits with code 1; this does not establish nonexistence."
        );
        return Ok(true);
    }
    let text = if let Some(path) = path.filter(|p| p != "-") {
        fs::read_to_string(path)?
    } else {
        let mut text = String::new();
        io::stdin().read_to_string(&mut text)?;
        text
    };
    let input: Input = serde_json::from_str(&text)?;
    let n = input.matrix.len();
    if n == 0 || input.matrix.iter().any(|row| row.len() != n) {
        return Err("matrix must be nonempty and square".into());
    }
    let u = Mat::from_fn(n, n, |i, j| input.matrix[i][j]);
    let started = Instant::now();
    let (method, result) = match n {
        2 => ("analytic_1q", analytic(u.as_ref(), input.solver.atol)?),
        4 => ("analytic_2q", analytic(u.as_ref(), input.solver.atol)?),
        _ => (
            "heuristic",
            SolverResult::Heuristic(input.solver.solve(u.as_ref())?),
        ),
    };
    let seconds = started.elapsed().as_secs_f64();
    let success = match &result {
        SolverResult::Heuristic(result) => result.success,
        SolverResult::Analytic { success, .. } => *success,
    };
    serde_json::to_writer(
        io::stdout().lock(),
        &serde_json::json!({
            "dimension": n, "method": method, "solve_seconds": seconds,
            "solver": input.solver, "result": result,
        }),
    )?;
    println!();
    Ok(success)
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(true) => std::process::ExitCode::SUCCESS,
        Ok(false) => std::process::ExitCode::FAILURE,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
