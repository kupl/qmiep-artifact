use qmiep_synthesis::{Complex64, Mat, Options, synthesize};
use serde::Deserialize;
use std::{
    env, fs,
    io::{self, Read},
    time::Instant,
};

const HELP: &str = "qmiep-synthesis [matrix.json|-] \
    [--algorithm analytic|heuristic|heuristic-opt] \
    [--mode exact|PD-equiv|D-equiv] [--seed N] [--atol 1e-10] \
    [--skip-input-check] [--verify] [--summary]\n\n\
    Input: {\"matrix\": [[[re,im], ...], ...]}. Qubit 0 is the most significant bit.\n\
    Output: timing and CX count, plus circuit W and correction U = D P W by default.\n\
    The default selection is --algorithm analytic --mode exact.\n\
    Legacy --kind and --method options also accept the QSD baseline.\n\
    --verify reconstructs the full matrix and checks it against the input \
    within --atol, outside synthesis timing.\n\
    --summary prints a JSON summary for debugging.";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    matrix: Vec<Vec<Complex64>>,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut options = Options::default();
    let (mut algorithm, mut mode, mut kind, mut method) = (None, None, None, None);
    // --verify reconstructs the full matrix and checks it against the input within --atol.
    let mut verify = false;
    // --summary prints a JSON summary for debugging.
    let mut summary = false;
    // Path to the input matrix JSON file.
    let mut path = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--algorithm" => algorithm = Some(args.next().ok_or("missing --algorithm value")?),
            "--mode" => mode = Some(args.next().ok_or("missing --mode value")?),
            "--kind" => kind = Some(args.next().ok_or("missing --kind value")?),
            "--method" => method = Some(args.next().ok_or("missing --method value")?),
            "--atol" => options.atol = args.next().ok_or("missing --atol value")?.parse()?,
            "--seed" => {
                options.solver.seed = Some(args.next().ok_or("missing --seed value")?.parse()?)
            }
            // Skip only the input unitarity check (U†U ≈ I).
            "--skip-input-check" => options.check_input = false,
            "--verify" => verify = true,
            "--summary" => summary = true,
            "--help" | "-h" => {
                println!("{HELP}");
                return Ok(());
            }
            _ if (arg == "-" || !arg.starts_with('-')) && path.is_none() => path = Some(arg),
            _ => return Err(format!("unexpected argument: {arg}").into()),
        }
    }
    let selection = Options::from_names(
        algorithm.as_deref(),
        mode.as_deref(),
        kind.as_deref(),
        method.as_deref(),
    )?;
    options.kind = selection.kind;
    options.method = selection.method;
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
    let start = Instant::now();
    let result = synthesize(u.as_ref(), options)?;
    let seconds = start.elapsed().as_secs_f64();
    let error = if verify {
        Some((result.reconstruct()? - &u).norm_max())
    } else {
        None
    };
    let mut output = serde_json::json!({
        "num_qubits": result.circuit.num_qubits,
        "kind": options.kind,
        "method": options.method,
        "optimizer": "newton",
        "numerical_policy": "qiskit_2_5_1_polar_demultiplex",
        "one_qubit_basis": "u",
        "circuit_backend": "native_vector_inline_dp",
        "cnot_count": result.circuit.cnot_count()?,
        "synthesis_seconds": seconds,
        "verification": if verify { "full_matrix" } else { "off" },
        "verification_max_error": error,
        "synthesis": if summary { None } else { Some(&result) },
    });
    if kind.is_none() && method.is_none() {
        output["algorithm"] = algorithm.as_deref().unwrap_or("analytic").into();
        output["mode"] = mode.as_deref().unwrap_or("exact").into();
    }
    serde_json::to_writer(io::stdout().lock(), &output)?;
    println!();
    if error.is_some_and(|error| !error.is_finite() || error > options.atol) {
        return Err(format!("reconstruction error exceeds atol: {error:?}").into());
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
