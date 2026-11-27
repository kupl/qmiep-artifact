mod circuit;
mod euler;
mod fragment;
mod kak;
mod linalg;
mod recursion;

pub use circuit::{
    Circuit, Instruction, MultiplexedDPMultiplier, MultiplexedPermutation, SynthesisResult,
};
pub use faer::{Mat, MatRef};
pub use num_complex::Complex64;
// Retain the existing synthesis-crate entry points for solver callers.
pub use qmiep_solver::{
    DPMultiplier, Error, QMIEPResult, QMIEPSolver, Result, permutation_candidates,
    solve_one_qubit_spectrum, solve_two_qubit_spectrum,
};

use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub(crate) type C = Complex64;

/// Algorithms 1–3 of the manuscript, independently of the output correction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Algorithm {
    #[default]
    Analytic,
    Heuristic,
    HeuristicOpt,
}

impl FromStr for Algorithm {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "analytic" => Ok(Self::Analytic),
            "heuristic" => Ok(Self::Heuristic),
            "heuristic-opt" => Ok(Self::HeuristicOpt),
            _ => Err(Error(
                "algorithm must be analytic, heuristic, or heuristic-opt".into(),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    #[default]
    #[serde(rename = "exact")]
    Exact,
    #[serde(rename = "PD-equiv")]
    PdEquiv,
    #[serde(rename = "D-equiv")]
    DEquiv,
}

impl FromStr for Mode {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "exact" => Ok(Self::Exact),
            "PD-equiv" => Ok(Self::PdEquiv),
            "D-equiv" => Ok(Self::DEquiv),
            _ => Err(Error("mode must be exact, PD-equiv, or D-equiv".into())),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    #[default]
    Exact,
    /// Exact recursion with improved PD-permitted children; requires Heuristic.
    #[serde(rename = "exact_further_D_optimized")]
    ExactFurtherDOptimized,
    /// Original terminal-only D synthesis (D_naive).
    #[serde(alias = "D_naive")]
    Diagonal,
    /// Single-pass improved synthesis requiring a diagonal output correction.
    #[serde(rename = "D_equivalent_further_optimal")]
    DiagonalFurtherOptimal,
    Pd,
    /// Improved recursion permitting a PD correction, also across backend calls.
    #[serde(rename = "pd_further_D_optimized")]
    PdFurtherDOptimized,
}

impl Kind {
    fn is_exact(self) -> bool {
        matches!(self, Self::Exact | Self::ExactFurtherDOptimized)
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    #[default]
    Analytic,
    Heuristic,
    /// Disable antipodal pairing while retaining diagonal propagation and the CX basis.
    Baseline,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub kind: Kind,
    /// ExactFurtherDOptimized requires Heuristic; other kinds retain their methods.
    pub method: Method,
    /// QMIEP settings used for numerical solves and two-qubit polishing at every recursion depth.
    pub solver: QMIEPSolver,
    pub atol: f64,
    /// Check U†U at the boundary. Shape and finite values are always checked.
    pub check_input: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            kind: Kind::Exact,
            method: Method::Analytic,
            solver: QMIEPSolver::default(),
            atol: 1e-10,
            check_input: true,
        }
    }
}

impl Options {
    /// Select the algorithm and the correction exposed at the root.
    pub fn new(algorithm: Algorithm, mode: Mode) -> Self {
        let optimized = algorithm == Algorithm::HeuristicOpt;
        let kind = match (mode, optimized) {
            (Mode::Exact, false) => Kind::Exact,
            (Mode::PdEquiv, false) => Kind::Pd,
            (Mode::DEquiv, false) => Kind::Diagonal,
            (Mode::Exact, true) => Kind::ExactFurtherDOptimized,
            (Mode::PdEquiv, true) => Kind::PdFurtherDOptimized,
            (Mode::DEquiv, true) => Kind::DiagonalFurtherOptimal,
        };
        Self {
            kind,
            method: if algorithm == Algorithm::Analytic {
                Method::Analytic
            } else {
                Method::Heuristic
            },
            ..Self::default()
        }
    }

    /// Parse CLI and PyO3 options, including the original kind/method interface.
    pub fn from_names(
        algorithm: Option<&str>,
        mode: Option<&str>,
        kind: Option<&str>,
        method: Option<&str>,
    ) -> Result<Self> {
        if kind.is_none() && method.is_none() {
            return Ok(Self::new(
                algorithm.unwrap_or("analytic").parse()?,
                mode.unwrap_or("exact").parse()?,
            ));
        }
        if algorithm.is_some() || mode.is_some() {
            return Err(Error(
                "use algorithm/mode or the legacy kind/method options separately".into(),
            ));
        }
        let kind = match kind.unwrap_or("exact") {
            "exact" => Kind::Exact,
            "pd" => Kind::Pd,
            "diagonal" | "D_naive" => Kind::Diagonal,
            "exact_further_D_optimized" => Kind::ExactFurtherDOptimized,
            "pd_further_D_optimized" => Kind::PdFurtherDOptimized,
            "D_equivalent_further_optimal" => Kind::DiagonalFurtherOptimal,
            _ => return Err(Error("invalid synthesis kind".into())),
        };
        let method = match method {
            None if kind == Kind::ExactFurtherDOptimized => Method::Heuristic,
            None | Some("analytic") => Method::Analytic,
            Some("heuristic") => Method::Heuristic,
            Some("baseline") => Method::Baseline,
            _ => {
                return Err(Error(
                    "method must be analytic, heuristic, or baseline".into(),
                ));
            }
        };
        Ok(Self {
            kind,
            method,
            ..Self::default()
        })
    }
}

/// Synthesize a 2^n by 2^n unitary (n >= 1).
/// Both exact kinds return no multiplier. Diagonal kinds return a multiplier with P = I.
/// In improved diagonal synthesis, N's PD correction propagates through the full right R into O.
/// Full matrix reconstruction is opt-in via [`SynthesisResult::reconstruct`].
pub fn synthesize(unitary: MatRef<'_, C>, options: Options) -> Result<SynthesisResult> {
    linalg::validate(unitary, options.atol, options.check_input)?;
    if options.kind == Kind::ExactFurtherDOptimized && options.method != Method::Heuristic {
        return Err(Error(
            "exact_further_D_optimized requires method='heuristic'; 'analytic' and 'baseline' are not supported".into(),
        ));
    }
    let uses_optimized_d = matches!(
        options.kind,
        Kind::DiagonalFurtherOptimal | Kind::ExactFurtherDOptimized | Kind::PdFurtherDOptimized
    );
    if uses_optimized_d && options.method == Method::Baseline {
        return Err(Error(
            "optimized D synthesis requires QMIEP; method cannot be 'baseline'".into(),
        ));
    }
    if options.method == Method::Heuristic || uses_optimized_d {
        options.solver.validate_options()?;
    }
    let n = unitary.nrows();
    if n < 2 || !n.is_power_of_two() {
        return Err(Error("matrix dimension must be 2^n with n >= 1".into()));
    }
    if n == 2 {
        let matrix = std::array::from_fn(|i| std::array::from_fn(|j| unitary[(i, j)]));
        let (gate, global_phase) = euler::synthesize(matrix, 0);
        return Ok(SynthesisResult {
            circuit: Circuit {
                num_qubits: 1,
                instructions: vec![gate],
                global_phase,
            },
            multiplier: (!options.kind.is_exact()).then(|| MultiplexedDPMultiplier {
                permutation: MultiplexedPermutation {
                    zero: vec![0],
                    one: vec![0],
                },
                diagonal: vec![C::new(1.0, 0.0); 2],
            }),
        });
    }
    recursion::synthesize(unitary, options)
}
