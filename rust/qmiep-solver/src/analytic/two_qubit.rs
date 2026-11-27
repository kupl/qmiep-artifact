use crate::{C, DPMultiplier, Error, MatRef, Result, validation::validate_unitary};
use std::f64::consts::PI;

/// Analytic U(4) QMIEP solver: select a polygon-feasible transversal, then
/// close its four diagonal vectors. For U(4), trace zero gives antipodal pairs.
pub fn solve_two_qubit_spectrum(u: MatRef<'_, C>, atol: f64) -> Result<DPMultiplier> {
    validate_unitary(u, atol)?;
    if u.nrows() != 4 {
        return Err(Error(
            "the analytic QMIEP solver requires a 4 by 4 unitary".into(),
        ));
    }
    let feasible = |rows: &[usize; 4]| {
        let lengths = std::array::from_fn::<_, 4, _>(|i| u[(rows[i], i)].norm());
        let longest = lengths.iter().copied().fold(0.0, f64::max);
        longest <= lengths.iter().sum::<f64>() - longest
    };
    let mut rows = [0, 1, 2, 3];
    if !feasible(&rows) {
        let mut largest = (0, 0);
        for i in 0..4 {
            for j in 0..4 {
                if u[(i, j)].norm_sqr() > u[largest].norm_sqr() {
                    largest = (i, j);
                }
            }
        }
        let remaining: Vec<_> = (0..4).filter(|&i| i != largest.0).collect();
        let columns: Vec<_> = (0..4).filter(|&i| i != largest.1).collect();
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut found = false;
        for p in permutations {
            rows[largest.1] = largest.0;
            for i in 0..3 {
                rows[columns[i]] = remaining[p[i]];
            }
            if feasible(&rows) {
                found = true;
                break;
            }
        }
        if !found {
            return Err(Error("no polygon-feasible row permutation found".into()));
        }
    }
    let z = std::array::from_fn::<_, 4, _>(|i| u[(rows[i], i)]);
    let lengths = z.map(|v| v.norm());
    let mut order = [0, 1, 2, 3];
    order.sort_by(|&a, &b| lengths[b].total_cmp(&lengths[a]));
    let scale = lengths[order[0]];
    let mut diagonal = [C::new(1.0, 0.0); 4];
    if scale > 0.0 {
        let [a0, a1, a2, a3] = order.map(|i| lengths[i] / scale);
        let r = a0 - a3;
        let (alpha, beta) = if r == 0.0 {
            (0.0, PI)
        } else if a2 == 0.0 {
            (0.0, 0.0)
        } else {
            let ratio = (a1 - a2) / r;
            let offset = ratio * (a1 + a2);
            let height = ((a1 + a2 - r) * (a1 + a2 + r) * (1.0 - ratio) * (1.0 + ratio))
                .max(0.0)
                .sqrt();
            (height.atan2(r + offset), height.atan2(r - offset))
        };
        for (i, angle) in order.into_iter().zip([PI, alpha, -beta, 0.0]) {
            diagonal[i] = C::from_polar(1.0, angle - z[i].arg());
        }
    }
    Ok(DPMultiplier {
        rows: rows.to_vec(),
        diagonal: diagonal.to_vec(),
    })
}
