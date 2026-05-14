"""Placebo Congress + leave-one-Congress-out CV for the BLI -> departure GEE.

Placebo Congress: reassign treatment cohort label to a non-treated wave
  (102, 106, 108) and re-run cohort DDD. Expect null effects.

LOOCV: drop each Congress (100-118) and refit the full-panel GEE. The
  distribution of BLI coefficients across 19 fits reports robustness.

Both are embarrassingly parallel across joblib workers.
"""
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bli_regression import build_panel
from config import CONGRESSES, RESULTS_DIR, SEED

warnings.filterwarnings("ignore")
COVARS = ["bli", "ideology_distance", "seniority", "is_republican"]
PLACEBO_COHORTS = [102, 106, 108]


def fit_full(panel):
    X = sm.add_constant(panel[COVARS].astype(float))
    y = panel["departed_within_2"].astype(float)
    return GEE(y, X, groups=panel["icpsr"], family=Binomial(),
               cov_struct=Independence()).fit(maxiter=80)


def loocv_one(panel, drop_c):
    sub = panel[panel["congress"] != drop_c].copy()
    if len(sub) < 100 or sub["departed_within_2"].nunique() < 2:
        return {"drop_congress": int(drop_c), "n": int(len(sub)), "error": "insufficient"}
    try:
        res = fit_full(sub)
        z = norm.ppf(0.975)
        b, se = float(res.params["bli"]), float(res.bse["bli"])
        return {
            "drop_congress": int(drop_c), "n": int(len(sub)),
            "bli_coef": b, "bli_se": se, "bli_p": float(res.pvalues["bli"]),
            "ci_lo": b - z * se, "ci_hi": b + z * se,
        }
    except Exception as e:
        return {"drop_congress": int(drop_c), "n": int(len(sub)), "error": str(e)}


def placebo_one(panel, fake_treat_c):
    """Pretend `fake_treat_c` is the treated cohort. Compute DDD-style
    coefficient = difference in BLI->departure slope at cohort vs. all
    other Congresses, holding the existing covariates fixed.
    """
    sub = panel.copy()
    sub["treated"] = (sub["congress"] == fake_treat_c).astype(int)
    sub["bli_x_treated"] = sub["bli"] * sub["treated"]
    fvars = ["bli", "treated", "bli_x_treated", "ideology_distance",
             "seniority", "is_republican"]
    X = sm.add_constant(sub[fvars].astype(float))
    y = sub["departed_within_2"].astype(float)
    try:
        res = GEE(y, X, groups=sub["icpsr"], family=Binomial(),
                  cov_struct=Independence()).fit(maxiter=80)
        b = float(res.params["bli_x_treated"])
        se = float(res.bse["bli_x_treated"])
        return {
            "placebo_congress": int(fake_treat_c), "n": int(len(sub)),
            "interaction_coef": b, "interaction_se": se,
            "interaction_p": float(res.pvalues["bli_x_treated"]),
        }
    except Exception as e:
        return {"placebo_congress": int(fake_treat_c), "error": str(e)}


def main():
    print("Building panel...")
    panel = build_panel()
    print(f"  n={len(panel)}, congresses={sorted(panel['congress'].unique())}")

    print("Fitting baseline GEE...")
    base = fit_full(panel)
    base_b, base_se = float(base.params["bli"]), float(base.bse["bli"])
    print(f"  Baseline BLI coef = {base_b:.4f} (SE={base_se:.4f})")

    print(f"LOOCV over {len(CONGRESSES)} congresses (parallel)...")
    congresses = sorted(panel["congress"].unique())
    loo = Parallel(n_jobs=-1, backend="loky")(
        delayed(loocv_one)(panel, int(c)) for c in congresses
    )

    print(f"Placebo cohorts: {PLACEBO_COHORTS} (parallel)...")
    plac = Parallel(n_jobs=len(PLACEBO_COHORTS), backend="loky")(
        delayed(placebo_one)(panel, int(c)) for c in PLACEBO_COHORTS
    )

    coefs = np.array([r["bli_coef"] for r in loo if "bli_coef" in r])
    summary = {
        "loocv_n": int(len(coefs)),
        "loocv_mean": float(coefs.mean()) if len(coefs) else None,
        "loocv_sd": float(coefs.std()) if len(coefs) else None,
        "loocv_min": float(coefs.min()) if len(coefs) else None,
        "loocv_max": float(coefs.max()) if len(coefs) else None,
        "loocv_q05": float(np.quantile(coefs, 0.05)) if len(coefs) else None,
        "loocv_q95": float(np.quantile(coefs, 0.95)) if len(coefs) else None,
    }

    output = {
        "baseline": {"bli_coef": base_b, "bli_se": base_se,
                     "bli_p": float(base.pvalues["bli"])},
        "loocv": loo,
        "loocv_summary": summary,
        "placebo_cohorts": plac,
    }
    out_path = RESULTS_DIR / "placebo_loocv_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  LOOCV coef range: [{summary['loocv_min']:.4f}, {summary['loocv_max']:.4f}]")
    for p in plac:
        if "interaction_p" in p:
            print(f"  Placebo C{p['placebo_congress']}: p={p['interaction_p']:.3g}")


if __name__ == "__main__":
    main()
