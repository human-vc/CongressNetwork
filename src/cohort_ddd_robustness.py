import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR

warnings.filterwarnings("ignore")

PANEL_PATH = RESULTS_DIR / "cohort_ddd_panel.csv"

def load_panel():
    p = pd.read_csv(PANEL_PATH)
    p = p.rename(columns={"cross_party_agreement": "outcome", "member_id": "icpsr", "prior_margin": "margin"})
    p["safe_margin"] = p["margin"].abs()
    return p

def fit(panel, formula, cluster="icpsr"):
    sub = panel.dropna(subset=[c for c in ["outcome", "freshman", "post_2010", "competitive", "safe_margin", "margin", "icpsr", "congress"] if c in panel.columns]).copy()
    try:
        m = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub[cluster].values})
        return m, sub
    except Exception as e:
        return None, sub

def coefs_p(m, names):
    out = {}
    for n in names:
        if n in m.params.index:
            out[n] = {
                "coef": float(m.params[n]),
                "se": float(m.bse[n]),
                "p": float(m.pvalues[n]),
                "ci": [float(m.conf_int().loc[n, 0]), float(m.conf_int().loc[n, 1])],
            }
    return out

def power_two_sided(coef, se, alpha=0.05):
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    ncp = abs(coef) / se if se > 0 else 0.0
    return float(1 - norm.cdf(z - ncp) + norm.cdf(-z - ncp))

def mde(se, alpha=0.05, power=0.8):
    from scipy.stats import norm
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)

def main():
    panel = load_panel()
    print(f"Panel: {len(panel)} obs, {panel['icpsr'].nunique()} members")

    has_competitive = "competitive" in panel.columns
    has_safe_margin = "safe_margin" in panel.columns
    has_freshman = "freshman" in panel.columns
    has_post = "post_2010" in panel.columns

    results = {}

    if has_freshman and has_post and has_competitive:
        m, _ = fit(panel, "outcome ~ freshman * post_2010 * competitive")
        if m is not None:
            triple = coefs_p(m, ["freshman:post_2010:competitive"])
            t = triple.get("freshman:post_2010:competitive", {})
            results["triple_baseline_power"] = {
                **t,
                "power_at_observed": power_two_sided(t.get("coef", 0), t.get("se", 1.0)) if t else None,
                "mde_for_80pct_power": mde(t.get("se", 1.0)) if t else None,
            }

    if has_freshman and has_post:
        m, _ = fit(panel, "outcome ~ freshman * post_2010")
        results["two_way_freshman_x_post2010"] = coefs_p(m, ["freshman", "post_2010", "freshman:post_2010"]) if m is not None else {}

    if has_post and has_competitive:
        m, _ = fit(panel, "outcome ~ post_2010 * competitive")
        results["two_way_post2010_x_competitive"] = coefs_p(m, ["post_2010", "competitive", "post_2010:competitive"]) if m is not None else {}

    if has_safe_margin and has_post:
        m, _ = fit(panel, "outcome ~ post_2010 * safe_margin")
        results["continuous_safe_margin_x_post2010"] = coefs_p(m, ["post_2010", "safe_margin", "post_2010:safe_margin"]) if m is not None else {}

    if has_freshman and "tea_party_cohort" not in panel.columns and "congress" in panel.columns:
        panel["tea_party_cohort"] = ((panel["congress"].isin([112, 113])) & (panel["freshman"] == 1)).astype(int)

    if "tea_party_cohort" in panel.columns:
        m, _ = fit(panel, "outcome ~ tea_party_cohort + freshman + C(congress)")
        results["tea_party_cohort_binary"] = coefs_p(m, ["tea_party_cohort"]) if m is not None else {}

    if has_freshman and has_post and has_competitive and "margin" in panel.columns:
        sweep = {}
        for cutoff in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
            p2 = panel.copy()
            p2["competitive"] = (p2["margin"].abs() < cutoff).astype(int)
            m, _ = fit(p2, "outcome ~ freshman * post_2010 * competitive")
            if m is not None:
                t = coefs_p(m, ["freshman:post_2010:competitive"])
                sweep[f"cutoff_{cutoff:.2f}"] = t.get("freshman:post_2010:competitive", {})
        results["triple_interaction_margin_cutoff_sweep"] = sweep

    if has_freshman and has_post:
        m, _ = fit(panel, "outcome ~ freshman * post_2010 + C(icpsr) + C(congress)", cluster="icpsr")
        if m is not None:
            results["two_way_freshman_x_post2010_member_fe"] = coefs_p(m, ["freshman:post_2010"])

    out = RESULTS_DIR / "cohort_ddd_robustness.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
