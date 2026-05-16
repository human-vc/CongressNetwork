import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bli_regression import build_panel, iqr_scale_effects
from config import RESULTS_DIR

warnings.filterwarnings("ignore")

def build_panel_with_lag():
    panel = build_panel()
    panel = panel.sort_values(["icpsr", "congress"]).reset_index(drop=True)
    panel["bli_lag1"] = panel.groupby("icpsr")["bli"].shift(1)
    panel["bli_lag2"] = panel.groupby("icpsr")["bli"].shift(2)
    panel["bli_change"] = panel["bli"] - panel["bli_lag1"]
    return panel

def within_between_variance(panel, var="bli"):
    member_means = panel.groupby("icpsr")[var].transform("mean")
    within = panel[var] - member_means
    between = member_means - panel[var].mean()
    return {
        "var_within_member": float(within.var()),
        "var_between_member": float(between.var()),
        "rho_intraclass": float(between.var() / (within.var() + between.var())),
    }

def _scale_panel(scales, bli_series):
    return {
        "sd":      float(bli_series.std()),
        "iqr":     float(bli_series.quantile(0.75) - bli_series.quantile(0.25)),
        "p90_p10": float(bli_series.quantile(0.90) - bli_series.quantile(0.10)),
        "p95_p05": float(bli_series.quantile(0.95) - bli_series.quantile(0.05)),
    }

def cox_ph_time_varying(panel, outcome="departed_within_2", extra_covars=None):
    cols = ["bli", "ideology_distance", "seniority", "is_republican"]
    if extra_covars:
        cols = cols + list(extra_covars)
    sub = panel.dropna(subset=[outcome] + cols).copy()
    sub = sub.sort_values(["icpsr", "congress"]).reset_index(drop=True)
    sub["start"] = sub["congress"].astype(int) - 100
    sub["stop"] = sub["start"] + 2
    sub["event"] = sub[outcome].astype(int)
    fit_cols = ["icpsr", "start", "stop", "event"] + cols
    try:
        from lifelines import CoxTimeVaryingFitter
        ctv = CoxTimeVaryingFitter(penalizer=0.001)
        ctv.fit(sub[fit_cols], id_col="icpsr", event_col="event",
                start_col="start", stop_col="stop")
        summary = ctv.summary
        coef = float(summary.loc["bli", "coef"])
        se = float(summary.loc["bli", "se(coef)"])
        p = float(summary.loc["bli", "p"])
        scales = _scale_panel(None, sub["bli"])
        out = {
            "engine": "lifelines.CoxTimeVaryingFitter",
            "coef": coef,
            "se": se,
            "p": p,
            "scales": scales,
            "hazard_ratio_per_sd":       float(np.exp(coef * scales["sd"])),
            "hazard_ratio_per_iqr":      float(np.exp(coef * scales["iqr"])),
            "hazard_ratio_p90_vs_p10":   float(np.exp(coef * scales["p90_p10"])),
            "hazard_ratio_p95_vs_p05":   float(np.exp(coef * scales["p95_p05"])),
            "n_obs": int(len(sub)),
            "n_events": int(sub["event"].sum()),
            "n_members": int(sub["icpsr"].nunique()),
            "outcome": outcome,
            "extra_covars": list(extra_covars) if extra_covars else [],
        }
        if extra_covars:
            try:
                from scipy.stats import chi2
                joint_vars = ["bli"] + list(extra_covars)
                params = ctv.params_
                param_names = list(params.index)
                positions = [param_names.index(v) for v in joint_vars]
                V_raw = ctv.variance_matrix_
                V_arr = V_raw.values if hasattr(V_raw, "values") else np.asarray(V_raw)
                V_sub = V_arr[np.ix_(positions, positions)].astype(float)
                betas = params.values[positions].astype(float)
                wald = float(betas @ np.linalg.solve(V_sub, betas))
                p_joint = float(1 - chi2.cdf(wald, df=len(joint_vars)))
                out["wald_joint"] = {
                    "chi2": wald,
                    "df": len(joint_vars),
                    "p": p_joint,
                    "vars": joint_vars,
                    "individual_coefs": {v: float(params[v]) for v in joint_vars},
                    "individual_pvals": {v: float(summary.loc[v, "p"]) for v in joint_vars},
                }
            except Exception as e:
                out["wald_joint_error"] = str(e)
        return out
    except ImportError:
        return _cox_ph_via_r(sub, outcome)
    except Exception as e:
        return {"error": f"lifelines failure: {e}", "outcome": outcome}

def _cox_ph_via_r(sub, outcome):
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(sub[["icpsr", "start", "stop", "event", "bli", "ideology_distance", "seniority", "is_republican"]])
        ro.globalenv["panel_r"] = r_df
        ro.r("library(survival)")
        ro.r("m <- coxph(Surv(start, stop, event) ~ bli + ideology_distance + seniority + is_republican + cluster(icpsr), data=panel_r)")
        coef = float(np.asarray(ro.r("coef(m)['bli']")).item())
        se = float(np.asarray(ro.r("sqrt(vcov(m)['bli','bli'])")).item())
        p = float(np.asarray(ro.r("summary(m)$coefficients['bli','Pr(>|z|)']")).item())
        bli_iqr = float(sub["bli"].quantile(0.75) - sub["bli"].quantile(0.25))
        return {
            "engine": "R survival::coxph (cluster=icpsr)",
            "coef": coef,
            "se": se,
            "p": p,
            "hazard_ratio_unit": float(np.exp(coef)),
            "hazard_ratio_iqr": float(np.exp(coef * bli_iqr)),
            "bli_iqr": bli_iqr,
            "n_obs": int(len(sub)),
            "n_events": int(sub["event"].sum()),
            "n_members": int(sub["icpsr"].nunique()),
            "outcome": outcome,
        }
    except Exception as e:
        return {"error": f"rpy2 fallback failure: {e}", "outcome": outcome}

def lagged_fe_lpm(panel, outcome="departed_within_2", lag_var="bli_lag1"):
    try:
        import pyfixest as pf
    except ImportError:
        return {"error": "pyfixest not installed", "outcome": outcome}
    sub = panel.dropna(subset=[outcome, lag_var, "ideology_distance", "seniority"]).copy()
    sub[outcome] = sub[outcome].astype(float)
    try:
        m = pf.feols(
            f"{outcome} ~ {lag_var} + ideology_distance + seniority | icpsr + congress",
            data=sub,
            vcov={"CRV1": "icpsr"},
        )
        coef = float(m.coef()[lag_var])
        se = float(m.se()[lag_var])
        p = float(m.pvalue()[lag_var])
        iqr_eff = iqr_scale_effects(coef, se, sub, var=lag_var)
        return {
            "engine": "pyfixest two-way FE LPM",
            "lag_var": lag_var,
            "coef": coef,
            "se": se,
            "p": p,
            "n_obs": int(m._N),
            "n_members": int(sub["icpsr"].nunique()),
            "iqr_scale": iqr_eff,
            "outcome": outcome,
        }
    except Exception as e:
        return {"error": str(e), "outcome": outcome, "lag_var": lag_var}

def event_study_from_cohort_ddd():
    p = RESULTS_DIR / "cohort_ddd_results.json"
    if not p.exists():
        return {"error": "cohort_ddd_results.json not present"}
    with open(p) as f:
        d = json.load(f)
    event = d.get("fits", {}).get("event_study", {}).get("coefficients", {})
    summary = {}
    for k, v in event.items():
        summary[k] = {
            "estimate": v.get("estimate"),
            "p_value": v.get("p_value"),
            "ci_low": v.get("ci_low"),
            "ci_high": v.get("ci_high"),
        }
    return {"engine": "imported from cohort_ddd_results.json", "coefficients": summary}

def main():
    print("Building panel with BLI lags...")
    panel = build_panel_with_lag()
    print(f"  n={len(panel)} obs, {panel.icpsr.nunique()} members")

    vc = within_between_variance(panel, "bli")
    print("\n--- BLI variance decomposition ---")
    print(f"  Within-member  variance: {vc['var_within_member']:.3e}")
    print(f"  Between-member variance: {vc['var_between_member']:.3e}")
    print(f"  Intraclass rho:          {vc['rho_intraclass']:.3f}  (closer to 1 means trait-like)")

    print("\n=== A. Cox PH with time-varying BLI (full panel) ===")
    cox_full = cox_ph_time_varying(panel, outcome="departed_within_2")
    print(json.dumps(cox_full, indent=2))

    print("\n=== A2. Cox PH with distributed lag BLI(t,t-1,t-2) joint Wald ===")
    cox_lag = cox_ph_time_varying(
        panel.dropna(subset=["bli_lag1", "bli_lag2"]),
        outcome="departed_within_2",
        extra_covars=["bli_lag1", "bli_lag2"],
    )
    print(json.dumps(cox_lag, indent=2))

    print("\n=== A'. Cox PH on voluntary_or_primary subset ===")
    panel_v = panel.copy()
    if "did_not_seek_general" in panel_v.columns:
        panel_v["voluntary_or_primary"] = panel_v["did_not_seek_general"].fillna(0).astype(int)
        cox_vol = cox_ph_time_varying(panel_v, outcome="voluntary_or_primary")
        print(json.dumps(cox_vol, indent=2))
    else:
        cox_vol = {"error": "did_not_seek_general missing"}

    print("\n=== B. Lagged BLI (lag 1 congress) FE LPM on departed_within_2 ===")
    lag1 = lagged_fe_lpm(panel, outcome="departed_within_2", lag_var="bli_lag1")
    print(json.dumps(lag1, indent=2))

    print("\n=== B'. Lagged BLI (lag 2 congresses) FE LPM on departed_within_2 ===")
    lag2 = lagged_fe_lpm(panel, outcome="departed_within_2", lag_var="bli_lag2")
    print(json.dumps(lag2, indent=2))

    print("\n=== B''. BLI change (yoy) FE LPM on departed_within_2 ===")
    chg = lagged_fe_lpm(panel, outcome="departed_within_2", lag_var="bli_change")
    print(json.dumps(chg, indent=2))

    print("\n=== C. Event-study coefficients (from cohort_ddd) ===")
    es = event_study_from_cohort_ddd()
    print(json.dumps(es, indent=2))

    output = {
        "bli_variance_decomposition": vc,
        "cox_ph_full_panel": cox_full,
        "cox_ph_distributed_lag": cox_lag,
        "cox_ph_voluntary_or_primary": cox_vol,
        "lag1_fe_lpm": lag1,
        "lag2_fe_lpm": lag2,
        "bli_change_fe_lpm": chg,
        "event_study_from_cohort_ddd": es,
    }
    out = RESULTS_DIR / "within_member_robustness.json"
    out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved {out}")

if __name__ == "__main__":
    main()
