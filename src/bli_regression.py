import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR, CONGRESSES

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Independence, Exchangeable, Autoregressive

try:
    import pyfixest as pf
    _HAS_PYFIXEST = True
except Exception:
    _HAS_PYFIXEST = False


def attach_departure_types(panel):
    """Merge in the 5-way departure classification when available.
    Adds: in_next, departed_voluntary_or_primary, lost_general, died_in_office,
          ran_for_higher_office, departure_type."""
    p = RESULTS_DIR / "departure_types.csv"
    if not p.exists():
        return panel
    dt = pd.read_csv(p)
    keep = ["icpsr", "congress", "in_next", "did_not_seek_general",
            "lost_general", "died_in_office", "ran_for_higher_office",
            "departure_type"]
    return panel.merge(dt[keep], on=["icpsr", "congress"], how="left")


def build_panel():
    bli_path = RESULTS_DIR / "bli_results.json"
    with open(bli_path) as f:
        bli_data = json.load(f)

    congress_rosters = {}
    for c in CONGRESSES:
        path = PROCESSED_DIR / f"congress_{c}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        congress_rosters[c] = set(data["member_ids"].tolist())

    rows = []
    for c in CONGRESSES:
        cs = str(c)
        if cs not in bli_data:
            continue
        if c + 2 > max(CONGRESSES):
            continue

        path = PROCESSED_DIR / f"congress_{c}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)

        member_ids = data["member_ids"]
        party_codes = data["party_codes"]
        nominate_dim1 = data["nominate_dim1"]
        bli_values = np.array(bli_data[cs]["bli_values"])

        party_medians = {}
        for p in [100, 200]:
            p_mask = party_codes == p
            if p_mask.any():
                party_medians[p] = np.median(nominate_dim1[p_mask])
            else:
                party_medians[p] = 0.0

        prev_congresses = [cc for cc in CONGRESSES if cc < c]
        prev_roster = set()
        for cc in prev_congresses:
            if cc in congress_rosters:
                prev_roster.update(congress_rosters[cc])

        future_c = c + 2
        future_roster = congress_rosters.get(future_c, set())

        for i, icpsr in enumerate(member_ids):
            departed = int(icpsr not in future_roster)
            party = int(party_codes[i])
            nom = float(nominate_dim1[i])
            ideology_distance = abs(nom - party_medians.get(party, 0.0))
            bli = float(bli_values[i])

            seniority = 0
            for cc in prev_congresses:
                if cc in congress_rosters and int(icpsr) in congress_rosters[cc]:
                    seniority += 1

            is_republican = int(party == 200)

            rows.append({
                "congress": c,
                "icpsr": int(icpsr),
                "departed_within_2": departed,
                "bli": bli,
                "ideology_distance": ideology_distance,
                "seniority": seniority,
                "is_republican": is_republican,
                "nominate_dim1": nom,
            })

    panel = pd.DataFrame(rows)
    panel = attach_departure_types(panel)
    return panel


def iqr_scale_effects(coef, se, panel, var="bli", base_rate=None):
    """Translate a logit coefficient into IQR-scale magnitudes that reviewers
    can read without unit confusion. Returns effect_iqr (logit), odds_ratio_iqr,
    risk_ratio_iqr (at base_rate), and E-value.
    """
    x = panel[var].dropna().values
    iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
    eff = coef * iqr
    or_iqr = float(np.exp(eff))
    if base_rate is None:
        base_rate = float(panel["departed_within_2"].mean())
    # Risk ratio approximation from odds ratio + base rate (VanderWeele 2017).
    rr_iqr = or_iqr / (1 - base_rate + base_rate * or_iqr)
    # E-value (VanderWeele & Ding 2017).
    rr_e = max(rr_iqr, 1 / rr_iqr)
    e_value = rr_e + np.sqrt(rr_e * (rr_e - 1)) if rr_e >= 1 else np.nan
    se_eff_iqr = abs(se * iqr)
    return {
        "bli_iqr": iqr,
        "effect_iqr_logit": float(eff),
        "effect_iqr_se": float(se_eff_iqr),
        "odds_ratio_iqr": or_iqr,
        "risk_ratio_iqr": float(rr_iqr),
        "base_rate": base_rate,
        "e_value": float(e_value) if np.isfinite(e_value) else None,
    }


def fit_member_fe_lpm(panel, outcome="departed_within_2"):
    """Linear probability model with two-way fixed effects (member + congress).
    Singletons are dropped by pyfixest. Standard errors clustered by icpsr.
    """
    if not _HAS_PYFIXEST:
        return {"error": "pyfixest not installed"}
    sub = panel.dropna(subset=[outcome, "bli", "ideology_distance", "seniority"]).copy()
    sub[outcome] = sub[outcome].astype(float)
    try:
        m = pf.feols(
            f"{outcome} ~ bli + ideology_distance + seniority | icpsr + congress",
            data=sub,
            vcov={"CRV1": "icpsr"},
        )
        coef = float(m.coef()["bli"])
        se = float(m.se()["bli"])
        pval = float(m.pvalue()["bli"])
        iqr_eff = iqr_scale_effects(coef, se, sub, var="bli")
        return {
            "coef": coef, "se": se, "p": pval,
            "n_obs": int(m._N),
            "n_members": int(sub["icpsr"].nunique()),
            "iqr_scale": iqr_eff,
            "outcome": outcome,
            "note": "LPM coefficients are probability points; iqr_scale.effect_iqr_logit is on the LPM (linear-probability) scale.",
        }
    except Exception as e:
        return {"error": str(e), "outcome": outcome}


def run_gee(panel, include_bli=True, label=""):
    if include_bli:
        formula_vars = ["bli", "ideology_distance", "seniority", "is_republican"]
    else:
        formula_vars = ["ideology_distance", "seniority", "is_republican"]

    X = panel[formula_vars].copy()
    X = sm.add_constant(X)
    y = panel["departed_within_2"]
    groups = panel["icpsr"]

    model = GEE(
        y, X, groups=groups,
        family=Binomial(),
        cov_struct=Independence(),
    )
    result = model.fit()
    print(f"\n{'='*60}")
    print(f"GEE: {label}")
    print(f"{'='*60}")
    print(result.summary())
    return result


def run_interaction_model(panel):
    X = panel[["bli", "ideology_distance", "seniority", "is_republican"]].copy()
    X["bli_x_republican"] = X["bli"] * X["is_republican"]
    X = sm.add_constant(X)
    y = panel["departed_within_2"]
    groups = panel["icpsr"]

    model = GEE(
        y, X, groups=groups,
        family=Binomial(),
        cov_struct=Independence(),
    )
    result = model.fit()
    print(f"\n{'='*60}")
    print("GEE: BLI x Party Interaction")
    print(f"{'='*60}")
    print(result.summary())
    return result


def run_era_splits(panel):
    eras = {
        "early (100-106)": (100, 106),
        "middle (107-112)": (107, 112),
        "late (113-116)": (113, 116),
    }
    era_results = {}
    for era_name, (start, end) in eras.items():
        sub = panel[(panel["congress"] >= start) & (panel["congress"] <= end)]
        if len(sub) < 50 or sub["departed_within_2"].nunique() < 2:
            print(f"Skipping era {era_name}: insufficient data")
            continue
        result = run_gee(sub, include_bli=True, label=f"Era: {era_name}")
        era_results[era_name] = {
            "n": len(sub),
            "n_departed": int(sub["departed_within_2"].sum()),
            "params": dict(zip(result.params.index.tolist(), result.params.values.tolist())),
            "pvalues": dict(zip(result.pvalues.index.tolist(), result.pvalues.values.tolist())),
        }
    return era_results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building member-congress panel...")
    panel = build_panel()
    print(f"Panel: {len(panel)} member-congress observations")
    print(f"Departed: {panel['departed_within_2'].sum()} ({panel['departed_within_2'].mean()*100:.1f}%)")
    print(f"Unique members: {panel['icpsr'].nunique()}")

    result_no_bli = run_gee(panel, include_bli=False, label="Without BLI")
    result_with_bli = run_gee(panel, include_bli=True, label="With BLI")

    interaction_result = run_interaction_model(panel)
    era_results = run_era_splits(panel)

    print("\n--- Absolute NOMINATE Robustness ---")
    panel["abs_nominate"] = panel["nominate_dim1"].abs()
    fvars_abs = ["bli", "ideology_distance", "abs_nominate", "seniority", "is_republican"]
    X_abs = sm.add_constant(panel[fvars_abs])
    y_abs = panel["departed_within_2"]
    g_abs = panel["icpsr"]
    m_abs = GEE(y_abs, X_abs, groups=g_abs, family=Binomial(), cov_struct=Independence())
    r_abs = m_abs.fit()
    abs_nominate_result = {
        "params": dict(zip(r_abs.params.index.tolist(), r_abs.params.values.tolist())),
        "pvalues": dict(zip(r_abs.pvalues.index.tolist(), r_abs.pvalues.values.tolist())),
        "bse": dict(zip(r_abs.bse.index.tolist(), r_abs.bse.values.tolist())),
    }
    print(f"  BLI: coef={r_abs.params['bli']:.1f}, p={r_abs.pvalues['bli']:.2e}")
    print(f"  abs_nominate: coef={r_abs.params['abs_nominate']:.3f}, p={r_abs.pvalues['abs_nominate']:.2e}")

    print("\n--- GEE Correlation Structure Sensitivity ---")
    corr_sensitivity = {}

    panel_sorted = panel.sort_values(["icpsr", "congress"]).reset_index(drop=True)
    formula_vars = ["bli", "ideology_distance", "seniority", "is_republican"]

    congress_vals = sorted(panel_sorted["congress"].unique())
    congress_to_time = {c: i for i, c in enumerate(congress_vals)}
    panel_sorted["time"] = panel_sorted["congress"].map(congress_to_time)

    scale_vars = ["bli", "ideology_distance", "seniority"]
    means = {v: panel_sorted[v].mean() for v in scale_vars}
    stds = {v: panel_sorted[v].std() for v in scale_vars}

    panel_std = panel_sorted.copy()
    for v in scale_vars:
        if stds[v] > 0:
            panel_std[v] = (panel_std[v] - means[v]) / stds[v]

    X_std = sm.add_constant(panel_std[formula_vars].values.astype(np.float64))
    y_np = panel_std["departed_within_2"].values.astype(np.float64)
    groups_np = panel_std["icpsr"].values
    time_np = panel_std["time"].values
    col_names = ["const"] + formula_vars

    indep_start = None

    import warnings
    warnings.filterwarnings("ignore")

    panel_sorted["bli_scaled"] = panel_sorted["bli"] * 1000.0
    fvars_scaled = ["bli_scaled", "ideology_distance", "seniority", "is_republican"]

    X_full = sm.add_constant(panel_sorted[fvars_scaled].values.astype(np.float64))
    y_full = panel_sorted["departed_within_2"].values.astype(np.float64)
    g_full = panel_sorted["icpsr"].values
    col_names_s = ["const"] + fvars_scaled

    counts = panel_sorted.groupby("icpsr").size()
    bal_ids = counts[(counts >= 2) & (counts <= 6)].index
    panel_bal = panel_sorted[panel_sorted["icpsr"].isin(bal_ids)].reset_index(drop=True)
    X_bal = sm.add_constant(panel_bal[fvars_scaled].values.astype(np.float64))
    y_bal = panel_bal["departed_within_2"].values.astype(np.float64)
    g_bal = panel_bal["icpsr"].values
    time_bal = panel_bal["time"].values.astype(np.float64)

    configs = [
        ("independence", Independence, False),
        ("exchangeable", Exchangeable, True),
        ("ar1", Autoregressive, True),
    ]

    for struct_name, struct_cls, use_balanced in configs:
        try:
            kw = {}
            if struct_name == "ar1":
                kw["time"] = time_bal if use_balanced else time_np

            X_use = X_bal if use_balanced else X_full
            y_use = y_bal if use_balanced else y_full
            g_use = g_bal if use_balanced else g_full
            n_use = len(y_use)

            model = GEE(
                y_use, X_use, groups=g_use,
                family=Binomial(),
                cov_struct=struct_cls(),
                **kw,
            )
            res = model.fit(maxiter=100)

            params_raw = dict(zip(col_names_s, res.params.tolist()))
            pvals = dict(zip(col_names_s, res.pvalues.tolist()))
            bses_raw = dict(zip(col_names_s, res.bse.tolist()))

            if np.isnan(params_raw["bli_scaled"]):
                raise ValueError("NaN coefficients")

            params_orig = {
                "const": params_raw["const"],
                "bli": params_raw["bli_scaled"] * 1000,
                "ideology_distance": params_raw["ideology_distance"],
                "seniority": params_raw["seniority"],
                "is_republican": params_raw["is_republican"],
            }
            bses_orig = {
                "const": bses_raw["const"],
                "bli": bses_raw["bli_scaled"] * 1000,
                "ideology_distance": bses_raw["ideology_distance"],
                "seniority": bses_raw["seniority"],
                "is_republican": bses_raw["is_republican"],
            }
            pvals_orig = {
                "const": pvals["const"],
                "bli": pvals["bli_scaled"],
                "ideology_distance": pvals["ideology_distance"],
                "seniority": pvals["seniority"],
                "is_republican": pvals["is_republican"],
            }

            corr_sensitivity[struct_name] = {
                "params": params_orig,
                "pvalues": pvals_orig,
                "bse": bses_orig,
                "n": n_use,
                "balanced_subsample": use_balanced,
            }
            print(f"  {struct_name} (n={n_use}): BLI coef={params_orig['bli']:.1f}, "
                  f"p={pvals_orig['bli']:.2e}")
        except Exception as e:
            print(f"  {struct_name}: failed ({e})")
            corr_sensitivity[struct_name] = {"error": str(e)}

    # --- Headline IQR-scale effect on the main GEE coefficient ---
    bli_coef_main = float(result_with_bli.params["bli"])
    bli_se_main = float(result_with_bli.bse["bli"])
    iqr_main = iqr_scale_effects(bli_coef_main, bli_se_main, panel, var="bli")
    print(f"\n--- Headline IQR-scale effects ---")
    print(f"  BLI IQR: {iqr_main['bli_iqr']:.5f}")
    print(f"  Effect at IQR (logit): {iqr_main['effect_iqr_logit']:.3f}")
    print(f"  Odds-ratio at IQR: {iqr_main['odds_ratio_iqr']:.3f}")
    print(f"  Risk-ratio at IQR: {iqr_main['risk_ratio_iqr']:.3f}")
    print(f"  E-value: {iqr_main['e_value']}")

    # --- Member + Congress fixed-effects LPM (pyfixest) ---
    print("\n--- Two-way FE LPM (member + congress) on departed_within_2 ---")
    fe_default = fit_member_fe_lpm(panel, outcome="departed_within_2")
    print(f"  result: {fe_default}")

    # --- FE LPM on substantive 'departed by choice / primary' subset ---
    fe_voluntary = None
    if "did_not_seek_general" in panel.columns:
        panel_v = panel.copy()
        panel_v["voluntary_or_primary"] = panel_v["did_not_seek_general"].fillna(0).astype(int)
        print("\n--- FE LPM on did_not_seek_general (voluntary + primary loss) ---")
        fe_voluntary = fit_member_fe_lpm(panel_v, outcome="voluntary_or_primary")
        print(f"  result: {fe_voluntary}")

    output = {
        "panel_stats": {
            "n_observations": len(panel),
            "n_departed": int(panel["departed_within_2"].sum()),
            "n_unique_members": int(panel["icpsr"].nunique()),
            "departure_rate": float(panel["departed_within_2"].mean()),
        },
        "headline_iqr_scale_effects": iqr_main,
        "member_congress_fe_lpm": fe_default,
        "member_congress_fe_lpm_voluntary_or_primary": fe_voluntary,
        "without_bli": {
            "params": dict(zip(result_no_bli.params.index.tolist(), result_no_bli.params.values.tolist())),
            "pvalues": dict(zip(result_no_bli.pvalues.index.tolist(), result_no_bli.pvalues.values.tolist())),
            "bse": dict(zip(result_no_bli.bse.index.tolist(), result_no_bli.bse.values.tolist())),
        },
        "with_bli": {
            "params": dict(zip(result_with_bli.params.index.tolist(), result_with_bli.params.values.tolist())),
            "pvalues": dict(zip(result_with_bli.pvalues.index.tolist(), result_with_bli.pvalues.values.tolist())),
            "bse": dict(zip(result_with_bli.bse.index.tolist(), result_with_bli.bse.values.tolist())),
        },
        "interaction": {
            "params": dict(zip(interaction_result.params.index.tolist(), interaction_result.params.values.tolist())),
            "pvalues": dict(zip(interaction_result.pvalues.index.tolist(), interaction_result.pvalues.values.tolist())),
        },
        "era_splits": era_results,
        "correlation_sensitivity": corr_sensitivity,
        "abs_nominate_robustness": abs_nominate_result,
    }

    with open(RESULTS_DIR / "bli_regression_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nBLI regression analysis complete.")


if __name__ == "__main__":
    main()
