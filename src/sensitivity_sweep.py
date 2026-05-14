import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, SEED, CONGRESSES

import sensemakr as smkr

warnings.filterwarnings("ignore")

OUTCOME = "departed"
TREATMENT = "bli"
BASE_CONTROLS = ["ideology_distance", "seniority", "is_republican"]
CENTRISM_CONTROL = "abs_nominate"
LOOK_FORWARD = 2


def load_panel():
    bli_path = RESULTS_DIR / "bli_results.json"
    with open(bli_path) as f:
        bli_data = json.load(f)

    congress_rosters = {}
    for c in CONGRESSES:
        path = PROCESSED_DIR / f"congress_{c}.npz"
        if path.exists():
            d = np.load(path, allow_pickle=True)
            congress_rosters[c] = set(int(m) for m in d["member_ids"])

    rows = []
    for c in sorted(bli_data.keys()):
        if not c.isdigit():
            continue
        c_int = int(c)
        if c_int + LOOK_FORWARD > max(CONGRESSES):
            continue
        path = PROCESSED_DIR / f"congress_{c_int}.npz"
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=True)
        bli_values = np.array(bli_data[c]["bli_values"])
        future = congress_rosters.get(c_int + LOOK_FORWARD, set())
        prev = [cc for cc in CONGRESSES if cc < c_int]
        seniority_map = {}
        for cc in prev:
            if cc in congress_rosters:
                for m in congress_rosters[cc]:
                    seniority_map[m] = seniority_map.get(m, 0) + 1

        party_codes = d["party_codes"]
        nom = d["nominate_dim1"]
        party_medians = {}
        for p in [100, 200]:
            mask = party_codes == p
            if mask.any():
                party_medians[p] = float(np.median(nom[mask]))

        for i, mid in enumerate(d["member_ids"]):
            mid_int = int(mid)
            p = int(party_codes[i])
            rows.append({
                "icpsr": mid_int,
                "congress": c_int,
                "departed": int(mid_int not in future),
                "bli": float(bli_values[i]),
                "ideology_distance": float(abs(nom[i] - party_medians.get(p, 0.0))),
                "abs_nominate": float(abs(nom[i])),
                "seniority": seniority_map.get(mid_int, 0),
                "is_republican": int(p == 200),
                "nominate_dim1": float(nom[i]),
                "party": "R" if p == 200 else "D",
            })
    return pd.DataFrame(rows)


def fit_lpm(df, controls, treatment=TREATMENT, outcome=OUTCOME):
    keep = df[[treatment, outcome] + controls].dropna()
    X = sm.add_constant(keep[[treatment] + controls])
    y = keep[outcome]
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df.loc[keep.index, "icpsr"]})
    return model, len(keep)


def section_sensemakr(panel):
    base_model, n_base = fit_lpm(panel, BASE_CONTROLS)
    centrism_model, n_c = fit_lpm(panel, BASE_CONTROLS + [CENTRISM_CONTROL])

    bli_b = float(base_model.params[TREATMENT])
    bli_se = float(base_model.bse[TREATMENT])
    bli_t = bli_b / bli_se
    dof_base = int(base_model.df_resid)

    bli_centrism_b = float(centrism_model.params[TREATMENT])
    bli_centrism_se = float(centrism_model.bse[TREATMENT])
    bli_centrism_t = bli_centrism_b / bli_centrism_se
    dof_c = int(centrism_model.df_resid)

    rv_q1 = float(np.atleast_1d(smkr.robustness_value(t_statistic=bli_t, dof=dof_base, q=1.0, alpha=1.0)).item())
    rv_q05 = float(np.atleast_1d(smkr.robustness_value(t_statistic=bli_t, dof=dof_base, q=0.5, alpha=1.0)).item())
    rv_alpha = float(np.atleast_1d(smkr.robustness_value(t_statistic=bli_t, dof=dof_base, q=1.0, alpha=0.05)).item())
    partial_r2 = float(np.atleast_1d(smkr.partial_r2(t_statistic=bli_t, dof=dof_base)).item())

    benchmarks = {}
    for ctrl in BASE_CONTROLS:
        if ctrl in base_model.params.index:
            t_bench = float(base_model.params[ctrl] / base_model.bse[ctrl])
            r2_bench = float(np.atleast_1d(smkr.partial_r2(t_statistic=t_bench, dof=dof_base)).item())
            benchmarks[ctrl] = {
                "t": t_bench,
                "partial_r2": r2_bench,
                "ratio_to_RV_q1": r2_bench / rv_q1 if rv_q1 > 0 else None,
            }

    rv_centrism = float(np.atleast_1d(smkr.robustness_value(t_statistic=abs(bli_centrism_t), dof=dof_c, q=1.0, alpha=1.0)).item())
    partial_r2_centrism = float(np.atleast_1d(smkr.partial_r2(t_statistic=bli_centrism_t, dof=dof_c)).item())

    return {
        "main_specification": {
            "n_obs": n_base,
            "bli_coefficient": bli_b,
            "bli_se": bli_se,
            "bli_t_statistic": bli_t,
            "dof": dof_base,
            "partial_r2": partial_r2,
            "RV_q1": rv_q1,
            "RV_q0.5": rv_q05,
            "RV_alpha_0.05": rv_alpha,
            "benchmarks": benchmarks,
        },
        "centrism_adjusted_specification": {
            "n_obs": n_c,
            "bli_coefficient": bli_centrism_b,
            "bli_se": bli_centrism_se,
            "bli_t_statistic": bli_centrism_t,
            "dof": dof_c,
            "partial_r2": partial_r2_centrism,
            "RV_q1": rv_centrism,
        },
    }


def section_oster(panel):
    short_model, n_short = fit_lpm(panel, BASE_CONTROLS)
    long_model, n_long = fit_lpm(panel, BASE_CONTROLS + [CENTRISM_CONTROL])

    b_short = float(short_model.params[TREATMENT])
    b_long = float(long_model.params[TREATMENT])
    r2_short = float(short_model.rsquared)
    r2_long = float(long_model.rsquared)

    def oster_delta(r2_max):
        denom = (b_short - b_long) * (r2_max - r2_long)
        if abs(denom) < 1e-12:
            return None
        return b_long * (r2_long - r2_short) / denom

    def oster_beta_star(r2_max, delta=1.0):
        if abs(r2_long - r2_short) < 1e-12:
            return None
        return b_long - delta * (b_short - b_long) * (r2_max - r2_long) / (r2_long - r2_short)

    results = {}
    for mult in [1.0, 1.1, 1.3, 1.5, 2.0]:
        r2max = min(mult * r2_long, 1.0)
        results[f"r2_max_{mult}x_long"] = {
            "r2_max": r2max,
            "delta_star": oster_delta(r2max),
            "beta_star_at_delta_1": oster_beta_star(r2max, 1.0),
        }

    swap_results = {}
    for ctrl in BASE_CONTROLS:
        base_minus_ctrl = [c for c in BASE_CONTROLS if c != ctrl]
        short_minus, _ = fit_lpm(panel, base_minus_ctrl)
        long_minus, _ = fit_lpm(panel, BASE_CONTROLS)
        bs = float(short_minus.params[TREATMENT])
        bl = float(long_minus.params[TREATMENT])
        rs = float(short_minus.rsquared)
        rl = float(long_minus.rsquared)
        r2max = min(1.3 * rl, 1.0)
        try:
            d_swap = bl * (rl - rs) / ((bs - bl) * (r2max - rl)) if abs((bs - bl) * (r2max - rl)) > 1e-12 else None
        except Exception:
            d_swap = None
        swap_results[ctrl] = {
            "b_short": bs, "b_long": bl, "r2_short": rs, "r2_long": rl, "delta_at_1.3": d_swap,
        }

    return {
        "centrism_as_added_control": {
            "b_short_no_centrism": b_short,
            "b_long_with_centrism": b_long,
            "r2_short": r2_short,
            "r2_long": r2_long,
            "results_by_r2_max": results,
        },
        "swap_in_each_base_control": swap_results,
    }


def section_e_value(panel):
    base_model, _ = fit_lpm(panel, BASE_CONTROLS)
    bli_b = float(base_model.params[TREATMENT])
    bli_se = float(base_model.bse[TREATMENT])

    bli_iqr = float(panel[TREATMENT].quantile(0.9) - panel[TREATMENT].quantile(0.1))
    effect_iqr = bli_b * bli_iqr
    base_rate = float(panel[OUTCOME].mean())
    ratio = (base_rate + effect_iqr) / base_rate if base_rate > 0 else None
    if ratio is None or ratio <= 0:
        e_value = None
    else:
        rr = max(ratio, 1.0 / ratio)
        e_value = float(rr + np.sqrt(rr * (rr - 1)))

    ci_low_iqr = (bli_b - 1.96 * bli_se) * bli_iqr
    ratio_low = (base_rate + ci_low_iqr) / base_rate if base_rate > 0 else None
    if ratio_low is None or ratio_low <= 0:
        e_value_ci = None
    else:
        rr_low = max(ratio_low, 1.0 / ratio_low)
        e_value_ci = float(rr_low + np.sqrt(rr_low * (rr_low - 1)))

    return {
        "bli_coefficient": bli_b,
        "bli_iqr": bli_iqr,
        "effect_at_iqr": effect_iqr,
        "base_departure_rate": base_rate,
        "risk_ratio_at_iqr": ratio,
        "e_value_point": e_value,
        "e_value_ci_lower": e_value_ci,
        "interpretation": "minimum risk-ratio-scale association an unobserved confounder must have with both BLI and departure to nullify the effect",
    }


def section_specification_curve(panel):
    bli_definitions = {"contemp": TREATMENT}
    era_windows = {
        "full_100_116": (100, 116),
        "early_100_106": (100, 106),
        "middle_107_112": (107, 112),
        "late_113_116": (113, 116),
    }
    control_sets = {
        "minimal": [],
        "ideology": ["ideology_distance"],
        "centrism": ["abs_nominate"],
        "seniority": ["seniority"],
        "party": ["is_republican"],
        "ideo_plus_seniority": ["ideology_distance", "seniority"],
        "all_base": ["ideology_distance", "seniority", "is_republican"],
        "all_plus_centrism": ["ideology_distance", "seniority", "is_republican", "abs_nominate"],
    }
    samples = {
        "all": None,
        "republicans": ("party", "R"),
        "democrats": ("party", "D"),
    }
    estimators = ["lpm", "logit"]

    rows = []
    for bli_label, bli_col in bli_definitions.items():
        for era_label, (lo, hi) in era_windows.items():
            for ctrl_label, ctrls in control_sets.items():
                for sample_label, sample_spec in samples.items():
                    for estimator in estimators:
                        df = panel[(panel["congress"] >= lo) & (panel["congress"] <= hi)].copy()
                        if sample_spec is not None:
                            df = df[df[sample_spec[0]] == sample_spec[1]].copy()
                        if len(df) < 200:
                            continue
                        keep = df[[bli_col, OUTCOME] + ctrls].dropna()
                        if len(keep) < 100:
                            continue
                        X = sm.add_constant(keep[[bli_col] + ctrls])
                        y = keep[OUTCOME]
                        try:
                            if estimator == "lpm":
                                m = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df.loc[keep.index, "icpsr"]})
                            else:
                                m = sm.Logit(y, X).fit(disp=0)
                                m_se = m.get_robustcov_results(cov_type="cluster", groups=df.loc[keep.index, "icpsr"].values)
                                m = m_se
                            b = float(m.params[bli_col]) if bli_col in m.params else float(m.params[0])
                            se = float(m.bse[bli_col]) if bli_col in m.bse else float(m.bse[0])
                            p = float(m.pvalues[bli_col]) if bli_col in m.pvalues else float(m.pvalues[0])
                        except Exception as e:
                            continue
                        rows.append({
                            "bli_definition": bli_label,
                            "era": era_label,
                            "controls": ctrl_label,
                            "sample": sample_label,
                            "estimator": estimator,
                            "n": int(len(keep)),
                            "coefficient": b,
                            "std_error": se,
                            "p_value": p,
                            "significant_05": int(p < 0.05),
                            "sign_positive": int(b > 0),
                        })

    spec_df = pd.DataFrame(rows)
    spec_df = spec_df.sort_values("coefficient").reset_index(drop=True)
    spec_df["rank"] = np.arange(1, len(spec_df) + 1)
    return spec_df


def plot_specification_curve(spec_df, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2]}, sharex=True)
    ax = axes[0]
    colors = ["#1f77b4" if (s == 1) else "#cccccc" for s in spec_df["significant_05"]]
    ax.scatter(spec_df["rank"], spec_df["coefficient"], c=colors, s=10)
    ax.errorbar(spec_df["rank"], spec_df["coefficient"],
                yerr=1.96 * spec_df["std_error"], fmt="none",
                ecolor="lightgrey", alpha=0.4, linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("BLI coefficient")
    ax.set_title(f"Specification curve, {len(spec_df)} specifications")

    ax2 = axes[1]
    factors = ["era", "controls", "sample", "estimator"]
    factor_levels = [sorted(spec_df[f].unique()) for f in factors]
    all_levels = []
    for f, levels in zip(factors, factor_levels):
        for lev in levels:
            all_levels.append((f, lev))
    y_pos = np.arange(len(all_levels))
    for ranks_, level_info in enumerate(all_levels):
        f, lev = level_info
        mask = (spec_df[f] == lev).values
        x_pts = spec_df["rank"].values[mask]
        ax2.scatter(x_pts, np.full(len(x_pts), y_pos[ranks_]), s=2, c="black", alpha=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{f}: {lev}" for f, lev in all_levels], fontsize=7)
    ax2.set_xlabel("Specification (ordered by coefficient)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def section_causal_forest(panel):
    from econml.dml import CausalForestDML
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    df = panel.dropna(subset=[TREATMENT, OUTCOME] + BASE_CONTROLS + [CENTRISM_CONTROL]).copy()
    df["era_early"] = (df["congress"] <= 106).astype(int)
    df["era_middle"] = ((df["congress"] >= 107) & (df["congress"] <= 112)).astype(int)
    df["era_late"] = (df["congress"] >= 113).astype(int)

    hetero_cols = ["ideology_distance", "seniority", "is_republican", "abs_nominate",
                   "era_early", "era_middle", "era_late"]

    Y = df[OUTCOME].values.astype(int)
    T = df[TREATMENT].values.astype(float)
    X = df[hetero_cols].values
    W = df[hetero_cols].values
    groups = df["icpsr"].values

    est = CausalForestDML(
        model_y=GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=SEED),
        model_t=GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=SEED),
        discrete_outcome=True,
        discrete_treatment=False,
        n_estimators=1000,
        min_samples_leaf=20,
        max_depth=None,
        honest=True,
        inference=True,
        cv=3,
        random_state=SEED,
    )
    est.fit(Y, T, X=X, W=W, groups=groups)
    cate = est.effect(X)
    lower, upper = est.effect_interval(X, alpha=0.05)
    ate_inf = est.ate_inference(X)

    fi = est.feature_importances_

    df["cate"] = cate
    df["cate_lower"] = lower
    df["cate_upper"] = upper

    by_era = {}
    for label, mask in [("early", df["era_early"] == 1), ("middle", df["era_middle"] == 1), ("late", df["era_late"] == 1)]:
        sub = df[mask]
        by_era[label] = {
            "n": int(len(sub)),
            "cate_mean": float(sub["cate"].mean()),
            "cate_std": float(sub["cate"].std()),
            "cate_median": float(sub["cate"].median()),
        }

    by_party = {}
    for p in ["R", "D"]:
        sub = df[df["party"] == p]
        by_party[p] = {
            "n": int(len(sub)),
            "cate_mean": float(sub["cate"].mean()),
            "cate_std": float(sub["cate"].std()),
        }

    return {
        "n_obs": int(len(df)),
        "ate_estimate": float(ate_inf.mean_point),
        "ate_se": float(ate_inf.stderr_mean),
        "ate_ci": [float(ate_inf.conf_int_mean()[0]), float(ate_inf.conf_int_mean()[1])],
        "feature_importance": {col: float(fi[i]) for i, col in enumerate(hetero_cols)},
        "cate_summary": {
            "mean": float(np.mean(cate)),
            "std": float(np.std(cate)),
            "median": float(np.median(cate)),
            "p10": float(np.percentile(cate, 10)),
            "p90": float(np.percentile(cate, 90)),
        },
        "heterogeneity_by_era": by_era,
        "heterogeneity_by_party": by_party,
        "panel_with_cate_path": str(RESULTS_DIR / "cate_by_observation.csv"),
    }


def plot_causal_forest(panel_with_cate, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    df = panel_with_cate.dropna(subset=["cate"])

    axes[0].hist(df["cate"], bins=50, color="#1f77b4", alpha=0.7)
    axes[0].axvline(df["cate"].mean(), color="red", linestyle="--", label=f"ATE = {df['cate'].mean():.4f}")
    axes[0].set_xlabel("CATE (effect of unit BLI on Pr(departure))")
    axes[0].set_ylabel("Members")
    axes[0].set_title("Distribution of CATE")
    axes[0].legend(fontsize=9)

    era_data = []
    for label, mask in [("Early\n100-106", df["era_early"] == 1),
                         ("Middle\n107-112", df["era_middle"] == 1),
                         ("Late\n113-116", df["era_late"] == 1)]:
        era_data.append(df.loc[mask, "cate"].values)
    axes[1].boxplot(era_data, labels=["Early", "Middle", "Late"], showfliers=False)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("CATE")
    axes[1].set_title("Heterogeneity by era")

    party_data = [df.loc[df["party"] == "R", "cate"].values, df.loc[df["party"] == "D", "cate"].values]
    axes[2].boxplot(party_data, labels=["Republicans", "Democrats"], showfliers=False)
    axes[2].axhline(0, color="black", linewidth=0.6)
    axes[2].set_ylabel("CATE")
    axes[2].set_title("Heterogeneity by party")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("Loading panel...")
    panel = load_panel()
    print(f"Panel: {len(panel)} obs, {panel['icpsr'].nunique()} members")
    panel.to_csv(RESULTS_DIR / "sensitivity_panel.csv", index=False)

    print()
    print("Running sensemakr...")
    sensemakr_results = section_sensemakr(panel)
    print(f"  Main BLI t-stat: {sensemakr_results['main_specification']['bli_t_statistic']:.3f}")
    print(f"  RV_q1: {sensemakr_results['main_specification']['RV_q1']:.4f}")
    print(f"  Partial R²: {sensemakr_results['main_specification']['partial_r2']:.4f}")
    print(f"  Centrism-adjusted RV_q1: {sensemakr_results['centrism_adjusted_specification']['RV_q1']:.4f}")

    print()
    print("Running Oster delta bounds...")
    oster_results = section_oster(panel)
    centrism_o = oster_results["centrism_as_added_control"]
    print(f"  b_short (no centrism): {centrism_o['b_short_no_centrism']:.4f}")
    print(f"  b_long (with centrism): {centrism_o['b_long_with_centrism']:.4f}")
    print(f"  R² short: {centrism_o['r2_short']:.4f}, long: {centrism_o['r2_long']:.4f}")
    for k, v in centrism_o["results_by_r2_max"].items():
        print(f"  {k}: delta_star={v['delta_star']}")

    print()
    print("Running E-value...")
    e_value_results = section_e_value(panel)
    print(f"  Risk ratio at IQR: {e_value_results['risk_ratio_at_iqr']:.4f}")
    print(f"  E-value point: {e_value_results['e_value_point']}")
    print(f"  E-value CI lower: {e_value_results['e_value_ci_lower']}")

    print()
    print("Running specification curve...")
    spec_df = section_specification_curve(panel)
    spec_df.to_csv(RESULTS_DIR / "specification_curve.csv", index=False)
    print(f"  Total specifications: {len(spec_df)}")
    print(f"  Median coefficient: {spec_df['coefficient'].median():.4f}")
    print(f"  Share significant (p<0.05): {spec_df['significant_05'].mean()*100:.1f}%")
    print(f"  Share with positive sign: {spec_df['sign_positive'].mean()*100:.1f}%")
    plot_specification_curve(spec_df, FIGURES_DIR / "specification_curve.pdf")

    spec_summary = {
        "n_specifications": int(len(spec_df)),
        "coefficient_median": float(spec_df["coefficient"].median()),
        "coefficient_iqr": [float(spec_df["coefficient"].quantile(0.25)), float(spec_df["coefficient"].quantile(0.75))],
        "share_significant_05": float(spec_df["significant_05"].mean()),
        "share_positive_sign": float(spec_df["sign_positive"].mean()),
        "share_significant_positive": float(((spec_df["significant_05"] == 1) & (spec_df["sign_positive"] == 1)).mean()),
    }

    print()
    print("Running causal forest...")
    cf_results = section_causal_forest(panel)
    print(f"  ATE: {cf_results['ate_estimate']:.4f} (SE {cf_results['ate_se']:.4f})")
    print(f"  CATE range: [{cf_results['cate_summary']['p10']:.4f}, {cf_results['cate_summary']['p90']:.4f}]")
    print(f"  Top features for heterogeneity:")
    sorted_fi = sorted(cf_results["feature_importance"].items(), key=lambda x: -x[1])
    for name, val in sorted_fi[:5]:
        print(f"    {name}: {val:.4f}")
    print(f"  By era: {cf_results['heterogeneity_by_era']}")

    df_for_plot = panel.dropna(subset=[TREATMENT, OUTCOME] + BASE_CONTROLS + [CENTRISM_CONTROL]).copy()
    df_for_plot["era_early"] = (df_for_plot["congress"] <= 106).astype(int)
    df_for_plot["era_middle"] = ((df_for_plot["congress"] >= 107) & (df_for_plot["congress"] <= 112)).astype(int)
    df_for_plot["era_late"] = (df_for_plot["congress"] >= 113).astype(int)

    output = {
        "sensemakr": sensemakr_results,
        "oster": oster_results,
        "e_value": e_value_results,
        "specification_curve": spec_summary,
        "causal_forest": cf_results,
    }
    with open(RESULTS_DIR / "sensitivity_sweep_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"All results saved to {RESULTS_DIR / 'sensitivity_sweep_results.json'}")


if __name__ == "__main__":
    main()
