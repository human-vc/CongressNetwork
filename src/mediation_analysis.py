import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, SEED, CONGRESSES

import rpy2.robjects as ro
from rpy2.robjects import packages, pandas2ri
from rpy2.robjects.conversion import localconverter

if not hasattr(np, "alltrue"):
    np.alltrue = np.all

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MEDIATION = packages.importr("mediation")
BASE = packages.importr("base")
STATS = packages.importr("stats")

ERAS = {
    "early": (100, 106),
    "middle": (107, 112),
    "late": (113, 116),
}

DEFAULT_BOOT_REPS = 2000
DEFAULT_SIMS = 1000
LOOK_FORWARD = 2
LOW_PCTL = 0.10
HIGH_PCTL = 0.90

def load_congress(c):
    p = PROCESSED_DIR / f"congress_{c}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)

def compute_bli_for_member(adjacency, idx):
    n = adjacency.shape[0]
    from scipy import sparse
    from scipy.sparse.linalg import eigsh

    def fiedler(adj):
        A = sparse.csr_matrix(adj)
        degrees = np.array(A.sum(axis=1)).flatten()
        keep = degrees > 0
        if keep.sum() < 3:
            return 0.0
        A_sub = A[np.ix_(keep, keep)]
        d_sub = degrees[keep]
        d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
        L = sparse.eye(A_sub.shape[0]) - d_inv_sqrt @ A_sub @ d_inv_sqrt
        try:
            eigs, _ = eigsh(L, k=2, which="SM")
            return float(np.sort(eigs)[1])
        except Exception:
            return 0.0

    base = fiedler(adjacency)
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    removed = fiedler(adjacency[np.ix_(mask, mask)])
    return base - removed

def load_bli_results():
    path = RESULTS_DIR / "bli_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def build_panel():
    bli_data = load_bli_results()
    if bli_data is None:
        raise RuntimeError("Run spectral_analysis.py first to generate bli_results.json")

    congress_rosters = {}
    for c in CONGRESSES:
        d = load_congress(c)
        if d is not None:
            congress_rosters[c] = set(int(m) for m in d["member_ids"])

    rows = []
    for c in sorted(bli_data.keys()):
        if not c.isdigit():
            continue
        c_int = int(c)
        if c_int + LOOK_FORWARD > max(CONGRESSES):
            continue
        d = load_congress(c_int)
        if d is None:
            continue
        ids = d["member_ids"]
        nom = d["nominate_dim1"]
        party = d["party_codes"]
        bli_values = np.array(bli_data[c]["bli_values"])
        future = congress_rosters.get(c_int + LOOK_FORWARD, set())

        prev_congresses = [cc for cc in CONGRESSES if cc < c_int]
        prev_centrism_map = {}
        for cc in prev_congresses:
            dprev = load_congress(cc)
            if dprev is None:
                continue
            for j, mid in enumerate(dprev["member_ids"]):
                prev_centrism_map[int(mid)] = float(-abs(dprev["nominate_dim1"][j]))

        seniority_map = {}
        for cc in prev_congresses:
            if cc in congress_rosters:
                for m in congress_rosters[cc]:
                    seniority_map[m] = seniority_map.get(m, 0) + 1

        for i, mid in enumerate(ids):
            mid_int = int(mid)
            departed = int(mid_int not in future)
            centrism = float(-abs(nom[i]))
            lagged_centrism = prev_centrism_map.get(mid_int, np.nan)
            rows.append({
                "icpsr": mid_int,
                "congress": c_int,
                "bli": float(bli_values[i]),
                "centrism": centrism,
                "lagged_centrism": lagged_centrism,
                "ideology_distance_from_zero": float(abs(nom[i])),
                "seniority": seniority_map.get(mid_int, 0),
                "is_republican": int(party[i] == 200),
                "departed": departed,
            })

    panel = pd.DataFrame(rows)

    pu_path = RESULTS_DIR / "party_unity.csv"
    if pu_path.exists():
        pu = pd.read_csv(pu_path)[["congress", "icpsr", "party_unity"]]
        pu["one_minus_party_unity"] = 1.0 - pu["party_unity"]
        panel = panel.merge(pu, on=["congress", "icpsr"], how="left")
    return panel

def push_panel_to_r(df, name="panel"):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].astype(str)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_clean)
    ro.globalenv[name] = r_df
    return r_df

def fit_mediation_r(df, treatment, mediator, outcome, controls, label, boot_reps=DEFAULT_BOOT_REPS, sims=DEFAULT_SIMS):
    raw_controls = [c.replace("factor(", "").replace("C(", "").replace(")", "") for c in controls]
    df_sub = df.dropna(subset=[treatment, mediator, outcome] + raw_controls).copy()
    if len(df_sub) < 50 or df_sub[outcome].nunique() < 2:
        return {"label": label, "error": "insufficient data", "n_obs": int(len(df_sub))}

    df_push = df_sub.copy()
    df_push["congress"] = df_push["congress"].astype(str)
    push_panel_to_r(df_push, name="panel_r")
    ro.r("panel_r$congress <- as.factor(panel_r$congress)")

    clean_controls = [c.replace("factor(", "").replace("C(", "").replace(")", "") for c in controls]
    control_terms = " + ".join(clean_controls) if clean_controls else "1"
    mediator_formula = f"{mediator} ~ {treatment} + {control_terms}"
    outcome_formula = f"{outcome} ~ {treatment} * {mediator} + {control_terms}"

    ro.r(f"med_model <- lm({mediator_formula}, data=panel_r)")
    ro.r(f"out_model <- glm({outcome_formula}, data=panel_r, family=binomial(link='probit'))")

    t_low = float(df_sub[treatment].quantile(LOW_PCTL))
    t_high = float(df_sub[treatment].quantile(HIGH_PCTL))

    ro.r(f"""
        set.seed({SEED})
        fit <- mediate(
            med_model, out_model,
            treat='{treatment}', mediator='{mediator}',
            control.value={t_low}, treat.value={t_high},
            boot=TRUE, boot.ci.type='bca', sims={sims}
        )
        fit_summary <- summary(fit)
    """)

    def safe_get(r_expr):
        try:
            obj = ro.r(r_expr)
            if obj is ro.NULL:
                return None
            arr = np.asarray(obj)
            if arr.size == 1:
                return float(arr.item())
            return arr.tolist()
        except Exception as e:
            return None

    acme_avg = safe_get("fit$d.avg")
    acme_avg_ci = safe_get("fit$d.avg.ci")
    acme_avg_p = safe_get("fit$d.avg.p")
    ade_avg = safe_get("fit$z.avg")
    ade_avg_ci = safe_get("fit$z.avg.ci")
    ade_avg_p = safe_get("fit$z.avg.p")
    total = safe_get("fit$tau.coef")
    total_ci = safe_get("fit$tau.ci")
    total_p = safe_get("fit$tau.p")
    prop_med = safe_get("fit$n.avg")
    prop_med_ci = safe_get("fit$n.avg.ci")
    acme_treated = safe_get("fit$d1")
    acme_control = safe_get("fit$d0")
    ade_treated = safe_get("fit$z1")
    ade_control = safe_get("fit$z0")

    sens_result = None
    try:
        ro.r(f"""
            out_model_noint <- glm({outcome} ~ {treatment} + {mediator} + {control_terms},
                                   data=panel_r, family=binomial(link='probit'))
            fit_noint <- mediate(
                med_model, out_model_noint,
                treat='{treatment}', mediator='{mediator}',
                control.value={t_low}, treat.value={t_high},
                boot=FALSE, sims={max(sims // 2, 500)}
            )
            sens <- medsens(fit_noint, rho.by=0.05, effect.type='indirect', sims={max(sims // 2, 500)})
            sens_rho_grid <- sens$rho
            sens_acme_grid <- sens$d0
            sens_acme_lower <- sens$d0.lb
            sens_acme_upper <- sens$d0.ub
            sens_rho_zero_idx <- which(diff(sign(sens$d0)) != 0)[1]
            sens_rho_zero <- if (!is.na(sens_rho_zero_idx)) sens$rho[sens_rho_zero_idx] else NA
            sens_r2_star <- sens$err.cr.d
            sens_r2_tilde <- sens$err.cr.d.tilde
        """)
        sens_result = {
            "rho_grid": safe_get("sens_rho_grid"),
            "acme_grid": safe_get("sens_acme_grid"),
            "acme_lower": safe_get("sens_acme_lower"),
            "acme_upper": safe_get("sens_acme_upper"),
            "breakdown_rho": safe_get("sens_rho_zero"),
            "r2_star_breakdown": safe_get("sens_r2_star"),
            "r2_tilde_breakdown": safe_get("sens_r2_tilde"),
        }
    except Exception as e:
        sens_result = {"error": str(e)}

    return {
        "label": label,
        "n_obs": int(len(df_sub)),
        "n_members": int(df_sub["icpsr"].nunique()) if "icpsr" in df_sub.columns else None,
        "departure_rate": float(df_sub[outcome].mean()),
        "treatment_contrast": {"low": t_low, "high": t_high},
        "acme_average": {"estimate": acme_avg, "ci": acme_avg_ci, "p_value": acme_avg_p},
        "acme_treated": {"estimate": acme_treated},
        "acme_control": {"estimate": acme_control},
        "ade_average": {"estimate": ade_avg, "ci": ade_avg_ci, "p_value": ade_avg_p},
        "ade_treated": {"estimate": ade_treated},
        "ade_control": {"estimate": ade_control},
        "total_effect": {"estimate": total, "ci": total_ci, "p_value": total_p},
        "proportion_mediated": {"estimate": prop_med, "ci": prop_med_ci},
        "sensitivity": sens_result,
    }

def cluster_bootstrap_mediation(df, treatment, mediator, outcome, controls, label, boot_reps=1000, sims=100):
    df_sub = df.dropna(subset=[treatment, mediator, outcome] + controls).copy()
    if len(df_sub) < 50:
        return {"label": label, "error": "insufficient data"}

    rng = np.random.default_rng(SEED)
    unique_ids = df_sub["icpsr"].unique()

    estimates = {"acme": [], "ade": [], "total": [], "prop_med": []}
    successful = 0
    for b in range(boot_reps):
        sampled = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        boot_df = pd.concat([df_sub[df_sub["icpsr"] == m] for m in sampled], ignore_index=True)
        try:
            push_panel_to_r(boot_df, name="boot_panel")
            control_terms = " + ".join(controls)
            mediator_formula = f"{mediator} ~ {treatment} + {control_terms}"
            outcome_formula = f"{outcome} ~ {treatment} * {mediator} + {control_terms}"
            ro.r(f"med_b <- lm({mediator_formula}, data=boot_panel)")
            ro.r(f"out_b <- glm({outcome_formula}, data=boot_panel, family=binomial(link='probit'))")
            t_low = float(boot_df[treatment].quantile(LOW_PCTL))
            t_high = float(boot_df[treatment].quantile(HIGH_PCTL))
            ro.r(f"""
                fit_b <- mediate(med_b, out_b,
                    treat='{treatment}', mediator='{mediator}',
                    control.value={t_low}, treat.value={t_high},
                    boot=FALSE, sims={sims})
            """)
            estimates["acme"].append(float(np.asarray(ro.r("fit_b$d.avg")).item()))
            estimates["ade"].append(float(np.asarray(ro.r("fit_b$z.avg")).item()))
            estimates["total"].append(float(np.asarray(ro.r("fit_b$tau.coef")).item()))
            estimates["prop_med"].append(float(np.asarray(ro.r("fit_b$n.avg")).item()))
            successful += 1
        except Exception:
            continue

    def percentile_ci(arr, alpha=0.05):
        if len(arr) < 10:
            return [None, None]
        return [float(np.percentile(arr, 100 * alpha / 2)), float(np.percentile(arr, 100 * (1 - alpha / 2)))]

    return {
        "label": label,
        "n_successful_bootstraps": successful,
        "n_attempted": boot_reps,
        "acme_cluster_ci": percentile_ci(estimates["acme"]),
        "ade_cluster_ci": percentile_ci(estimates["ade"]),
        "total_cluster_ci": percentile_ci(estimates["total"]),
        "prop_med_cluster_ci": percentile_ci(estimates["prop_med"]),
    }

def plot_sensitivity(sens_dict, title, out_path):
    import matplotlib.pyplot as plt
    if sens_dict is None or "rho_grid" not in sens_dict or sens_dict["rho_grid"] is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rho = np.asarray(sens_dict["rho_grid"], dtype=float)
    acme = np.asarray(sens_dict["acme_grid"], dtype=float)
    lo = np.asarray(sens_dict["acme_lower"], dtype=float) if sens_dict["acme_lower"] is not None else None
    hi = np.asarray(sens_dict["acme_upper"], dtype=float) if sens_dict["acme_upper"] is not None else None
    ax.plot(rho, acme, color="#1f77b4", linewidth=2, label="ACME")
    if lo is not None and hi is not None and lo.shape == acme.shape:
        ax.fill_between(rho, lo, hi, alpha=0.2, color="#1f77b4", label="95% CI")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle=":", linewidth=0.6)
    if sens_dict.get("breakdown_rho") is not None:
        ax.axvline(sens_dict["breakdown_rho"], color="red", linestyle="--", linewidth=1, label=f"breakdown rho = {sens_dict['breakdown_rho']:.2f}")
    ax.set_xlabel(r"$\rho$ (sensitivity parameter)")
    ax.set_ylabel("ACME")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_panel()
    panel_path = RESULTS_DIR / "mediation_panel.csv"
    panel.to_csv(panel_path, index=False)

    base_controls = ["seniority", "is_republican", "C(congress)"]

    controls_for_r = ["seniority", "is_republican", "factor(congress)"]
    print(f"Panel: {len(panel)} member-Congress observations, {panel['icpsr'].nunique()} unique members")
    print(f"Departure rate: {panel['departed'].mean():.3f}")
    print()

    main_result = None
    if "one_minus_party_unity" in panel.columns and panel["one_minus_party_unity"].notna().sum() > 100:
        cor_pu_bli = float(panel[["bli", "one_minus_party_unity"]].dropna().corr().iloc[0, 1])
        cor_centrism_bli = float(panel[["bli", "centrism"]].dropna().corr().iloc[0, 1])
        print(f"  cor(BLI, 1-party_unity) = {cor_pu_bli:.3f}")
        print(f"  cor(BLI, centrism)      = {cor_centrism_bli:.3f}  (near-tautology cf. CKLY 2010)")
        main_result = fit_mediation_r(
            panel, "bli", "one_minus_party_unity", "departed", controls_for_r,
            label="main_party_unity_defection",
        )
        print(f"Main (party-unity defection) ACME: {main_result['acme_average']['estimate']}")
        print(f"  CI: {main_result['acme_average']['ci']}")
        print(f"  Proportion mediated: {main_result['proportion_mediated']['estimate']}")
        plot_sensitivity(
            main_result.get("sensitivity"),
            "Mediation sensitivity: BLI -> defection (1 - party unity) -> departure",
            FIGURES_DIR / "mediation_sensitivity_party_unity.pdf",
        )

    full_result = fit_mediation_r(
        panel, "bli", "centrism", "departed", controls_for_r,
        label="comparator_centrism_near_tautology",
    )
    print(f"Comparator (centrism) ACME: {full_result['acme_average']['estimate']}")
    print(f"  Proportion mediated: {full_result['proportion_mediated']['estimate']}")
    print(f"  WARNING: this mediator is mechanically related to BLI; see paper for caveats.")
    print()

    plot_sensitivity(
        full_result.get("sensitivity"),
        "Mediation sensitivity: BLI -> centrism -> departure (near-tautology comparator)",
        FIGURES_DIR / "mediation_sensitivity_centrism_comparator.pdf",
    )

    era_results = {}
    for era, (lo, hi) in ERAS.items():
        sub = panel[(panel["congress"] >= lo) & (panel["congress"] <= hi)].copy()
        if len(sub) < 100:
            era_results[era] = {"error": "insufficient data", "n_obs": int(len(sub))}
            continue
        result = fit_mediation_r(
            sub, "bli", "centrism", "departed", controls_for_r,
            label=f"era_{era}",
            sims=500,
        )
        era_results[era] = result
        plot_sensitivity(
            result.get("sensitivity"),
            f"Mediation sensitivity ({era} era, Congresses {lo}-{hi})",
            FIGURES_DIR / f"mediation_sensitivity_{era}.pdf",
        )
        print(f"Era {era}: ACME = {result['acme_average']['estimate']}, "
              f"prop med = {result['proportion_mediated']['estimate']}, "
              f"breakdown rho = {result.get('sensitivity', {}).get('breakdown_rho')}")

    print()
    lagged_result = fit_mediation_r(
        panel.dropna(subset=["lagged_centrism"]),
        "bli", "lagged_centrism", "departed", controls_for_r,
        label="lagged_centrism_robustness",
        sims=500,
    )
    print(f"Lagged centrism ACME: {lagged_result['acme_average']['estimate']}")

    output = {
        "config": {
            "boot_reps": DEFAULT_BOOT_REPS,
            "sims": DEFAULT_SIMS,
            "look_forward_congresses": LOOK_FORWARD,
            "treatment_low_percentile": LOW_PCTL,
            "treatment_high_percentile": HIGH_PCTL,
            "panel_path": str(panel_path),
        },
        "panel_summary": {
            "n_obs": int(len(panel)),
            "n_members": int(panel["icpsr"].nunique()),
            "departure_rate": float(panel["departed"].mean()),
            "bli_mean": float(panel["bli"].mean()),
            "bli_std": float(panel["bli"].std()),
            "centrism_mean": float(panel["centrism"].mean()),
            "centrism_std": float(panel["centrism"].std()),
        },
        "main_party_unity_defection": main_result,
        "comparator_centrism_near_tautology": full_result,
        "by_era": era_results,
        "lagged_centrism_robustness": lagged_result,
    }

    out_path = RESULTS_DIR / "mediation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
