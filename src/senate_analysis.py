import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse, stats
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR, CONGRESSES, SEED

warnings.filterwarnings("ignore")

SENATE_DIR = DATA_DIR / "processed_senate"
LOOK_FORWARD = 1


def load_senate_congress(c):
    p = SENATE_DIR / f"congress_{c}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def fiedler_value(adjacency):
    A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    keep = degrees > 0
    if keep.sum() < 3:
        return 0.0, None
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    L = sparse.eye(A_sub.shape[0]) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    try:
        eigs, _ = eigsh(L, k=2, which="SM")
        return float(np.sort(eigs)[1]), None
    except Exception:
        return 0.0, None


def build_senate_panel():
    with open(RESULTS_DIR / "senate_bli_results.json") as f:
        bli_data = json.load(f)

    congress_rosters = {}
    for c in CONGRESSES:
        d = load_senate_congress(c)
        if d is not None:
            congress_rosters[c] = set(int(m) for m in d["member_ids"])

    rows = []
    for c in sorted(bli_data.keys()):
        if not c.isdigit():
            continue
        c_int = int(c)
        if c_int + LOOK_FORWARD > max(CONGRESSES):
            continue
        d = load_senate_congress(c_int)
        if d is None:
            continue
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

        prior_pool = set()
        for cc in prev:
            if cc in congress_rosters:
                prior_pool.update(congress_rosters[cc])

        for i, mid in enumerate(d["member_ids"]):
            mid_int = int(mid)
            p = int(party_codes[i])
            rows.append({
                "icpsr": mid_int,
                "congress": c_int,
                "chamber": "Senate",
                "state": str(d["state_abbrev"][i]),
                "departed_within_1": int(mid_int not in future),
                "is_freshman": int(mid_int not in prior_pool),
                "bli": float(bli_values[i]),
                "ideology_distance": float(abs(nom[i] - party_medians.get(p, 0.0))),
                "abs_nominate": float(abs(nom[i])),
                "seniority": seniority_map.get(mid_int, 0),
                "is_republican": int(p == 200),
                "nominate_dim1": float(nom[i]),
                "party": "R" if p == 200 else "D",
            })
    return pd.DataFrame(rows)


def gee_departure_regression(panel):
    df = panel.dropna(subset=["bli", "ideology_distance", "seniority", "is_republican", "departed_within_1"]).copy()
    formula_vars = ["bli", "ideology_distance", "seniority", "is_republican"]
    X = sm.add_constant(df[formula_vars])
    y = df["departed_within_1"]

    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Independence

    model = GEE(y, X, groups=df["icpsr"], family=Binomial(), cov_struct=Independence())
    result = model.fit()

    df["abs_nominate"] = df["abs_nominate"]
    X_abs = sm.add_constant(df[formula_vars + ["abs_nominate"]])
    model_abs = GEE(y, X_abs, groups=df["icpsr"], family=Binomial(), cov_struct=Independence())
    result_abs = model_abs.fit()

    return {
        "without_centrism": {
            "params": dict(zip(result.params.index.tolist(), result.params.values.tolist())),
            "pvalues": dict(zip(result.pvalues.index.tolist(), result.pvalues.values.tolist())),
            "bse": dict(zip(result.bse.index.tolist(), result.bse.values.tolist())),
            "n_obs": int(len(df)),
        },
        "with_centrism": {
            "params": dict(zip(result_abs.params.index.tolist(), result_abs.params.values.tolist())),
            "pvalues": dict(zip(result_abs.pvalues.index.tolist(), result_abs.pvalues.values.tolist())),
            "bse": dict(zip(result_abs.bse.index.tolist(), result_abs.bse.values.tolist())),
            "n_obs": int(len(df)),
        },
    }


def counterfactual_perturbation(panel, k_values=(5, 10, 15, 20)):
    with open(RESULTS_DIR / "senate_bli_results.json") as f:
        bli_data = json.load(f)

    results_by_congress = {}
    for c in sorted([int(k) for k in bli_data if k.isdigit()]):
        d = load_senate_congress(c)
        if d is None:
            continue
        adjacency = d["adjacency"]
        bli_values = np.array(bli_data[str(c)]["bli_values"])
        nominate = d["nominate_dim1"]

        base_fiedler, _ = fiedler_value(adjacency)
        n = adjacency.shape[0]

        congress_results = {"base_fiedler": base_fiedler, "n": int(n), "removals": {}}
        for k in k_values:
            if k >= n:
                continue
            top_bli_idx = np.argsort(-bli_values)[:k]
            mask = np.ones(n, dtype=bool)
            mask[top_bli_idx] = False
            f_no_bli, _ = fiedler_value(adjacency[np.ix_(mask, mask)])

            ideology_extremity = np.abs(nominate - np.median(nominate))
            top_ideology = np.argsort(-ideology_extremity)[:k]
            mask_ideo = np.ones(n, dtype=bool)
            mask_ideo[top_ideology] = False
            f_no_ideo, _ = fiedler_value(adjacency[np.ix_(mask_ideo, mask_ideo)])

            rng = np.random.default_rng(SEED + k + c)
            random_f = []
            for _ in range(50):
                rand_idx = rng.choice(n, size=k, replace=False)
                mask_rand = np.ones(n, dtype=bool)
                mask_rand[rand_idx] = False
                rf, _ = fiedler_value(adjacency[np.ix_(mask_rand, mask_rand)])
                random_f.append(rf)
            f_random = float(np.mean(random_f))

            congress_results["removals"][k] = {
                "remove_top_bli": f_no_bli,
                "remove_top_ideology": f_no_ideo,
                "remove_random_mean": f_random,
                "delta_bli": base_fiedler - f_no_bli,
                "delta_ideology": base_fiedler - f_no_ideo,
                "delta_random": base_fiedler - f_random,
            }
        results_by_congress[c] = congress_results
    return results_by_congress


def freshman_cohort_comparison(panel):
    cwa_congress = 104
    tea_congress = 112
    df_cwa = panel[(panel["congress"] == cwa_congress) & (panel["is_freshman"] == 1)]
    df_tea = panel[(panel["congress"] == tea_congress) & (panel["is_freshman"] == 1)]

    inc_cwa = panel[(panel["congress"] == cwa_congress) & (panel["is_freshman"] == 0)]
    inc_tea = panel[(panel["congress"] == tea_congress) & (panel["is_freshman"] == 0)]

    def safe_stat(arr, fn):
        if len(arr) == 0:
            return None
        return float(fn(arr))

    result = {
        "contract_with_america_104": {
            "n_freshmen": int(len(df_cwa)),
            "n_rep_fresh": int((df_cwa["party"] == "R").sum()),
            "n_dem_fresh": int((df_cwa["party"] == "D").sum()),
            "fresh_extremity_mean": safe_stat(df_cwa["abs_nominate"].values, np.mean),
            "fresh_bli_mean": safe_stat(df_cwa["bli"].values, np.mean),
            "incumbent_extremity_mean": safe_stat(inc_cwa["abs_nominate"].values, np.mean),
        },
        "tea_party_112": {
            "n_freshmen": int(len(df_tea)),
            "n_rep_fresh": int((df_tea["party"] == "R").sum()),
            "n_dem_fresh": int((df_tea["party"] == "D").sum()),
            "fresh_extremity_mean": safe_stat(df_tea["abs_nominate"].values, np.mean),
            "fresh_bli_mean": safe_stat(df_tea["bli"].values, np.mean),
            "incumbent_extremity_mean": safe_stat(inc_tea["abs_nominate"].values, np.mean),
        },
    }
    if len(df_cwa) > 1 and len(df_tea) > 1:
        ks_stat, ks_p = stats.ks_2samp(df_cwa["abs_nominate"].dropna(), df_tea["abs_nominate"].dropna())
        result["ks_extremity"] = {"statistic": float(ks_stat), "p_value": float(ks_p)}
    return result


def bicameral_synthetic_did():
    with open(RESULTS_DIR / "senate_spectral_results.json") as f:
        senate_spec = json.load(f)
    with open(RESULTS_DIR / "spectral_results.json") as f:
        house_spec = json.load(f)

    rows = []
    for c_str, info in senate_spec.items():
        if not c_str.isdigit():
            continue
        c = int(c_str)
        rows.append({"chamber": "Senate", "congress": c, "fiedler": info["fiedler"], "sri": info["sri"]})
    for c_str, info in house_spec.items():
        if not c_str.isdigit():
            continue
        c = int(c_str)
        if isinstance(info, dict) and "fiedler" in info:
            rows.append({"chamber": "House", "congress": c, "fiedler": info["fiedler"], "sri": info["sri"]})
    bc = pd.DataFrame(rows)
    bc = bc.dropna(subset=["fiedler"])

    bc["post_2010"] = (bc["congress"] >= 112).astype(int)
    bc["is_house"] = (bc["chamber"] == "House").astype(int)
    bc["interaction"] = bc["post_2010"] * bc["is_house"]

    X = sm.add_constant(bc[["post_2010", "is_house", "interaction"]])
    y = bc["fiedler"]
    model = sm.OLS(y, X).fit(cov_type="HC3")

    return {
        "panel_size": int(len(bc)),
        "panel": bc.to_dict(orient="records"),
        "ols_did": {
            "params": dict(zip(model.params.index.tolist(), model.params.values.tolist())),
            "bse": dict(zip(model.bse.index.tolist(), model.bse.values.tolist())),
            "pvalues": dict(zip(model.pvalues.index.tolist(), model.pvalues.values.tolist())),
        },
        "fiedler_house_pre_post": {
            "pre_2010": float(bc[(bc["chamber"] == "House") & (bc["post_2010"] == 0)]["fiedler"].mean()),
            "post_2010": float(bc[(bc["chamber"] == "House") & (bc["post_2010"] == 1)]["fiedler"].mean()),
        },
        "fiedler_senate_pre_post": {
            "pre_2010": float(bc[(bc["chamber"] == "Senate") & (bc["post_2010"] == 0)]["fiedler"].mean()),
            "post_2010": float(bc[(bc["chamber"] == "Senate") & (bc["post_2010"] == 1)]["fiedler"].mean()),
        },
    }


def plot_bicameral_trajectory(bicameral_result, out_path):
    import matplotlib.pyplot as plt
    panel = pd.DataFrame(bicameral_result["panel"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for chamber, color, marker in [("House", "#1f77b4", "o"), ("Senate", "#d62728", "s")]:
        sub = panel[panel["chamber"] == chamber].sort_values("congress")
        ax.plot(sub["congress"], sub["fiedler"], "-" + marker, color=color, label=chamber, markersize=5)
    ax.axvline(112, color="grey", linestyle=":", alpha=0.6, label="Tea Party (2010)")
    ax.set_xlabel("Congress")
    ax.set_ylabel("Fiedler value")
    ax.set_title("Bicameral Fiedler trajectory, House vs Senate")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_senate_perturbation(perturbation_results, out_path):
    import matplotlib.pyplot as plt
    congresses = sorted(perturbation_results.keys())
    delta_bli = []
    delta_ideo = []
    delta_random = []
    for c in congresses:
        r = perturbation_results[c]["removals"].get(10)
        if r is None:
            r = perturbation_results[c]["removals"].get(5)
        if r is None:
            continue
        delta_bli.append(r["delta_bli"])
        delta_ideo.append(r["delta_ideology"])
        delta_random.append(r["delta_random"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(congresses[:len(delta_bli)], delta_bli, "-o", color="#d62728", label="Remove top BLI")
    ax.plot(congresses[:len(delta_ideo)], delta_ideo, "-s", color="#1f77b4", label="Remove top ideology extremes")
    ax.plot(congresses[:len(delta_random)], delta_random, "-^", color="#7f7f7f", label="Remove random")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Congress")
    ax.set_ylabel(r"$\Delta$ Fiedler (10 members removed)")
    ax.set_title("Senate counterfactual perturbation")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("Building Senate panel...")
    panel = build_senate_panel()
    print(f"Panel: {len(panel)} senator-Congress obs, {panel['icpsr'].nunique()} unique senators")
    panel.to_csv(RESULTS_DIR / "senate_panel.csv", index=False)

    print("\nRunning Senate GEE departure regression...")
    gee_results = gee_departure_regression(panel)
    print(f"  Without centrism: BLI coef = {gee_results['without_centrism']['params'].get('bli', 'NA'):.4f}, "
          f"p = {gee_results['without_centrism']['pvalues'].get('bli', 'NA'):.3e}")
    print(f"  With centrism: BLI coef = {gee_results['with_centrism']['params'].get('bli', 'NA'):.4f}, "
          f"p = {gee_results['with_centrism']['pvalues'].get('bli', 'NA'):.3e}")

    print("\nRunning Senate counterfactual perturbation...")
    perturbation_results = counterfactual_perturbation(panel)
    for c in [100, 104, 107, 111, 112, 118]:
        if c in perturbation_results:
            r = perturbation_results[c]["removals"].get(10, {})
            if r:
                print(f"  {c}: base={perturbation_results[c]['base_fiedler']:.4f}, "
                      f"delta_BLI={r['delta_bli']:.4f}, delta_random={r['delta_random']:.4f}")
    plot_senate_perturbation(perturbation_results, FIGURES_DIR / "senate_perturbation.pdf")

    print("\nRunning Senate freshman cohort comparison...")
    cohort_results = freshman_cohort_comparison(panel)
    print(f"  104th (Contract) freshmen: {cohort_results['contract_with_america_104']['n_freshmen']} "
          f"(R: {cohort_results['contract_with_america_104']['n_rep_fresh']})")
    print(f"  112th (Tea Party) freshmen: {cohort_results['tea_party_112']['n_freshmen']} "
          f"(R: {cohort_results['tea_party_112']['n_rep_fresh']})")
    if "ks_extremity" in cohort_results:
        print(f"  KS test on extremity: D={cohort_results['ks_extremity']['statistic']:.3f}, "
              f"p={cohort_results['ks_extremity']['p_value']:.3f}")

    print("\nRunning bicameral synthetic DiD...")
    bicameral_results = bicameral_synthetic_did()
    print(f"  House pre-2010 Fiedler: {bicameral_results['fiedler_house_pre_post']['pre_2010']:.4f}")
    print(f"  House post-2010 Fiedler: {bicameral_results['fiedler_house_pre_post']['post_2010']:.4f}")
    print(f"  Senate pre-2010 Fiedler: {bicameral_results['fiedler_senate_pre_post']['pre_2010']:.4f}")
    print(f"  Senate post-2010 Fiedler: {bicameral_results['fiedler_senate_pre_post']['post_2010']:.4f}")
    print(f"  DiD interaction coefficient: {bicameral_results['ols_did']['params'].get('interaction'):.4f} "
          f"(p={bicameral_results['ols_did']['pvalues'].get('interaction'):.3f})")
    plot_bicameral_trajectory(bicameral_results, FIGURES_DIR / "bicameral_fiedler.pdf")

    output = {
        "panel_summary": {
            "n_obs": int(len(panel)),
            "n_senators": int(panel["icpsr"].nunique()),
            "departure_rate": float(panel["departed_within_1"].mean()),
            "fiedler_by_congress": panel.groupby("congress")["bli"].count().to_dict(),
        },
        "gee_departure_regression": gee_results,
        "counterfactual_perturbation": perturbation_results,
        "freshman_cohort_comparison": cohort_results,
        "bicameral_synthetic_did": {
            "panel_size": bicameral_results["panel_size"],
            "ols_did": bicameral_results["ols_did"],
            "fiedler_house_pre_post": bicameral_results["fiedler_house_pre_post"],
            "fiedler_senate_pre_post": bicameral_results["fiedler_senate_pre_post"],
        },
    }
    with open(RESULTS_DIR / "senate_analysis_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to {RESULTS_DIR / 'senate_analysis_results.json'}")


if __name__ == "__main__":
    main()
