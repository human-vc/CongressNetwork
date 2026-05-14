import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, SEED, CONGRESSES

from scipy import sparse, stats
from scipy.sparse.linalg import eigsh

if not hasattr(np, "alltrue"):
    np.alltrue = np.all

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


N_PERMUTATIONS = 9999
N_PROPENSITY_REPLICATES = 2000
BLI_HIGH_PERCENTILE = 66.67
BLI_LOW_PERCENTILE = 33.33
EXOGENEITY_LEVELS = {
    "strict": {"exogenous_flag": 1, "max_age": 85, "exclude_retired": True},
    "borderline": {"exogenous_flag": [1], "max_age": 90, "exclude_retired": False, "include_short_cancer": True},
    "permissive": {"exogenous_flag": [0, 1], "max_age": 100, "exclude_retired": False},
}


def load_congress(c):
    p = PROCESSED_DIR / f"congress_{c}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def fiedler_value(adjacency):
    A = sparse.csr_matrix(adjacency)
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


def load_bli_results():
    path = RESULTS_DIR / "bli_results.json"
    if not path.exists():
        raise RuntimeError("Run spectral_analysis.py first")
    with open(path) as f:
        return json.load(f)


def cross_party_rate(agreement, party_codes, idx):
    row = agreement[idx]
    other = party_codes != party_codes[idx]
    vals = row[other]
    vals = vals[vals > 0]
    if len(vals) < 3:
        return np.nan
    return float(vals.mean())


def load_deaths():
    path = DATA_DIR / "house_deaths_1987_2025.csv"
    deaths = pd.read_csv(path)
    deaths["date_of_death"] = pd.to_datetime(deaths["date_of_death"])
    deaths["age_at_death"] = deaths["age_at_death"].astype(int)
    deaths["congress"] = deaths["congress"].astype(int)
    return deaths


def match_deaths_to_voteview(deaths):
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    house = members[members["chamber"] == "House"].copy()
    house["bioname_upper"] = house["bioname"].str.upper().str.strip()

    matched = []
    for _, d in deaths.iterrows():
        target_name = d["name"].upper().strip()
        candidates = house[house["congress"] == d["congress"]].copy()
        last_name = target_name.split(",")[0].strip()
        candidates["score"] = candidates["bioname_upper"].apply(
            lambda x: 1.0 if last_name in x else (0.5 if any(part in x for part in target_name.split()[:2]) else 0.0)
        )
        best = candidates.sort_values("score", ascending=False).iloc[0] if len(candidates) else None
        if best is not None and best["score"] > 0:
            matched.append({
                **d.to_dict(),
                "icpsr": int(best["icpsr"]),
                "matched_bioname": best["bioname"],
                "matched_score": float(best["score"]),
            })
    return pd.DataFrame(matched)


def filter_exogenous(deaths_df, criterion="strict"):
    df = deaths_df.copy()
    if criterion == "strict":
        df = df[(df["exogenous_flag"] == 1) & (df["age_at_death"] <= 85) & (df["retirement_announced"] == 0)]
    elif criterion == "borderline":
        keep_cause = df["cause_category"].isin(["sudden", "sudden_illness", "sudden_surgical", "accident", "cancer_short"])
        df = df[keep_cause & (df["age_at_death"] <= 90) & (df["retirement_announced"] == 0)]
    elif criterion == "permissive":
        df = df[df["age_at_death"] <= 100]
    return df


def compute_bli_at_death(deaths_df, bli_data):
    rows = []
    for _, d in deaths_df.iterrows():
        c = int(d["congress"])
        key = str(c)
        if key not in bli_data:
            continue
        info = bli_data[key]
        member_ids = info["member_ids"]
        bli_values = info["bli_values"]
        if int(d["icpsr"]) not in [int(m) for m in member_ids]:
            continue
        idx = [int(m) for m in member_ids].index(int(d["icpsr"]))
        bli = float(bli_values[idx])
        all_bli = np.array(bli_values)
        high_threshold = np.percentile(all_bli, BLI_HIGH_PERCENTILE)
        low_threshold = np.percentile(all_bli, BLI_LOW_PERCENTILE)
        if bli >= high_threshold:
            bli_category = "high"
        elif bli <= low_threshold:
            bli_category = "low"
        else:
            bli_category = "middle"
        rows.append({
            **d.to_dict(),
            "deceased_bli": bli,
            "bli_high_threshold": float(high_threshold),
            "bli_low_threshold": float(low_threshold),
            "bli_category": bli_category,
            "bli_rank": float(stats.rankdata(all_bli)[idx] / len(all_bli)),
        })
    return pd.DataFrame(rows)


def build_member_outcomes(deaths_df, agreement_threshold=0.5):
    bli_data = load_bli_results()
    outcomes = []

    for _, d in deaths_df.iterrows():
        c = int(d["congress"])
        next_c = c + 1
        d_curr = load_congress(c)
        d_next = load_congress(next_c)
        if d_curr is None or d_next is None:
            continue

        curr_ids = d_curr["member_ids"]
        next_ids = d_next["member_ids"]
        curr_id_list = [int(m) for m in curr_ids]
        next_id_list = [int(m) for m in next_ids]
        deceased_icpsr = int(d["icpsr"])

        if deceased_icpsr not in curr_id_list:
            continue
        deceased_idx = curr_id_list.index(deceased_icpsr)
        deceased_party_code = int(d_curr["party_codes"][deceased_idx])

        agreement_curr = d_curr["agreement"]
        deceased_agreement = agreement_curr[deceased_idx]

        bli_curr = np.array(bli_data[str(c)]["bli_values"]) if str(c) in bli_data else None
        bli_next = np.array(bli_data[str(next_c)]["bli_values"]) if str(next_c) in bli_data else None
        if bli_curr is None or bli_next is None:
            continue

        for j, member_id in enumerate(curr_id_list):
            if member_id == deceased_icpsr:
                continue
            member_party = int(d_curr["party_codes"][j])
            same_party = int(member_party == deceased_party_code)
            agreement_with_deceased = float(deceased_agreement[j])
            was_co_voter = int(agreement_with_deceased > agreement_threshold)

            if member_id not in next_id_list:
                continue
            j_next = next_id_list.index(member_id)

            bli_change = float(bli_next[j_next] - bli_curr[j])
            curr_xparty = cross_party_rate(agreement_curr, d_curr["party_codes"], j)
            next_xparty = cross_party_rate(d_next["agreement"], d_next["party_codes"], j_next)
            if np.isnan(curr_xparty) or np.isnan(next_xparty):
                continue
            xparty_change = next_xparty - curr_xparty

            outcomes.append({
                "death_event_id": f"{c}_{deceased_icpsr}",
                "death_congress": c,
                "deceased_icpsr": deceased_icpsr,
                "deceased_bli_category": d["bli_category"],
                "deceased_age": int(d["age_at_death"]),
                "deceased_party": d["party"],
                "deceased_state": d["state"],
                "member_icpsr": member_id,
                "same_party": same_party,
                "agreement_with_deceased": agreement_with_deceased,
                "was_co_voter": was_co_voter,
                "is_xparty_covoter": int(was_co_voter == 1 and same_party == 0),
                "is_within_party_covoter": int(was_co_voter == 1 and same_party == 1),
                "delta_bli": bli_change,
                "delta_cross_party": xparty_change,
                "curr_bli": float(bli_curr[j]),
                "curr_cross_party": curr_xparty,
            })

    return pd.DataFrame(outcomes)


def build_chamber_outcomes(deaths_df):
    bli_data = load_bli_results()
    rows = []
    for _, d in deaths_df.iterrows():
        c = int(d["congress"])
        next_c = c + 1
        if str(c) not in bli_data or str(next_c) not in bli_data:
            continue
        fiedler_curr = float(bli_data[str(c)]["base_fiedler"])
        fiedler_next = float(bli_data[str(next_c)]["base_fiedler"])
        rows.append({
            "death_event_id": f"{c}_{int(d['icpsr'])}",
            "death_congress": c,
            "deceased_icpsr": int(d["icpsr"]),
            "deceased_bli_category": d["bli_category"],
            "deceased_bli_rank": float(d["bli_rank"]),
            "deceased_age": int(d["age_at_death"]),
            "deceased_party": d["party"],
            "fiedler_curr": fiedler_curr,
            "fiedler_next": fiedler_next,
            "delta_fiedler": fiedler_next - fiedler_curr,
        })
    return pd.DataFrame(rows)


def assign_exposure(outcomes_df):
    df = outcomes_df.copy()
    df["exposure"] = "d_0_non_covoter"
    df.loc[(df["was_co_voter"] == 1) & (df["deceased_bli_category"] == "high"), "exposure"] = "d_1H_high_bli_covoter"
    df.loc[(df["was_co_voter"] == 1) & (df["deceased_bli_category"] == "low"), "exposure"] = "d_1L_low_bli_covoter"
    df.loc[(df["was_co_voter"] == 1) & (df["deceased_bli_category"] == "middle"), "exposure"] = "d_1M_mid_bli_covoter"

    df["refined_exposure"] = "e_0_non_covoter"
    df.loc[
        (df["is_xparty_covoter"] == 1) & (df["deceased_bli_category"] == "high"),
        "refined_exposure",
    ] = "e_1H_xparty_covoter"
    df.loc[
        (df["is_within_party_covoter"] == 1) & (df["deceased_bli_category"] == "high"),
        "refined_exposure",
    ] = "e_1H_within_party_covoter"
    df.loc[
        (df["is_xparty_covoter"] == 1) & (df["deceased_bli_category"] == "low"),
        "refined_exposure",
    ] = "e_1L_xparty_covoter"
    df.loc[
        (df["is_within_party_covoter"] == 1) & (df["deceased_bli_category"] == "low"),
        "refined_exposure",
    ] = "e_1L_within_party_covoter"
    df.loc[
        (df["is_xparty_covoter"] == 1) & (df["deceased_bli_category"] == "middle"),
        "refined_exposure",
    ] = "e_1M_xparty_covoter"
    df.loc[
        (df["is_within_party_covoter"] == 1) & (df["deceased_bli_category"] == "middle"),
        "refined_exposure",
    ] = "e_1M_within_party_covoter"
    return df


def horvitz_thompson(df, outcome, exposure_levels=None):
    if exposure_levels is None:
        exposure_levels = df["exposure"].unique()
    results = {}
    for level in exposure_levels:
        sub = df[df["exposure"] == level]
        if len(sub) == 0:
            results[level] = {"mean": None, "n": 0, "se": None}
            continue
        vals = sub[outcome].dropna().values
        if len(vals) < 2:
            results[level] = {"mean": float(vals.mean()) if len(vals) else None, "n": int(len(vals)), "se": None}
            continue
        results[level] = {
            "mean": float(vals.mean()),
            "n": int(len(vals)),
            "se": float(vals.std(ddof=1) / np.sqrt(len(vals))),
            "median": float(np.median(vals)),
            "p25": float(np.percentile(vals, 25)),
            "p75": float(np.percentile(vals, 75)),
        }
    return results


def studentized_diff(df, outcome, level_a, level_b):
    a = df[df["exposure"] == level_a][outcome].dropna().values
    b = df[df["exposure"] == level_b][outcome].dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    var_a = a.var(ddof=1) / len(a)
    var_b = b.var(ddof=1) / len(b)
    se = np.sqrt(var_a + var_b)
    if se == 0:
        return np.nan
    return (a.mean() - b.mean()) / se


def permutation_test(df, outcome, level_a, level_b, n_perms=N_PERMUTATIONS, seed=SEED, stratify=None):
    rng = np.random.default_rng(seed)
    t_obs = studentized_diff(df, outcome, level_a, level_b)
    if np.isnan(t_obs):
        return {"t_observed": None, "p_value": None, "n_permutations": 0}

    mask_relevant = df["exposure"].isin([level_a, level_b])
    sub = df[mask_relevant].copy()
    labels = sub["exposure"].values
    values = sub[outcome].values
    valid = ~np.isnan(values)
    sub = sub[valid]
    labels = labels[valid]
    values = values[valid]

    t_perms = np.empty(n_perms)
    if stratify is not None and stratify in sub.columns:
        strata = sub[stratify].values
        for b in range(n_perms):
            perm_labels = labels.copy()
            for s in np.unique(strata):
                idx = np.where(strata == s)[0]
                if len(idx) > 1:
                    perm_labels[idx] = rng.permutation(labels[idx])
            a_vals = values[perm_labels == level_a]
            b_vals = values[perm_labels == level_b]
            if len(a_vals) < 2 or len(b_vals) < 2:
                t_perms[b] = 0
                continue
            se = np.sqrt(a_vals.var(ddof=1) / len(a_vals) + b_vals.var(ddof=1) / len(b_vals))
            t_perms[b] = (a_vals.mean() - b_vals.mean()) / se if se > 0 else 0
    else:
        for b in range(n_perms):
            perm = rng.permutation(labels)
            a_vals = values[perm == level_a]
            b_vals = values[perm == level_b]
            if len(a_vals) < 2 or len(b_vals) < 2:
                t_perms[b] = 0
                continue
            se = np.sqrt(a_vals.var(ddof=1) / len(a_vals) + b_vals.var(ddof=1) / len(b_vals))
            t_perms[b] = (a_vals.mean() - b_vals.mean()) / se if se > 0 else 0

    p_value = float((np.sum(np.abs(t_perms) >= np.abs(t_obs)) + 1) / (n_perms + 1))
    return {
        "t_observed": float(t_obs),
        "p_value": p_value,
        "n_permutations": n_perms,
        "perm_mean": float(t_perms.mean()),
        "perm_std": float(t_perms.std()),
    }


def chamber_level_contrast(chamber_df, outcome="delta_fiedler", seed=SEED, n_perms=N_PERMUTATIONS):
    high = chamber_df[chamber_df["deceased_bli_category"] == "high"][outcome].dropna().values
    low = chamber_df[chamber_df["deceased_bli_category"] == "low"][outcome].dropna().values
    if len(high) < 2 or len(low) < 2:
        return {"n_high": int(len(high)), "n_low": int(len(low)), "error": "insufficient_data"}

    obs_diff = float(high.mean() - low.mean())
    pooled = np.concatenate([high, low])
    rng = np.random.default_rng(seed)
    n_high = len(high)
    perms = np.empty(n_perms)
    for b in range(n_perms):
        shuffled = rng.permutation(pooled)
        perms[b] = shuffled[:n_high].mean() - shuffled[n_high:].mean()
    p_value = float((np.sum(np.abs(perms) >= abs(obs_diff)) + 1) / (n_perms + 1))
    return {
        "n_high": int(n_high),
        "n_low": int(len(low)),
        "high_mean": float(high.mean()),
        "low_mean": float(low.mean()),
        "observed_diff": obs_diff,
        "p_value": p_value,
        "high_median": float(np.median(high)),
        "low_median": float(np.median(low)),
    }


def era_residualized_chamber(chamber_df, outcome="delta_fiedler", seed=SEED, n_perms=N_PERMUTATIONS):
    bli_data = load_bli_results()
    fiedler_series = {}
    for k, v in bli_data.items():
        if k.isdigit() and isinstance(v, dict) and "base_fiedler" in v:
            fiedler_series[int(k)] = float(v["base_fiedler"])
    congresses = sorted(fiedler_series.keys())
    deltas = np.array([fiedler_series[c + 1] - fiedler_series[c] for c in congresses if (c + 1) in fiedler_series])
    delta_congresses = np.array([c for c in congresses if (c + 1) in fiedler_series])

    df = chamber_df.dropna(subset=[outcome]).copy()
    df["expected_delta"] = df["death_congress"].apply(
        lambda c: float(deltas[delta_congresses == c][0]) if c in delta_congresses else np.nan
    )
    df = df.dropna(subset=["expected_delta"]).copy()
    df["residualized"] = df[outcome] - df["expected_delta"]

    high = df[df["deceased_bli_category"] == "high"]["residualized"].values
    low = df[df["deceased_bli_category"] == "low"]["residualized"].values
    if len(high) < 2 or len(low) < 2:
        return {"n_high": int(len(high)), "n_low": int(len(low)), "error": "insufficient_data"}

    obs_diff = float(high.mean() - low.mean())
    pooled = np.concatenate([high, low])
    rng = np.random.default_rng(seed)
    n_high = len(high)
    perms = np.empty(n_perms)
    for b in range(n_perms):
        shuffled = rng.permutation(pooled)
        perms[b] = shuffled[:n_high].mean() - shuffled[n_high:].mean()
    p_value = float((np.sum(np.abs(perms) >= abs(obs_diff)) + 1) / (n_perms + 1))

    return {
        "n_high": int(len(high)),
        "n_low": int(len(low)),
        "high_residualized_mean": float(high.mean()),
        "low_residualized_mean": float(low.mean()),
        "observed_residualized_diff": obs_diff,
        "p_value": p_value,
        "note": "residual = observed_delta_fiedler - chamber_wide_delta_for_that_congress",
    }


def era_regression_chamber(chamber_df, outcome="delta_fiedler"):
    import statsmodels.api as sm
    df = chamber_df.dropna(subset=[outcome]).copy()
    df["era_early"] = (df["death_congress"] <= 106).astype(int)
    df["era_middle"] = ((df["death_congress"] >= 107) & (df["death_congress"] <= 112)).astype(int)
    df["era_late"] = (df["death_congress"] >= 113).astype(int)
    df["bli_high_indicator"] = (df["deceased_bli_category"] == "high").astype(int)
    df["bli_low_indicator"] = (df["deceased_bli_category"] == "low").astype(int)
    df["age_centered"] = df["deceased_age"] - df["deceased_age"].mean()

    X = df[["bli_high_indicator", "era_middle", "era_late", "age_centered"]]
    X = sm.add_constant(X)
    y = df[outcome]
    try:
        model = sm.OLS(y, X).fit(cov_type="HC3")
        return {
            "n_obs": int(len(df)),
            "coefficients": {k: float(v) for k, v in model.params.items()},
            "std_errors": {k: float(v) for k, v in model.bse.items()},
            "p_values": {k: float(v) for k, v in model.pvalues.items()},
            "r_squared": float(model.rsquared),
        }
    except Exception as e:
        return {"error": str(e)}


def pre_trend_test(deaths_df):
    bli_data = load_bli_results()
    rows = []
    for _, d in deaths_df.iterrows():
        c = int(d["congress"])
        traj = []
        for lookback in range(3, 0, -1):
            prev_c = c - lookback
            if str(prev_c) not in bli_data:
                continue
            ids = [int(m) for m in bli_data[str(prev_c)]["member_ids"]]
            if int(d["icpsr"]) in ids:
                idx = ids.index(int(d["icpsr"]))
                traj.append({"lookback": -lookback, "bli": float(bli_data[str(prev_c)]["bli_values"][idx])})
        if traj:
            rows.append({"death_event_id": f"{c}_{int(d['icpsr'])}", "trajectory": traj, "bli_category": d["bli_category"]})
    return rows


def plot_results(member_results, chamber_results, perm_test_results, out_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    levels = ["d_1H_high_bli_covoter", "d_1M_mid_bli_covoter", "d_1L_low_bli_covoter", "d_0_non_covoter", "d_2_other"]
    means = [member_results.get(lev, {}).get("mean", np.nan) for lev in levels]
    ses = [member_results.get(lev, {}).get("se", 0) for lev in levels]
    ns = [member_results.get(lev, {}).get("n", 0) for lev in levels]
    labels_short = ["High-BLI\nco-voter", "Mid-BLI\nco-voter", "Low-BLI\nco-voter", "Non-co-voter", "Other"]
    x = np.arange(len(levels))
    valid = [i for i, m in enumerate(means) if m is not None and not np.isnan(m)]
    ax.errorbar([x[i] for i in valid], [means[i] for i in valid], yerr=[ses[i] for i in valid],
                fmt="o", color="#1f77b4", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=8)
    ax.set_ylabel("Mean Δ BLI")
    ax.set_title("Exposure effects on surviving members' BLI")
    for i, n in enumerate(ns):
        ax.annotate(f"n={n}", (x[i], ax.get_ylim()[0]), fontsize=7, ha="center")

    ax = axes[1]
    if "high_mean" in chamber_results and "low_mean" in chamber_results:
        means_c = [chamber_results["high_mean"], chamber_results["low_mean"]]
        ax.bar([0, 1], means_c, color=["#d62728", "#2ca02c"], alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"High-BLI death\n(n={chamber_results['n_high']})",
                            f"Low-BLI death\n(n={chamber_results['n_low']})"], fontsize=9)
        ax.set_ylabel("Δ Fiedler value")
        ax.set_title(f"Chamber-level effect\np = {chamber_results['p_value']:.3f}")

    ax = axes[2]
    if perm_test_results and "t_observed" in perm_test_results and perm_test_results["t_observed"] is not None:
        t_obs = perm_test_results["t_observed"]
        ax.axvline(t_obs, color="red", linestyle="--", label=f"t_obs = {t_obs:.2f}")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Studentized t")
        ax.set_ylabel("Density")
        ax.set_title(f"Permutation test\np = {perm_test_results['p_value']:.3f}")
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_full_analysis(criterion="strict"):
    deaths_raw = load_deaths()
    deaths_matched = match_deaths_to_voteview(deaths_raw)
    deaths_filtered = filter_exogenous(deaths_matched, criterion=criterion)

    if len(deaths_filtered) == 0:
        return {"criterion": criterion, "error": "no deaths after filter"}

    bli_data = load_bli_results()
    deaths_with_bli = compute_bli_at_death(deaths_filtered, bli_data)

    member_outcomes = build_member_outcomes(deaths_with_bli)
    if len(member_outcomes) == 0:
        return {"criterion": criterion, "error": "no member outcomes"}
    member_outcomes = assign_exposure(member_outcomes)

    chamber_outcomes = build_chamber_outcomes(deaths_with_bli)

    bli_ht = horvitz_thompson(member_outcomes, "delta_bli")
    xparty_ht = horvitz_thompson(member_outcomes, "delta_cross_party")

    perm_bli = permutation_test(member_outcomes, "delta_bli",
                                 "d_1H_high_bli_covoter", "d_1L_low_bli_covoter",
                                 stratify="deceased_party")
    perm_xparty = permutation_test(member_outcomes, "delta_cross_party",
                                    "d_1H_high_bli_covoter", "d_1L_low_bli_covoter",
                                    stratify="deceased_party")

    chamber_contrast = chamber_level_contrast(chamber_outcomes, "delta_fiedler")
    chamber_residualized = era_residualized_chamber(chamber_outcomes, "delta_fiedler")
    chamber_regression = era_regression_chamber(chamber_outcomes, "delta_fiedler")

    refined_bli = horvitz_thompson(member_outcomes, "delta_bli", exposure_levels=sorted(member_outcomes["refined_exposure"].unique()))
    member_outcomes_refined = member_outcomes.rename(columns={"exposure": "exposure_orig", "refined_exposure": "exposure"})
    refined_xparty = horvitz_thompson(member_outcomes_refined, "delta_cross_party",
                                       exposure_levels=sorted(member_outcomes_refined["exposure"].unique()))

    perm_xparty_refined = permutation_test(
        member_outcomes_refined, "delta_cross_party",
        "e_1H_xparty_covoter", "e_1L_xparty_covoter",
        stratify="deceased_party",
    )

    pretrend = pre_trend_test(deaths_with_bli)

    return {
        "criterion": criterion,
        "n_deaths_raw": int(len(deaths_raw)),
        "n_deaths_matched": int(len(deaths_matched)),
        "n_deaths_filtered": int(len(deaths_filtered)),
        "n_deaths_with_bli": int(len(deaths_with_bli)),
        "n_member_outcomes": int(len(member_outcomes)),
        "n_chamber_events": int(len(chamber_outcomes)),
        "deaths_by_bli_category": deaths_with_bli["bli_category"].value_counts().to_dict(),
        "deaths_by_congress": deaths_with_bli.groupby("congress").size().to_dict(),
        "ht_delta_bli": bli_ht,
        "ht_delta_cross_party": xparty_ht,
        "permutation_test_bli_high_vs_low": perm_bli,
        "permutation_test_xparty_high_vs_low": perm_xparty,
        "chamber_level_high_vs_low": chamber_contrast,
        "chamber_residualized_against_trend": chamber_residualized,
        "chamber_regression_with_era_controls": chamber_regression,
        "ht_refined_delta_bli": refined_bli,
        "ht_refined_delta_cross_party": refined_xparty,
        "permutation_xparty_refined_high_vs_low": perm_xparty_refined,
        "deceased_pretrends": pretrend,
        "member_outcomes_summary": {
            "delta_bli_mean": float(member_outcomes["delta_bli"].mean()),
            "delta_bli_std": float(member_outcomes["delta_bli"].std()),
            "delta_xparty_mean": float(member_outcomes["delta_cross_party"].mean()),
            "delta_xparty_std": float(member_outcomes["delta_cross_party"].std()),
        },
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    results = {}
    for criterion in ["strict", "borderline", "permissive"]:
        print(f"\n=== Running {criterion} filter ===")
        results[criterion] = run_full_analysis(criterion=criterion)
        r = results[criterion]
        if "error" in r:
            print(f"  Error: {r['error']}")
            continue
        print(f"  Deaths filtered: {r['n_deaths_filtered']}")
        print(f"  Deaths with BLI: {r['n_deaths_with_bli']}")
        print(f"  Member outcomes: {r['n_member_outcomes']}")
        print(f"  BLI category: {r['deaths_by_bli_category']}")
        if "permutation_test_bli_high_vs_low" in r:
            pt = r["permutation_test_bli_high_vs_low"]
            print(f"  Δ BLI permutation test: t={pt.get('t_observed')}, p={pt.get('p_value')}")
        cl = r.get("chamber_level_high_vs_low", {})
        if "observed_diff" in cl:
            print(f"  Chamber-level Δ Fiedler high-vs-low: {cl['observed_diff']:.4f}, p={cl['p_value']:.3f}")

    if "strict" in results and "error" not in results["strict"]:
        deaths_raw = load_deaths()
        deaths_matched = match_deaths_to_voteview(deaths_raw)
        deaths_filtered = filter_exogenous(deaths_matched, criterion="strict")
        bli_data = load_bli_results()
        deaths_with_bli = compute_bli_at_death(deaths_filtered, bli_data)
        member_outcomes = build_member_outcomes(deaths_with_bli)
        member_outcomes = assign_exposure(member_outcomes)
        chamber_outcomes = build_chamber_outcomes(deaths_with_bli)
        plot_results(
            results["strict"]["ht_delta_bli"],
            results["strict"]["chamber_level_high_vs_low"],
            results["strict"]["permutation_test_bli_high_vs_low"],
            FIGURES_DIR / "death_in_office_main.pdf",
        )
        member_outcomes.to_csv(RESULTS_DIR / "death_member_outcomes.csv", index=False)
        chamber_outcomes.to_csv(RESULTS_DIR / "death_chamber_outcomes.csv", index=False)
        deaths_with_bli.to_csv(RESULTS_DIR / "death_events.csv", index=False)

    out_path = RESULTS_DIR / "death_in_office_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
