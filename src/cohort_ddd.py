import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from medsl_data import district_competitiveness_panel, congress_to_election_year

import pyfixest as pf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


COHORTS = [102, 104, 106, 108, 110, 112, 114, 116]
TREATMENT_CONGRESS = 112
COMPETITIVE_MARGIN_CUTOFF = 0.10
MIN_OPP_PARTNERS = 5
LOOKBACK_ELECTIONS = 2


def load_congress(c):
    path = PROCESSED_DIR / f"congress_{c}.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def member_cross_party_rate(agreement, party_codes, idx):
    row = agreement[idx]
    other = party_codes != party_codes[idx]
    vals = row[other]
    vals = vals[vals > 0]
    if len(vals) < MIN_OPP_PARTNERS:
        return np.nan
    return float(vals.mean())


def members_seen_before(through_congress):
    seen = set()
    for cc in range(100, through_congress):
        d = load_congress(cc)
        if d is not None:
            seen.update(int(m) for m in d["member_ids"])
    return seen


def load_member_district_map():
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    house = members[
        (members["chamber"] == "House")
        & (members["congress"].between(min(COHORTS), max(COHORTS) + 1))
    ].copy()
    house["district_code"] = pd.to_numeric(house["district_code"], errors="coerce").astype("Int64")
    house["state_abbrev"] = house["state_abbrev"].astype(str).str.upper()
    return house[["congress", "icpsr", "state_abbrev", "district_code"]]


def lookup_prior_margin(panel_lookup, state, district, congress):
    target_year = congress_to_election_year(congress)
    margins = []
    leans = []
    for back in range(1, LOOKBACK_ELECTIONS + 1):
        key = (state, district, target_year - 2 * back)
        if key in panel_lookup.index:
            row = panel_lookup.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            m = row["margin"]
            d = row["dem_share"]
            if pd.notna(m):
                margins.append(float(m))
                leans.append(float(d) - 0.5)
    if not margins:
        return np.nan, np.nan
    return float(np.mean(margins)), float(np.mean(leans))


def build_panel():
    district_panel = district_competitiveness_panel()
    panel_lookup = district_panel.set_index(["state_po", "district", "year"]).sort_index()
    member_dist = load_member_district_map()
    member_dist_idx = member_dist.set_index(["congress", "icpsr"])

    records = []
    for c in COHORTS:
        d = load_congress(c)
        if d is None:
            continue
        ids = d["member_ids"]
        party = d["party_codes"]
        states = d["state_abbrev"]
        nom = d["nominate_dim1"]
        agreement = d["agreement"]
        prior = members_seen_before(c)

        for i, mid in enumerate(ids):
            rate = member_cross_party_rate(agreement, party, i)
            if np.isnan(rate):
                continue
            try:
                drec = member_dist_idx.loc[(c, int(mid))]
                if isinstance(drec, pd.DataFrame):
                    drec = drec.iloc[0]
                state_po = str(drec["state_abbrev"]).upper()
                district_code = drec["district_code"]
                if pd.isna(district_code):
                    continue
                district_code = int(district_code)
            except KeyError:
                continue
            prior_margin, prior_lean = lookup_prior_margin(panel_lookup, state_po, district_code, c)
            if np.isnan(prior_margin):
                continue
            is_fresh = int(int(mid) not in prior)
            records.append({
                "member_id": int(mid),
                "state": state_po,
                "district": district_code,
                "congress": int(c),
                "party": "D" if party[i] == 100 else "R",
                "freshman": is_fresh,
                "post_2010": int(c >= TREATMENT_CONGRESS),
                "extremity": float(abs(nom[i])),
                "prior_margin": prior_margin,
                "prior_dem_lean": prior_lean,
                "competitive": int(prior_margin < COMPETITIVE_MARGIN_CUTOFF),
                "cross_party_agreement": rate,
            })
    panel = pd.DataFrame(records)
    panel["event_time"] = (panel["congress"] - TREATMENT_CONGRESS) // 2
    return panel


def fit_baseline(panel):
    df = panel.copy()
    for col in ["freshman", "post_2010", "competitive"]:
        df[col] = df[col].astype(float)
    return pf.feols(
        "cross_party_agreement ~ freshman * post_2010 * competitive | congress + state",
        data=df,
        vcov={"CRV1": "state"},
    )


def fit_member_fe(panel):
    df = panel.copy()
    for col in ["freshman", "post_2010", "competitive"]:
        df[col] = df[col].astype(float)
    return pf.feols(
        "cross_party_agreement ~ freshman * post_2010 * competitive | congress + member_id",
        data=df,
        vcov={"CRV1": "state"},
    )


def fit_party_split(panel, party):
    df = panel[panel["party"] == party].copy()
    for col in ["freshman", "post_2010", "competitive"]:
        df[col] = df[col].astype(float)
    return pf.feols(
        "cross_party_agreement ~ freshman * post_2010 * competitive | congress + state",
        data=df,
        vcov={"CRV1": "state"},
    )


def fit_event_study(panel):
    df = panel[panel["freshman"] == 1].copy()
    df["event_time"] = df["event_time"].astype(int)
    df["competitive"] = df["competitive"].astype(float)
    return pf.feols(
        "cross_party_agreement ~ i(event_time, competitive, ref=-1) | congress + state",
        data=df,
        vcov={"CRV1": "state"},
    )


def fit_continuous_treatment(panel):
    df = panel.copy()
    df["safe_margin"] = df["prior_margin"]
    for col in ["freshman", "post_2010"]:
        df[col] = df[col].astype(float)
    return pf.feols(
        "cross_party_agreement ~ freshman * post_2010 * safe_margin | congress + state",
        data=df,
        vcov={"CRV1": "state"},
    )


def fit_placebo(panel, placebo_year):
    df = panel.copy()
    df = df[df["congress"] < TREATMENT_CONGRESS].copy()
    df["placebo_post"] = (df["congress"] >= placebo_year).astype(float)
    for col in ["freshman", "competitive"]:
        df[col] = df[col].astype(float)
    return pf.feols(
        "cross_party_agreement ~ freshman * placebo_post * competitive | congress + state",
        data=df,
        vcov={"CRV1": "state"},
    )


def summarize_fit(fit, label):
    tidy = fit.tidy()
    return {
        "label": label,
        "n_obs": int(fit._N),
        "coefficients": {
            row.name: {
                "estimate": float(row["Estimate"]),
                "std_error": float(row["Std. Error"]),
                "t_stat": float(row["t value"]),
                "p_value": float(row["Pr(>|t|)"]),
                "ci_low": float(row["2.5%"]),
                "ci_high": float(row["97.5%"]),
            }
            for _, row in tidy.iterrows()
        },
    }


def manual_wild_cluster_bootstrap(panel, term_pattern="freshman:post_2010:competitive", reps=9999, seed=SEED):
    df = panel.copy()
    for col in ["freshman", "post_2010", "competitive"]:
        df[col] = df[col].astype(float)
    df["congress_factor"] = df["congress"].astype("category")
    df["state_factor"] = df["state"].astype("category")

    formula = "cross_party_agreement ~ freshman * post_2010 * competitive + C(congress_factor) + C(state_factor)"
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")
    y_arr = np.asarray(y).ravel().astype(np.float64)
    X_arr = np.asarray(X, dtype=np.float64)
    col_names = list(X.columns)

    triple_idx = None
    for i, name in enumerate(col_names):
        normalized = name.replace(" ", "")
        if normalized.endswith(term_pattern) or normalized == term_pattern:
            triple_idx = i
            break
    if triple_idx is None:
        return {"error": f"Could not locate column matching {term_pattern}"}

    n, k = X_arr.shape
    XtX = X_arr.T @ X_arr
    XtX_inv = np.linalg.pinv(XtX)
    XtX_inv_Xt = XtX_inv @ X_arr.T

    beta_full = XtX_inv_Xt @ y_arr
    resid_full = y_arr - X_arr @ beta_full
    cluster_ids, _ = pd.factorize(df["state"].astype(str))
    cluster_ids = cluster_ids.astype(np.int64)
    n_clusters = int(cluster_ids.max() + 1)

    def cr1_se_for_coef(beta, residuals, coef_idx):
        g = n_clusters
        meat = np.zeros((k, k))
        for c in range(g):
            mask = cluster_ids == c
            Xc = X_arr[mask]
            uc = residuals[mask]
            score = Xc.T @ uc
            meat += np.outer(score, score)
        adj = (g / (g - 1)) * ((n - 1) / (n - k))
        vcov = adj * (XtX_inv @ meat @ XtX_inv)
        return float(np.sqrt(vcov[coef_idx, coef_idx]))

    se_full = cr1_se_for_coef(beta_full, resid_full, triple_idx)
    t_orig = float(beta_full[triple_idx] / se_full)

    restricted_mask = np.ones(k, dtype=bool)
    restricted_mask[triple_idx] = False
    X_restricted = X_arr[:, restricted_mask]
    XtX_r = X_restricted.T @ X_restricted
    XtX_r_inv = np.linalg.pinv(XtX_r)
    beta_restricted = XtX_r_inv @ X_restricted.T @ y_arr
    y_tilde = X_restricted @ beta_restricted
    u_tilde = y_arr - y_tilde

    rng = np.random.default_rng(seed)
    t_stars = np.empty(reps, dtype=np.float64)

    for b in range(reps):
        w_clusters = rng.choice([-1.0, 1.0], size=n_clusters)
        w_obs = w_clusters[cluster_ids]
        u_star = w_obs * u_tilde
        y_star = y_tilde + u_star
        beta_star = XtX_inv_Xt @ y_star
        resid_star = y_star - X_arr @ beta_star
        se_star = cr1_se_for_coef(beta_star, resid_star, triple_idx)
        t_stars[b] = (beta_star[triple_idx] - 0.0) / se_star if se_star > 0 else np.nan

    finite = t_stars[np.isfinite(t_stars)]
    p_value = float((np.abs(finite) >= abs(t_orig)).mean())

    return {
        "term": col_names[triple_idx],
        "coefficient": float(beta_full[triple_idx]),
        "cluster_robust_se": se_full,
        "t_original": t_orig,
        "wild_bootstrap_p_value": p_value,
        "n_reps": int(len(finite)),
        "n_clusters": n_clusters,
    }


def balance_table(panel):
    fresh = panel[(panel["freshman"] == 1) & panel["congress"].isin([104, 112])]
    rows = []
    for c in [104, 112]:
        sub = fresh[fresh["congress"] == c]
        rows.append({
            "cohort": "Contract" if c == 104 else "Tea Party",
            "n": int(len(sub)),
            "prior_margin_mean": float(sub["prior_margin"].mean()),
            "prior_margin_sd": float(sub["prior_margin"].std()),
            "competitive_share": float(sub["competitive"].mean()),
            "rep_share": float((sub["party"] == "R").mean()),
            "extremity_mean": float(sub["extremity"].mean()),
            "cross_party_mean": float(sub["cross_party_agreement"].mean()),
            "cross_party_sd": float(sub["cross_party_agreement"].std()),
        })
    return rows


def main():
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_panel()
    panel_path = RESULTS_DIR / "cohort_ddd_panel.csv"
    panel.to_csv(panel_path, index=False)

    fits = {
        "baseline": fit_baseline(panel),
        "member_fe": fit_member_fe(panel),
        "republicans_only": fit_party_split(panel, "R"),
        "democrats_only": fit_party_split(panel, "D"),
        "continuous_treatment": fit_continuous_treatment(panel),
        "event_study": fit_event_study(panel),
        "placebo_2006": fit_placebo(panel, 106),
        "placebo_2002": fit_placebo(panel, 102),
    }

    summaries = {name: summarize_fit(fit, name) for name, fit in fits.items()}

    wb = manual_wild_cluster_bootstrap(panel, reps=4999)

    output = {
        "config": {
            "cohorts": COHORTS,
            "treatment_congress": TREATMENT_CONGRESS,
            "competitive_margin_cutoff": COMPETITIVE_MARGIN_CUTOFF,
            "lookback_elections": LOOKBACK_ELECTIONS,
            "min_opposite_party_partners": MIN_OPP_PARTNERS,
            "panel_path": str(panel_path),
        },
        "panel_summary": {
            "n_observations": int(len(panel)),
            "n_freshmen": int((panel["freshman"] == 1).sum()),
            "n_incumbents": int((panel["freshman"] == 0).sum()),
            "n_states": int(panel["state"].nunique()),
            "n_members": int(panel["member_id"].nunique()),
            "cohort_sizes": panel.groupby("congress").size().to_dict(),
        },
        "balance_table": balance_table(panel),
        "fits": summaries,
        "wild_bootstrap": {
            "baseline_triple_interaction": wb,
        },
    }

    out_path = RESULTS_DIR / "cohort_ddd_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Wrote panel to {panel_path} ({len(panel)} rows)")
    print(f"Wrote results to {out_path}")
    print()
    print("Baseline DDD:")
    print(fits["baseline"].tidy())
    print()
    print(f"Triple interaction wild bootstrap p-value: {wb}")


if __name__ == "__main__":
    main()
