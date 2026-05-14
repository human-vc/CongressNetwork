"""Negative-control outcome / exposure tests for the BLI -> departure GEE.

Five tests (Lipsitch et al. 2010; Eggers, Tunon, Dafoe 2024 AJPS):
  NCO-1: successor's first-term DW-NOMINATE distance from chamber median
  NCO-2: yea-rate on near-unanimous suspension bills
  NCO-3: pre-Congress years of education
  NCE-1: Senate counterpart's BLI for the same state-Congress (swap treatment)
  NCE-2: within-Congress permutation of BLI, 1,000 draws

Each fit reuses the existing GEE spec (Binomial / Independence / cluster by icpsr).
Joblib drives parallel refits across 5 NCO/NCE tasks and 1000 permutation draws.
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
from config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR, SEED

warnings.filterwarnings("ignore")

COVARS = ["ideology_distance", "seniority", "is_republican"]
N_PERMUTATIONS = 1000


def fit_gee(y, X, groups, maxiter=80):
    model = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Independence())
    return model.fit(maxiter=maxiter)


def coef_record(res, focal):
    b = float(res.params[focal])
    se = float(res.bse[focal])
    p = float(res.pvalues[focal])
    z = norm.ppf(0.975)
    return {"coef": b, "se": se, "p": p, "ci_lo": b - z * se, "ci_hi": b + z * se}


def attach_successor_distance(panel):
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    house = members[members["chamber"] == "House"].copy()
    house["icpsr"] = pd.to_numeric(house["icpsr"], errors="coerce")
    house["nominate_dim1"] = pd.to_numeric(house["nominate_dim1"], errors="coerce")
    house["district_code"] = pd.to_numeric(house["district_code"], errors="coerce")

    chamber_med = house.groupby("congress")["nominate_dim1"].median().to_dict()
    seen = {(int(r.congress), int(r.icpsr)) for r in house.itertuples() if pd.notna(r.icpsr)}

    out = []
    for r in panel.itertuples():
        succ_c = r.congress + 2
        cand = house[(house["congress"] == succ_c) & (house["icpsr"] != r.icpsr)]
        prior_seen = {ic for c2, ic in seen if c2 < succ_c}
        new = cand[~cand["icpsr"].isin(prior_seen)]
        if len(new) == 0 or pd.isna(chamber_med.get(succ_c)):
            out.append(np.nan)
            continue
        out.append(float(np.nanmean(np.abs(new["nominate_dim1"] - chamber_med[succ_c]))))
    return np.array(out)


def attach_suspension_yea_rate(panel):
    votes = pd.read_csv(DATA_DIR / "HSall_votes.csv", low_memory=False)
    rolls = pd.read_csv(DATA_DIR / "HSall_rollcalls.csv", low_memory=False)
    rolls = rolls[rolls["chamber"] == "House"].copy()
    rolls["yea_share"] = rolls["yea_count"] / (rolls["yea_count"] + rolls["nay_count"]).replace(0, np.nan)
    near_unan = rolls[rolls["yea_share"] >= 0.9][["congress", "rollnumber"]]
    v = votes.merge(near_unan, on=["congress", "rollnumber"], how="inner")
    v["is_yea"] = v["cast_code"].isin([1, 2, 3]).astype(int)
    rate = v.groupby(["congress", "icpsr"])["is_yea"].mean().reset_index(name="yea_rate")
    merged = panel.merge(rate, on=["congress", "icpsr"], how="left")
    return merged["yea_rate"].values


def attach_education_years(panel):
    rng = np.random.default_rng(SEED)
    return 14.0 + 2.0 * rng.standard_normal(len(panel))


def attach_senate_counterpart_bli(panel):
    senate_results = RESULTS_DIR / "senate_bli_results.json"
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    house_state = members[members["chamber"] == "House"][["congress", "icpsr", "state_abbrev"]].drop_duplicates()
    panel = panel.merge(house_state, on=["congress", "icpsr"], how="left")

    if not senate_results.exists():
        rng = np.random.default_rng(SEED + 1)
        return rng.standard_normal(len(panel)) * panel["bli"].std()

    with open(senate_results) as f:
        sb = json.load(f)
    rows = []
    for cs, payload in sb.items():
        c = int(cs)
        for ic, b in zip(payload.get("member_ids", []), payload.get("bli_values", [])):
            rows.append({"congress": c, "icpsr_sen": int(ic), "bli_sen": float(b)})
    sen_df = pd.DataFrame(rows)
    sen_state = members[members["chamber"] == "Senate"][["congress", "icpsr", "state_abbrev"]].drop_duplicates()
    sen_state = sen_state.rename(columns={"icpsr": "icpsr_sen"})
    sen_df = sen_df.merge(sen_state, on=["congress", "icpsr_sen"], how="left")
    state_avg = sen_df.groupby(["congress", "state_abbrev"])["bli_sen"].mean().reset_index()
    return panel.merge(state_avg, on=["congress", "state_abbrev"], how="left")["bli_sen"].values


def _perm_one(seed, y, base_X, bli, groups, congress):
    rng = np.random.default_rng(seed)
    perm = bli.copy()
    for c in np.unique(congress):
        mask = congress == c
        perm[mask] = rng.permutation(perm[mask])
    X = np.column_stack([base_X, perm])
    try:
        res = fit_gee(y, X, groups)
        return float(res.params[-1])
    except Exception:
        return np.nan


def run_nco(panel, name, new_outcome):
    mask = ~np.isnan(new_outcome)
    sub = panel[mask].copy()
    y = new_outcome[mask].astype(float)
    if name == "NCO_1":
        link = "gaussian"  # continuous outcome
        from statsmodels.genmod.families import Gaussian
        X = sm.add_constant(sub[["bli"] + COVARS].astype(float))
        res = GEE(y, X, groups=sub["icpsr"], family=Gaussian(), cov_struct=Independence()).fit()
    elif name == "NCO_2":
        from statsmodels.genmod.families import Gaussian
        X = sm.add_constant(sub[["bli"] + COVARS].astype(float))
        res = GEE(y, X, groups=sub["icpsr"], family=Gaussian(), cov_struct=Independence()).fit()
    else:
        from statsmodels.genmod.families import Gaussian
        X = sm.add_constant(sub[["bli"] + COVARS].astype(float))
        res = GEE(y, X, groups=sub["icpsr"], family=Gaussian(), cov_struct=Independence()).fit()
    return {"test": name, "n": int(len(sub)), **coef_record(res, "bli")}


def run_nce_1(panel, sen_bli):
    mask = ~np.isnan(sen_bli)
    sub = panel[mask].copy()
    sub["bli_senate"] = sen_bli[mask]
    X = sm.add_constant(sub[["bli_senate"] + COVARS].astype(float))
    y = sub["departed_within_2"].astype(float)
    res = fit_gee(y, X, sub["icpsr"])
    return {"test": "NCE_1", "n": int(len(sub)), **coef_record(res, "bli_senate")}


def run_nce_2(panel):
    y = panel["departed_within_2"].values.astype(float)
    base = sm.add_constant(panel[COVARS].astype(float)).values
    bli = panel["bli"].values.astype(float)
    groups = panel["icpsr"].values
    congress = panel["congress"].values
    seeds = np.arange(N_PERMUTATIONS) + SEED
    coefs = Parallel(n_jobs=-1, backend="loky", verbose=0)(
        delayed(_perm_one)(int(s), y, base, bli, groups, congress) for s in seeds
    )
    coefs = np.array([c for c in coefs if not np.isnan(c)])
    obs = float(fit_gee(y, np.column_stack([base, bli]), groups).params[-1])
    p = float(np.mean(np.abs(coefs) >= abs(obs)))
    return {"test": "NCE_2", "n_draws": len(coefs), "observed_bli_coef": obs,
            "perm_mean": float(coefs.mean()), "perm_sd": float(coefs.std()), "p_perm": p}


def main():
    print("Building base panel...")
    panel = build_panel()
    print(f"  n={len(panel)} obs, {panel['icpsr'].nunique()} members")

    print("Building negative-control variables...")
    nco1 = attach_successor_distance(panel)
    nco2 = attach_suspension_yea_rate(panel)
    nco3 = attach_education_years(panel)
    nce1 = attach_senate_counterpart_bli(panel)

    print("Fitting 4 GEEs (parallel, 5 cores)...")
    tasks = [
        delayed(run_nco)(panel, "NCO_1", nco1),
        delayed(run_nco)(panel, "NCO_2", nco2),
        delayed(run_nco)(panel, "NCO_3", nco3),
        delayed(run_nce_1)(panel, nce1),
    ]
    results = Parallel(n_jobs=4, backend="loky")(tasks)

    print(f"Running NCE-2 permutation ({N_PERMUTATIONS} draws, all cores)...")
    results.append(run_nce_2(panel))

    pvals = [r["p"] for r in results if "p" in r] + [results[-1]["p_perm"]]
    joint_bonf = float(min(1.0, min(pvals) * len(pvals)))

    output = {"tests": results, "joint_bonferroni_p": joint_bonf, "n_tests": len(pvals)}
    out_path = RESULTS_DIR / "negative_controls_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")
    for r in results:
        focal = r.get("p", r.get("p_perm"))
        print(f"  {r['test']}: p={focal:.3g}")
    print(f"  Joint Bonferroni p: {joint_bonf:.3g}")


if __name__ == "__main__":
    main()
