import sys
import json
import warnings
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import norm
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bli_regression import build_panel
from config import DATA_DIR, RESULTS_DIR, SEED

warnings.filterwarnings("ignore")

COVARS = ["ideology_distance", "seniority", "is_republican"]
N_PERMUTATIONS = 1000

def fit_gee_binomial(y, X, groups, maxiter=80):
    return GEE(y, X, groups=groups, family=Binomial(), cov_struct=Independence()).fit(maxiter=maxiter)

def fit_gee_gaussian(y, X, groups, maxiter=80):
    return GEE(y, X, groups=groups, family=Gaussian(), cov_struct=Independence()).fit(maxiter=maxiter)

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

def attach_birth_year(panel):
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    house = members[members["chamber"] == "House"][["congress", "icpsr", "born"]].drop_duplicates()
    house["icpsr"] = pd.to_numeric(house["icpsr"], errors="coerce")
    house["born"] = pd.to_numeric(house["born"], errors="coerce")
    merged = panel.merge(house, on=["congress", "icpsr"], how="left")
    return merged["born"].values.astype(float)

def attach_surname_initial(panel):
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    house = members[members["chamber"] == "House"][["congress", "icpsr", "bioname"]].drop_duplicates()
    house["icpsr"] = pd.to_numeric(house["icpsr"], errors="coerce")
    def _initial(b):
        if not isinstance(b, str) or not b.strip():
            return np.nan
        ch = b.strip()[0].upper()
        if not ch.isalpha():
            return np.nan
        return float(ord(ch) - ord("A") + 1)
    house["surname_initial_idx"] = house["bioname"].apply(_initial)
    merged = panel.merge(
        house[["congress", "icpsr", "surname_initial_idx"]],
        on=["congress", "icpsr"], how="left"
    )
    return merged["surname_initial_idx"].values.astype(float)

def attach_cross_state_bli(panel):
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    house_state = members[members["chamber"] == "House"][["congress", "icpsr", "state_abbrev"]].drop_duplicates()
    house_state["icpsr"] = pd.to_numeric(house_state["icpsr"], errors="coerce")
    panel_in = panel.merge(house_state, on=["congress", "icpsr"], how="left")
    panel_in = panel_in.reset_index(drop=True)
    rng = np.random.default_rng(SEED + 12345)
    out = np.full(len(panel_in), np.nan)
    for c, sub in panel_in.groupby("congress"):
        if len(sub) < 2:
            continue
        for i, row in sub.iterrows():
            other = sub[sub["state_abbrev"] != row["state_abbrev"]]
            if len(other) == 0:
                continue
            pick = other.iloc[int(rng.integers(0, len(other)))]
            out[i] = float(pick["bli"])
    return out

def _perm_one(seed, y, base_X, bli, groups, congress):
    rng = np.random.default_rng(seed)
    perm = bli.copy()
    for c in np.unique(congress):
        mask = congress == c
        perm[mask] = rng.permutation(perm[mask])
    X = np.column_stack([base_X, perm])
    try:
        res = fit_gee_binomial(y, X, groups)
        return float(res.params[-1])
    except Exception:
        return np.nan

def run_nco(panel, name, new_outcome):
    mask = ~np.isnan(new_outcome)
    sub = panel[mask].copy()
    y = new_outcome[mask].astype(float)
    X = sm.add_constant(sub[["bli"] + COVARS].astype(float))
    res = fit_gee_gaussian(y, X, sub["icpsr"])
    return {"test": name, "n": int(len(sub)), **coef_record(res, "bli")}

def run_nce_1(panel, cross_bli):
    mask = ~np.isnan(cross_bli)
    sub = panel[mask].copy()
    sub["bli_cross_state"] = cross_bli[mask]
    X = sm.add_constant(sub[["bli_cross_state"] + COVARS].astype(float))
    y = sub["departed_within_2"].astype(float)
    res = fit_gee_binomial(y, X, sub["icpsr"])
    return {"test": "NCE_1", "n": int(len(sub)), **coef_record(res, "bli_cross_state")}

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
    obs = float(fit_gee_binomial(y, np.column_stack([base, bli]), groups).params[-1])
    p = float(np.mean(np.abs(coefs) >= abs(obs)))
    return {"test": "NCE_2", "n_draws": len(coefs), "observed_bli_coef": obs,
            "perm_mean": float(coefs.mean()), "perm_sd": float(coefs.std()), "p_perm": p}

def main():
    print("Building base panel...")
    panel = build_panel()
    print(f"  n={len(panel)} obs, {panel['icpsr'].nunique()} members")

    print("Building negative-control variables...")
    nco1 = attach_successor_distance(panel)
    nco2 = attach_birth_year(panel)
    nco3 = attach_surname_initial(panel)
    nce1 = attach_cross_state_bli(panel)

    print("Fitting NCO tests + NCE_1 in parallel...")
    tasks = [
        delayed(run_nco)(panel, "NCO_1_successor_distance", nco1),
        delayed(run_nco)(panel, "NCO_2_birth_year", nco2),
        delayed(run_nco)(panel, "NCO_3_surname_initial", nco3),
        delayed(run_nce_1)(panel, nce1),
    ]
    results = Parallel(n_jobs=4, backend="loky")(tasks)

    print(f"Running NCE-2 permutation ({N_PERMUTATIONS} draws)...")
    results.append(run_nce_2(panel))

    pvals = [r["p"] for r in results if "p" in r] + [results[-1]["p_perm"]]
    joint_bonf = float(min(1.0, min(pvals) * len(pvals)))

    output = {
        "tests": results,
        "joint_bonferroni_p": joint_bonf,
        "n_tests": len(pvals),
        "rubric": {
            "NCO_1_successor_distance": "structural placebo: successor's NOMINATE distance is determined by next election",
            "NCO_2_birth_year": "pre-determined covariate; expected null with small cohort residual",
            "NCO_3_surname_initial": "truly null lexical covariate",
            "NCE_1_cross_state_bli": "BLI of a randomly matched member in a different state; no causal pathway",
            "NCE_2_within_congress_permutation": "structural placebo for member identity",
        },
    }
    out_path = RESULTS_DIR / "negative_controls_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")
    for r in results:
        focal = r.get("p", r.get("p_perm"))
        print(f"  {r['test']:35s}: p={focal:.3g}")
    print(f"  Joint Bonferroni p: {joint_bonf:.3g}")

if __name__ == "__main__":
    main()
