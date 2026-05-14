import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from medsl_data import district_competitiveness_panel, congress_to_election_year

import pyfixest as pf

if not hasattr(np, "alltrue"):
    np.alltrue = np.all

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


WAVE_CONGRESSES = [104, 110, 112, 116, 118]
PANEL_CONGRESSES = list(range(100, 119))
COMPETITIVE_MARGIN_CUTOFF = 0.10
MIN_OPP_PARTNERS = 5
WINDOW_PRE = 2
WINDOW_POST = 2


CYCLE_BOUNDS = [
    ("C0", 100, 102),
    ("C1", 103, 107),
    ("C2", 108, 112),
    ("C3", 113, 117),
    ("C4", 118, 118),
]


def cycle_for_congress(c):
    for name, lo, hi in CYCLE_BOUNDS:
        if lo <= c <= hi:
            return name
    return "OUT"


def load_congress(c):
    p = PROCESSED_DIR / f"congress_{c}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def cross_party_rate(agreement, party_codes, idx):
    row = agreement[idx]
    other = party_codes != party_codes[idx]
    vals = row[other]
    vals = vals[vals > 0]
    if len(vals) < MIN_OPP_PARTNERS:
        return np.nan
    return float(vals.mean())


def freshmen_set(c):
    prior = set()
    for cc in range(100, c):
        d = load_congress(cc)
        if d is not None:
            prior.update(int(m) for m in d["member_ids"])
    d = load_congress(c)
    if d is None:
        return set()
    return {int(m) for m in d["member_ids"] if int(m) not in prior}


def load_member_district_map():
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    house = members[
        (members["chamber"] == "House")
        & (members["congress"].between(min(PANEL_CONGRESSES), max(PANEL_CONGRESSES)))
    ].copy()
    house["district_code"] = pd.to_numeric(house["district_code"], errors="coerce").astype("Int64")
    house["state_abbrev"] = house["state_abbrev"].astype(str).str.upper()
    house["district_norm"] = house["district_code"].fillna(0).astype(int).clip(lower=0).where(
        ~house["district_code"].isna(), 0
    )
    house.loc[house["district_norm"].isin([1, 98, 99]) & house.duplicated(subset=["congress", "state_abbrev"]) == False, "district_norm"] = (
        house.loc[house["district_norm"].isin([1, 98, 99]), "district_norm"]
    )
    return house[["congress", "icpsr", "state_abbrev", "district_code", "district_norm"]]


def lookup_prior_margin(panel_lookup, state, district, congress, lookback=2):
    target = congress_to_election_year(congress)
    margins = []
    for back in range(1, lookback + 1):
        key = (state, district, target - 2 * back)
        if key in panel_lookup.index:
            row = panel_lookup.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if pd.notna(row["margin"]):
                margins.append(float(row["margin"]))
    return float(np.mean(margins)) if margins else np.nan


def build_district_panel():
    dist_panel = district_competitiveness_panel()
    panel_lookup = dist_panel.set_index(["state_po", "district", "year"]).sort_index()
    member_map = load_member_district_map()
    member_idx = member_map.set_index(["congress", "icpsr"])

    rows = []
    for c in PANEL_CONGRESSES:
        d = load_congress(c)
        if d is None:
            continue
        ids = d["member_ids"]
        party = d["party_codes"]
        agreement = d["agreement"]
        fresh = freshmen_set(c)

        for i, mid in enumerate(ids):
            rate = cross_party_rate(agreement, party, i)
            if np.isnan(rate):
                continue
            try:
                rec = member_idx.loc[(c, int(mid))]
                if isinstance(rec, pd.DataFrame):
                    rec = rec.iloc[0]
                state = str(rec["state_abbrev"]).upper()
                district_code = rec["district_code"]
                if pd.isna(district_code):
                    continue
                district_code = int(district_code)
            except KeyError:
                continue

            district_norm = district_code if district_code not in (98, 99) else 0
            margin = lookup_prior_margin(panel_lookup, state, district_norm, c)
            if np.isnan(margin):
                continue

            cycle = cycle_for_congress(c)
            district_id = f"{state}_{district_norm:02d}_{cycle}"

            rows.append({
                "district_id": district_id,
                "state": state,
                "district_norm": district_norm,
                "cycle": cycle,
                "congress": int(c),
                "member_id": int(mid),
                "party": "D" if party[i] == 100 else "R",
                "is_freshman": int(int(mid) in fresh),
                "prior_margin": margin,
                "competitive": int(margin < COMPETITIVE_MARGIN_CUTOFF),
                "cross_party_agreement": rate,
                "is_wave_congress": int(c in WAVE_CONGRESSES),
            })
    panel = pd.DataFrame(rows)
    return panel


def assign_cohorts(panel):
    df = panel.copy()
    wave_fresh = df[(df["is_wave_congress"] == 1) & (df["is_freshman"] == 1)]
    first_wave_by_district = wave_fresh.groupby("district_id")["congress"].min()
    df["cohort"] = df["district_id"].map(first_wave_by_district).fillna(0).astype(int)
    return df


def build_stacked_panel(panel):
    stacks = []
    for w in WAVE_CONGRESSES:
        lo = w - 2 * WINDOW_PRE
        hi = w + 2 * WINDOW_POST
        cycle_w = cycle_for_congress(w)
        for back in range(1, WINDOW_PRE + 1):
            if cycle_for_congress(w - 2 * back) != cycle_w:
                lo = w - 2 * (back - 1)
                break
        for fwd in range(1, WINDOW_POST + 1):
            if cycle_for_congress(w + 2 * fwd) != cycle_w:
                hi = w + 2 * (fwd - 1)
                break

        wave_pool = panel[(panel["congress"] >= lo) & (panel["congress"] <= hi)].copy()
        wave_fresh = panel[(panel["congress"] == w) & (panel["is_freshman"] == 1)]
        treated_districts = set(wave_fresh["district_id"].unique())

        wave_pool["stack_id"] = w
        wave_pool["event_time"] = (wave_pool["congress"] - w) // 2
        wave_pool["treated"] = wave_pool["district_id"].isin(treated_districts).astype(int)
        wave_pool["treated_post"] = wave_pool["treated"] * (wave_pool["event_time"] >= 0).astype(int)

        keep = (
            wave_pool.groupby("district_id")["cycle"].nunique() == 1
        )
        valid_districts = keep[keep].index
        wave_pool = wave_pool[wave_pool["district_id"].isin(valid_districts)].copy()
        stacks.append(wave_pool)

    stacked = pd.concat(stacks, ignore_index=True)
    stacked["stack_district"] = stacked["stack_id"].astype(str) + "_" + stacked["district_id"]
    stacked["stack_congress"] = stacked["stack_id"].astype(str) + "_" + stacked["congress"].astype(str)
    return stacked


def fit_stacked_event_study(stacked):
    df = stacked.copy()
    df["event_time"] = df["event_time"].astype(int)
    df["treated"] = df["treated"].astype(float)
    return pf.feols(
        "cross_party_agreement ~ i(event_time, treated, ref=-1) | stack_district + stack_congress",
        data=df,
        vcov={"CRV1": "stack_district"},
    )


def fit_stacked_ddd(stacked):
    df = stacked.copy()
    df["event_time"] = df["event_time"].astype(int)
    df["treated"] = df["treated"].astype(float)
    df["competitive"] = df["competitive"].astype(float)
    df["treated_competitive"] = df["treated"] * df["competitive"]
    return pf.feols(
        "cross_party_agreement ~ i(event_time, treated, ref=-1) + i(event_time, treated_competitive, ref=-1) | stack_district + stack_congress",
        data=df,
        vcov={"CRV1": "stack_district"},
    )


def fit_stacked_by_stratum(stacked, stratum_value):
    df = stacked[stacked["competitive"] == stratum_value].copy()
    df["event_time"] = df["event_time"].astype(int)
    df["treated"] = df["treated"].astype(float)
    return pf.feols(
        "cross_party_agreement ~ i(event_time, treated, ref=-1) | stack_district + stack_congress",
        data=df,
        vcov={"CRV1": "stack_district"},
    )


def fit_wave_specific(stacked):
    out = {}
    for w in WAVE_CONGRESSES:
        sub = stacked[stacked["stack_id"] == w].copy()
        if len(sub) < 50 or sub["treated"].sum() < 5:
            continue
        sub["event_time"] = sub["event_time"].astype(int)
        sub["treated"] = sub["treated"].astype(float)
        try:
            fit = pf.feols(
                "cross_party_agreement ~ i(event_time, treated, ref=-1) | district_id + congress",
                data=sub,
                vcov={"CRV1": "district_id"},
            )
            out[w] = fit
        except Exception as e:
            out[w] = {"error": str(e)}
    return out


def _flatten_aggregate(df):
    if df is None:
        return None
    if df.columns.nlevels > 1:
        df = df.copy()
        df.columns = ["__".join([str(x) for x in tup if x]).strip("_") for tup in df.columns]
    return df.reset_index().to_dict(orient="records")


def run_callaway_santanna(panel):
    try:
        from differences import ATTgt
    except ImportError:
        return {"error": "differences package not installed"}

    df = panel.copy()
    df = df[df["competitive"].notna()].copy()
    df["cohort"] = df["cohort"].replace(0, np.nan)
    df["entity"] = df["district_id"]
    df["time"] = df["congress"]
    cs_panel = df.set_index(["entity", "time"]).sort_index()

    att = ATTgt(
        data=cs_panel,
        cohort_column="cohort",
        base_period="varying",
    )
    result = att.fit(
        formula="cross_party_agreement",
        est_method="dr",
        control_group="not_yet_treated",
        sample_split_column="competitive",
        boot_iterations=499,
        random_state=SEED,
        n_jobs=1,
        progress_bar=False,
    )

    output = {}
    output["event"] = _flatten_aggregate(result.aggregate("event"))
    output["simple"] = _flatten_aggregate(result.aggregate("simple"))
    output["cohort"] = _flatten_aggregate(result.aggregate("cohort"))
    try:
        output["ddd_event"] = _flatten_aggregate(
            result.aggregate("event", difference=["competitive = 1", "competitive = 0"])
        )
    except Exception as e:
        output["ddd_event_error"] = str(e)
    try:
        output["ddd_simple"] = _flatten_aggregate(
            result.aggregate("simple", difference=["competitive = 1", "competitive = 0"])
        )
    except Exception as e:
        output["ddd_simple_error"] = str(e)

    return output


def summarize_pyfixest(fit, label):
    try:
        tidy = fit.tidy()
        coefs = {
            row.name: {
                "estimate": float(row["Estimate"]),
                "std_error": float(row["Std. Error"]),
                "t_stat": float(row["t value"]),
                "p_value": float(row["Pr(>|t|)"]),
                "ci_low": float(row["2.5%"]),
                "ci_high": float(row["97.5%"]),
            }
            for _, row in tidy.iterrows()
        }
        return {
            "label": label,
            "n_obs": int(fit._N),
            "coefficients": coefs,
        }
    except Exception as e:
        return {"label": label, "error": str(e)}


def main():
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_district_panel()
    panel = assign_cohorts(panel)
    panel_path = RESULTS_DIR / "staggered_district_panel.csv"
    panel.to_csv(panel_path, index=False)

    stacked = build_stacked_panel(panel)
    stacked_path = RESULTS_DIR / "staggered_stacked_panel.csv"
    stacked.to_csv(stacked_path, index=False)

    headline_fit = fit_stacked_event_study(stacked)
    ddd_fit = fit_stacked_ddd(stacked)
    safe_fit = fit_stacked_by_stratum(stacked, 0)
    comp_fit = fit_stacked_by_stratum(stacked, 1)
    wave_fits = fit_wave_specific(stacked)

    summaries = {
        "headline_stacked_event_study": summarize_pyfixest(headline_fit, "headline"),
        "stacked_ddd": summarize_pyfixest(ddd_fit, "stacked_ddd"),
        "safe_stratum": summarize_pyfixest(safe_fit, "safe"),
        "competitive_stratum": summarize_pyfixest(comp_fit, "competitive"),
        "wave_specific": {
            str(w): summarize_pyfixest(f, f"wave_{w}") if not isinstance(f, dict) else f
            for w, f in wave_fits.items()
        },
    }

    cs_results = run_callaway_santanna(panel)

    output = {
        "config": {
            "wave_congresses": WAVE_CONGRESSES,
            "competitive_margin_cutoff": COMPETITIVE_MARGIN_CUTOFF,
            "window_pre": WINDOW_PRE,
            "window_post": WINDOW_POST,
            "panel_path": str(panel_path),
            "stacked_path": str(stacked_path),
        },
        "panel_summary": {
            "n_district_congress_obs": int(len(panel)),
            "n_districts": int(panel["district_id"].nunique()),
            "n_congresses": int(panel["congress"].nunique()),
            "n_wave_freshmen": int(panel[(panel["is_wave_congress"] == 1) & (panel["is_freshman"] == 1)].shape[0]),
            "cohort_distribution": panel.groupby("cohort").size().to_dict(),
        },
        "stacked_summary": {
            "n_stacked_obs": int(len(stacked)),
            "n_stack_districts": int(stacked["stack_district"].nunique()),
            "treated_per_wave": {int(w): int(stacked[(stacked["stack_id"] == w) & (stacked["treated"] == 1) & (stacked["event_time"] == 0)].shape[0]) for w in WAVE_CONGRESSES},
        },
        "pyfixest_fits": summaries,
        "callaway_santanna": cs_results,
    }

    out_path = RESULTS_DIR / "staggered_did_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Panel: {len(panel)} district-Congress obs, {panel['district_id'].nunique()} districts")
    print(f"Stacked: {len(stacked)} obs, {stacked['stack_district'].nunique()} stack-districts")
    print()
    print("Headline stacked event study:")
    print(headline_fit.tidy())
    print()
    print("Treated per wave (event time 0):")
    for w in WAVE_CONGRESSES:
        n = stacked[(stacked["stack_id"] == w) & (stacked["treated"] == 1) & (stacked["event_time"] == 0)].shape[0]
        print(f"  {w}: {n}")
    print()
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
