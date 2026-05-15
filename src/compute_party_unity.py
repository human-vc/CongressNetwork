import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RESULTS_DIR

YEA_CODES = {1, 2, 3}
NAY_CODES = {4, 5, 6}

def compute_party_unity(votes, members, rollcalls, exclude_procedural=False):
    members_h = members[(members.chamber == "House") & (members.party_code.isin([100, 200]))].copy()
    members_h["icpsr"] = pd.to_numeric(members_h["icpsr"], errors="coerce")
    members_h["congress"] = pd.to_numeric(members_h["congress"], errors="coerce")
    members_h = members_h.dropna(subset=["icpsr", "congress"])
    members_h["icpsr"] = members_h["icpsr"].astype(int)
    members_h["congress"] = members_h["congress"].astype(int)
    party_map = members_h.set_index(["congress", "icpsr"])["party_code"].to_dict()

    v = votes[votes.chamber == "House"].copy()
    v["yea"] = v["cast_code"].isin(YEA_CODES)
    v["nay"] = v["cast_code"].isin(NAY_CODES)
    v = v[v["yea"] | v["nay"]].copy()
    v["congress"] = v["congress"].astype(int)
    v["icpsr"] = v["icpsr"].astype(int)
    v["party_code"] = v.apply(lambda r: party_map.get((r["congress"], r["icpsr"]), np.nan), axis=1)
    v = v.dropna(subset=["party_code"])
    v["party_code"] = v["party_code"].astype(int)
    v = v[v["party_code"].isin([100, 200])]
    v["vote_yea"] = v["yea"].astype(int)

    if exclude_procedural and "vote_question" in rollcalls.columns:
        rc = rollcalls[rollcalls.chamber == "House"][["congress", "rollnumber", "vote_question"]].copy()
        rc["procedural"] = rc["vote_question"].astype(str).str.lower().str.contains(
            "procedural|previous question|order of business|motion to recommit",
            na=False,
        )
        proc = rc[rc["procedural"]][["congress", "rollnumber"]]
        before = len(v)
        v = v.merge(proc, on=["congress", "rollnumber"], how="left", indicator=True)
        v = v[v["_merge"] == "left_only"].drop(columns=["_merge"])
        print(f"  excluded procedural rollcalls: dropped {before - len(v):,} of {before:,} votes")

    party_pos = (
        v.groupby(["congress", "rollnumber", "party_code"])["vote_yea"]
        .mean()
        .unstack("party_code")
        .reset_index()
    )
    party_pos.columns.name = None
    party_pos = party_pos.rename(columns={100: "dem_yea_share", 200: "rep_yea_share"})
    party_pos["dem_pos"] = (party_pos["dem_yea_share"] >= 0.5).astype(int)
    party_pos["rep_pos"] = (party_pos["rep_yea_share"] >= 0.5).astype(int)
    party_pos["party_line"] = (party_pos["dem_pos"] != party_pos["rep_pos"]).astype(int)
    party_line_rolls = party_pos[party_pos["party_line"] == 1][
        ["congress", "rollnumber", "dem_pos", "rep_pos"]
    ]

    v_pl = v.merge(party_line_rolls, on=["congress", "rollnumber"], how="inner")
    v_pl["own_party_pos"] = np.where(v_pl["party_code"] == 100, v_pl["dem_pos"], v_pl["rep_pos"])
    v_pl["with_party"] = (v_pl["vote_yea"] == v_pl["own_party_pos"]).astype(int)
    agg = (
        v_pl.groupby(["congress", "icpsr", "party_code"])
        .agg(party_unity=("with_party", "mean"), n_party_line_votes=("with_party", "size"))
        .reset_index()
    )
    return agg

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading votes (this is the big file)...")
    votes = pd.read_csv(DATA_DIR / "HSall_votes.csv", low_memory=False)
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    rollcalls = pd.read_csv(DATA_DIR / "HSall_rollcalls.csv", low_memory=False)
    print(f"  votes: {len(votes):,}, members: {len(members):,}, rollcalls: {len(rollcalls):,}")

    print("\n=== All party-line votes ===")
    pu_all = compute_party_unity(votes, members, rollcalls, exclude_procedural=False)
    pu_all = pu_all[(pu_all.congress >= 100) & (pu_all.congress <= 116)]
    pu_all.to_csv(RESULTS_DIR / "party_unity.csv", index=False)
    print(pu_all.head())
    print(f"\nWrote {RESULTS_DIR / 'party_unity.csv'}: {len(pu_all)} member-congress rows")
    print(f"  party_unity summary: mean={pu_all.party_unity.mean():.3f} sd={pu_all.party_unity.std():.3f}")
    print(f"  n_party_line_votes: median={pu_all.n_party_line_votes.median():.0f}")

    print("\n=== Cox-McCubbins adjusted (no procedural) ===")
    pu_adj = compute_party_unity(votes, members, rollcalls, exclude_procedural=True)
    pu_adj = pu_adj[(pu_adj.congress >= 100) & (pu_adj.congress <= 116)]
    pu_adj = pu_adj.rename(columns={"party_unity": "party_unity_adj", "n_party_line_votes": "n_party_line_votes_adj"})
    pu_adj.to_csv(RESULTS_DIR / "party_unity_adjusted.csv", index=False)
    print(f"Wrote {RESULTS_DIR / 'party_unity_adjusted.csv'}: {len(pu_adj)} rows")

if __name__ == "__main__":
    main()
