import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, PROCESSED_DIR, RESULTS_DIR, CONGRESSES,
    MIN_VOTES, MIN_SHARED_VOTES, THRESHOLD_TAU,
)

import numpy as np
import pandas as pd
from spectral_analysis import fiedler_value


PROCEDURAL_PATTERNS = [
    "On Approving the Journal",
    "On Motion to Adjourn",
    "On Ordering the Previous Question",
    "On Motion that the Committee Rise",
    "Table Appeal of the Ruling of the Chair",
    "Table Motion to Reconsider",
    "On Motion to Table",
    "QUORUM",
    "ADJOURN",
    "JOURNAL",
]

SUSPENSION_PATTERNS = [
    "On Motion to Suspend the Rules and Pass",
    "On Motion to Suspend the Rules and Agree",
    "Suspend the rules and pass",
    "SUSPEND THE RULES AND PASS",
    "Suspend the rules and agree",
    "SUSPEND THE RULES AND AGREE",
]


def is_procedural(vote_question, dtl_desc):
    vq = str(vote_question) if pd.notna(vote_question) else ""
    dd = str(dtl_desc).upper() if pd.notna(dtl_desc) else ""

    for pattern in PROCEDURAL_PATTERNS:
        if pattern.lower() in vq.lower() or pattern.upper() in dd:
            return True
    return False


def is_suspension(vote_question):
    vq = str(vote_question) if pd.notna(vote_question) else ""
    for pattern in SUSPENSION_PATTERNS:
        if pattern.lower() in vq.lower():
            return True
    return False


def compute_filtered_fiedler(congress_num, members_all, votes_all, rollcalls_all,
                              exclude_procedural=True, exclude_suspension=True):
    rc = rollcalls_all[
        (rollcalls_all["congress"] == congress_num)
        & (rollcalls_all["chamber"] == "House")
    ].copy()

    excluded_rolls = set()
    for _, row in rc.iterrows():
        if exclude_procedural and is_procedural(row.get("vote_question"), row.get("dtl_desc")):
            excluded_rolls.add(row["rollnumber"])
        elif exclude_suspension and is_suspension(row.get("vote_question")):
            excluded_rolls.add(row["rollnumber"])

    members = members_all[
        (members_all["congress"] == congress_num)
        & (members_all["chamber"] == "House")
        & (members_all["party_code"].isin([100, 200]))
    ].copy()

    votes = votes_all[
        (votes_all["congress"] == congress_num)
        & (votes_all["chamber"] == "House")
        & (votes_all["cast_code"].isin([1, 2, 3, 4, 5, 6]))
    ].copy()

    votes = votes[~votes["rollnumber"].isin(excluded_rolls)]
    votes["vote"] = (votes["cast_code"].isin([1, 2, 3])).astype(float)

    vote_counts = votes.groupby("icpsr").size()
    min_votes_filtered = max(MIN_VOTES // 2, 25)
    valid_icpsrs = vote_counts[vote_counts >= min_votes_filtered].index
    members = members[members["icpsr"].isin(valid_icpsrs)].reset_index(drop=True)
    votes = votes[votes["icpsr"].isin(valid_icpsrs)]

    if len(members) < 10:
        return None, 0, 0

    icpsr_list = members["icpsr"].values
    icpsr_to_idx = {icpsr: i for i, icpsr in enumerate(icpsr_list)}
    n = len(icpsr_list)

    rollcalls_kept = sorted(votes["rollnumber"].unique())
    roll_to_col = {r: j for j, r in enumerate(rollcalls_kept)}
    n_rolls = len(rollcalls_kept)

    vote_matrix = np.full((n, n_rolls), np.nan, dtype=np.float32)
    for _, row in votes.iterrows():
        i = icpsr_to_idx.get(row["icpsr"])
        j = roll_to_col.get(row["rollnumber"])
        if i is not None and j is not None:
            vote_matrix[i, j] = row["vote"]

    valid_mask = (~np.isnan(vote_matrix)).astype(np.float32)
    vm_filled = np.where(valid_mask > 0, vote_matrix, 0.0).astype(np.float32)

    both_valid = valid_mask @ valid_mask.T
    both_yea = vm_filled @ vm_filled.T
    both_nay = ((1.0 - vm_filled) * valid_mask) @ ((1.0 - vm_filled) * valid_mask).T
    agree_count = both_yea + both_nay

    agreement = np.zeros_like(both_valid)
    mask = both_valid >= MIN_SHARED_VOTES
    agreement[mask] = agree_count[mask] / both_valid[mask]
    np.fill_diagonal(agreement, 0.0)

    adjacency = (agreement > THRESHOLD_TAU).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)

    f_val, _ = fiedler_value(adjacency)
    total_rolls = len(rc)
    kept_rolls = n_rolls

    return f_val, total_rolls, kept_rolls


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Voteview data...")
    members_all = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    votes_all = pd.read_csv(DATA_DIR / "HSall_votes.csv", low_memory=False)
    rollcalls_all = pd.read_csv(DATA_DIR / "HSall_rollcalls.csv", low_memory=False)

    results = {}

    for c in CONGRESSES:
        print(f"Congress {c}:", end=" ", flush=True)

        path = PROCESSED_DIR / f"congress_{c}.npz"
        if path.exists():
            data = np.load(path, allow_pickle=True)
            full_fiedler, _ = fiedler_value(data["adjacency"])
        else:
            full_fiedler = None

        f_subst, total, kept = compute_filtered_fiedler(
            c, members_all, votes_all, rollcalls_all,
            exclude_procedural=True, exclude_suspension=True,
        )

        f_no_proc, _, _ = compute_filtered_fiedler(
            c, members_all, votes_all, rollcalls_all,
            exclude_procedural=True, exclude_suspension=False,
        )

        f_no_susp, _, _ = compute_filtered_fiedler(
            c, members_all, votes_all, rollcalls_all,
            exclude_procedural=False, exclude_suspension=True,
        )

        excluded = total - kept if total and kept else 0
        results[str(c)] = {
            "full_votes_fiedler": float(full_fiedler) if full_fiedler is not None else None,
            "substantive_only_fiedler": float(f_subst) if f_subst is not None else None,
            "no_procedural_fiedler": float(f_no_proc) if f_no_proc is not None else None,
            "no_suspension_fiedler": float(f_no_susp) if f_no_susp is not None else None,
            "total_rollcalls": total,
            "substantive_rollcalls": kept,
            "excluded_rollcalls": excluded,
        }
        print(f"full={full_fiedler:.4f}, subst={f_subst:.4f}, "
              f"kept={kept}/{total} votes"
              if f_subst is not None and full_fiedler is not None
              else "skipped")

    full_vals = []
    subst_vals = []
    for c in CONGRESSES:
        cs = str(c)
        if cs in results:
            fv = results[cs]["full_votes_fiedler"]
            sv = results[cs]["substantive_only_fiedler"]
            if fv is not None and sv is not None:
                full_vals.append(fv)
                subst_vals.append(sv)
    if full_vals:
        corr = float(np.corrcoef(full_vals, subst_vals)[0, 1])
        results["correlation"] = corr
        print(f"\nFull vs substantive Fiedler correlation: r = {corr:.4f}")

    with open(RESULTS_DIR / "vote_filtering_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Vote filtering analysis complete.")


if __name__ == "__main__":
    main()
