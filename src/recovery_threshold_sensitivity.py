"""Compute recovery ratios at tau=0.45, 0.50, 0.55."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RESULTS_DIR, MIN_VOTES, MIN_SHARED_VOTES

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh


def fiedler_at_threshold(members_all, votes_all, congress, tau):
    members = members_all[
        (members_all["congress"] == congress)
        & (members_all["chamber"] == "House")
        & (members_all["party_code"].isin([100, 200]))
    ].copy()
    votes = votes_all[
        (votes_all["congress"] == congress)
        & (votes_all["chamber"] == "House")
        & (votes_all["cast_code"].isin([1, 2, 3, 4, 5, 6]))
    ].copy()
    votes["vote"] = (votes["cast_code"].isin([1, 2, 3])).astype(float)
    vote_counts = votes.groupby("icpsr").size()
    valid_icpsrs = vote_counts[vote_counts >= MIN_VOTES].index
    members = members[members["icpsr"].isin(valid_icpsrs)].reset_index(drop=True)
    votes = votes[votes["icpsr"].isin(valid_icpsrs)]
    if len(members) < 5:
        return 0.0
    icpsr_list = members["icpsr"].values
    icpsr_to_idx = {icpsr: i for i, icpsr in enumerate(icpsr_list)}
    n = len(icpsr_list)
    rollcalls = sorted(votes["rollnumber"].unique())
    roll_to_col = {r: j for j, r in enumerate(rollcalls)}
    n_rolls = len(rollcalls)
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
    adjacency = (agreement > tau).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)

    A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    keep = degrees > 0
    if keep.sum() < 3:
        return 0.0
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    nn = A_sub.shape[0]
    L = sparse.eye(nn) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    try:
        eigenvalues, _ = eigsh(L, k=2, which="SM")
        return float(sorted(eigenvalues)[1])
    except Exception:
        return 0.0


def main():
    print("Loading vote data...")
    members_all = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    votes_all = pd.read_csv(DATA_DIR / "HSall_votes.csv", low_memory=False)

    congresses_needed = [100, 103, 105, 106, 107, 109, 111, 112, 114]
    thresholds = [0.45, 0.50, 0.55]

    shocks = {
        "Contract with America": {"pre": 100, "shock": 103, "post": 105},
        "9/11 Rally": {"pre": 106, "shock": 107, "post": 109},
        "Tea Party": {"pre": 111, "shock": 112, "post": 114},
    }

    results = {}
    for tau in thresholds:
        print(f"\nThreshold tau = {tau}")
        fiedlers = {}
        for c in congresses_needed:
            f = fiedler_at_threshold(members_all, votes_all, c, tau)
            fiedlers[c] = round(f, 4)
            print(f"  Congress {c}: lambda_2 = {f:.4f}")
        ratios = {}
        for name, spec in shocks.items():
            pre_f = fiedlers[spec["pre"]]
            ratio = fiedlers[spec["post"]] / pre_f if pre_f > 0 else 0
            ratios[name] = {
                "pre": fiedlers[spec["pre"]],
                "shock": fiedlers[spec["shock"]],
                "post": fiedlers[spec["post"]],
                "ratio": round(ratio, 2),
            }
            print(f"  {name}: {fiedlers[spec['pre']]} -> {fiedlers[spec['shock']]} -> {fiedlers[spec['post']]} (R={ratio:.2f})")
        results[str(tau)] = {"fiedlers": {str(k): v for k, v in fiedlers.items()}, "ratios": ratios}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "recovery_threshold_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'recovery_threshold_sensitivity.json'}")


if __name__ == "__main__":
    main()
