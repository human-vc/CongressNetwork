import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RESULTS_DIR, CONGRESSES, MIN_SHARED_VOTES, THRESHOLD_TAU

from scipy import sparse
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore")

SENATE_DIR = DATA_DIR / "processed_senate"
SENATE_MIN_VOTES = 30
SENATE_THRESHOLD_TAU = 0.6


def process_senate_congress(c, members_all, votes_all):
    members = members_all[
        (members_all["congress"] == c)
        & (members_all["chamber"] == "Senate")
        & (members_all["party_code"].isin([100, 200]))
    ].copy()

    votes = votes_all[
        (votes_all["congress"] == c)
        & (votes_all["chamber"] == "Senate")
        & (votes_all["cast_code"].isin([1, 2, 3, 4, 5, 6]))
    ].copy()
    votes["vote"] = (votes["cast_code"].isin([1, 2, 3])).astype(float)

    vote_counts = votes.groupby("icpsr").size()
    valid_icpsrs = vote_counts[vote_counts >= SENATE_MIN_VOTES].index
    members = members[members["icpsr"].isin(valid_icpsrs)].reset_index(drop=True)
    votes = votes[votes["icpsr"].isin(valid_icpsrs)]

    if len(members) < 5:
        return None

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
    adjacency = (agreement > SENATE_THRESHOLD_TAU).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)

    party_codes = members["party_code"].values
    party_binary = (party_codes == 200).astype(np.float32)
    nom1 = members["nominate_dim1"].fillna(0.0).values.astype(np.float32)
    nom2 = members["nominate_dim2"].fillna(0.0).values.astype(np.float32)

    participation = valid_mask.sum(axis=1) / n_rolls
    yea_rate = np.nanmean(vote_matrix, axis=1)
    yea_rate = np.nan_to_num(yea_rate, nan=0.5)

    mean_agreement = np.zeros(n, dtype=np.float32)
    mean_cross = np.zeros(n, dtype=np.float32)
    mean_within = np.zeros(n, dtype=np.float32)
    for i in range(n):
        others = agreement[i]
        nonzero = others > 0
        if nonzero.any():
            mean_agreement[i] = others[nonzero].mean()
        cross_mask = nonzero & (party_binary != party_binary[i])
        if cross_mask.any():
            mean_cross[i] = others[cross_mask].mean()
        within_mask = nonzero & (party_binary == party_binary[i])
        if within_mask.any():
            mean_within[i] = others[within_mask].mean()

    features = np.stack([
        nom1, nom2, party_binary, participation, yea_rate,
        mean_agreement, mean_cross, mean_within,
    ], axis=1).astype(np.float32)

    defection_rates = np.zeros(n, dtype=np.float32)
    for i in range(n):
        voted_mask = ~np.isnan(vote_matrix[i])
        if voted_mask.sum() == 0:
            continue
        my_party = party_codes[i]
        n_defections = 0
        n_voted = 0
        for j in np.where(voted_mask)[0]:
            same_party_mask = (party_codes == my_party) & (~np.isnan(vote_matrix[:, j]))
            if same_party_mask.sum() < 3:
                continue
            party_mean = vote_matrix[same_party_mask, j].mean()
            party_maj = 1.0 if party_mean > 0.5 else 0.0
            n_voted += 1
            if vote_matrix[i, j] != party_maj:
                n_defections += 1
        if n_voted > 0:
            defection_rates[i] = n_defections / n_voted

    return {
        "adjacency": adjacency,
        "agreement": agreement,
        "features": features,
        "defection_rates": defection_rates,
        "member_ids": icpsr_list,
        "party_codes": party_codes,
        "member_names": members["bioname"].values,
        "nominate_dim1": nom1,
        "state_abbrev": members["state_abbrev"].values,
    }


def normalized_laplacian(adjacency):
    A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    keep = degrees > 0
    if keep.sum() < 3:
        return None, None
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    L = sparse.eye(A_sub.shape[0]) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    return L, keep


def fiedler_value(adjacency):
    L, keep = normalized_laplacian(adjacency)
    if L is None or L.shape[0] < 3:
        return 0.0, None
    try:
        eigenvalues, eigenvectors = eigsh(L, k=2, which="SM")
        idx = np.argsort(eigenvalues)
        return float(eigenvalues[idx[1]]), eigenvectors[:, idx[1]]
    except Exception:
        return 0.0, None


def compute_bli(adjacency):
    n = adjacency.shape[0]
    base_fiedler, _ = fiedler_value(adjacency)
    bli = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sub_adj = adjacency[np.ix_(mask, mask)]
        removed_fiedler, _ = fiedler_value(sub_adj)
        bli[i] = base_fiedler - removed_fiedler
    return bli, base_fiedler


def compute_sri(fiedler_prev, fiedler_curr, ids_prev, ids_curr):
    shared = sorted(set(ids_prev) & set(ids_curr))
    if len(shared) < 5 or fiedler_prev is None or fiedler_curr is None:
        return 0.0
    prev_map = {icpsr: i for i, icpsr in enumerate(ids_prev)}
    curr_map = {icpsr: i for i, icpsr in enumerate(ids_curr)}
    v_prev = np.array([fiedler_prev[prev_map[icpsr]] for icpsr in shared])
    v_curr = np.array([fiedler_curr[curr_map[icpsr]] for icpsr in shared])
    if np.dot(v_prev, v_curr) < 0:
        v_curr = -v_curr
    return float(np.linalg.norm(v_curr - v_prev))


def main():
    SENATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Voteview data...")
    members_all = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    votes_all = pd.read_csv(DATA_DIR / "HSall_votes.csv", low_memory=False)

    spectral_results = {}
    bli_results = {}
    prev_fiedler_vec = None
    prev_ids = None

    for congress_num in CONGRESSES:
        print(f"Congress {congress_num}: processing Senate...", end=" ", flush=True)
        result = process_senate_congress(congress_num, members_all, votes_all)
        if result is None:
            print("skipped")
            continue

        out_path = SENATE_DIR / f"congress_{congress_num}.npz"
        np.savez(
            out_path,
            adjacency=result["adjacency"],
            agreement=result["agreement"],
            features=result["features"],
            defection_rates=result["defection_rates"],
            member_ids=result["member_ids"],
            party_codes=result["party_codes"],
            member_names=result["member_names"],
            nominate_dim1=result["nominate_dim1"],
            state_abbrev=result["state_abbrev"],
        )

        adj = result["adjacency"]
        f_val, f_vec = fiedler_value(adj)
        L, keep = normalized_laplacian(adj)
        fiedler_full = np.zeros(len(result["member_ids"]))
        if f_vec is not None and keep is not None:
            fiedler_full[keep] = f_vec
        sri = 0.0
        if prev_fiedler_vec is not None and prev_ids is not None:
            sri = compute_sri(prev_fiedler_vec, fiedler_full,
                              list(prev_ids), list(result["member_ids"]))
        prev_fiedler_vec = fiedler_full
        prev_ids = result["member_ids"]

        n_edges = int(adj.sum()) // 2
        spectral_results[str(congress_num)] = {
            "fiedler": f_val,
            "sri": sri,
            "n_members": int(len(result["member_ids"])),
            "n_edges": n_edges,
        }

        print(f"n={len(result['member_ids'])}, fiedler={f_val:.4f}, sri={sri:.4f}, computing BLI...", end=" ", flush=True)
        bli, base_f = compute_bli(adj)

        top_idx = np.argsort(-bli)[:10]
        top_members = []
        for i in top_idx:
            top_members.append({
                "name": str(result["member_names"][i]),
                "icpsr": int(result["member_ids"][i]),
                "state": str(result["state_abbrev"][i]),
                "bli": float(bli[i]),
                "party": int(result["party_codes"][i]),
                "nominate": float(result["nominate_dim1"][i]),
            })
        bli_results[str(congress_num)] = {
            "base_fiedler": base_f,
            "bli_values": bli.tolist(),
            "member_ids": result["member_ids"].tolist(),
            "top_bli": top_members,
        }
        print("done")

    with open(RESULTS_DIR / "senate_spectral_results.json", "w") as f:
        json.dump(spectral_results, f, indent=2)
    with open(RESULTS_DIR / "senate_bli_results.json", "w") as f:
        json.dump(bli_results, f, indent=2)

    print()
    print(f"Senate Fiedler trajectory:")
    for c in sorted([int(k) for k in spectral_results]):
        s = spectral_results[str(c)]
        print(f"  {c}: fiedler={s['fiedler']:.4f}, sri={s['sri']:.4f}, n={s['n_members']}")
    print(f"\nSaved spectral to results/senate_spectral_results.json")
    print(f"Saved BLI to results/senate_bli_results.json")


if __name__ == "__main__":
    main()
