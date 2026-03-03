import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, PROCESSED_DIR, RESULTS_DIR, CONGRESSES,
    SEED, MIN_VOTES, MIN_SHARED_VOTES, THRESHOLD_TAU,
)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def normalized_laplacian(adjacency):
    A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    isolated = degrees == 0
    if isolated.all():
        return None, None
    keep = ~isolated
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    n = A_sub.shape[0]
    L = sparse.eye(n) - d_inv_sqrt @ A_sub @ d_inv_sqrt
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
    if len(shared) < 5:
        return 0.0
    if fiedler_prev is None or fiedler_curr is None:
        return 0.0
    prev_map = {icpsr: i for i, icpsr in enumerate(ids_prev)}
    curr_map = {icpsr: i for i, icpsr in enumerate(ids_curr)}
    v_prev = np.array([fiedler_prev[prev_map[icpsr]] for icpsr in shared])
    v_curr = np.array([fiedler_curr[curr_map[icpsr]] for icpsr in shared])
    if np.dot(v_prev, v_curr) < 0:
        v_curr = -v_curr
    return float(np.linalg.norm(v_curr - v_prev))


def counterfactual_removal(adjacency, bli_values, nominate_dim1, k=10):
    base_fiedler, _ = fiedler_value(adjacency)
    n = adjacency.shape[0]

    top_bli = np.argsort(-bli_values)[:k]
    mask_bli = np.ones(n, dtype=bool)
    mask_bli[top_bli] = False
    fiedler_no_bli, _ = fiedler_value(adjacency[np.ix_(mask_bli, mask_bli)])

    ideology_extremity = np.abs(nominate_dim1 - np.median(nominate_dim1))
    top_ideology = np.argsort(-ideology_extremity)[:k]
    mask_ideo = np.ones(n, dtype=bool)
    mask_ideo[top_ideology] = False
    fiedler_no_ideo, _ = fiedler_value(adjacency[np.ix_(mask_ideo, mask_ideo)])

    rng = np.random.RandomState(SEED)
    random_drops = []
    for _ in range(50):
        rand_idx = rng.choice(n, size=k, replace=False)
        mask_rand = np.ones(n, dtype=bool)
        mask_rand[rand_idx] = False
        f_rand, _ = fiedler_value(adjacency[np.ix_(mask_rand, mask_rand)])
        random_drops.append(f_rand)
    fiedler_no_random = float(np.mean(random_drops))

    return {
        "base": base_fiedler,
        "remove_top_bli": fiedler_no_bli,
        "remove_top_ideology": fiedler_no_ideo,
        "remove_random": fiedler_no_random,
        "delta_bli": base_fiedler - fiedler_no_bli,
        "delta_ideology": base_fiedler - fiedler_no_ideo,
        "delta_random": base_fiedler - fiedler_no_random,
    }


def process_senate_congress(congress_num, members_all, votes_all):
    members = members_all[
        (members_all["congress"] == congress_num)
        & (members_all["chamber"] == "Senate")
        & (members_all["party_code"].isin([100, 200]))
    ].copy()

    votes = votes_all[
        (votes_all["congress"] == congress_num)
        & (votes_all["chamber"] == "Senate")
        & (votes_all["cast_code"].isin([1, 2, 3, 4, 5, 6]))
    ].copy()

    votes["vote"] = (votes["cast_code"].isin([1, 2, 3])).astype(float)

    vote_counts = votes.groupby("icpsr").size()
    valid_icpsrs = vote_counts[vote_counts >= MIN_VOTES].index
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

    adjacency = (agreement > THRESHOLD_TAU).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)

    return adjacency


def main():
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    spectral_results = {}
    bli_results = {}
    prev_fiedler_vec = None
    prev_ids = None

    for congress_num in CONGRESSES:
        path = PROCESSED_DIR / f"congress_{congress_num}.npz"
        if not path.exists():
            print(f"Congress {congress_num}: no data, skipping")
            continue

        data = np.load(path, allow_pickle=True)
        adjacency = data["adjacency"]
        member_ids = data["member_ids"]
        nominate_dim1 = data["nominate_dim1"]
        party_codes = data["party_codes"]
        member_names = data["member_names"]

        print(f"Congress {congress_num}: computing Fiedler...", end=" ", flush=True)
        f_val, f_vec = fiedler_value(adjacency)

        L, keep = normalized_laplacian(adjacency)
        fiedler_full = np.zeros(len(member_ids))
        if f_vec is not None and keep is not None:
            fiedler_full[keep] = f_vec

        sri = 0.0
        if prev_fiedler_vec is not None and prev_ids is not None:
            sri = compute_sri(prev_fiedler_vec, fiedler_full, prev_ids, member_ids)

        prev_fiedler_vec = fiedler_full
        prev_ids = member_ids

        spectral_results[str(congress_num)] = {
            "fiedler": f_val,
            "sri": sri,
            "n_members": int(len(member_ids)),
            "n_edges": int(adjacency.sum()) // 2,
        }

        print(f"fiedler={f_val:.4f}, sri={sri:.4f}, computing BLI...", end=" ", flush=True)
        bli, base_f = compute_bli(adjacency)

        top_bli_idx = np.argsort(-bli)[:10]
        top_bli_members = []
        for idx in top_bli_idx:
            top_bli_members.append({
                "name": str(member_names[idx]),
                "icpsr": int(member_ids[idx]),
                "bli": float(bli[idx]),
                "party": int(party_codes[idx]),
                "nominate": float(nominate_dim1[idx]),
            })

        bli_results[str(congress_num)] = {
            "base_fiedler": base_f,
            "bli_values": bli.tolist(),
            "member_ids": member_ids.tolist(),
            "top_bli": top_bli_members,
        }

        if len(member_ids) >= 20:
            cf = counterfactual_removal(adjacency, bli, nominate_dim1, k=10)
            spectral_results[str(congress_num)]["counterfactual"] = cf

        print("done")

    print("\nComputing Senate trajectory...")
    members_all = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    votes_all = pd.read_csv(DATA_DIR / "HSall_votes.csv", low_memory=False)

    senate_fiedler = {}
    for congress_num in CONGRESSES:
        adj = process_senate_congress(congress_num, members_all, votes_all)
        if adj is None:
            continue
        f_val, _ = fiedler_value(adj)
        senate_fiedler[str(congress_num)] = f_val
        print(f"  Senate {congress_num}: fiedler={f_val:.4f}")

    spectral_results["senate_fiedler"] = senate_fiedler

    nominate_distance = {}
    for congress_num in CONGRESSES:
        path = PROCESSED_DIR / f"congress_{congress_num}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        nom = data["nominate_dim1"]
        party = data["party_codes"]
        dem_med = np.median(nom[party == 100])
        rep_med = np.median(nom[party == 200])
        nominate_distance[str(congress_num)] = float(abs(rep_med - dem_med))

    spectral_results["nominate_distance"] = nominate_distance

    with open(RESULTS_DIR / "spectral_results.json", "w") as f:
        json.dump(spectral_results, f, indent=2)

    with open(RESULTS_DIR / "bli_results.json", "w") as f:
        json.dump(bli_results, f, indent=2)

    print("Spectral analysis complete.")


if __name__ == "__main__":
    main()
