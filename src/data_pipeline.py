import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR, PROCESSED_DIR, CONGRESSES,
    MIN_VOTES, MIN_SHARED_VOTES, THRESHOLD_TAU, DEFECTION_THRESHOLD,
)

import numpy as np
import pandas as pd


def load_voteview():
    members = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    votes = pd.read_csv(DATA_DIR / "HSall_votes.csv", low_memory=False)
    return members, votes


def process_congress(congress_num, members_all, votes_all):
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

    votes["vote"] = (votes["cast_code"].isin([1, 2, 3])).astype(float)

    vote_counts = votes.groupby("icpsr").size()
    valid_icpsrs = vote_counts[vote_counts >= MIN_VOTES].index
    members = members[members["icpsr"].isin(valid_icpsrs)].reset_index(drop=True)
    votes = votes[votes["icpsr"].isin(valid_icpsrs)]

    if len(members) < 10:
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

    party_majority = np.zeros(n_rolls, dtype=np.float32)
    for j in range(n_rolls):
        col = vote_matrix[:, j]
        for p_code in [100, 200]:
            p_mask = party_codes == p_code
            p_votes = col[p_mask & ~np.isnan(col)]
            if len(p_votes) > 0:
                party_majority[j] = 1.0 if p_votes.mean() > 0.5 else 0.0

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

    labels = (defection_rates >= DEFECTION_THRESHOLD).astype(np.int32)

    return {
        "adjacency": adjacency,
        "agreement": agreement,
        "features": features,
        "labels": labels,
        "defection_rates": defection_rates,
        "member_ids": icpsr_list,
        "party_codes": party_codes,
        "member_names": members["bioname"].values,
        "nominate_dim1": nom1,
        "state_abbrev": members["state_abbrev"].values,
    }


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Voteview data...")
    members_all, votes_all = load_voteview()

    for congress_num in CONGRESSES:
        print(f"Processing congress {congress_num}...", end=" ")
        result = process_congress(congress_num, members_all, votes_all)
        if result is None:
            print("skipped (too few members)")
            continue

        out_path = PROCESSED_DIR / f"congress_{congress_num}.npz"
        np.savez(
            out_path,
            adjacency=result["adjacency"],
            agreement=result["agreement"],
            features=result["features"],
            labels=result["labels"],
            defection_rates=result["defection_rates"],
            member_ids=result["member_ids"],
            party_codes=result["party_codes"],
            member_names=result["member_names"],
            nominate_dim1=result["nominate_dim1"],
            state_abbrev=result["state_abbrev"],
        )

        n = len(result["member_ids"])
        n_edges = int(result["adjacency"].sum()) // 2
        n_defectors = int(result["labels"].sum())
        print(f"n={n}, edges={n_edges}, defectors={n_defectors}")

    print("Data pipeline complete.")


if __name__ == "__main__":
    main()
