"""Counterfactual sensitivity grid: vary k, tau, and overlap filter."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR, SEED

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def fiedler_value(adjacency):
    A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    keep = degrees > 0
    if keep.sum() < 3:
        return 0.0
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    n = A_sub.shape[0]
    L = sparse.eye(n) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    try:
        eigenvalues, _ = eigsh(L, k=2, which="SM")
        return float(sorted(eigenvalues)[1])
    except Exception:
        return 0.0


def run_sensitivity():
    rng = np.random.RandomState(SEED)

    data_118 = np.load(PROCESSED_DIR / "congress_118.npz", allow_pickle=True)
    adj_118 = data_118["adjacency"].astype(np.float64)
    nom_118 = data_118["nominate_dim1"]
    party_118 = data_118["party_codes"]
    n = adj_118.shape[0]

    data_103 = np.load(PROCESSED_DIR / "congress_103.npz", allow_pickle=True)
    party_103 = data_103["party_codes"]
    adj_103 = data_103["adjacency"].astype(np.float64)
    nom_103 = data_103["nominate_dim1"]

    # 103rd cross-party edge density among 40 most moderate
    med_103 = np.median(nom_103)
    mod_mask_103 = np.argsort(np.abs(nom_103 - med_103))[:40]
    cross_edges_103 = 0
    cross_pairs_103 = 0
    for i in mod_mask_103:
        for j in range(adj_103.shape[0]):
            if party_103[i] != party_103[j]:
                cross_pairs_103 += 1
                if adj_103[i, j] > 0:
                    cross_edges_103 += 1
    cross_density_103 = cross_edges_103 / max(cross_pairs_103, 1)
    print(f"103rd cross-party edge density among moderates: {cross_density_103:.3f}")

    med_118 = np.median(nom_118)
    mod_order = np.argsort(np.abs(nom_118 - med_118))

    base_fiedler = fiedler_value(adj_118)
    print(f"118th base Fiedler: {base_fiedler:.4f}")

    k_values = [20, 30, 40, 50]
    overlap_filters = {"none": 1.0, "loose": 0.60, "strict": 0.30}

    results = {}
    print(f"\n{'k':>4} {'overlap':>8} {'edges_added':>12} {'new_fiedler':>12} {'ratio':>8}")
    print("-" * 50)

    for k in k_values:
        moderates = mod_order[:k]
        for overlap_name, overlap_frac in overlap_filters.items():
            adj_new = adj_118.copy()
            edges_added = 0
            for i in moderates:
                cross_party_targets = []
                for j in range(n):
                    if party_118[j] != party_118[i] and adj_new[i, j] == 0 and i != j:
                        cross_party_targets.append(j)
                n_to_add = int(len(cross_party_targets) * overlap_frac * cross_density_103)
                if n_to_add > 0 and len(cross_party_targets) > 0:
                    chosen = rng.choice(cross_party_targets,
                                       size=min(n_to_add, len(cross_party_targets)),
                                       replace=False)
                    for j in chosen:
                        adj_new[i, j] = 1.0
                        adj_new[j, i] = 1.0
                        edges_added += 1
            new_fiedler = fiedler_value(adj_new)
            ratio = new_fiedler / base_fiedler if base_fiedler > 0 else 0
            key = f"k={k}_overlap={overlap_name}"
            results[key] = {
                "k": k, "overlap": overlap_name,
                "edges_added": edges_added,
                "fiedler": round(new_fiedler, 4),
                "ratio_vs_base": round(ratio, 2),
            }
            print(f"{k:>4} {overlap_name:>8} {edges_added:>12} {new_fiedler:>12.4f} {ratio:>8.2f}")

    results["base_fiedler_118"] = round(base_fiedler, 4)
    results["cross_density_103"] = round(cross_density_103, 3)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "counterfactual_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'counterfactual_sensitivity.json'}")


if __name__ == "__main__":
    run_sensitivity()
