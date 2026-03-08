import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DIR, RESULTS_DIR, CONGRESSES, SEED,
    THRESHOLD_TAU, MIN_SHARED_VOTES,
)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import networkx as nx


def fiedler_value(adjacency):
    A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    isolated = degrees == 0
    if isolated.all():
        return 0.0
    keep = ~isolated
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    n = A_sub.shape[0]
    L = sparse.eye(n) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    if n < 3:
        return 0.0
    try:
        eigenvalues, _ = eigsh(L, k=2, which="SM")
        return float(sorted(eigenvalues)[1])
    except Exception:
        return 0.0


def configuration_model_null(adjacency, n_samples, rng):
    degrees = adjacency.astype(int).sum(axis=1).tolist()
    fiedler_vals = []
    for _ in range(n_samples):
        G = nx.configuration_model(degrees, create_using=nx.Graph, seed=rng)
        G.remove_edges_from(nx.selfloop_edges(G))
        adj = nx.to_numpy_array(G).astype(np.float32)
        fiedler_vals.append(fiedler_value(adj))
    return np.array(fiedler_vals)


def temporal_null(congress_list, agreement_matrices, party_codes_list,
                  member_ids_list, n_samples, rng):
    cross_party_rates = []
    for c_idx, c in enumerate(congress_list):
        agreement = agreement_matrices[c_idx]
        party = party_codes_list[c_idx]
        n = len(party)
        cross_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                if party[i] != party[j] and agreement[i, j] > 0:
                    cross_vals.append(agreement[i, j])
        if cross_vals:
            cross_party_rates.append(np.mean(cross_vals))
        else:
            cross_party_rates.append(0.0)

    congress_arr = np.array(congress_list, dtype=float)
    rates_arr = np.array(cross_party_rates)
    slope, intercept = np.polyfit(congress_arr, rates_arr, 1)

    results = {}
    for c_idx, c in enumerate(congress_list):
        agreement = agreement_matrices[c_idx].copy()
        party = party_codes_list[c_idx]
        n = len(party)

        predicted_rate = slope * c + intercept
        actual_rate = cross_party_rates[c_idx]
        if actual_rate <= 0:
            results[str(c)] = {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
            continue

        fiedler_vals = []
        for _ in range(n_samples):
            sim_agreement = agreement.copy()
            for i in range(n):
                for j in range(i + 1, n):
                    if party[i] != party[j] and sim_agreement[i, j] > 0:
                        noise = rng.normal(0, 0.03)
                        sim_agreement[i, j] = predicted_rate + noise
                        sim_agreement[j, i] = sim_agreement[i, j]

            sim_agreement = np.clip(sim_agreement, 0, 1)
            np.fill_diagonal(sim_agreement, 0)
            adj = (sim_agreement > THRESHOLD_TAU).astype(np.float32)
            np.fill_diagonal(adj, 0)
            fiedler_vals.append(fiedler_value(adj))

        fiedler_vals = np.array(fiedler_vals)
        results[str(c)] = {
            "mean": float(np.mean(fiedler_vals)),
            "ci_lo": float(np.percentile(fiedler_vals, 2.5)),
            "ci_hi": float(np.percentile(fiedler_vals, 97.5)),
        }

    return results, {"slope": float(slope), "intercept": float(intercept),
                     "observed_rates": cross_party_rates}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(SEED)
    N_SAMPLES = 1000

    config_results = {}
    agreement_matrices = []
    party_codes_list = []
    member_ids_list = []
    valid_congresses = []

    for c in CONGRESSES:
        path = PROCESSED_DIR / f"congress_{c}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        adjacency = data["adjacency"]
        agreement = data["agreement"]
        party_codes = data["party_codes"]
        member_ids = data["member_ids"]

        valid_congresses.append(c)
        agreement_matrices.append(agreement)
        party_codes_list.append(party_codes)
        member_ids_list.append(member_ids)

        print(f"Congress {c}: configuration model null ({N_SAMPLES} samples)...",
              end=" ", flush=True)
        empirical_fiedler = fiedler_value(adjacency)
        null_fiedlers = configuration_model_null(adjacency, N_SAMPLES, rng)

        p_above = float(np.mean(null_fiedlers >= empirical_fiedler))
        p_below = float(np.mean(null_fiedlers <= empirical_fiedler))

        config_results[str(c)] = {
            "empirical": empirical_fiedler,
            "null_mean": float(np.mean(null_fiedlers)),
            "null_std": float(np.std(null_fiedlers)),
            "null_ci_lo": float(np.percentile(null_fiedlers, 2.5)),
            "null_ci_hi": float(np.percentile(null_fiedlers, 97.5)),
            "p_above": p_above,
            "p_below": p_below,
        }
        print(f"empirical={empirical_fiedler:.4f}, "
              f"null={np.mean(null_fiedlers):.4f} "
              f"[{np.percentile(null_fiedlers, 2.5):.4f}, "
              f"{np.percentile(null_fiedlers, 97.5):.4f}]")

    print("\nTemporal null model (linear decline)...", flush=True)
    temporal_results, trend_info = temporal_null(
        valid_congresses, agreement_matrices, party_codes_list,
        member_ids_list, n_samples=200, rng=rng,
    )

    output = {
        "configuration_model": config_results,
        "temporal_null": temporal_results,
        "temporal_trend": {
            "slope": trend_info["slope"],
            "intercept": trend_info["intercept"],
            "observed_cross_party_rates": {
                str(c): float(r)
                for c, r in zip(valid_congresses, trend_info["observed_rates"])
            },
        },
        "n_samples_config": N_SAMPLES,
        "n_samples_temporal": 200,
    }

    with open(RESULTS_DIR / "null_model_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Null model analysis complete.")


if __name__ == "__main__":
    main()
