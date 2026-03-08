import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR, CONGRESSES

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def weighted_fiedler(agreement_matrix):
    W = agreement_matrix.copy()
    np.fill_diagonal(W, 0)
    W[W < 0] = 0

    degrees = W.sum(axis=1)
    isolated = degrees == 0
    if isolated.all():
        return 0.0
    keep = ~isolated
    W_sub = W[np.ix_(keep, keep)]
    d_sub = degrees[keep]

    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_sub))
    n = W_sub.shape[0]
    L = np.eye(n) - D_inv_sqrt @ W_sub @ D_inv_sqrt

    if n < 3:
        return 0.0

    L_sparse = sparse.csr_matrix(L)
    try:
        eigenvalues, _ = eigsh(L_sparse, k=2, which="SM")
        return float(sorted(eigenvalues)[1])
    except Exception:
        return 0.0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for c in CONGRESSES:
        path = PROCESSED_DIR / f"congress_{c}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        agreement = data["agreement"]
        adjacency = data["adjacency"]

        w_fiedler = weighted_fiedler(agreement)

        from spectral_analysis import fiedler_value
        b_fiedler, _ = fiedler_value(adjacency)

        results[str(c)] = {
            "weighted_fiedler": w_fiedler,
            "binary_fiedler": b_fiedler,
        }
        print(f"Congress {c}: weighted={w_fiedler:.4f}, binary={b_fiedler:.4f}")

    weighted_vals = [results[str(c)]["weighted_fiedler"] for c in CONGRESSES
                     if str(c) in results]
    binary_vals = [results[str(c)]["binary_fiedler"] for c in CONGRESSES
                   if str(c) in results]
    if weighted_vals and binary_vals:
        corr = np.corrcoef(weighted_vals, binary_vals)[0, 1]
        results["correlation"] = float(corr)
        print(f"\nWeighted-binary Fiedler correlation: r = {corr:.4f}")

    with open(RESULTS_DIR / "weighted_spectral_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Weighted spectral analysis complete.")


if __name__ == "__main__":
    main()
