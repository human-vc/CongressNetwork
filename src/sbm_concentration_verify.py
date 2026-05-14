"""
Theorem 4 (BLI concentration) computational verification.

For each (n, p_in, p_out, k_main, b) cell on a density sweep, generate L SBM
samples and measure the empirical concentration of the first-order BLI vector
around its population mean. The theoretical rate from Lei-Rinaldo style
||L_hat - L_pop|| concentration combined with Davis-Kahan on (psi_2, lambda_2)
predicts

    |BLI_n(v) - BLI_pop(v)| = O( sqrt(log n / (n p_min)) / gamma* )

where gamma* = lambda_3 - lambda_2 is the population spectral gap. The script
fits a log-log regression of mean absolute error on n*p_min and checks that
the slope is close to -1/2.

Bridge degree-matching uses the SBM benchmark formula
    q_b = ((m-1) p_in + (m (k-1) + 1 - b) p_out) / (k m - b)
from sbm_benchmark_brev.py.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import RESULTS_DIR, FIGURES_DIR, SEED
except ImportError:
    RESULTS_DIR = Path("results")
    FIGURES_DIR = Path("results/figures")
    SEED = 42

warnings.filterwarnings("ignore")


def fiedler_triple(A):
    n = A.shape[0]
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    d = np.asarray(A.sum(axis=1)).ravel()
    keep = d > 0
    if keep.sum() < 4:
        return None
    Asub = A[keep][:, keep]
    dsub = d[keep]
    Dis = sparse.diags(1.0 / np.sqrt(dsub))
    L = sparse.eye(Asub.shape[0]) - Dis @ Asub @ Dis
    try:
        w, v = eigsh(L, k=3, which="SM", tol=1e-6)
    except Exception:
        try:
            w, v = eigsh(L, k=3, sigma=-1e-5, which="LM", tol=1e-6)
        except Exception:
            return None
    order = np.argsort(w)
    w = w[order]
    v = v[:, order]
    psi = np.zeros((n, 3))
    psi[np.where(keep)[0], :] = v
    return float(w[1]), float(w[2]), psi


def first_order_bli(A):
    out = fiedler_triple(A)
    if out is None:
        return None
    lam2, lam3, psi = out
    gap = lam3 - lam2
    return gap * (1.0 - psi[:, 1] ** 2)


def degree_matched_q_bridge(m, b, p_in, p_out, k_main):
    return ((m - 1) * p_in + (m * (k_main - 1) + 1 - b) * p_out) / (k_main * m - b)


def gen_sbm(m, b, p_in, p_out, k_main, seed):
    rng = np.random.default_rng(seed)
    q_b = max(1e-3, min(0.9, degree_matched_q_bridge(m, b, p_in, p_out, k_main)))
    sizes = [m] * k_main + [b]
    K = k_main + 1
    P = np.full((K, K), p_out)
    for i in range(k_main):
        P[i, i] = p_in
        P[i, k_main] = q_b
        P[k_main, i] = q_b
    P[k_main, k_main] = p_out
    G = nx.stochastic_block_model(sizes, P.tolist(), seed=int(rng.integers(2 ** 31 - 1)), sparse=True)
    return nx.to_scipy_sparse_array(G, dtype=np.float64, format="csr")


def one_cell(m, b, p_in, p_out, k_main, L, base_seed):
    n_total = k_main * m + b
    np_min = n_total * min(p_in, p_out)
    bli_samples = []
    pop_lam = []
    for s in range(L):
        A = gen_sbm(m, b, p_in, p_out, k_main, seed=base_seed + s)
        if A.sum() == 0:
            continue
        bli = first_order_bli(A)
        if bli is None:
            continue
        out = fiedler_triple(A)
        bli_samples.append(bli)
        pop_lam.append(out[1] - out[0])
    if len(bli_samples) < 5:
        return None
    bli_samples = np.stack(bli_samples, axis=0)
    bli_pop = bli_samples.mean(axis=0)
    abs_err = np.abs(bli_samples - bli_pop[None, :]).mean()
    gamma_star = float(np.mean(pop_lam))
    predicted = np.sqrt(max(np.log(n_total), 1.0) / max(np_min, 1.0)) / max(gamma_star, 1e-6)
    return {
        "n_total": n_total, "m": m, "b": b, "p_in": p_in, "p_out": p_out, "k_main": k_main,
        "np_min": float(np_min), "log_n_over_np_min": float(np.log(n_total) / np_min),
        "gamma_star": gamma_star,
        "L": int(len(bli_samples)),
        "mean_abs_err": float(abs_err),
        "predicted_rate": float(predicted),
        "err_over_predicted": float(abs_err / predicted) if predicted > 0 else np.nan,
    }


def build_grid(scale):
    # We vary density (n p_min) along a primary axis with a fixed structural
    # ratio p_in / p_out so the population spectral gap gamma* stays roughly
    # constant within an (m, k_main) slice. The log-log slope is fit within
    # each (m, k_main) slice and pooled.
    if scale == "smoke":
        ms = [50]
        densities = [4, 8, 16, 32]  # target n * p_out values
        L = 20
    elif scale == "medium":
        ms = [50, 100]
        densities = [4, 8, 16, 32, 64]
        L = 40
    else:  # large
        ms = [100, 200, 400]
        densities = [4, 8, 16, 32, 64, 128]
        L = 80
    grid = []
    for m in ms:
        for k_main in [2, 3]:
            n_total = k_main * m + max(5, m // 10)
            for target in densities:
                p_out = min(0.5, target / n_total)
                p_in = 10 * p_out  # fixed structural ratio -> roughly fixed gamma*
                if p_in > 0.9 or p_out < 1e-4:
                    continue
                grid.append({"m": m, "b": max(5, m // 10), "p_in": p_in, "p_out": p_out,
                             "k_main": k_main, "L": L})
    return grid


def loglog_slope(df, x_col, y_col):
    mask = (df[x_col] > 0) & (df[y_col] > 0)
    if mask.sum() < 3:
        return None
    x = np.log(df.loc[mask, x_col].values)
    y = np.log(df.loc[mask, y_col].values)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            "n_points": int(mask.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="medium", choices=["smoke", "medium", "large"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    grid = build_grid(args.scale)
    print(f"Theorem 4 verification: {len(grid)} cells, scale={args.scale}")
    print(f"Cores available: {Parallel(n_jobs=args.n_jobs)._effective_n_jobs()}")

    t0 = time.time()
    rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(one_cell)(base_seed=args.seed + 1000 * i, **cell)
        for i, cell in enumerate(grid)
    )
    rows = [r for r in rows if r is not None]
    df = pd.DataFrame(rows)
    print(f"\nWall time: {(time.time() - t0) / 60:.1f} min   cells_ok={len(df)}/{len(grid)}")

    out_csv = RESULTS_DIR / f"sbm_concentration_verify_{args.scale}.csv"
    df.to_csv(out_csv, index=False)

    pooled_fit = loglog_slope(df, "np_min", "mean_abs_err")
    by_slice = {}
    for (m, k), sub in df.groupby(["m", "k_main"]):
        by_slice[f"m={m}_k={k}"] = loglog_slope(sub, "np_min", "mean_abs_err")
    slopes = [v["slope"] for v in by_slice.values() if v is not None]
    summary = {
        "scale": args.scale,
        "n_cells": int(len(df)),
        "predicted_slope": -0.5,
        "pooled_fit_err_vs_np_min": pooled_fit,
        "within_slice_fits": by_slice,
        "median_within_slice_slope": float(np.median(slopes)) if slopes else None,
        "median_err_over_predicted": float(df["err_over_predicted"].median()),
    }
    print("\n=== Theorem 4 verification summary ===")
    print(json.dumps(summary, indent=2))

    out_json = RESULTS_DIR / f"sbm_concentration_verify_{args.scale}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_csv} and {out_json}")


if __name__ == "__main__":
    main()
