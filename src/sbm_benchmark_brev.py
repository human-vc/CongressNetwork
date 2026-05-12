import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse, stats
from scipy.sparse.linalg import eigsh
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import RESULTS_DIR, FIGURES_DIR, SEED
except ImportError:
    RESULTS_DIR = Path("results")
    FIGURES_DIR = Path("paper/paper_figures")
    SEED = 42

warnings.filterwarnings("ignore")


def fiedler_pair_sparse(adjacency, k=3):
    n = adjacency.shape[0]
    if sparse.issparse(adjacency):
        A = adjacency.tocsr()
    else:
        A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    keep = degrees > 0
    if keep.sum() < k + 1:
        return None, None, None, None
    keep_idx = np.where(keep)[0]
    A_sub = A[keep_idx][:, keep_idx]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    L = sparse.eye(A_sub.shape[0]) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    try:
        eigs, vecs = eigsh(L, k=k, sigma=0, which="LM", tol=1e-7)
        order = np.argsort(eigs)
        eigs = eigs[order]
        vecs = vecs[:, order]
        psi_full = np.zeros((n, k))
        psi_full[keep_idx, :] = vecs
        return float(eigs[1]), float(eigs[2]) if k >= 3 else None, psi_full, keep
    except Exception:
        try:
            eigs, vecs = eigsh(L, k=k, which="SM", tol=1e-6)
            order = np.argsort(eigs)
            eigs = eigs[order]
            vecs = vecs[:, order]
            psi_full = np.zeros((n, k))
            psi_full[keep_idx, :] = vecs
            return float(eigs[1]), float(eigs[2]) if k >= 3 else None, psi_full, keep
        except Exception:
            return None, None, None, None


def fiedler_value_only(adjacency):
    lam2, _, _, _ = fiedler_pair_sparse(adjacency, k=2)
    return lam2 if lam2 is not None else 0.0


def first_order_bli(A):
    n = A.shape[0]
    lam2, lam3, psi, _ = fiedler_pair_sparse(A, k=3)
    if lam2 is None or psi is None:
        return np.zeros(n)
    psi2 = psi[:, 1]
    gap = (lam3 - lam2) if lam3 is not None else lam2
    return gap * (1.0 - psi2 ** 2)


def brute_force_bli(A, n_jobs=1):
    n = A.shape[0]
    base = fiedler_value_only(A)
    if base == 0.0:
        return np.zeros(n)

    def single(i):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        return base - fiedler_value_only(A[np.ix_(mask, mask)])

    if n_jobs > 1:
        results = Parallel(n_jobs=n_jobs)(delayed(single)(i) for i in range(n))
        return np.array(results)
    return np.array([single(i) for i in range(n)])


def comparator_centralities(A):
    n = A.shape[0]
    G = nx.from_numpy_array(A)
    isolated = [i for i in range(n) if G.degree(i) == 0]
    G_keep = G.copy()
    G_keep.remove_nodes_from(isolated)

    def fill(d):
        arr = np.zeros(n)
        for i, v in d.items():
            arr[i] = v
        return arr

    out = {}
    for name, fn in [
        ("betweenness", lambda: nx.betweenness_centrality(G_keep, normalized=True)),
        ("closeness", lambda: nx.closeness_centrality(G_keep)),
        ("pagerank", lambda: nx.pagerank(G_keep, alpha=0.85)),
    ]:
        try:
            out[name] = fill(fn())
        except Exception:
            out[name] = np.zeros(n)

    try:
        out["eigenvector"] = fill(nx.eigenvector_centrality_numpy(G_keep))
    except Exception:
        out["eigenvector"] = np.zeros(n)

    try:
        kc = nx.core_number(G_keep)
        if kc:
            mc = max(kc.values())
            out["kcore"] = fill({k: v / mc for k, v in kc.items()} if mc > 0 else kc)
        else:
            out["kcore"] = np.zeros(n)
    except Exception:
        out["kcore"] = np.zeros(n)

    deg = np.array([G.degree(i) for i in range(n)], dtype=float)
    if deg.max() > 0:
        deg = deg / deg.max()
    out["degree"] = deg
    return out


def degree_matched_q_bridge(m, b, p_in, p_out, k_main):
    return ((m - 1) * p_in + (m * (k_main - 1) + 1 - b) * p_out) / (k_main * m - b)


def generate_degree_matched_sbm(m, b, p_in, p_out, k_main=2, rng=None):
    rng = rng or np.random.default_rng()
    q_b = degree_matched_q_bridge(m, b, p_in, p_out, k_main)
    q_b = max(0.001, min(0.9, q_b))
    sizes = [m] * k_main + [b]
    K = k_main + 1
    P = np.full((K, K), p_out, dtype=float)
    for i in range(k_main):
        P[i, i] = p_in
    for i in range(k_main):
        P[i, k_main] = q_b
        P[k_main, i] = q_b
    P[k_main, k_main] = p_out
    G = nx.stochastic_block_model(sizes, P.tolist(), seed=int(rng.integers(0, 2**31 - 1)), sparse=True)
    A = nx.to_scipy_sparse_array(G, dtype=np.float64, format="csr")
    bridge_indices = np.arange(k_main * m, k_main * m + b)
    return A, bridge_indices, q_b


def verify_degree_match(A, bridge_indices, alpha=0.05):
    n = A.shape[0]
    if sparse.issparse(A):
        degrees = np.array(A.sum(axis=1)).flatten()
    else:
        degrees = A.sum(axis=1)
    bridge_mask = np.zeros(n, dtype=bool)
    bridge_mask[bridge_indices] = True
    bd = degrees[bridge_mask]
    nd = degrees[~bridge_mask]
    if len(bd) < 2 or len(nd) < 2:
        return {"matched": False, "mannwhitney_p": None}
    try:
        stat, p = stats.mannwhitneyu(bd, nd, alternative="two-sided")
        return {
            "mean_bridge_degree": float(bd.mean()),
            "mean_non_bridge_degree": float(nd.mean()),
            "mannwhitney_p": float(p),
            "matched": bool(p > alpha),
            "abs_mean_diff_normalized": float(abs(bd.mean() - nd.mean()) / nd.mean()) if nd.mean() > 0 else None,
        }
    except Exception:
        return {"matched": False, "mannwhitney_p": None}


def auc_safe(scores, bridge_indices, n):
    y = np.zeros(n)
    y[bridge_indices] = 1
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return None


def run_one_replicate(m, b, p_in, p_out, k_main, rng_seed, use_brute_force):
    t0 = time.time()
    rng = np.random.default_rng(rng_seed)
    A_sparse, bridge_idx, q_b = generate_degree_matched_sbm(m, b, p_in, p_out, k_main=k_main, rng=rng)
    if A_sparse.sum() == 0:
        return None

    match = verify_degree_match(A_sparse, bridge_idx)
    A_dense = A_sparse.toarray()
    n = A_dense.shape[0]

    fo_bli = first_order_bli(A_sparse)
    scores = {"bli_first_order": fo_bli}

    if use_brute_force and n <= 200:
        bli = brute_force_bli(A_dense, n_jobs=1)
        scores["bli_brute_force"] = bli
    else:
        scores["bli_brute_force"] = np.full(n, np.nan)

    comparators = comparator_centralities(A_dense)
    scores.update(comparators)

    aucs = {name: auc_safe(s, bridge_idx, n) if not np.isnan(s).all() else None for name, s in scores.items()}

    elapsed = time.time() - t0
    return {
        "n_total": n, "m": m, "b": b, "p_in": p_in, "p_out": p_out, "k_main": k_main,
        "q_bridge": q_b,
        "degrees_matched": match.get("matched"),
        "degree_diff_normalized": match.get("abs_mean_diff_normalized"),
        "elapsed_sec": elapsed,
        **{f"auc_{name}": val for name, val in aucs.items()},
    }


def parameter_sweep(grid, n_reps, base_seed, n_jobs, use_brute_force):
    rng = np.random.default_rng(base_seed)
    jobs = []
    for cell in grid:
        for rep in range(n_reps):
            jobs.append((cell, int(rng.integers(0, 2**31 - 1))))

    print(f"Total jobs: {len(jobs)} ({len(grid)} cells × {n_reps} reps)")
    print(f"Parallel workers: {n_jobs}")
    print(f"Use brute-force BLI: {use_brute_force}")

    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(run_one_replicate)(**cell, rng_seed=seed, use_brute_force=use_brute_force)
        for cell, seed in jobs
    )
    return pd.DataFrame([r for r in rows if r is not None])


def paired_bootstrap_auc_diff(df, ref_col, other_col, n_boot=2000, seed=SEED):
    valid = df[[ref_col, other_col]].dropna().values
    if len(valid) < 5:
        return None
    diffs = valid[:, 0] - valid[:, 1]
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(diffs, size=len(diffs), replace=True)
        boot_means[b] = sample.mean()
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(boot_means, 2.5)),
        "ci_high": float(np.percentile(boot_means, 97.5)),
        "n_pairs": int(len(diffs)),
        "share_positive": float((diffs > 0).mean()),
    }


def romano_wolf_stepdown(df, ref_col, other_cols, n_boot=2000, seed=SEED):
    valid_cols = [c for c in [ref_col] + other_cols if c in df.columns]
    panel = df[valid_cols].dropna()
    if len(panel) < 5:
        return {"error": "insufficient data"}
    ref = panel[ref_col].values
    others = panel[other_cols].values

    rng = np.random.default_rng(seed)
    diffs = ref[:, None] - others
    n = len(panel)
    se = diffs.std(axis=0, ddof=1) / np.sqrt(n)
    t_obs = diffs.mean(axis=0) / np.where(se > 0, se, 1e-12)

    boot_t = np.empty((n_boot, len(other_cols)))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d_boot = diffs[idx]
        d_centered = d_boot - diffs.mean(axis=0)
        se_b = d_centered.std(axis=0, ddof=1) / np.sqrt(n)
        boot_t[b] = d_centered.mean(axis=0) / np.where(se_b > 0, se_b, 1e-12)

    abs_t_obs = np.abs(t_obs)
    abs_boot_t = np.abs(boot_t)
    order = np.argsort(-abs_t_obs)

    p_init = np.zeros(len(other_cols))
    for s_idx, comp_idx in enumerate(order):
        remaining = order[s_idx:]
        max_stat = abs_boot_t[:, remaining].max(axis=1)
        p_init[comp_idx] = (np.sum(max_stat >= abs_t_obs[comp_idx]) + 1) / (n_boot + 1)

    corr_padj = np.zeros(len(other_cols))
    for s_idx, comp_idx in enumerate(order):
        if s_idx == 0:
            corr_padj[comp_idx] = p_init[comp_idx]
        else:
            prev_comp = order[s_idx - 1]
            corr_padj[comp_idx] = max(p_init[comp_idx], corr_padj[prev_comp])

    return {
        comp: {
            "t_observed": float(t_obs[i]),
            "p_unadjusted": float(p_init[i]),
            "p_romano_wolf": float(corr_padj[i]),
            "n_pairs": int(n),
            "mean_diff": float(diffs[:, i].mean()),
        }
        for i, comp in enumerate(other_cols)
    }


def plot_forest(bootstrap_results, out_path, ref_name="BLI"):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(bootstrap_results.keys())
    means = [bootstrap_results[n]["mean_diff"] for n in names]
    lows = [bootstrap_results[n]["ci_low"] for n in names]
    highs = [bootstrap_results[n]["ci_high"] for n in names]
    y_pos = np.arange(len(names))
    ax.errorbar(means, y_pos, xerr=[[m - l for m, l in zip(means, lows)],
                                     [h - m for m, h in zip(means, highs)]],
                fmt="o", color="#1f77b4", capsize=4, markersize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(f"AUC difference ({ref_name} minus comparator)")
    ax.set_title(f"Paired bootstrap CIs: {ref_name} vs comparators")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_n_scaling(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    centralities = [c.replace("auc_", "") for c in df.columns if c.startswith("auc_")]
    grouped = df.groupby("n_total")[[f"auc_{c}" for c in centralities]].mean()
    for c in centralities:
        col = f"auc_{c}"
        if col in grouped.columns and grouped[col].notna().any():
            ax.plot(grouped.index, grouped[col], "-o", label=c, markersize=5)
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Total graph size n")
    ax.set_ylabel("Mean AUC")
    ax.set_title("Bridge recovery AUC across graph sizes")
    ax.legend(loc="best", fontsize=8)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_k_scaling(df, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    centralities = [c.replace("auc_", "") for c in df.columns if c.startswith("auc_")]
    grouped = df.groupby("k_main")[[f"auc_{c}" for c in centralities]].mean()
    x = grouped.index.values
    for c in centralities:
        col = f"auc_{c}"
        if col in grouped.columns and grouped[col].notna().any():
            ax.plot(x, grouped[col], "-o", label=c, markersize=6)
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Number of main blocks K")
    ax.set_ylabel("Mean AUC")
    ax.set_title("Bridge recovery AUC across K (number of blocks)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="medium", choices=["smoke", "medium", "large", "xlarge"])
    p.add_argument("--n-reps", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--brute-force", action="store_true", help="Compute brute-force BLI for graphs n <= 200")
    p.add_argument("--output-suffix", default="brev")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def build_grid(scale):
    if scale == "smoke":
        cells = []
        for k_main in [2, 3]:
            for m in [25]:
                for b in [5]:
                    for p_in in [0.20]:
                        for p_out in [0.02]:
                            cells.append({"m": m, "b": b, "p_in": p_in, "p_out": p_out, "k_main": k_main})
        return cells, 3
    if scale == "medium":
        cells = []
        for k_main in [2, 3, 5]:
            for m in [25, 50, 100]:
                for b in [5, 10]:
                    for p_in in [0.15, 0.25]:
                        for p_out in [0.01, 0.03]:
                            cells.append({"m": m, "b": b, "p_in": p_in, "p_out": p_out, "k_main": k_main})
        return cells, 20
    if scale == "large":
        cells = []
        for k_main in [2, 3, 5, 10]:
            for m in [50, 100, 200]:
                for b in [5, 10, 20]:
                    for p_in in [0.10, 0.15, 0.20, 0.30]:
                        for p_out in [0.005, 0.01, 0.02, 0.05]:
                            cells.append({"m": m, "b": b, "p_in": p_in, "p_out": p_out, "k_main": k_main})
        return cells, 30
    if scale == "xlarge":
        cells = []
        for k_main in [2, 3, 5, 10]:
            for m in [100, 200, 500]:
                for b in [10, 25, 50]:
                    for p_in in [0.10, 0.15, 0.20, 0.30]:
                        for p_out in [0.005, 0.01, 0.02, 0.05]:
                            cells.append({"m": m, "b": b, "p_in": p_in, "p_out": p_out, "k_main": k_main})
        return cells, 50
    raise ValueError(f"unknown scale {scale}")


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    grid, default_reps = build_grid(args.scale)
    n_reps = args.n_reps if args.n_reps is not None else default_reps

    print(f"=== SBM Benchmark Brev ({args.scale}) ===")
    print(f"Grid cells: {len(grid)}")
    print(f"Reps per cell: {n_reps}")
    print(f"Total graphs: {len(grid) * n_reps}")
    print(f"Brute-force BLI: {args.brute_force}")
    print(f"Available cores: {os.cpu_count()}")

    t0 = time.time()
    full_df = parameter_sweep(grid, n_reps=n_reps, base_seed=args.seed,
                              n_jobs=args.n_jobs, use_brute_force=args.brute_force)
    t1 = time.time()
    print(f"\nSweep wall time: {(t1 - t0) / 60:.1f} min")

    suffix = f"_{args.output_suffix}_{args.scale}"
    full_csv = RESULTS_DIR / f"sbm_benchmark{suffix}_full.csv"
    full_df.to_csv(full_csv, index=False)
    print(f"Saved {len(full_df)} rows to {full_csv}")

    print(f"\nDegree-matching: {full_df['degrees_matched'].mean():.3f} share matched")
    print(f"Median runtime per graph: {full_df['elapsed_sec'].median():.3f}s")

    auc_cols = [c for c in full_df.columns if c.startswith("auc_")]
    print(f"\nMean AUC by centrality (sorted):")
    summary = full_df[auc_cols].mean().sort_values(ascending=False)
    for col, val in summary.items():
        print(f"  {col.replace('auc_', ''):25s}: {val:.4f}")

    print(f"\nMean AUC by K:")
    by_k = full_df.groupby("k_main")[auc_cols].mean()
    print(by_k.to_string())

    print(f"\nMean AUC by n:")
    by_n = full_df.groupby("n_total")[auc_cols].mean()
    print(by_n.to_string())

    primary_ref = "auc_bli_first_order"
    if args.brute_force and full_df["auc_bli_brute_force"].notna().sum() > 100:
        primary_ref = "auc_bli_brute_force"

    other_cols = [c for c in auc_cols if c != primary_ref]
    bootstrap_results = {}
    for other in other_cols:
        result = paired_bootstrap_auc_diff(full_df, primary_ref, other, n_boot=2000, seed=args.seed)
        if result is not None:
            bootstrap_results[other.replace("auc_", "")] = result

    print(f"\nPaired bootstrap CIs ({primary_ref.replace('auc_', '')} minus comparator):")
    for name, res in bootstrap_results.items():
        sig = "*" if res["ci_low"] > 0 or res["ci_high"] < 0 else " "
        print(f"  {sig} {name:25s}: diff={res['mean_diff']:+.4f}, CI=[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}], share+ {res['share_positive']:.2f}")

    rw_results = romano_wolf_stepdown(full_df, primary_ref, other_cols, n_boot=2000, seed=args.seed)
    print(f"\nRomano-Wolf stepdown p-values:")
    for name, res in rw_results.items():
        if isinstance(res, dict) and "p_romano_wolf" in res:
            print(f"  {name.replace('auc_', ''):25s}: t={res['t_observed']:+.2f}, p_RW={res['p_romano_wolf']:.4f}")

    plot_forest(bootstrap_results, FIGURES_DIR / f"sbm_benchmark{suffix}_forest.pdf",
                ref_name=primary_ref.replace("auc_", ""))
    plot_n_scaling(full_df, FIGURES_DIR / f"sbm_benchmark{suffix}_n_scaling.pdf")
    plot_k_scaling(full_df, FIGURES_DIR / f"sbm_benchmark{suffix}_k_scaling.pdf")

    output = {
        "scale": args.scale,
        "n_grid_cells": int(len(grid)),
        "n_reps_per_cell": n_reps,
        "n_total_graphs": int(len(full_df)),
        "wall_time_min": float((t1 - t0) / 60),
        "degree_match_share": float(full_df["degrees_matched"].mean()),
        "auc_means": {c.replace("auc_", ""): float(full_df[c].mean()) for c in auc_cols},
        "auc_medians": {c.replace("auc_", ""): float(full_df[c].median()) for c in auc_cols},
        "auc_by_k": by_k.to_dict(),
        "auc_by_n": by_n.to_dict(),
        "primary_ref": primary_ref,
        "paired_bootstrap": bootstrap_results,
        "romano_wolf": rw_results,
    }
    out_path = RESULTS_DIR / f"sbm_benchmark{suffix}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved JSON to {out_path}")


if __name__ == "__main__":
    main()
