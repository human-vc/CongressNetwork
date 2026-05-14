import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse, stats
from scipy.sparse.linalg import eigsh
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FIGURES_DIR, SEED

warnings.filterwarnings("ignore")


def degree_matched_q_bridge(m, b, p_in, p_out):
    return ((m - 1) * p_in + (m - b + 1) * p_out) / (2 * m - b)


def fiedler_value(adjacency):
    A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    keep = degrees > 0
    if keep.sum() < 3:
        return 0.0
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    L = sparse.eye(A_sub.shape[0]) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    try:
        eigs, _ = eigsh(L, k=2, which="SM")
        return float(np.sort(eigs)[1])
    except Exception:
        return 0.0


def fiedler_pair(adjacency, k=3):
    A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    keep = degrees > 0
    if keep.sum() < k + 1:
        return None, None, None, None
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    L = sparse.eye(A_sub.shape[0]) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    try:
        eigs, vecs = eigsh(L, k=k, which="SM")
        order = np.argsort(eigs)
        eigs = eigs[order]
        vecs = vecs[:, order]
        psi_full = np.zeros((adjacency.shape[0], k))
        psi_full[keep, :] = vecs
        return float(eigs[1]), float(eigs[2]) if k >= 3 else None, psi_full, keep
    except Exception:
        return None, None, None, None


def brute_force_bli(A):
    n = A.shape[0]
    base = fiedler_value(A)
    bli = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        bli[i] = base - fiedler_value(A[np.ix_(mask, mask)])
    return bli


def first_order_bli(A):
    n = A.shape[0]
    lam2, lam3, psi, _ = fiedler_pair(A, k=3)
    if lam2 is None or psi is None:
        return np.zeros(n)
    psi2 = psi[:, 1]
    gap = (lam3 - lam2) if lam3 is not None else lam2
    return gap * (1.0 - psi2 ** 2)


def compute_comparator_centralities(A):
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
        ("eigenvector", lambda: nx.eigenvector_centrality_numpy(G_keep)),
        ("closeness", lambda: nx.closeness_centrality(G_keep)),
        ("pagerank", lambda: nx.pagerank(G_keep, alpha=0.85)),
        ("kcore", lambda: nx.core_number(G_keep)),
    ]:
        try:
            out[name] = fill(fn())
        except Exception:
            out[name] = np.zeros(n)
    out["degree"] = np.array([G.degree(i) for i in range(n)], dtype=float)
    if out["degree"].max() > 0:
        out["degree"] = out["degree"] / out["degree"].max()
    if out["kcore"].max() > 0:
        out["kcore"] = out["kcore"] / out["kcore"].max()
    return out


def generate_degree_matched_sbm(m, b, p_in, p_out, k_main=2, rng=None):
    rng = rng or np.random.default_rng()
    q_b = degree_matched_q_bridge(m, b, p_in, p_out)
    sizes = [m] * k_main + [b]
    K = k_main + 1
    P = np.full((K, K), p_out, dtype=float)
    for i in range(k_main):
        P[i, i] = p_in
    for i in range(k_main):
        P[i, k_main] = q_b
        P[k_main, i] = q_b
    P[k_main, k_main] = p_out
    G = nx.stochastic_block_model(sizes, P.tolist(), seed=int(rng.integers(0, 2**31 - 1)))
    A = nx.to_numpy_array(G, dtype=np.float64)
    bridge_indices = np.arange(k_main * m, k_main * m + b)
    return A, bridge_indices, q_b


def verify_degree_match(A, bridge_indices, alpha=0.05):
    n = A.shape[0]
    degrees = A.sum(axis=1)
    bridge_mask = np.zeros(n, dtype=bool)
    bridge_mask[bridge_indices] = True
    bd = degrees[bridge_mask]
    nd = degrees[~bridge_mask]
    if len(bd) < 2 or len(nd) < 2:
        return {"matched": False, "p_value": None}
    stat, p = stats.mannwhitneyu(bd, nd, alternative="two-sided")
    return {
        "mean_bridge_degree": float(bd.mean()),
        "mean_non_bridge_degree": float(nd.mean()),
        "std_bridge_degree": float(bd.std()),
        "std_non_bridge_degree": float(nd.std()),
        "mannwhitney_u": float(stat),
        "mannwhitney_p": float(p),
        "matched": bool(p > alpha),
        "abs_mean_diff_normalized": float(abs(bd.mean() - nd.mean()) / nd.mean()) if nd.mean() > 0 else None,
    }


def auc_for_scores(scores, bridge_indices, n):
    y = np.zeros(n)
    y[bridge_indices] = 1
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return None


def run_one_replicate(m, b, p_in, p_out, k_main, rng_seed):
    rng = np.random.default_rng(rng_seed)
    A, bridge_idx, q_b = generate_degree_matched_sbm(m, b, p_in, p_out, k_main=k_main, rng=rng)
    if A.sum() == 0:
        return None
    match = verify_degree_match(A, bridge_idx)

    bli = brute_force_bli(A)
    fo_bli = first_order_bli(A)
    comparators = compute_comparator_centralities(A)
    scores = {"bli_brute_force": bli, "bli_first_order": fo_bli, **comparators}

    n = A.shape[0]
    aucs = {name: auc_for_scores(s, bridge_idx, n) for name, s in scores.items()}

    return {
        "m": m, "b": b, "p_in": p_in, "p_out": p_out, "k_main": k_main,
        "q_bridge": q_b,
        "degrees_matched": match["matched"],
        "degree_diff_normalized": match["abs_mean_diff_normalized"],
        **{f"auc_{name}": val for name, val in aucs.items()},
    }


def parameter_sweep(grid, n_reps, base_seed):
    rng = np.random.default_rng(base_seed)
    seeds_per_cell = [int(rng.integers(0, 2**31 - 1)) for _ in range(len(grid) * n_reps)]
    jobs = []
    idx = 0
    for cell in grid:
        for rep in range(n_reps):
            jobs.append((cell, seeds_per_cell[idx]))
            idx += 1

    rows = Parallel(n_jobs=-1, backend="loky", verbose=5)(
        delayed(run_one_replicate)(**cell, rng_seed=seed) for cell, seed in jobs
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


def plot_paired_bootstrap_forest(bootstrap_results, out_path):
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
    ax.set_xlabel("AUC difference (BLI minus comparator)")
    ax.set_title("Paired bootstrap CIs: BLI vs comparators")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_diagram_v2(df, out_path, centrality_col="auc_bli_brute_force"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    pivot = df.pivot_table(index="p_in", columns="p_out", values=centrality_col, aggfunc="mean")
    im = ax.imshow(pivot.values, cmap="RdYlBu_r", aspect="auto", origin="lower", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.3f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("p_out")
    ax.set_ylabel("p_in")
    ax.set_title("BLI bridge-recovery AUC")
    fig.colorbar(im, ax=ax, label="AUC")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v - 0.5) > 0.25 else "black", fontsize=8)

    ax = axes[1]
    pivot_bet = df.pivot_table(index="p_in", columns="p_out", values="auc_betweenness", aggfunc="mean")
    diff = pivot.values - pivot_bet.values
    im2 = ax.imshow(diff, cmap="RdBu_r", aspect="auto", origin="lower", vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.3f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("p_out")
    ax.set_ylabel("p_in")
    ax.set_title("BLI AUC minus betweenness AUC")
    fig.colorbar(im2, ax=ax, label="AUC difference")
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            v = diff[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.2 else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    grid = []
    for k_main in [2, 3]:
        for m in [25, 40]:
            for b in [5, 10]:
                for p_in in [0.15, 0.20, 0.25, 0.30]:
                    for p_out in [0.01, 0.02, 0.05]:
                        grid.append({"m": m, "b": b, "p_in": p_in, "p_out": p_out, "k_main": k_main})

    print(f"Total parameter cells: {len(grid)}")
    print(f"Running with 20 reps per cell, parallel...")

    full_df = parameter_sweep(grid, n_reps=20, base_seed=SEED)
    full_df.to_csv(RESULTS_DIR / "sbm_benchmark_v2_full.csv", index=False)

    print(f"\nDegree-matching verification:")
    matched_share = full_df["degrees_matched"].mean()
    mean_diff = full_df["degree_diff_normalized"].mean()
    print(f"  Share of graphs where Mann-Whitney fails to reject (matched): {matched_share:.3f}")
    print(f"  Mean normalized degree difference: {mean_diff:.4f}")

    auc_cols = [c for c in full_df.columns if c.startswith("auc_")]
    print(f"\nMean AUC by centrality:")
    summary = full_df[auc_cols].mean().sort_values(ascending=False)
    for col, val in summary.items():
        print(f"  {col.replace('auc_', ''):25s}: {val:.4f}")

    bootstrap_results = {}
    ref_col = "auc_bli_brute_force"
    other_cols = [c for c in auc_cols if c != ref_col]
    for other in other_cols:
        result = paired_bootstrap_auc_diff(full_df, ref_col, other, n_boot=2000)
        if result is not None:
            bootstrap_results[other.replace("auc_", "")] = result

    print(f"\nPaired bootstrap CIs (BLI minus comparator):")
    for name, res in bootstrap_results.items():
        print(f"  {name:25s}: diff={res['mean_diff']:+.4f}, CI=[{res['ci_low']:+.4f}, {res['ci_high']:+.4f}], share+ {res['share_positive']:.2f}")

    rw_results = romano_wolf_stepdown(full_df, ref_col, other_cols, n_boot=2000)
    print(f"\nRomano-Wolf stepdown p-values (FWE controlled):")
    for name, res in rw_results.items():
        if isinstance(res, dict) and "p_romano_wolf" in res:
            print(f"  {name.replace('auc_', ''):25s}: t={res['t_observed']:+.2f}, p_RW={res['p_romano_wolf']:.4f}")

    plot_paired_bootstrap_forest(bootstrap_results, FIGURES_DIR / "sbm_benchmark_v2_forest.pdf")
    plot_phase_diagram_v2(full_df, FIGURES_DIR / "sbm_benchmark_v2_phase.pdf")

    for k in [2, 3]:
        sub = full_df[full_df["k_main"] == k]
        if len(sub) > 0:
            sub_means = sub[auc_cols].mean().sort_values(ascending=False)
            print(f"\nK={k} mean AUC ranking:")
            for col, val in sub_means.items():
                print(f"  {col.replace('auc_', ''):25s}: {val:.4f}")

    output = {
        "n_total_graphs": int(len(full_df)),
        "degree_match_share": float(matched_share),
        "degree_diff_mean": float(mean_diff),
        "auc_means": {c.replace("auc_", ""): float(full_df[c].mean()) for c in auc_cols},
        "auc_medians": {c.replace("auc_", ""): float(full_df[c].median()) for c in auc_cols},
        "auc_stds": {c.replace("auc_", ""): float(full_df[c].std()) for c in auc_cols},
        "paired_bootstrap_ci": bootstrap_results,
        "romano_wolf_stepdown": rw_results,
        "by_k": {
            str(k): {c.replace("auc_", ""): float(full_df[full_df["k_main"] == k][c].mean())
                     for c in auc_cols if c in full_df.columns}
            for k in [2, 3]
        },
    }
    with open(RESULTS_DIR / "sbm_benchmark_v2_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR / 'sbm_benchmark_v2_results.json'}")


if __name__ == "__main__":
    main()
