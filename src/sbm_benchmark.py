import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FIGURES_DIR, SEED

warnings.filterwarnings("ignore")


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


def brute_force_bli(A):
    n = A.shape[0]
    base = fiedler_value(A)
    bli = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        bli[i] = base - fiedler_value(A[np.ix_(mask, mask)])
    return bli


def fiedler_vector_squared(A):
    n = A.shape[0]
    A_sp = sparse.csr_matrix(A)
    degrees = np.array(A_sp.sum(axis=1)).flatten()
    keep = degrees > 0
    if keep.sum() < 3:
        return np.zeros(n)
    A_sub = A_sp[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    L = sparse.eye(A_sub.shape[0]) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    try:
        eigs, vecs = eigsh(L, k=2, which="SM")
        order = np.argsort(eigs)
        psi2_sub = vecs[:, order[1]]
        psi2_full = np.zeros(n)
        psi2_full[keep] = psi2_sub
        return psi2_full ** 2
    except Exception:
        return np.zeros(n)


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
    try:
        out["betweenness"] = fill(nx.betweenness_centrality(G_keep, normalized=True))
    except Exception:
        out["betweenness"] = np.zeros(n)
    try:
        out["eigenvector"] = fill(nx.eigenvector_centrality_numpy(G_keep))
    except Exception:
        out["eigenvector"] = np.zeros(n)
    try:
        out["closeness"] = fill(nx.closeness_centrality(G_keep))
    except Exception:
        out["closeness"] = np.zeros(n)
    try:
        h = nx.harmonic_centrality(G_keep)
        m = max(h.values()) if h else 1.0
        out["harmonic"] = fill({k: v / m for k, v in h.items()} if m > 0 else h)
    except Exception:
        out["harmonic"] = np.zeros(n)
    try:
        out["pagerank"] = fill(nx.pagerank(G_keep, alpha=0.85))
    except Exception:
        out["pagerank"] = np.zeros(n)
    try:
        kc = nx.core_number(G_keep)
        mc = max(kc.values()) if kc else 1
        out["kcore"] = fill({k: v / mc for k, v in kc.items()} if mc > 0 else kc)
    except Exception:
        out["kcore"] = np.zeros(n)
    deg = np.array([G.degree(i) for i in range(n)], dtype=float)
    if deg.max() > 0:
        deg = deg / deg.max()
    out["degree"] = deg
    out["fiedler_squared_inverted"] = 1.0 - fiedler_vector_squared(A)
    return out


def generate_planted_bridge_sbm(n_main, n_bridges, p_in, p_out, p_bridge_to_blocs=None, rng=None):
    rng = rng or np.random.default_rng()
    if p_bridge_to_blocs is None:
        p_bridge_to_blocs = p_in / 2.0
    sizes = [n_main, n_main, n_bridges]
    P = [
        [p_in, p_out, p_bridge_to_blocs],
        [p_out, p_in, p_bridge_to_blocs],
        [p_bridge_to_blocs, p_bridge_to_blocs, p_out],
    ]
    G = nx.stochastic_block_model(sizes, P, seed=int(rng.integers(0, 2**31 - 1)))
    A = nx.to_numpy_array(G, dtype=np.float64)
    bridge_indices = np.arange(2 * n_main, 2 * n_main + n_bridges)
    return A, bridge_indices


def measure_recovery(scores, bridge_indices, n):
    y = np.zeros(n)
    y[bridge_indices] = 1
    n_bridges = len(bridge_indices)
    out = {}
    for name, s in scores.items():
        try:
            auc = float(roc_auc_score(y, s))
        except ValueError:
            auc = None
        order = np.argsort(-s)
        top_k_recovery = {}
        for k_mult in [1, 2, 3]:
            k = min(n - 1, k_mult * n_bridges)
            top_set = set(order[:k].tolist())
            bridge_set = set(bridge_indices.tolist())
            top_k_recovery[f"top_{k_mult}x"] = len(top_set & bridge_set) / n_bridges
        ranks = np.argsort(np.argsort(-s)) + 1
        mean_rank_norm = float(np.mean(ranks[bridge_indices]) / n)
        out[name] = {
            "auc": auc,
            "mean_rank_normalized": mean_rank_norm,
            **{f"top_{k}x_recovery": v for k, v in top_k_recovery.items()},
        }
    return out


def run_cell(n_main, n_bridges, p_in, p_out, n_reps, rng_seed):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for rep in range(n_reps):
        A, bridge_idx = generate_planted_bridge_sbm(n_main, n_bridges, p_in, p_out, rng=rng)
        if A.sum() == 0:
            continue
        bli = brute_force_bli(A)
        comparators = compute_comparator_centralities(A)
        scores = {"bli_brute_force": bli, **comparators}
        recovery = measure_recovery(scores, bridge_idx, A.shape[0])
        for name, metrics in recovery.items():
            rows.append({
                "n_main": n_main,
                "n_bridges": n_bridges,
                "p_in": p_in,
                "p_out": p_out,
                "p_ratio": p_in / p_out,
                "rep": rep,
                "centrality": name,
                **metrics,
            })
    return pd.DataFrame(rows)


def parameter_sweep():
    rng = np.random.default_rng(SEED)
    grid = []
    for n_main in [25, 50]:
        for n_bridges in [3, 5, 10]:
            for p_in in [0.15, 0.25]:
                for p_out in [0.01, 0.03]:
                    grid.append({"n_main": n_main, "n_bridges": n_bridges,
                                 "p_in": p_in, "p_out": p_out})
    all_rows = []
    for cell in grid:
        seed = int(rng.integers(0, 2**31 - 1))
        df = run_cell(**cell, n_reps=20, rng_seed=seed)
        all_rows.append(df)
        n_total = cell["n_main"] * 2 + cell["n_bridges"]
        sub_bli = df[df["centrality"] == "bli_brute_force"]["auc"].dropna()
        sub_bet = df[df["centrality"] == "betweenness"]["auc"].dropna()
        print(f"  n={n_total}, n_b={cell['n_bridges']}, p_in={cell['p_in']}, p_out={cell['p_out']}: "
              f"BLI AUC mean = {sub_bli.mean():.3f}, betweenness AUC mean = {sub_bet.mean():.3f}")
    return pd.concat(all_rows, ignore_index=True)


def summarize(full_df):
    agg_dict = {"auc": ["mean", "std", "median"], "mean_rank_normalized": ["mean"]}
    for col in [c for c in full_df.columns if c.startswith("top_") and c.endswith("_recovery")]:
        agg_dict[col] = ["mean"]
    summary = full_df.groupby("centrality").agg(agg_dict).reset_index()
    summary.columns = ["centrality"] + ["_".join(c).strip("_") for c in summary.columns[1:]]
    summary = summary.sort_values("auc_mean", ascending=False)
    return summary


def plot_auc_comparison(full_df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    centrality_order = (
        full_df.groupby("centrality")["auc"].mean().sort_values(ascending=False).index.tolist()
    )
    box_data = [full_df[full_df["centrality"] == c]["auc"].dropna().values for c in centrality_order]
    bp = ax.boxplot(box_data, labels=centrality_order, showfliers=False, patch_artist=True)
    colors = ["#1f77b4" if c == "bli_brute_force" else "#cccccc" for c in centrality_order]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0.5, color="black", linewidth=0.5, linestyle=":")
    ax.set_ylabel("AUC for bridge identification")
    ax.set_title("Bridge recovery AUC by centrality")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    ax = axes[1]
    pivot = full_df.pivot_table(index=["p_in", "p_out"], columns="centrality", values="auc", aggfunc="mean")
    if "bli_brute_force" in pivot.columns:
        bli_minus_best_comp = pivot["bli_brute_force"] - pivot.drop(columns=["bli_brute_force"]).max(axis=1)
        bli_minus_best_comp = bli_minus_best_comp.reset_index()
        bli_minus_best_comp["ratio"] = bli_minus_best_comp["p_in"] / bli_minus_best_comp["p_out"]
        ax.scatter(bli_minus_best_comp["ratio"], bli_minus_best_comp[0], s=80, color="#1f77b4")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("p_in / p_out (signal-to-noise ratio)")
        ax.set_ylabel("BLI AUC minus best competitor AUC")
        ax.set_title("BLI advantage by partition strength")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_phase_diagram(full_df, out_path):
    sub = full_df[full_df["centrality"] == "bli_brute_force"]
    pivot = sub.pivot_table(index="p_in", columns="p_out", values="auc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="RdYlBu_r", aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.3f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.2f}" for r in pivot.index])
    ax.set_xlabel("p_out (inter-bloc density)")
    ax.set_ylabel("p_in (intra-bloc density)")
    ax.set_title("BLI bridge-recovery AUC across (p_in, p_out)")
    fig.colorbar(im, ax=ax, label="AUC")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v - 0.5) > 0.25 else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    print("Running SBM bridge-recovery benchmark...")
    full_df = parameter_sweep()
    full_df.to_csv(RESULTS_DIR / "sbm_benchmark_full.csv", index=False)

    print()
    print("Summary by centrality across all parameter cells:")
    summary = summarize(full_df)
    print(summary.to_string(index=False))

    plot_auc_comparison(full_df, FIGURES_DIR / "sbm_benchmark_auc.pdf")
    plot_phase_diagram(full_df, FIGURES_DIR / "sbm_benchmark_phase.pdf")

    summary.to_csv(RESULTS_DIR / "sbm_benchmark_summary.csv", index=False)

    output = {
        "n_total_graphs": int(len(full_df) // full_df["centrality"].nunique()),
        "n_centralities": int(full_df["centrality"].nunique()),
        "summary": summary.to_dict(orient="records"),
        "bli_dominance_by_p_ratio": full_df[full_df["centrality"] == "bli_brute_force"].groupby("p_ratio")["auc"].mean().to_dict(),
    }
    with open(RESULTS_DIR / "sbm_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR / 'sbm_benchmark_results.json'}")


if __name__ == "__main__":
    main()
