import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, CONGRESSES, SEED

from scipy import sparse, stats
from scipy.sparse.linalg import eigsh

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HIGHLIGHT_CONGRESSES = [104, 107, 111, 112, 114, 117]


def load_congress(c):
    p = PROCESSED_DIR / f"congress_{c}.npz"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=True)


def fiedler_vector(adjacency):
    A = sparse.csr_matrix(adjacency)
    degrees = np.array(A.sum(axis=1)).flatten()
    keep = degrees > 0
    if keep.sum() < 3:
        return None, None, None
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    d_inv_sqrt = sparse.diags(1.0 / np.sqrt(d_sub))
    L = sparse.eye(A_sub.shape[0]) - d_inv_sqrt @ A_sub @ d_inv_sqrt
    try:
        eigvals, eigvecs = eigsh(L, k=2, which="SM")
        order = np.argsort(eigvals)
        return float(eigvals[order[1]]), eigvecs[:, order[1]], keep
    except Exception:
        return None, None, None


def restrepo_ott_hunt_di(adjacency):
    fval, fvec, keep = fiedler_vector(adjacency)
    n = adjacency.shape[0]
    di = np.zeros(n)
    if fvec is None:
        return di
    fvec_norm = fvec / np.linalg.norm(fvec)
    di[keep] = fvec_norm ** 2
    return di


def first_order_bli_approximation(adjacency):
    fval, fvec, keep = fiedler_vector(adjacency)
    n = adjacency.shape[0]
    approx = np.zeros(n)
    if fvec is None:
        return approx
    psi = fvec / np.linalg.norm(fvec)
    psi_full = np.zeros(n)
    psi_full[keep] = psi
    denom = 1.0 - psi_full ** 2
    denom = np.where(denom > 1e-12, denom, 1e-12)
    approx = fval * (psi_full ** 2) / denom
    return approx


def compute_centralities(adjacency):
    G = nx.from_numpy_array(adjacency)
    n = adjacency.shape[0]
    isolated = [i for i in range(n) if G.degree(i) == 0]
    G_keep = G.copy()
    G_keep.remove_nodes_from(isolated)

    def fill_full(d):
        arr = np.zeros(n)
        for i, v in d.items():
            arr[i] = v
        return arr

    bet = nx.betweenness_centrality(G_keep, normalized=True)
    try:
        eig = nx.eigenvector_centrality_numpy(G_keep)
    except Exception:
        eig = {i: 0.0 for i in G_keep.nodes()}
    try:
        clo = nx.closeness_centrality(G_keep)
    except Exception:
        clo = {i: 0.0 for i in G_keep.nodes()}
    try:
        harm = nx.harmonic_centrality(G_keep)
        max_harm = max(harm.values()) if harm else 1.0
        harm = {k: v / max_harm for k, v in harm.items()} if max_harm > 0 else harm
    except Exception:
        harm = {i: 0.0 for i in G_keep.nodes()}
    try:
        pr = nx.pagerank(G_keep, alpha=0.85)
    except Exception:
        pr = {i: 0.0 for i in G_keep.nodes()}
    try:
        kcore = nx.core_number(G_keep)
        max_core = max(kcore.values()) if kcore else 1
        kcore = {k: v / max_core for k, v in kcore.items()} if max_core > 0 else kcore
    except Exception:
        kcore = {i: 0.0 for i in G_keep.nodes()}

    di_arr = restrepo_ott_hunt_di(adjacency)
    bli_approx_arr = first_order_bli_approximation(adjacency)

    degree_arr = np.array([G.degree(i) for i in range(n)], dtype=float)
    if degree_arr.max() > 0:
        degree_arr = degree_arr / degree_arr.max()

    return {
        "betweenness": fill_full(bet),
        "eigenvector": fill_full(eig),
        "closeness": fill_full(clo),
        "harmonic": fill_full(harm),
        "pagerank": fill_full(pr),
        "kcore": fill_full(kcore),
        "degree": degree_arr,
        "dynamical_importance": di_arr,
        "bli_first_order": bli_approx_arr,
    }


def load_bli_results():
    path = RESULTS_DIR / "bli_results.json"
    if not path.exists():
        raise RuntimeError("Run spectral_analysis.py first")
    with open(path) as f:
        return json.load(f)


def compare_per_congress(c, bli_data):
    d = load_congress(c)
    if d is None or str(c) not in bli_data:
        return None
    adjacency = d["adjacency"]
    member_ids = [int(m) for m in d["member_ids"]]
    member_names = d["member_names"]
    party_codes = d["party_codes"]

    bli_values = np.array(bli_data[str(c)]["bli_values"])
    bli_member_ids = [int(m) for m in bli_data[str(c)]["member_ids"]]

    id_to_idx = {m: i for i, m in enumerate(member_ids)}
    bli_by_member_idx = np.full(len(member_ids), np.nan)
    for m, b in zip(bli_member_ids, bli_values):
        if m in id_to_idx:
            bli_by_member_idx[id_to_idx[m]] = b

    centralities = compute_centralities(adjacency)

    valid = ~np.isnan(bli_by_member_idx)
    bli_v = bli_by_member_idx[valid]

    correlations = {}
    for name, arr in centralities.items():
        arr_v = arr[valid]
        if arr_v.std() == 0 or bli_v.std() == 0:
            correlations[name] = {"spearman_rho": None, "pearson_r": None}
            continue
        rho, p_rho = stats.spearmanr(bli_v, arr_v)
        r, p_r = stats.pearsonr(bli_v, arr_v)
        correlations[name] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "pearson_r": float(r),
            "pearson_p": float(p_r),
        }

    member_table = pd.DataFrame({
        "icpsr": [int(m) for m in member_ids],
        "name": [str(n) for n in member_names],
        "party": ["D" if p == 100 else "R" if p == 200 else "I" for p in party_codes],
        "bli": bli_by_member_idx,
        **{k: centralities[k] for k in centralities},
    })

    pairwise_matrix = compute_pairwise_correlations(member_table, valid)

    return {
        "congress": int(c),
        "n_members": int(valid.sum()),
        "correlations": correlations,
        "pairwise_correlations": pairwise_matrix,
        "member_table": member_table,
    }


def compute_pairwise_correlations(member_table, valid):
    cols = ["bli", "betweenness", "eigenvector", "closeness", "harmonic", "pagerank", "kcore", "degree", "dynamical_importance", "bli_first_order"]
    sub = member_table.loc[valid, cols].dropna()
    if len(sub) < 5:
        return None
    matrix = {}
    for c1 in cols:
        matrix[c1] = {}
        for c2 in cols:
            a = sub[c1].values
            b = sub[c2].values
            if a.std() == 0 or b.std() == 0:
                matrix[c1][c2] = None
                continue
            rho, _ = stats.spearmanr(a, b)
            matrix[c1][c2] = float(rho)
    return matrix


def top_n_table(member_table, columns, n=10):
    rows = []
    for col in columns:
        sorted_df = member_table.sort_values(col, ascending=False).head(n)
        for rank, (_, row) in enumerate(sorted_df.iterrows(), start=1):
            rows.append({
                "centrality": col,
                "rank": rank,
                "name": row["name"],
                "party": row["party"],
                "icpsr": row["icpsr"],
                "value": float(row[col]),
                "bli": float(row["bli"]) if pd.notna(row["bli"]) else None,
            })
    return pd.DataFrame(rows)


def plot_correlation_trajectory(corr_df, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    measures = [
        ("betweenness", "#1f77b4", "Betweenness"),
        ("eigenvector", "#ff7f0e", "Eigenvector"),
        ("closeness", "#9467bd", "Closeness"),
        ("harmonic", "#8c564b", "Harmonic"),
        ("pagerank", "#e377c2", "PageRank"),
        ("kcore", "#7f7f7f", "K-core"),
        ("degree", "#bcbd22", "Degree"),
        ("dynamical_importance", "#2ca02c", "RestreOttHunt DI"),
    ]
    for m, color, lab in measures:
        col = f"rho_{m}"
        if col in corr_df.columns:
            ax.plot(corr_df["congress"], corr_df[col], "-o", color=color, label=lab, markersize=4, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Congress")
    ax.set_ylabel(r"Spearman rank correlation with BLI")
    ax.set_title("BLI versus standard centralities over time, 100th-118th Congress")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_heatmap(per_congress_results, out_path, congress=112):
    import matplotlib.pyplot as plt
    if congress not in per_congress_results:
        return
    matrix = per_congress_results[congress]["pairwise_correlations"]
    if matrix is None:
        return
    labels = list(matrix.keys())
    n = len(labels)
    M = np.zeros((n, n))
    for i, k1 in enumerate(labels):
        for j, k2 in enumerate(labels):
            v = matrix[k1].get(k2)
            M[i, j] = v if v is not None else 0.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i,j]) > 0.5 else "black")
    ax.set_title(f"Pairwise Spearman correlations, Congress {congress}")
    fig.colorbar(im, ax=ax, shrink=0.8, label="rank correlation")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_panel(per_congress_results, out_path, congresses=(104, 112)):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(congresses), figsize=(5.5 * len(congresses), 4.5))
    if len(congresses) == 1:
        axes = [axes]
    for ax, c in zip(axes, congresses):
        if c not in per_congress_results:
            continue
        mt = per_congress_results[c]["member_table"].dropna(subset=["bli"]).copy()
        ax.scatter(mt["betweenness"], mt["bli"], alpha=0.4, s=12, c="#1f77b4")
        rho = per_congress_results[c]["correlations"]["betweenness"]["spearman_rho"]
        ax.set_xlabel("Betweenness centrality")
        ax.set_ylabel("BLI")
        ax.set_title(f"Congress {c}  (Spearman rho = {rho:.3f})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    bli_data = load_bli_results()

    per_congress_results = {}
    rows = []
    top_table_records = []

    for c in CONGRESSES:
        result = compare_per_congress(c, bli_data)
        if result is None:
            continue
        per_congress_results[c] = result
        row = {"congress": c, "n_members": result["n_members"]}
        for measure, stats_dict in result["correlations"].items():
            row[f"rho_{measure}"] = stats_dict.get("spearman_rho")
            row[f"r_{measure}"] = stats_dict.get("pearson_r")
        rows.append(row)
        print(f"Congress {c}: BLI vs betweenness rho = {row.get('rho_betweenness'):.3f}, "
              f"vs eigenvector = {row.get('rho_eigenvector'):.3f}, "
              f"vs DI = {row.get('rho_dynamical_importance'):.3f}")

        if c in HIGHLIGHT_CONGRESSES:
            mt = result["member_table"].dropna(subset=["bli"])
            top = top_n_table(mt, columns=["bli", "betweenness", "eigenvector", "dynamical_importance"], n=10)
            top["congress"] = c
            top_table_records.append(top)

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(RESULTS_DIR / "centrality_correlations.csv", index=False)

    if top_table_records:
        top_df = pd.concat(top_table_records, ignore_index=True)
        top_df.to_csv(RESULTS_DIR / "centrality_top_members.csv", index=False)

    plot_correlation_trajectory(corr_df, FIGURES_DIR / "bli_vs_centralities.pdf")
    plot_scatter_panel(per_congress_results, FIGURES_DIR / "bli_betweenness_scatter.pdf", congresses=(104, 112))
    plot_pairwise_heatmap(per_congress_results, FIGURES_DIR / "centrality_heatmap_103.pdf", congress=103)
    plot_pairwise_heatmap(per_congress_results, FIGURES_DIR / "centrality_heatmap_112.pdf", congress=112)
    plot_pairwise_heatmap(per_congress_results, FIGURES_DIR / "centrality_heatmap_118.pdf", congress=118)

    centrality_columns = [c for c in corr_df.columns if c.startswith("rho_")]
    summary = {
        "trajectory": corr_df.to_dict(orient="records"),
        "mean_correlations": {col.replace("rho_", ""): float(corr_df[col].mean()) for col in centrality_columns},
        "median_correlations": {col.replace("rho_", ""): float(corr_df[col].median()) for col in centrality_columns},
        "min_correlations": {col.replace("rho_", ""): float(corr_df[col].min()) for col in centrality_columns},
        "max_correlations": {col.replace("rho_", ""): float(corr_df[col].max()) for col in centrality_columns},
        "pairwise_by_congress": {
            str(c): per_congress_results[c]["pairwise_correlations"]
            for c in per_congress_results
            if per_congress_results[c].get("pairwise_correlations") is not None
        },
    }

    with open(RESULTS_DIR / "centrality_comparison_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Mean Spearman rank correlations with BLI:")
    for k, v in summary["mean_correlations"].items():
        print(f"  {k}: {v:.3f}")
    print()
    print(f"Saved to {RESULTS_DIR / 'centrality_comparison_results.json'}")


if __name__ == "__main__":
    main()
