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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FIGURES_DIR, SEED

warnings.filterwarnings("ignore")


def normalized_laplacian_dense(A):
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    degrees = A.sum(axis=1)
    keep = degrees > 0
    if keep.sum() < 3:
        return None, None
    A_sub = A[np.ix_(keep, keep)]
    d_sub = degrees[keep]
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_sub))
    L = np.eye(A_sub.shape[0]) - D_inv_sqrt @ A_sub @ D_inv_sqrt
    return L, keep


def fiedler_pair(A, k=3):
    L, keep = normalized_laplacian_dense(A)
    if L is None or L.shape[0] < k + 1:
        return None, None, None, None
    eigs, vecs = np.linalg.eigh(L)
    order = np.argsort(eigs)
    eigs = eigs[order]
    vecs = vecs[:, order]
    psi_full = np.zeros((len(keep), k))
    for j in range(min(k, vecs.shape[1])):
        psi_full[keep, j] = vecs[:, j]
    return float(eigs[1]), float(eigs[2]) if len(eigs) > 2 else None, psi_full, keep


def brute_force_bli(A):
    n = A.shape[0]
    base_eigs, base_eigs3, _, _ = fiedler_pair(A)
    if base_eigs is None:
        return np.zeros(n)
    bli = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sub = A[np.ix_(mask, mask)]
        sub_eigs, _, _, _ = fiedler_pair(sub)
        if sub_eigs is not None:
            bli[i] = base_eigs - sub_eigs
    return bli


def candidate_predictors(A):
    base_lambda2, base_lambda3, psi, keep = fiedler_pair(A)
    n = A.shape[0]
    if base_lambda2 is None or psi is None:
        return {
            "candidate_A": np.zeros(n),
            "candidate_B": np.zeros(n),
            "candidate_C": np.zeros(n),
            "candidate_D_naive_ROH": np.zeros(n),
            "candidate_E_kirkland": np.zeros(n),
            "candidate_F_chen_hero": np.zeros(n),
        }

    psi2 = psi[:, 1]
    psi3 = psi[:, 2] if psi.shape[1] > 2 and base_lambda3 is not None else np.zeros_like(psi2)

    gap = (base_lambda3 - base_lambda2) if base_lambda3 is not None else 0.0

    cand_A = base_lambda2 * (1.0 - psi2 ** 2)
    cand_B = base_lambda2 * np.maximum(psi3 ** 2 - psi2 ** 2, 0.0)
    cand_C = gap * (1.0 - psi2 ** 2)
    denom = 1.0 - psi2 ** 2
    denom = np.where(np.abs(denom) > 1e-12, denom, np.sign(denom) * 1e-12 + 1e-12)
    cand_D = base_lambda2 * (psi2 ** 2) / denom

    degrees = A.sum(axis=1)
    cand_E = np.zeros(n)
    cand_F = np.zeros(n)
    for i in range(n):
        if degrees[i] == 0:
            continue
        neighbors = np.where(A[i] > 0)[0]
        if len(neighbors) == 0:
            continue
        cand_F[i] = float(np.sum((psi2[i] - psi2[neighbors]) ** 2))
        cand_E[i] = base_lambda2 - max(0.0, base_lambda2 - gap * (1 - psi2[i] ** 2))

    return {
        "candidate_A_lambda2_times_1minus_psi2sq": cand_A,
        "candidate_B_lambda2_times_psi3sq_minus_psi2sq": cand_B,
        "candidate_C_gap_times_1minus_psi2sq": cand_C,
        "candidate_D_naive_ROH": cand_D,
        "candidate_E_kirkland_normalized": cand_E,
        "candidate_F_chen_hero_lfvc": cand_F,
    }


def measure_correlations(bli, predictors):
    out = {}
    valid = np.isfinite(bli)
    for name, pred in predictors.items():
        v = valid & np.isfinite(pred)
        if v.sum() < 5 or np.std(bli[v]) == 0 or np.std(pred[v]) == 0:
            out[name] = {"pearson": None, "spearman": None, "n": int(v.sum())}
            continue
        r, _ = stats.pearsonr(bli[v], pred[v])
        rho, _ = stats.spearmanr(bli[v], pred[v])
        out[name] = {"pearson": float(r), "spearman": float(rho), "n": int(v.sum())}
    return out


def cauchy_interlacing_check(A):
    base_lambda2, base_lambda3, _, _ = fiedler_pair(A)
    if base_lambda2 is None:
        return None
    n = A.shape[0]
    violations = 0
    blis = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        sub = A[np.ix_(mask, mask)]
        sub_lambda2, sub_lambda3, _, _ = fiedler_pair(sub)
        if sub_lambda2 is None:
            continue
        bli = base_lambda2 - sub_lambda2
        blis.append(bli)
        if bli < -1e-9:
            violations += 1
        if base_lambda3 is not None and bli > (base_lambda3 - base_lambda2) + 1e-6:
            violations += 1
    return {
        "bli_min": float(min(blis)) if blis else None,
        "bli_max": float(max(blis)) if blis else None,
        "spectral_gap_upper_bound": float(base_lambda3 - base_lambda2) if base_lambda3 is not None else None,
        "interlacing_violations": int(violations),
    }


def generate_sbm(n, p_in, p_out, n_blocks=2, rng=None):
    rng = rng or np.random.default_rng()
    block_sizes = [n // n_blocks] * n_blocks
    block_sizes[0] += n - sum(block_sizes)
    P = np.full((n_blocks, n_blocks), p_out)
    np.fill_diagonal(P, p_in)
    G = nx.stochastic_block_model(block_sizes, P, seed=int(rng.integers(0, 2**31 - 1)))
    return nx.to_numpy_array(G, dtype=np.float64)


def generate_erdos_renyi(n, p, rng=None):
    rng = rng or np.random.default_rng()
    G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))
    return nx.to_numpy_array(G, dtype=np.float64)


def generate_barabasi_albert(n, m, rng=None):
    rng = rng or np.random.default_rng()
    G = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**31 - 1)))
    return nx.to_numpy_array(G, dtype=np.float64)


def generate_planted_bridge(n_main, n_bridges, p_in, p_out, rng=None):
    rng = rng or np.random.default_rng()
    p_bridge = min(0.8, 5 * p_in)
    sizes = [n_main, n_main, n_bridges]
    P = [
        [p_in, p_out, p_bridge],
        [p_out, p_in, p_bridge],
        [p_bridge, p_bridge, p_out],
    ]
    G = nx.stochastic_block_model(sizes, P, seed=int(rng.integers(0, 2**31 - 1)))
    A = nx.to_numpy_array(G, dtype=np.float64)
    bridge_indices = np.arange(2 * n_main, 2 * n_main + n_bridges)
    return A, bridge_indices


def run_ensemble(name, generator, params_list, n_reps, rng):
    rows = []
    for params in params_list:
        for rep in range(n_reps):
            try:
                A = generator(**params, rng=rng)
                if A.shape[0] < 10:
                    continue
                bli = brute_force_bli(A)
                preds = candidate_predictors(A)
                corrs = measure_correlations(bli, preds)
                interlacing = cauchy_interlacing_check(A)
                row = {"ensemble": name, "rep": rep, **params}
                for cand_name, c_info in corrs.items():
                    row[f"pearson_{cand_name}"] = c_info["pearson"]
                    row[f"spearman_{cand_name}"] = c_info["spearman"]
                row["interlacing_violations"] = interlacing["interlacing_violations"] if interlacing else None
                row["bli_min"] = interlacing["bli_min"] if interlacing else None
                row["bli_max"] = interlacing["bli_max"] if interlacing else None
                row["spectral_gap"] = interlacing["spectral_gap_upper_bound"] if interlacing else None
                rows.append(row)
            except Exception as e:
                continue
    return pd.DataFrame(rows)


def sympy_symbolic_verification():
    import sympy as sp
    results = {}

    def normalized_laplacian_sympy(A):
        n = A.rows
        D = sp.Matrix.zeros(n, n)
        for i in range(n):
            D[i, i] = sum(A.row(i))
        D_inv_sqrt = sp.Matrix.zeros(n, n)
        for i in range(n):
            if D[i, i] != 0:
                D_inv_sqrt[i, i] = 1 / sp.sqrt(D[i, i])
        L = sp.eye(n) - D_inv_sqrt * A * D_inv_sqrt
        return L

    A_path = sp.Matrix([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    L_path = normalized_laplacian_sympy(A_path)
    eigs_path = sorted([float(e.evalf()) for e in L_path.eigenvals().keys()])
    results["path_n4"] = {
        "laplacian_eigenvalues": eigs_path,
        "lambda_2": eigs_path[1],
        "lambda_3": eigs_path[2] if len(eigs_path) > 2 else None,
    }

    A_barbell = sp.Matrix([
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ])
    L_barbell = normalized_laplacian_sympy(A_barbell)
    eigs_b = sorted([float(e.evalf()) for e in L_barbell.eigenvals().keys()])
    results["barbell_n6"] = {
        "laplacian_eigenvalues": eigs_b,
        "lambda_2": eigs_b[1],
        "lambda_3": eigs_b[2] if len(eigs_b) > 2 else None,
        "bridge_node_index": 3,
        "comment": "node 3 is the bridge; expected high BLI",
    }

    A_np = np.array(A_barbell.tolist(), dtype=np.float64)
    bli_np = brute_force_bli(A_np)
    base_lambda2, base_lambda3, psi, _ = fiedler_pair(A_np)
    psi2 = psi[:, 1] if psi is not None else None
    psi3 = psi[:, 2] if psi is not None and psi.shape[1] > 2 else None
    results["barbell_n6"]["bli_brute_force"] = bli_np.tolist()
    if psi2 is not None:
        results["barbell_n6"]["psi_2"] = psi2.tolist()
        results["barbell_n6"]["psi_2_squared"] = (psi2 ** 2).tolist()
        results["barbell_n6"]["candidate_A_predictions"] = (base_lambda2 * (1 - psi2 ** 2)).tolist()
        if base_lambda3 is not None:
            results["barbell_n6"]["candidate_C_predictions"] = ((base_lambda3 - base_lambda2) * (1 - psi2 ** 2)).tolist()

    A_path_np = np.array(A_path.tolist(), dtype=np.float64)
    bli_path_np = brute_force_bli(A_path_np)
    base_lambda2_p, base_lambda3_p, psi_p, _ = fiedler_pair(A_path_np)
    psi2_p = psi_p[:, 1]
    results["path_n4"]["bli_brute_force"] = bli_path_np.tolist()
    results["path_n4"]["psi_2_squared"] = (psi2_p ** 2).tolist()
    results["path_n4"]["candidate_A_predictions"] = (base_lambda2_p * (1 - psi2_p ** 2)).tolist()
    if base_lambda3_p is not None:
        results["path_n4"]["candidate_C_predictions"] = ((base_lambda3_p - base_lambda2_p) * (1 - psi2_p ** 2)).tolist()

    return results


def plot_verification_summary(df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    candidate_cols = [c for c in df.columns if c.startswith("spearman_candidate_")]
    candidates = [c.replace("spearman_", "") for c in candidate_cols]

    ax = axes[0]
    ensemble_data = []
    labels = []
    for ens in sorted(df["ensemble"].unique()):
        sub = df[df["ensemble"] == ens]
        for col in candidate_cols:
            vals = sub[col].dropna().values
            if len(vals) > 0:
                ensemble_data.append(vals)
                labels.append(f"{ens}\n{col.replace('spearman_candidate_', '')[:10]}")
    if ensemble_data:
        ax.boxplot(ensemble_data, showfliers=False, widths=0.5)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(0.9, color="red", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Spearman rho with brute-force BLI")
    ax.set_title("Candidate formula correlations by ensemble")

    ax = axes[1]
    summary = {}
    for col in candidate_cols:
        summary[col.replace("spearman_candidate_", "")] = df[col].mean()
    names = list(summary.keys())
    values = [summary[n] for n in names]
    ax.barh(names, values, color=["#1f77b4" if v > 0 else "#d62728" for v in values])
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Mean Spearman rho across all ensembles")
    ax.set_title("Overall ranking of candidate predictors")
    ax.tick_params(axis='y', labelsize=8)

    ax = axes[2]
    if "interlacing_violations" in df.columns:
        violations = df["interlacing_violations"].fillna(0)
        ax.hist(violations, bins=20, color="#2ca02c")
        ax.set_xlabel("Cauchy interlacing violations per graph")
        ax.set_ylabel("Number of graphs")
        ax.set_title(f"Interlacing constraint check\n(0 = constraint holds for all nodes)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Running SymPy symbolic verification on small graphs...")
    symbolic_results = sympy_symbolic_verification()
    print(f"  Path n=4: lambda_2 = {symbolic_results['path_n4']['lambda_2']:.4f}")
    print(f"  Path n=4 BLI: {[f'{v:.4f}' for v in symbolic_results['path_n4']['bli_brute_force']]}")
    print(f"  Path n=4 Candidate A: {[f'{v:.4f}' for v in symbolic_results['path_n4']['candidate_A_predictions']]}")
    print(f"  Barbell n=6: lambda_2 = {symbolic_results['barbell_n6']['lambda_2']:.4f}, lambda_3 = {symbolic_results['barbell_n6']['lambda_3']:.4f}")
    print(f"  Barbell n=6 BLI: {[f'{v:.4f}' for v in symbolic_results['barbell_n6']['bli_brute_force']]}")
    print(f"  Bridge nodes 2 and 3 should have highest BLI")

    print("\nRunning numerical verification across ensembles...")

    ensembles = {
        "sbm_strong_partition": ("sbm", [
            {"n": 50, "p_in": 0.4, "p_out": 0.02, "n_blocks": 2},
            {"n": 50, "p_in": 0.3, "p_out": 0.03, "n_blocks": 2},
            {"n": 100, "p_in": 0.3, "p_out": 0.02, "n_blocks": 2},
            {"n": 100, "p_in": 0.2, "p_out": 0.01, "n_blocks": 2},
        ], 30),
        "sbm_weak_partition": ("sbm", [
            {"n": 50, "p_in": 0.2, "p_out": 0.1, "n_blocks": 2},
            {"n": 100, "p_in": 0.15, "p_out": 0.08, "n_blocks": 2},
        ], 30),
        "sbm_three_blocks": ("sbm", [
            {"n": 60, "p_in": 0.3, "p_out": 0.02, "n_blocks": 3},
            {"n": 90, "p_in": 0.25, "p_out": 0.02, "n_blocks": 3},
        ], 20),
        "erdos_renyi": ("er", [
            {"n": 50, "p": 0.1},
            {"n": 50, "p": 0.2},
            {"n": 100, "p": 0.1},
        ], 20),
        "barabasi_albert": ("ba", [
            {"n": 50, "m": 3},
            {"n": 100, "m": 4},
        ], 20),
        "planted_bridge": ("pb", [
            {"n_main": 25, "n_bridges": 3, "p_in": 0.3, "p_out": 0.02},
            {"n_main": 50, "n_bridges": 5, "p_in": 0.25, "p_out": 0.02},
        ], 20),
    }

    generators = {
        "sbm": generate_sbm,
        "er": generate_erdos_renyi,
        "ba": generate_barabasi_albert,
        "pb": lambda **kw: generate_planted_bridge(**kw)[0],
    }

    all_results = []
    for ens_name, (gen_key, params_list, n_reps) in ensembles.items():
        print(f"  {ens_name}: {len(params_list)} param settings x {n_reps} reps = {len(params_list)*n_reps} graphs")
        df = run_ensemble(ens_name, generators[gen_key], params_list, n_reps, rng)
        all_results.append(df)

    full_df = pd.concat(all_results, ignore_index=True)
    full_df.to_csv(RESULTS_DIR / "bli_theorem_verification.csv", index=False)

    print()
    print("Mean Spearman correlations with brute-force BLI:")
    candidate_cols = [c for c in full_df.columns if c.startswith("spearman_candidate_")]
    overall_means = {}
    for col in candidate_cols:
        name = col.replace("spearman_candidate_", "")
        mean_val = full_df[col].mean()
        overall_means[name] = float(mean_val) if pd.notna(mean_val) else None
        print(f"  {name}: {mean_val:.4f}")

    print()
    print("Spearman correlation by ensemble (rows) and candidate (columns):")
    ensemble_summary = {}
    for ens in sorted(full_df["ensemble"].unique()):
        sub = full_df[full_df["ensemble"] == ens]
        means = {}
        for col in candidate_cols:
            name = col.replace("spearman_candidate_", "")
            v = sub[col].mean()
            means[name] = float(v) if pd.notna(v) else None
        ensemble_summary[ens] = means
        print(f"  {ens}:")
        for name, val in means.items():
            if val is not None:
                print(f"    {name}: {val:.4f}")

    plot_verification_summary(full_df, FIGURES_DIR / "bli_theorem_verification.pdf")

    interlacing_failures = (full_df["interlacing_violations"].fillna(0) > 0).sum()
    bli_min_overall = full_df["bli_min"].min()
    bli_max_overall = full_df["bli_max"].max()
    print()
    print(f"Cauchy interlacing check:")
    print(f"  Graphs with >0 violations: {interlacing_failures} of {len(full_df)}")
    print(f"  Min BLI observed: {bli_min_overall:.6f}")
    print(f"  Max BLI observed: {bli_max_overall:.6f}")

    output = {
        "symbolic_verification": symbolic_results,
        "overall_mean_spearman": overall_means,
        "ensemble_means": ensemble_summary,
        "n_graphs_tested": int(len(full_df)),
        "interlacing": {
            "graphs_with_violations": int(interlacing_failures),
            "bli_min": float(bli_min_overall),
            "bli_max": float(bli_max_overall),
        },
    }
    with open(RESULTS_DIR / "bli_theorem_verification.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR / 'bli_theorem_verification.json'}")


if __name__ == "__main__":
    main()
