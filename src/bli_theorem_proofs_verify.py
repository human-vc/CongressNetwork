import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR, FIGURES_DIR, SEED

warnings.filterwarnings("ignore")


def normalized_laplacian_dense(A):
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    degrees = A.sum(axis=1)
    if (degrees <= 0).any():
        keep = degrees > 0
        if keep.sum() < 3:
            return None, None
        A_sub = A[np.ix_(keep, keep)]
        d_sub = degrees[keep]
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d_sub))
        L = np.eye(A_sub.shape[0]) - D_inv_sqrt @ A_sub @ D_inv_sqrt
        return L, keep
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    return L, np.ones(n, dtype=bool)


def fiedler_triple(A, k=3):
    L, keep = normalized_laplacian_dense(A)
    if L is None or L.shape[0] < k + 1:
        return None, None, None, None, None
    eigs, vecs = np.linalg.eigh(L)
    order = np.argsort(eigs)
    eigs = eigs[order]
    vecs = vecs[:, order]
    psi_full = np.zeros((A.shape[0], min(k, vecs.shape[1])))
    keep_idx = np.where(keep)[0]
    psi_full[keep_idx, :] = vecs[:, :psi_full.shape[1]]
    return float(eigs[0]), float(eigs[1]), float(eigs[2]) if k >= 3 else None, psi_full, keep


def principal_submatrix_normalized_laplacian(A, v):
    n = A.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[v] = False
    L_full, _ = normalized_laplacian_dense(A)
    if L_full is None:
        return None
    return L_full[np.ix_(mask, mask)]


def leave_one_out_laplacian(A, v):
    n = A.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[v] = False
    A_sub = A[np.ix_(mask, mask)]
    L_sub, _ = normalized_laplacian_dense(A_sub)
    return L_sub


def degree_correction_perturbation(A, v):
    L_principal = principal_submatrix_normalized_laplacian(A, v)
    L_leaveone = leave_one_out_laplacian(A, v)
    if L_principal is None or L_leaveone is None:
        return None, None
    if L_principal.shape != L_leaveone.shape:
        return None, None
    Delta = L_leaveone - L_principal
    return Delta, np.linalg.norm(Delta, ord=2)


def brute_force_bli(A):
    n = A.shape[0]
    L, _ = normalized_laplacian_dense(A)
    if L is None:
        return np.zeros(n)
    eigs, _ = np.linalg.eigh(L)
    base = float(np.sort(eigs)[1])
    bli = np.zeros(n)
    for i in range(n):
        L_sub = leave_one_out_laplacian(A, i)
        if L_sub is None:
            continue
        eigs_sub, _ = np.linalg.eigh(L_sub)
        bli[i] = base - float(np.sort(eigs_sub)[1])
    return bli


def first_order_approximation(A):
    n = A.shape[0]
    lam1, lam2, lam3, psi, _ = fiedler_triple(A, k=3)
    if lam2 is None or psi is None or lam3 is None:
        return np.zeros(n)
    psi2 = psi[:, 1]
    return (lam3 - lam2) * (1.0 - psi2 ** 2)


def verify_theorem_1(A):
    n = A.shape[0]
    lam1, lam2, lam3, _, _ = fiedler_triple(A, k=3)
    if lam2 is None or lam3 is None:
        return None
    bli = brute_force_bli(A)

    rows = []
    for v in range(n):
        Delta_v, delta_norm = degree_correction_perturbation(A, v)
        if delta_norm is None:
            continue
        lower_bound = -(lam3 - lam2) - delta_norm
        upper_bound = delta_norm
        loose_bound = (lam3 - lam2) + delta_norm
        in_tight = lower_bound - 1e-9 <= bli[v] <= upper_bound + 1e-9
        in_loose = -loose_bound - 1e-9 <= bli[v] <= loose_bound + 1e-9
        rows.append({
            "vertex": v,
            "bli": float(bli[v]),
            "delta_norm": float(delta_norm),
            "spectral_gap": float(lam3 - lam2),
            "tight_lower": float(lower_bound),
            "tight_upper": float(upper_bound),
            "loose_bound": float(loose_bound),
            "in_tight_bounds": bool(in_tight),
            "in_loose_bounds": bool(in_loose),
        })
    return rows


def verify_theorem_2(A):
    n = A.shape[0]
    bli = brute_force_bli(A)
    approx = first_order_approximation(A)
    valid = ~np.isnan(bli) & ~np.isnan(approx)
    if valid.sum() < 3:
        return None
    bli_v = bli[valid]
    approx_v = approx[valid]
    if bli_v.std() == 0 or approx_v.std() == 0:
        return None
    rho, p_rho = stats.spearmanr(bli_v, approx_v)
    r, p_r = stats.pearsonr(bli_v, approx_v)
    return {
        "n_vertices": int(valid.sum()),
        "spearman_rho": float(rho),
        "spearman_p": float(p_rho),
        "pearson_r": float(r),
        "pearson_p": float(p_r),
        "max_abs_bli": float(np.max(np.abs(bli_v))),
        "max_abs_approx": float(np.max(np.abs(approx_v))),
        "mean_residual": float(np.mean(bli_v - approx_v)),
        "std_residual": float(np.std(bli_v - approx_v)),
    }


def all_subsets_of_size_k(n, k):
    from itertools import combinations
    return list(combinations(range(n), k))


def evaluate_removal_set(A, S):
    n = A.shape[0]
    if len(S) >= n - 2:
        return None
    base, _ = normalized_laplacian_dense(A)
    if base is None:
        return None
    eigs_base, _ = np.linalg.eigh(base)
    lam2_base = float(np.sort(eigs_base)[1])

    mask = np.ones(n, dtype=bool)
    for v in S:
        mask[v] = False
    A_sub = A[np.ix_(mask, mask)]
    L_sub, _ = normalized_laplacian_dense(A_sub)
    if L_sub is None or L_sub.shape[0] < 3:
        return None
    eigs_sub, _ = np.linalg.eigh(L_sub)
    return lam2_base - float(np.sort(eigs_sub)[1])


def greedy_removal(A, k):
    n = A.shape[0]
    S = []
    A_current = A.copy()
    remaining_idx_map = list(range(n))
    for step in range(k):
        bli_current = brute_force_bli(A_current)
        best_local = int(np.argmax(bli_current))
        best_global = remaining_idx_map[best_local]
        S.append(best_global)
        mask = np.ones(A_current.shape[0], dtype=bool)
        mask[best_local] = False
        A_current = A_current[np.ix_(mask, mask)]
        remaining_idx_map = [remaining_idx_map[i] for i in range(len(remaining_idx_map)) if i != best_local]
        if A_current.shape[0] < 3:
            break
    return S


def verify_proposition_3(A, k_values=(1, 2, 3)):
    n = A.shape[0]
    if n > 10:
        return {"skipped": "n too large for exhaustive search"}
    out = {}
    for k in k_values:
        if k >= n - 2:
            continue
        all_S = all_subsets_of_size_k(n, k)
        best_f = -np.inf
        best_S = None
        for S in all_S:
            f_S = evaluate_removal_set(A, S)
            if f_S is None:
                continue
            if f_S > best_f:
                best_f = f_S
                best_S = S
        S_greedy = greedy_removal(A, k)
        f_greedy = evaluate_removal_set(A, S_greedy)
        ratio = None
        if best_f > 0 and f_greedy is not None:
            ratio = f_greedy / best_f
        out[k] = {
            "optimal_set": list(best_S) if best_S is not None else None,
            "optimal_f": float(best_f) if best_f != -np.inf else None,
            "greedy_set": S_greedy,
            "greedy_f": float(f_greedy) if f_greedy is not None else None,
            "ratio_greedy_to_optimal": float(ratio) if ratio is not None else None,
            "exceeds_1_minus_1_over_e": bool(ratio is not None and ratio >= 1.0 - 1.0 / np.e - 1e-9),
        }
    return out


def generate_graph(name, n, rng):
    if name == "sbm_strong":
        sizes = [n // 2, n - n // 2]
        P = [[0.30, 0.02], [0.02, 0.30]]
        G = nx.stochastic_block_model(sizes, P, seed=int(rng.integers(0, 2**31 - 1)))
    elif name == "sbm_weak":
        sizes = [n // 2, n - n // 2]
        P = [[0.20, 0.08], [0.08, 0.20]]
        G = nx.stochastic_block_model(sizes, P, seed=int(rng.integers(0, 2**31 - 1)))
    elif name == "erdos_renyi":
        G = nx.erdos_renyi_graph(n, 0.15, seed=int(rng.integers(0, 2**31 - 1)))
    elif name == "barabasi_albert":
        G = nx.barabasi_albert_graph(n, 3, seed=int(rng.integers(0, 2**31 - 1)))
    elif name == "planted_bridge":
        m = (n - 5) // 2
        b = 5
        sizes = [m, m, b]
        P = [[0.30, 0.02, 0.15], [0.02, 0.30, 0.15], [0.15, 0.15, 0.02]]
        G = nx.stochastic_block_model(sizes, P, seed=int(rng.integers(0, 2**31 - 1)))
    else:
        raise ValueError(name)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()
        G = nx.convert_node_labels_to_integers(G)
    return nx.to_numpy_array(G, dtype=np.float64)


def run_theorem_1_sweep(n_per_ensemble=50, graph_sizes=(10, 15, 20, 30)):
    rng = np.random.default_rng(SEED)
    ensembles = ["sbm_strong", "sbm_weak", "erdos_renyi", "barabasi_albert", "planted_bridge"]
    all_rows = []
    for ens in ensembles:
        for n in graph_sizes:
            successes_tight = 0
            successes_loose = 0
            total = 0
            for rep in range(n_per_ensemble):
                A = generate_graph(ens, n, rng)
                if A.shape[0] < 8:
                    continue
                rows = verify_theorem_1(A)
                if rows is None:
                    continue
                for r in rows:
                    if r["in_tight_bounds"]:
                        successes_tight += 1
                    if r["in_loose_bounds"]:
                        successes_loose += 1
                    total += 1
            if total > 0:
                all_rows.append({
                    "ensemble": ens,
                    "n_target": n,
                    "n_vertex_observations": int(total),
                    "share_satisfying_tight": float(successes_tight / total),
                    "share_satisfying_loose": float(successes_loose / total),
                })
    return pd.DataFrame(all_rows)


def run_theorem_2_sweep(n_per_ensemble=50, graph_sizes=(20, 40, 80)):
    rng = np.random.default_rng(SEED + 1)
    ensembles = ["sbm_strong", "sbm_weak", "erdos_renyi", "barabasi_albert", "planted_bridge"]
    rows = []
    for ens in ensembles:
        for n in graph_sizes:
            spearmans = []
            pearsons = []
            for rep in range(n_per_ensemble):
                A = generate_graph(ens, n, rng)
                if A.shape[0] < 10:
                    continue
                res = verify_theorem_2(A)
                if res is None:
                    continue
                spearmans.append(res["spearman_rho"])
                pearsons.append(res["pearson_r"])
            if spearmans:
                rows.append({
                    "ensemble": ens,
                    "n_target": n,
                    "n_graphs": len(spearmans),
                    "spearman_mean": float(np.mean(spearmans)),
                    "spearman_std": float(np.std(spearmans)),
                    "pearson_mean": float(np.mean(pearsons)),
                    "pearson_std": float(np.std(pearsons)),
                })
    return pd.DataFrame(rows)


def run_proposition_3_sweep(n_per_ensemble=30, graph_sizes=(8, 10)):
    rng = np.random.default_rng(SEED + 2)
    ensembles = ["sbm_strong", "sbm_weak", "planted_bridge"]
    rows = []
    for ens in ensembles:
        for n in graph_sizes:
            ratios_k1 = []
            ratios_k2 = []
            ratios_k3 = []
            for rep in range(n_per_ensemble):
                A = generate_graph(ens, n, rng)
                if A.shape[0] < 6 or A.shape[0] > 10:
                    continue
                res = verify_proposition_3(A, k_values=(1, 2, 3))
                if not isinstance(res, dict) or "skipped" in res:
                    continue
                if 1 in res and res[1]["ratio_greedy_to_optimal"] is not None:
                    ratios_k1.append(res[1]["ratio_greedy_to_optimal"])
                if 2 in res and res[2]["ratio_greedy_to_optimal"] is not None:
                    ratios_k2.append(res[2]["ratio_greedy_to_optimal"])
                if 3 in res and res[3]["ratio_greedy_to_optimal"] is not None:
                    ratios_k3.append(res[3]["ratio_greedy_to_optimal"])

            def stat_block(lst):
                if not lst:
                    return None
                arr = np.array(lst)
                return {
                    "n_graphs": int(len(arr)),
                    "mean_ratio": float(arr.mean()),
                    "min_ratio": float(arr.min()),
                    "share_above_1_minus_1_e": float((arr >= 1.0 - 1.0 / np.e - 1e-9).mean()),
                }

            rows.append({
                "ensemble": ens,
                "n_target": n,
                "k_1": stat_block(ratios_k1),
                "k_2": stat_block(ratios_k2),
                "k_3": stat_block(ratios_k3),
            })
    return rows


def verify_barbell_canonical():
    A = np.array([
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
    ], dtype=np.float64)

    lam1, lam2, lam3, psi, _ = fiedler_triple(A, k=3)
    bli = brute_force_bli(A)
    approx = first_order_approximation(A)

    deltas = []
    bounds = []
    for v in range(6):
        Delta_v, dnorm = degree_correction_perturbation(A, v)
        deltas.append(float(dnorm))
        upper = float(dnorm)
        lower = -(lam3 - lam2) - float(dnorm)
        in_tight = lower - 1e-9 <= bli[v] <= upper + 1e-9
        bounds.append({"vertex": v, "lower": lower, "upper": upper, "bli": float(bli[v]), "in_bounds": bool(in_tight)})

    return {
        "lambda_1": lam1, "lambda_2": lam2, "lambda_3": lam3,
        "spectral_gap": lam3 - lam2,
        "psi_2": psi[:, 1].tolist(),
        "psi_2_squared": (psi[:, 1] ** 2).tolist(),
        "bli_brute_force": bli.tolist(),
        "first_order_approximation": approx.tolist(),
        "delta_norms": deltas,
        "interlacing_check": bounds,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Barbell canonical case ===")
    barbell = verify_barbell_canonical()
    print(f"lambda_2 = {barbell['lambda_2']:.4f}, lambda_3 = {barbell['lambda_3']:.4f}")
    print(f"spectral gap = {barbell['spectral_gap']:.4f}")
    print(f"BLI brute force: {[f'{v:+.4f}' for v in barbell['bli_brute_force']]}")
    print(f"First-order approx: {[f'{v:+.4f}' for v in barbell['first_order_approximation']]}")
    print(f"Delta norms: {[f'{v:.4f}' for v in barbell['delta_norms']]}")
    print("Interlacing check:")
    for b in barbell["interlacing_check"]:
        ok = "OK" if b["in_bounds"] else "FAIL"
        print(f"  v={b['vertex']}: BLI={b['bli']:+.4f}, [lower, upper]=[{b['lower']:+.4f}, {b['upper']:+.4f}] {ok}")

    print("\n=== Theorem 1 sweep (interlacing bounds) ===")
    df1 = run_theorem_1_sweep()
    df1.to_csv(RESULTS_DIR / "bli_theorem1_verification.csv", index=False)
    print(df1.to_string(index=False))

    print("\n=== Theorem 2 sweep (first-order approximation) ===")
    df2 = run_theorem_2_sweep()
    df2.to_csv(RESULTS_DIR / "bli_theorem2_verification.csv", index=False)
    print(df2.to_string(index=False))

    print("\n=== Proposition 3 sweep (submodularity / greedy guarantee) ===")
    rows3 = run_proposition_3_sweep()
    print(json.dumps(rows3, indent=2, default=str))

    bound_threshold = 1.0 - 1.0 / np.e
    print(f"\nGreedy guarantee threshold (1 - 1/e) = {bound_threshold:.4f}")

    output = {
        "barbell_canonical": barbell,
        "theorem_1_sweep": df1.to_dict(orient="records"),
        "theorem_2_sweep": df2.to_dict(orient="records"),
        "proposition_3_sweep": rows3,
        "greedy_threshold": bound_threshold,
    }
    out_path = RESULTS_DIR / "bli_theorem_proofs_verification.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
