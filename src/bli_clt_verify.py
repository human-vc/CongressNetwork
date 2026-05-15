import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse, stats
from scipy.sparse.linalg import eigsh
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    Dis = sparse.diags(1.0 / np.sqrt(d[keep]))
    L = sparse.eye(Asub.shape[0]) - Dis @ Asub @ Dis
    try:
        w, v = eigsh(L, k=3, which="SM", tol=1e-6)
    except Exception:
        return None
    order = np.argsort(w)
    psi = np.zeros((n, 3))
    psi[np.where(keep)[0], :] = v[:, order]
    return float(w[order[1]]), float(w[order[2]]), psi

def first_order_bli(A):
    out = fiedler_triple(A)
    if out is None:
        return None
    lam2, lam3, psi = out
    return (lam3 - lam2) * (1.0 - psi[:, 1] ** 2)

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

def one_sample(m, b, p_in, p_out, k_main, seed):
    A = gen_sbm(m, b, p_in, p_out, k_main, seed)
    return first_order_bli(A)

def _wasserstein2_to_normal(Z):
    Z = np.asarray(Z, dtype=float)
    Z = Z[np.isfinite(Z)]
    if Z.size < 50:
        return float("nan")
    Z_sorted = np.sort(Z)
    n = Z_sorted.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    q = stats.norm.ppf(probs)
    return float(np.sqrt(np.mean((Z_sorted - q) ** 2)))

def _moment_bootstrap_ci(Z, n_boot=200, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    if Z.size < 100:
        return None
    n = Z.size
    moments = np.empty((n_boot, 4))
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        zb = Z[idx]
        moments[i, 0] = zb.mean()
        moments[i, 1] = zb.var(ddof=1)
        moments[i, 2] = stats.skew(zb)
        moments[i, 3] = stats.kurtosis(zb)
    lo = np.quantile(moments, alpha / 2, axis=0)
    hi = np.quantile(moments, 1 - alpha / 2, axis=0)
    return {
        "mean_ci": [float(lo[0]), float(hi[0])],
        "var_ci": [float(lo[1]), float(hi[1])],
        "skew_ci": [float(lo[2]), float(hi[2])],
        "kurt_excess_ci": [float(lo[3]), float(hi[3])],
    }

def run_cell(m, b, p_in, p_out, k_main, N, base_seed, n_jobs):
    samples = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(one_sample)(m, b, p_in, p_out, k_main, base_seed + s) for s in range(2 * N)
    )
    samples = [s for s in samples if s is not None]
    if len(samples) < 60:
        return None
    M = np.stack(samples, axis=0)
    half = M.shape[0] // 2
    M_moments = M[:half]
    M_test = M[half: 2 * half]
    n_total = k_main * m + b
    bridge_mask = np.zeros(n_total, dtype=bool)
    bridge_mask[k_main * m: k_main * m + b] = True

    out = {}
    for role, mask in [("bridge", bridge_mask), ("non_bridge", ~bridge_mask)]:
        block_m = M_moments[:, mask]
        block_t = M_test[:, mask]
        mu_hat = block_m.mean(axis=0)
        sigma_hat = block_m.std(axis=0, ddof=1)
        good = sigma_hat > 1e-12
        if good.sum() == 0:
            continue
        Z_mat = (block_t[:, good] - mu_hat[good][None, :]) / sigma_hat[good][None, :]
        Z = Z_mat.ravel()
        Z = Z[np.isfinite(Z)]
        if Z.size < 100:
            continue
        ks_stat, ks_p = stats.kstest(Z, "norm")
        ad = stats.anderson(Z, dist="norm")
        sw_stat, sw_p = stats.shapiro(Z[:5000]) if Z.size >= 8 else (np.nan, np.nan)
        out[role] = {
            "n_samples": int(Z.size),
            "mean": float(Z.mean()),
            "var": float(Z.var(ddof=1)),
            "skew": float(stats.skew(Z)),
            "kurt_excess": float(stats.kurtosis(Z)),
            "moment_bootstrap_ci": _moment_bootstrap_ci(Z, seed=base_seed),
            "wasserstein2_to_normal": _wasserstein2_to_normal(Z),
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "ad_stat": float(ad.statistic),
            "ad_crit_5pct": float(ad.critical_values[2]),
            "sw_stat": float(sw_stat), "sw_p": float(sw_p),
            "note": ("AD/KS/SW at N>=5000 have >0.99 power against W2 gaps of "
                     "order 1e-3; treat moment + Wasserstein evidence as the "
                     "substantive metric (Theorem 5 reports moment convergence)."),
            "Z": Z,
        }
    out["meta"] = {"n_total": n_total, "N_used_for_moments": int(half),
                   "N_used_for_tests": int(half),
                   "m": m, "b": b, "p_in": p_in, "p_out": p_out, "k_main": k_main}
    return out

def plot_clt(cell_out, path, label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    grid = np.linspace(-4, 4, 200)
    for ax, role in zip(axes, ["non_bridge", "bridge"]):
        if role not in cell_out:
            continue
        Z = cell_out[role]["Z"]
        ax.hist(Z, bins=40, density=True, alpha=0.55, color="#1f77b4", edgecolor="white")
        ax.plot(grid, stats.norm.pdf(grid), "k-", lw=1.8, label="N(0,1)")
        ax.set_title(f"{role}: KS p={cell_out[role]['ks_p']:.3f}, "
                     f"skew={cell_out[role]['skew']:.2f}, kurt={cell_out[role]['kurt_excess']:.2f}")
        ax.set_xlabel("Standardised BLI")
        ax.legend(loc="upper right")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="medium", choices=["smoke", "medium", "large"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if args.scale == "smoke":
        cells = [{"m": 25, "b": 5, "p_in": 0.20, "p_out": 0.02, "k_main": 2, "N": 100}]
    elif args.scale == "medium":
        cells = [{"m": m, "b": 5, "p_in": 0.20, "p_out": 0.02, "k_main": 2, "N": 300}
                 for m in [50, 100, 200]]
    else:
        cells = [{"m": m, "b": 10, "p_in": 0.20, "p_out": 0.02, "k_main": 2, "N": 500}
                 for m in [100, 200, 500]]

    print(f"CLT verification: {len(cells)} cells, scale={args.scale}")
    summary = []
    t0 = time.time()
    for i, cell in enumerate(cells):
        n_total = cell["k_main"] * cell["m"] + cell["b"]
        print(f"\n--- cell {i + 1}/{len(cells)}: n={n_total}, N={cell['N']} ---")
        ct = time.time()
        out = run_cell(base_seed=args.seed + 10_000 * i, n_jobs=args.n_jobs, **cell)
        print(f"  done in {time.time() - ct:.1f}s")
        if out is None:
            continue
        label = f"n={n_total}, N={out['meta']['N_used_for_tests']}, " \
                f"p_in={cell['p_in']}, p_out={cell['p_out']}, k={cell['k_main']}"
        plot_clt(out, FIGURES_DIR / f"bli_clt_n{n_total}.pdf", label)
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "Z"}
                for k, v in out.items() if k != "meta" and isinstance(v, dict)}
        slim["meta"] = out["meta"]
        summary.append(slim)
        for role in ["non_bridge", "bridge"]:
            if role in out:
                s = out[role]
                print(f"  {role:11s}: skew={s['skew']:+.3f}  kurt={s['kurt_excess']:+.3f}  "
                      f"W2={s['wasserstein2_to_normal']:.3f}  AD={s['ad_stat']:.2f} "
                      f"(crit5%={s['ad_crit_5pct']:.2f})")

    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")
    out_json = RESULTS_DIR / f"bli_clt_verify_{args.scale}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved {out_json}")

if __name__ == "__main__":
    main()
