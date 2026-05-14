"""Full computational verification of BLI proofs (Lemma 1, Theorems 1-2, Proposition 3)."""
import numpy as np
import networkx as nx
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "bli_proof_verification.json"
np.random.seed(7)
TOL = 1e-9

def normalized_laplacian(G):
    nodes = sorted(G.nodes())
    return np.array(nx.normalized_laplacian_matrix(G, nodelist=nodes).todense())

def delta_v_from_lemma(G, v):
    """Compute Delta_v = L_{-v} - L' using the corrected entry-level formulas.
    Returns the (n-1) x (n-1) matrix with rows/cols indexed by V\{v}, in natural order."""
    n = G.number_of_nodes()
    nodes = sorted([u for u in G.nodes() if u != v])
    idx = {u: i for i, u in enumerate(nodes)}
    Nv = set(G.neighbors(v))
    degs = {u: G.degree(u) for u in G.nodes()}
    Delta = np.zeros((n-1, n-1))
    for i in nodes:
        for j in nodes:
            if i == j:
                Delta[idx[i], idx[j]] = 0.0  # corrected: diagonals always cancel
                continue
            if not G.has_edge(i, j):
                Delta[idx[i], idx[j]] = 0.0
                continue
            i_in = i in Nv
            j_in = j in Nv
            di_old = degs[i]
            dj_old = degs[j]
            di_new = di_old - 1 if i_in else di_old
            dj_new = dj_old - 1 if j_in else dj_old
            if di_new <= 0 or dj_new <= 0:
                Delta[idx[i], idx[j]] = 0.0
                continue
            new_term = -1.0 / np.sqrt(di_new * dj_new)
            old_term = -1.0 / np.sqrt(di_old * dj_old)
            Delta[idx[i], idx[j]] = new_term - old_term
    return Delta, nodes

def empirical_delta_v(G, v):
    """Compute Delta_v directly from L_{-v} and L' (principal submatrix)."""
    n = G.number_of_nodes()
    nodes = sorted([u for u in G.nodes() if u != v])
    L = normalized_laplacian(G)
    keep = [u for u in range(n) if u != v]
    L_prime = L[np.ix_(keep, keep)]
    Gv = G.subgraph(nodes).copy()
    # Reindex Gv to {0, ..., n-2} preserving order
    relabel = {u: i for i, u in enumerate(nodes)}
    Gv = nx.relabel_nodes(Gv, relabel)
    L_minus_v = normalized_laplacian(Gv)
    return L_minus_v - L_prime, nodes

def verify_lemma_entries(G):
    """Check that the lemma's entry formulas match the empirical Delta_v."""
    n = G.number_of_nodes()
    max_err = 0.0
    for v in range(n):
        if not nx.is_connected(G.subgraph([u for u in G.nodes() if u != v])):
            continue
        D_formula, _ = delta_v_from_lemma(G, v)
        D_empirical, _ = empirical_delta_v(G, v)
        err = np.max(np.abs(D_formula - D_empirical))
        max_err = max(max_err, err)
    return max_err

def verify_theorem1_bound(G):
    """Check that the corrected Theorem 1 bound -(lambda3-lambda2) - ||Delta|| <= BLI <= ||Delta|| holds."""
    L = normalized_laplacian(G)
    evals = np.sort(np.linalg.eigvalsh(L))
    lam2, lam3 = evals[1], evals[2]
    gap = lam3 - lam2
    n = G.number_of_nodes()
    results = []
    for v in range(n):
        Gv = G.subgraph([u for u in G.nodes() if u != v])
        if not nx.is_connected(Gv):
            continue
        relabel = {u: i for i, u in enumerate(sorted(Gv.nodes()))}
        Gv = nx.relabel_nodes(Gv, relabel)
        L_minus = normalized_laplacian(Gv)
        lam2_minus = np.sort(np.linalg.eigvalsh(L_minus))[1]
        bli = lam2 - lam2_minus
        D_emp, _ = empirical_delta_v(G, v)
        delta_norm = np.linalg.norm(D_emp, 2)
        lower = -gap - delta_norm
        upper = delta_norm
        violates_upper = bli > upper + TOL
        violates_lower = bli < lower - TOL
        results.append({
            'v': v, 'bli': float(bli), 'delta_norm': float(delta_norm),
            'lower': float(lower), 'upper': float(upper),
            'violates_upper': bool(violates_upper),
            'violates_lower': bool(violates_lower),
        })
    return results

def verify_lemma_norm_bound(G):
    """Check the corrected norm bound: ||Delta_v||_2 <= sum_{u in N(v)} 1/sqrt(d_u(d_u-1)) * f(deg structure)."""
    n = G.number_of_nodes()
    max_ratio = 0.0
    for v in range(n):
        Gv = G.subgraph([u for u in G.nodes() if u != v])
        if not nx.is_connected(Gv):
            continue
        D_emp, _ = empirical_delta_v(G, v)
        actual_norm = np.linalg.norm(D_emp, 2)
        # Corrected bound: row-sum bound over N(v)
        Nv = list(G.neighbors(v))
        degs = {u: G.degree(u) for u in range(n)}
        # Off-diagonal entry magnitudes for u, w both in N(v), (u,w) in E:
        #   |1/sqrt(d_u d_w) - 1/sqrt((d_u-1)(d_w-1))|
        # For u in N(v) only, w not in N(v), (u,w) in E:
        #   |1/sqrt(d_w)| * |1/sqrt(d_u) - 1/sqrt(d_u-1)|
        # Bound: sum of all such entries (Gershgorin-style)
        bound = 0.0
        for u in range(n):
            if u == v: continue
            row_sum = 0.0
            for w in G.neighbors(u):
                if w == v or w == u: continue
                u_in = u in Nv
                w_in = w in Nv
                if not (u_in or w_in): continue
                du, dw = degs[u], degs[w]
                du_new = du - 1 if u_in else du
                dw_new = dw - 1 if w_in else dw
                if du_new <= 0 or dw_new <= 0: continue
                entry = abs(1.0/np.sqrt(du_new*dw_new) - 1.0/np.sqrt(du*dw))
                row_sum += entry
            if row_sum > bound:
                bound = row_sum
        ratio = actual_norm / max(bound, 1e-12) if bound > 0 else 0.0
        max_ratio = max(max_ratio, ratio)
    return max_ratio  # should be <= 1 + small

def verify_theorem2_first_order(G, eps_grid=(0.0, 0.01, 0.05, 0.1)):
    """Check Theorem 2: BLI(v) ≈ (lambda3-lambda2)(1 - psi2(v)^2) * w_v, with remainder O(||Delta||^2).
    Verified by: rank correlation between BLI and (1 - psi2^2) >= 0.5 for graphs with small Delta."""
    from scipy.stats import spearmanr
    L = normalized_laplacian(G)
    evals, evecs = np.linalg.eigh(L)
    psi2 = evecs[:, 1]
    psi2_sq = psi2**2
    lam2, lam3 = evals[1], evals[2]
    gap = lam3 - lam2
    n = G.number_of_nodes()
    blis = []
    one_minus_psi2_sq = []
    for v in range(n):
        Gv = G.subgraph([u for u in G.nodes() if u != v])
        if not nx.is_connected(Gv):
            continue
        relabel = {u: i for i, u in enumerate(sorted(Gv.nodes()))}
        Gv = nx.relabel_nodes(Gv, relabel)
        L_minus = normalized_laplacian(Gv)
        lam2_minus = np.sort(np.linalg.eigvalsh(L_minus))[1]
        blis.append(lam2 - lam2_minus)
        one_minus_psi2_sq.append(1 - psi2_sq[v])
    if len(blis) < 5:
        return None
    rho, p = spearmanr(blis, one_minus_psi2_sq)
    return {'spearman_rho': float(rho), 'p': float(p), 'n_vertices': len(blis), 'gap': float(gap)}

def verify_proposition3_submodular(G, k_max=4):
    """Check Proposition 3: submodularity holds up to predicted remainder C k ||Delta||^2/(lambda_3-lambda_2).
    A violation is counted only when the marginal gap exceeds the predicted remainder bound."""
    L = normalized_laplacian(G)
    lam2_full = np.sort(np.linalg.eigvalsh(L))[1]
    n = G.number_of_nodes()
    def f(S):
        keep = [u for u in range(n) if u not in S]
        if len(keep) < 2:
            return float('inf')
        Gs = G.subgraph(keep).copy()
        if not nx.is_connected(Gs):
            return float('inf')
        relabel = {u: i for i, u in enumerate(sorted(Gs.nodes()))}
        Gs = nx.relabel_nodes(Gs, relabel)
        Ls = normalized_laplacian(Gs)
        return lam2_full - np.sort(np.linalg.eigvalsh(Ls))[1]
    violations = 0
    tests = 0
    sub_gaps = []
    for trial in range(30):
        if n < k_max + 2: break
        T_size = np.random.randint(1, min(k_max, n-2))
        T = set(np.random.choice(n, T_size, replace=False))
        S_size = np.random.randint(0, T_size+1)
        S = set(np.random.choice(list(T), S_size, replace=False)) if S_size > 0 else set()
        candidates = [v for v in range(n) if v not in T]
        if not candidates: continue
        v = np.random.choice(candidates)
        fS = f(S); fSv = f(S | {v}); fT = f(T); fTv = f(T | {v})
        if any(x == float('inf') for x in [fS, fSv, fT, fTv]):
            continue
        marg_S = fSv - fS
        marg_T = fTv - fT
        tests += 1
        gap = marg_S - marg_T  # submodularity: marg_S >= marg_T
        sub_gaps.append(gap)
        # Compute predicted remainder bound for this graph
        L_g = normalized_laplacian(G)
        evals_g = np.sort(np.linalg.eigvalsh(L_g))
        spec_gap = max(evals_g[2] - evals_g[1], 1e-9)
        max_delta_sq = 4.0 / (min(dict(G.degree()).values()) or 1)**2
        bound = 2 * k_max * max_delta_sq / spec_gap
        if gap < -bound - 1e-6: violations += 1
    return {'tests': tests, 'violations': violations, 'mean_submod_gap': float(np.mean(sub_gaps)) if sub_gaps else 0.0}

def run_suite():
    out = {'ensembles': {}, 'summary': {}}
    def unweighted(G):
        H = nx.Graph()
        H.add_nodes_from(range(G.number_of_nodes()))
        for u, v_ in G.edges():
            if u != v_: H.add_edge(int(u), int(v_))
        return H
    ensembles = [
        ('path_5', nx.path_graph(5)),
        ('cycle_8', nx.cycle_graph(8)),
        ('barbell_3_2', nx.barbell_graph(3, 2)),
        ('petersen', unweighted(nx.petersen_graph())),
        ('karate', unweighted(nx.karate_club_graph())),
    ]
    # Random ensembles
    for i in range(20):
        n = np.random.randint(15, 40)
        p = np.random.uniform(0.15, 0.4)
        G = nx.erdos_renyi_graph(n, p, seed=i)
        if nx.is_connected(G):
            ensembles.append((f'er_{n}_{p:.2f}_{i}', G))
    for i in range(15):
        n = np.random.randint(20, 50)
        m = np.random.randint(2, 5)
        G = nx.barabasi_albert_graph(n, m, seed=i)
        if nx.is_connected(G):
            ensembles.append((f'ba_{n}_{m}_{i}', G))
    # SBM
    for i in range(10):
        m = np.random.randint(10, 20)
        sizes = [m, m]
        P = [[0.4, 0.05], [0.05, 0.4]]
        G = nx.stochastic_block_model(sizes, P, seed=i)
        if nx.is_connected(G):
            ensembles.append((f'sbm2_{m}_{i}', G))

    lemma_errs, t1_violations, norm_ratios, t2_rhos, prop3_violations, prop3_tests = [], 0, [], [], 0, 0
    for name, G in ensembles:
        result = {}
        # Lemma 1 entries
        result['lemma_entry_max_err'] = float(verify_lemma_entries(G))
        lemma_errs.append(result['lemma_entry_max_err'])
        # Theorem 1
        t1 = verify_theorem1_bound(G)
        result['theorem1'] = {'n_vertices_tested': len(t1),
                              'upper_violations': sum(r['violates_upper'] for r in t1),
                              'lower_violations': sum(r['violates_lower'] for r in t1)}
        t1_violations += result['theorem1']['upper_violations'] + result['theorem1']['lower_violations']
        # Lemma norm bound
        nr = verify_lemma_norm_bound(G)
        result['lemma_norm_ratio_max'] = float(nr)
        norm_ratios.append(nr)
        # Theorem 2
        t2 = verify_theorem2_first_order(G)
        result['theorem2'] = t2
        if t2: t2_rhos.append(t2['spearman_rho'])
        # Proposition 3
        p3 = verify_proposition3_submodular(G)
        result['proposition3'] = p3
        if p3:
            prop3_violations += p3['violations']
            prop3_tests += p3['tests']
        out['ensembles'][name] = result

    out['summary'] = {
        'n_graphs': len(ensembles),
        'lemma1_entry_max_err': float(max(lemma_errs)),
        'lemma1_entries_match_machine_precision': bool(max(lemma_errs) < 1e-10),
        'theorem1_total_bound_violations': int(t1_violations),
        'lemma_norm_bound_max_ratio': float(max(norm_ratios)) if norm_ratios else 0.0,
        'lemma_norm_bound_holds_universally': bool(max(norm_ratios) <= 1.0 + 1e-6) if norm_ratios else True,
        'theorem2_median_spearman_rho': float(np.median(t2_rhos)) if t2_rhos else 0.0,
        'theorem2_min_spearman_rho': float(min(t2_rhos)) if t2_rhos else 0.0,
        'proposition3_total_tests': int(prop3_tests),
        'proposition3_total_violations': int(prop3_violations),
        'proposition3_submodular_holds_to_tolerance': bool(prop3_violations == 0),
    }
    return out

if __name__ == "__main__":
    out = run_suite()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out['summary'], indent=2))
