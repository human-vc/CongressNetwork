"""Bridge Legislator Index for EP10 (current term) from HowTheyVote.eu data.

Pipeline:
  1) Load member_votes/members/group_memberships CSVs (polars for speed)
  2) Pivot to MEP x vote sparse matrix, encode +1=FOR, -1=AGAINST, 0=other
  3) Compute pairwise agreement via sparse matmul; threshold at tau=0.5
  4) Fiedler value + leave-one-out BLI on the agreement graph
  5) Dump JSON: fiedler, top-30 bridge MEPs (name, group, country)
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RESULTS_DIR, THRESHOLD_TAU, MIN_SHARED_VOTES

EP_RAW = DATA_DIR / "ep_raw" / "howtheyvote"
TERM = 10
TERM_START = "2024-07-16"
MIN_VOTES_EP = 50


def load_term(term: int, start_date: str):
    members = pl.read_csv(EP_RAW / "members.csv.gz")
    votes = pl.read_csv(EP_RAW / "votes.csv.gz", try_parse_dates=True)
    mv = pl.read_csv(EP_RAW / "member_votes.csv.gz")
    gm = pl.read_csv(EP_RAW / "group_memberships.csv.gz", try_parse_dates=True)

    term_votes = votes.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_datetime())
    mv = mv.join(term_votes.select("id").rename({"id": "vote_id"}), on="vote_id", how="inner")
    mv = mv.filter(pl.col("position").is_in(["FOR", "AGAINST"]))

    counts = mv.group_by("member_id").len().filter(pl.col("len") >= MIN_VOTES_EP)
    mv = mv.join(counts.select("member_id"), on="member_id", how="inner")

    gm_term = gm.filter(pl.col("term") == term).group_by("member_id").agg(pl.col("group_code").last())
    members = members.join(gm_term, left_on="id", right_on="member_id", how="inner")
    return mv, members, term_votes


def agreement_matrix(mv: pl.DataFrame, member_ids: np.ndarray, vote_ids: np.ndarray):
    m_idx = {m: i for i, m in enumerate(member_ids)}
    v_idx = {v: j for j, v in enumerate(vote_ids)}
    rows = mv["member_id"].to_numpy()
    cols = mv["vote_id"].to_numpy()
    pos = mv["position"].to_numpy()

    keep = np.array([(r in m_idx) and (c in v_idx) for r, c in zip(rows, cols)])
    r = np.fromiter((m_idx[x] for x in rows[keep]), dtype=np.int32)
    c = np.fromiter((v_idx[x] for x in cols[keep]), dtype=np.int32)
    p = pos[keep]

    n, k = len(member_ids), len(vote_ids)
    yes = sparse.csr_matrix(
        (np.ones(np.sum(p == "FOR"), dtype=np.float32),
         (r[p == "FOR"], c[p == "FOR"])), shape=(n, k))
    no = sparse.csr_matrix(
        (np.ones(np.sum(p == "AGAINST"), dtype=np.float32),
         (r[p == "AGAINST"], c[p == "AGAINST"])), shape=(n, k))
    cast = yes + no

    both_yes = (yes @ yes.T).toarray()
    both_no = (no @ no.T).toarray()
    both_cast = (cast @ cast.T).toarray()
    agree = both_yes + both_no

    agreement = np.zeros_like(both_cast, dtype=np.float32)
    mask = both_cast >= MIN_SHARED_VOTES
    agreement[mask] = agree[mask] / both_cast[mask]
    np.fill_diagonal(agreement, 0.0)
    return agreement


def fiedler_value(adj):
    A = sparse.csr_matrix(adj)
    deg = np.asarray(A.sum(axis=1)).ravel()
    keep = deg > 0
    if keep.sum() < 3:
        return 0.0, None, keep
    A_s = A[np.ix_(keep, keep)]
    d_inv = sparse.diags(1.0 / np.sqrt(deg[keep]))
    L = sparse.eye(A_s.shape[0]) - d_inv @ A_s @ d_inv
    try:
        w, v = eigsh(L, k=2, which="SM")
        order = np.argsort(w)
        return float(w[order[1]]), v[:, order[1]], keep
    except Exception:
        return 0.0, None, keep


def bli_leave_one_out(adj):
    n = adj.shape[0]
    base, _, _ = fiedler_value(adj)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        f_i, _, _ = fiedler_value(adj[np.ix_(mask, mask)])
        out[i] = base - f_i
    return out, base


def main():
    print(f"Loading EP{TERM} from HowTheyVote.eu...")
    mv, members, term_votes = load_term(TERM, TERM_START)
    member_ids = members["id"].to_numpy()
    vote_ids = term_votes["id"].to_numpy()
    print(f"  MEPs={len(member_ids)}, votes={len(vote_ids)}, member-votes={mv.height}")

    print("Building agreement matrix...")
    agreement = agreement_matrix(mv, member_ids, vote_ids)
    adj = (agreement > THRESHOLD_TAU).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    print(f"  edges={int(adj.sum()) // 2}, mean agreement={agreement[agreement > 0].mean():.3f}")

    print("Computing Fiedler value + BLI (leave-one-out)...")
    bli, base = bli_leave_one_out(adj)

    top = np.argsort(-bli)[:30]
    info = members.select(["id", "first_name", "last_name", "country_code", "group_code"]).to_dicts()
    info_by_id = {r["id"]: r for r in info}
    top_list = [
        {
            "rank": rk + 1,
            "name": f"{info_by_id[int(member_ids[i])]['first_name']} {info_by_id[int(member_ids[i])]['last_name']}",
            "member_id": int(member_ids[i]),
            "country": info_by_id[int(member_ids[i])]["country_code"],
            "group": info_by_id[int(member_ids[i])]["group_code"],
            "bli": float(bli[i]),
        }
        for rk, i in enumerate(top)
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "term": TERM,
        "n_meps": int(len(member_ids)),
        "n_votes": int(len(vote_ids)),
        "n_edges": int(adj.sum()) // 2,
        "base_fiedler": base,
        "tau": THRESHOLD_TAU,
        "top_bridge_meps": top_list,
    }
    (RESULTS_DIR / f"ep_bli_ep{TERM}.json").write_text(json.dumps(out, indent=2))
    print(f"  fiedler={base:.4f}, written to results/ep_bli_ep{TERM}.json")


if __name__ == "__main__":
    main()
