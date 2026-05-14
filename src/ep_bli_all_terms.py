"""Replicate the Bridge Legislator Index across all 10 directly-elected EP terms.

Data sources by era:
  EP1-EP6 (1979-2009): Hix-Noury-Roland archive (wide CSV: MEP rows x V1..Vn cols)
  EP7-EP8 (2009-2019): VoteWatch / EUI Cadmus dump (manual download, same wide layout)
  EP9-EP10 (2019-now): HowTheyVote.eu CSVs (long-format member_votes)

Outputs:
  results/ep_bli_all_terms.json  — per-term Fiedler, base BLI summary, top-30 bridges
  results/ep_fiedler_trajectory.csv — term, year_start, year_end, fiedler, n_meps, n_votes
"""

import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RESULTS_DIR, THRESHOLD_TAU, MIN_SHARED_VOTES
from ep_bli_compute import fiedler_value, bli_leave_one_out, agreement_matrix

EP_RAW = DATA_DIR / "ep_raw"

TERMS = [
    {"term": 1, "years": (1979, 1984), "src": "hix", "zip": "rcv_ep1.zip"},
    {"term": 2, "years": (1984, 1989), "src": "hix", "zip": "rcv_ep2.zip"},
    {"term": 3, "years": (1989, 1994), "src": "hix", "zip": "rcv_ep3.zip"},
    {"term": 4, "years": (1994, 1999), "src": "hix", "zip": "rcv_ep4.zip"},
    {"term": 5, "years": (1999, 2004), "src": "hix", "zip": "rcv_ep5.zip"},
    {"term": 6, "years": (2004, 2009), "src": "hix", "zip": "rcv_ep6.zip"},
    {"term": 7, "years": (2009, 2014), "src": "cadmus", "dir": "ep7"},
    {"term": 8, "years": (2014, 2019), "src": "cadmus", "dir": "ep8"},
    {"term": 9, "years": (2019, 2024), "src": "howtheyvote", "start": "2019-07-02", "end": "2024-07-15"},
    {"term": 10, "years": (2024, 2029), "src": "howtheyvote", "start": "2024-07-16", "end": None},
]

MIN_VOTES_EP = 50

# Hix vote codes: 1=Yes, 2=No, 3=Abstain, 4=Present, 0=Absent (EP3/4 fold absent + not-MEP into 0)
HIX_YES = 1
HIX_NO = 2


def load_hix_wide(zip_path: Path) -> pd.DataFrame:
    """Hix RCV files: MEPID, MEPNAME, MS, NP, EPG, V1, V2, ..., Vn (comma-delimited)."""
    with zipfile.ZipFile(zip_path) as z:
        txts = [n for n in z.namelist() if n.lower().endswith((".txt", ".csv"))]
        with z.open(txts[0]) as f:
            df = pd.read_csv(f, low_memory=False, encoding="latin-1")
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def agreement_from_wide(df: pd.DataFrame):
    meta_cols = [c for c in df.columns if not re.fullmatch(r"V\d+", c)]
    vote_cols = [c for c in df.columns if re.fullmatch(r"V\d+", c)]
    V = df[vote_cols].to_numpy(dtype=np.int8)

    yes = (V == HIX_YES).astype(np.float32)
    no = (V == HIX_NO).astype(np.float32)
    cast = yes + no

    n_votes = cast.sum(axis=1)
    keep = n_votes >= MIN_VOTES_EP
    df = df.loc[keep].reset_index(drop=True)
    yes, no, cast = yes[keep], no[keep], cast[keep]

    Y = sparse.csr_matrix(yes)
    N = sparse.csr_matrix(no)
    C = sparse.csr_matrix(cast)

    both_yes = (Y @ Y.T).toarray()
    both_no = (N @ N.T).toarray()
    both_cast = (C @ C.T).toarray()
    agree = both_yes + both_no

    agreement = np.zeros_like(both_cast, dtype=np.float32)
    mask = both_cast >= MIN_SHARED_VOTES
    agreement[mask] = agree[mask] / both_cast[mask]
    np.fill_diagonal(agreement, 0.0)
    return agreement, df[meta_cols]


def load_howtheyvote(term: int, start: str, end: str | None):
    from ep_bli_compute import EP_RAW as HTV_DIR
    import polars as pl
    members = pl.read_csv(HTV_DIR / "members.csv.gz")
    votes = pl.read_csv(HTV_DIR / "votes.csv.gz", try_parse_dates=True)
    mv = pl.read_csv(HTV_DIR / "member_votes.csv.gz")
    gm = pl.read_csv(HTV_DIR / "group_memberships.csv.gz", try_parse_dates=True)

    term_votes = votes.filter(pl.col("timestamp") >= pl.lit(start).str.to_datetime())
    if end is not None:
        term_votes = term_votes.filter(pl.col("timestamp") <= pl.lit(end).str.to_datetime())
    mv = mv.join(term_votes.select("id").rename({"id": "vote_id"}), on="vote_id", how="inner")
    mv = mv.filter(pl.col("position").is_in(["FOR", "AGAINST"]))
    counts = mv.group_by("member_id").len().filter(pl.col("len") >= MIN_VOTES_EP)
    mv = mv.join(counts.select("member_id"), on="member_id", how="inner")
    gm_t = gm.filter(pl.col("term") == term).group_by("member_id").agg(pl.col("group_code").last())
    members = members.join(gm_t, left_on="id", right_on="member_id", how="inner")
    members = members.filter(pl.col("id").is_in(mv["member_id"].unique()))

    member_ids = members["id"].to_numpy()
    vote_ids = term_votes["id"].to_numpy()
    agreement = agreement_matrix(mv, member_ids, vote_ids)
    meta = members.select(["id", "first_name", "last_name", "country_code", "group_code"]).to_pandas()
    meta = meta.rename(columns={"id": "MEPID", "country_code": "MS", "group_code": "EPG"})
    meta["MEPNAME"] = meta["first_name"] + " " + meta["last_name"]
    return agreement, meta[["MEPID", "MEPNAME", "MS", "EPG"]]


def process_term(spec):
    term = spec["term"]
    print(f"[EP{term}] loading {spec['src']}...")
    if spec["src"] == "hix":
        df = load_hix_wide(EP_RAW / "hix" / spec["zip"])
        agreement, meta = agreement_from_wide(df)
    elif spec["src"] == "cadmus":
        cadmus_csv = EP_RAW / spec["dir"] / "rcv.csv"
        if not cadmus_csv.exists():
            print(f"  EP{term}: {cadmus_csv} missing — skip (manual download required)")
            return None
        df = pd.read_csv(cadmus_csv, low_memory=False)
        df.columns = [c.strip().upper() for c in df.columns]
        agreement, meta = agreement_from_wide(df)
    else:
        agreement, meta = load_howtheyvote(term, spec["start"], spec.get("end"))

    adj = (agreement > THRESHOLD_TAU).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    bli, base = bli_leave_one_out(adj)

    top = np.argsort(-bli)[:30]
    top_list = []
    for rk, i in enumerate(top):
        row = meta.iloc[int(i)]
        top_list.append({
            "rank": rk + 1,
            "name": str(row.get("MEPNAME", "")),
            "country": str(row.get("MS", "")),
            "group": str(row.get("EPG", "")),
            "bli": float(bli[i]),
        })

    return {
        "term": term,
        "year_start": spec["years"][0],
        "year_end": spec["years"][1],
        "n_meps": int(adj.shape[0]),
        "n_edges": int(adj.sum()) // 2,
        "base_fiedler": base,
        "top_bridge_meps": top_list,
    }


def main(n_jobs: int = -1):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_term)(s) for s in TERMS
    )
    results = [r for r in results if r is not None]

    (RESULTS_DIR / "ep_bli_all_terms.json").write_text(json.dumps(results, indent=2))

    traj = pd.DataFrame([
        {"term": r["term"], "year_start": r["year_start"], "year_end": r["year_end"],
         "fiedler": r["base_fiedler"], "n_meps": r["n_meps"], "n_edges": r["n_edges"]}
        for r in results
    ])
    traj.to_csv(RESULTS_DIR / "ep_fiedler_trajectory.csv", index=False)
    print(traj.to_string(index=False))


if __name__ == "__main__":
    main()
