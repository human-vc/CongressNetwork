"""Build long-format coalition transitions for the alluvial plot.

Coalition rule: per Congress, within each party, the top 25% of |BLI| are
"Bridge" and the rest are "Other". Legislators absent from a given Congress
get an explicit "Out-of-house" stratum so the alluvial honestly shows
turnover. Output CSV columns: icpsr, congress, coalition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR

PANELS = [100, 105, 110, 115, 118]
BRIDGE_Q = 0.75
OUT_PATH = RESULTS_DIR / "coalition_transitions.csv"


def coalition_label(party: int, is_bridge: bool) -> str:
    p = "Dem" if party == 100 else "Rep" if party == 200 else "Ind"
    role = "Bridge" if is_bridge else "Other"
    return f"{p}-{role}"


def main() -> None:
    bli_all = json.load(open(RESULTS_DIR / "bli_results.json"))
    records = []
    members_per_congress = {}

    for c in PANELS:
        data = np.load(PROCESSED_DIR / f"congress_{c}.npz", allow_pickle=True)
        npz_ids = np.asarray(data["member_ids"], dtype=int)
        party = np.asarray(data["party_codes"], dtype=int)
        bli_ids = np.asarray(bli_all[str(c)]["member_ids"], dtype=int)
        bli_vals = np.asarray(bli_all[str(c)]["bli_values"], dtype=float)
        id_to_bli = dict(zip(bli_ids.tolist(), bli_vals.tolist()))
        bli = np.array([id_to_bli.get(int(m), 0.0) for m in npz_ids])

        members_per_congress[c] = set(npz_ids.tolist())

        for p in (100, 200):
            mask = party == p
            if mask.sum() == 0:
                continue
            absbli = np.abs(bli[mask])
            cutoff = np.quantile(absbli, BRIDGE_Q)
            ids_p = npz_ids[mask]
            for icpsr, val in zip(ids_p, absbli):
                is_bridge = bool(val >= cutoff)
                records.append((int(icpsr), int(c), coalition_label(p, is_bridge)))

    # Add Out-of-house rows for legislators who appear in at least one panel
    # but are absent from a given one.
    union = set().union(*members_per_congress.values())
    for icpsr in union:
        for c in PANELS:
            if icpsr in members_per_congress[c]:
                continue
            records.append((int(icpsr), int(c), "Out-of-house"))

    df = pd.DataFrame(records, columns=["icpsr", "congress", "coalition"])
    df = df.sort_values(["icpsr", "congress"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"  wrote {OUT_PATH.name}  ({len(df):,} rows, "
          f"{df['icpsr'].nunique():,} unique legislators)")
    print(df.groupby(["congress", "coalition"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
