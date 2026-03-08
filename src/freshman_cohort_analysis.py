"""Freshman cohort analysis: Contract with America (104th) vs Tea Party (112th)."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR, RESULTS_DIR

import numpy as np
from scipy import stats


def load_congress(c):
    data = np.load(PROCESSED_DIR / f"congress_{c}.npz", allow_pickle=True)
    return {
        "member_ids": data["member_ids"],
        "party_codes": data["party_codes"],
        "member_names": data["member_names"],
        "nominate_dim1": data["nominate_dim1"],
        "agreement": data["agreement"],
        "adjacency": data["adjacency"],
    }


def get_freshmen(curr_congress, prev_congress):
    curr = load_congress(curr_congress)
    prev = load_congress(prev_congress)
    prev_ids = set(prev["member_ids"].tolist())
    fresh_mask = np.array([mid not in prev_ids for mid in curr["member_ids"]])
    return curr, fresh_mask


def cross_party_agreement(congress_data, member_idx):
    party = congress_data["party_codes"][member_idx]
    other_party_mask = congress_data["party_codes"] != party
    agreement_row = congress_data["agreement"][member_idx]
    cross_vals = agreement_row[other_party_mask]
    # Filter out zeros (no shared votes)
    cross_vals = cross_vals[cross_vals > 0]
    return float(np.mean(cross_vals)) if len(cross_vals) > 0 else 0.0


def analyze_cohort(congress_num, prev_congress_num, label):
    curr, fresh_mask = get_freshmen(congress_num, prev_congress_num)
    n_fresh = fresh_mask.sum()
    n_total = len(curr["member_ids"])

    fresh_nom = curr["nominate_dim1"][fresh_mask]
    fresh_party = curr["party_codes"][fresh_mask]
    fresh_names = curr["member_names"][fresh_mask]

    # Cross-party agreement rate for each freshman
    fresh_indices = np.where(fresh_mask)[0]
    fresh_xparty = np.array([cross_party_agreement(curr, i) for i in fresh_indices])

    # Same for incumbents
    inc_mask = ~fresh_mask
    inc_indices = np.where(inc_mask)[0]
    inc_xparty = np.array([cross_party_agreement(curr, i) for i in inc_indices])
    inc_nom = curr["nominate_dim1"][inc_mask]

    # Party breakdown
    fresh_dem = (fresh_party == 100).sum()
    fresh_rep = (fresh_party == 200).sum()

    print(f"\n{'='*60}")
    print(f"{label}: Congress {congress_num}")
    print(f"{'='*60}")
    print(f"Total members: {n_total}")
    print(f"Freshmen: {n_fresh} ({n_fresh/n_total*100:.1f}%)")
    print(f"  Democrats: {fresh_dem}, Republicans: {fresh_rep}")
    print(f"\nFreshmen DW-NOMINATE:")
    print(f"  Mean: {fresh_nom.mean():.3f}, Median: {np.median(fresh_nom):.3f}")
    print(f"  Std:  {fresh_nom.std():.3f}")
    print(f"  Range: [{fresh_nom.min():.3f}, {fresh_nom.max():.3f}]")
    print(f"\nFreshmen cross-party agreement rate:")
    print(f"  Mean: {fresh_xparty.mean():.3f}, Median: {np.median(fresh_xparty):.3f}")
    print(f"  Std:  {fresh_xparty.std():.3f}")
    print(f"  >0.40: {(fresh_xparty > 0.40).sum()} ({(fresh_xparty > 0.40).mean()*100:.1f}%)")
    print(f"  >0.45: {(fresh_xparty > 0.45).sum()} ({(fresh_xparty > 0.45).mean()*100:.1f}%)")
    print(f"\nIncumbents cross-party agreement rate:")
    print(f"  Mean: {inc_xparty.mean():.3f}, Median: {np.median(inc_xparty):.3f}")

    # Top cooperators among freshmen
    top_idx = np.argsort(-fresh_xparty)[:10]
    print(f"\nTop 10 freshmen by cross-party agreement:")
    for i in top_idx:
        p = "D" if fresh_party[i] == 100 else "R"
        print(f"  {fresh_names[i]:30s} ({p})  xparty={fresh_xparty[i]:.3f}  nom={fresh_nom[i]:.3f}")

    return {
        "congress": congress_num,
        "n_total": int(n_total),
        "n_freshmen": int(n_fresh),
        "n_fresh_dem": int(fresh_dem),
        "n_fresh_rep": int(fresh_rep),
        "fresh_nominate": fresh_nom.tolist(),
        "fresh_xparty": fresh_xparty.tolist(),
        "inc_xparty": inc_xparty.tolist(),
        "fresh_nom_mean": float(fresh_nom.mean()),
        "fresh_nom_std": float(fresh_nom.std()),
        "fresh_xparty_mean": float(fresh_xparty.mean()),
        "fresh_xparty_std": float(fresh_xparty.std()),
        "inc_xparty_mean": float(inc_xparty.mean()),
        "fresh_xparty_above_40": int((fresh_xparty > 0.40).sum()),
        "fresh_xparty_above_45": int((fresh_xparty > 0.45).sum()),
    }


def main():
    cwa = analyze_cohort(104, 103, "Contract with America")
    tea = analyze_cohort(112, 111, "Tea Party Wave")

    # KS test on cross-party agreement distributions
    ks_stat, ks_p = stats.ks_2samp(cwa["fresh_xparty"], tea["fresh_xparty"])
    print(f"\n{'='*60}")
    print("Two-sample KS test (cross-party agreement: CwA vs Tea Party freshmen)")
    print(f"  KS statistic: {ks_stat:.3f}")
    print(f"  p-value: {ks_p:.2e}")

    # KS test on NOMINATE
    ks_nom, ks_nom_p = stats.ks_2samp(cwa["fresh_nominate"], tea["fresh_nominate"])
    print(f"\nTwo-sample KS test (DW-NOMINATE: CwA vs Tea Party freshmen)")
    print(f"  KS statistic: {ks_nom:.3f}")
    print(f"  p-value: {ks_nom_p:.2e}")

    # Mann-Whitney U on cross-party agreement
    mw_stat, mw_p = stats.mannwhitneyu(
        cwa["fresh_xparty"], tea["fresh_xparty"], alternative="greater"
    )
    print(f"\nMann-Whitney U (CwA > Tea Party cross-party agreement)")
    print(f"  U statistic: {mw_stat:.1f}")
    print(f"  p-value: {mw_p:.2e}")

    output = {
        "contract_with_america_104": cwa,
        "tea_party_112": tea,
        "ks_test_xparty": {"statistic": float(ks_stat), "pvalue": float(ks_p)},
        "ks_test_nominate": {"statistic": float(ks_nom), "pvalue": float(ks_nom_p)},
        "mann_whitney_xparty": {"statistic": float(mw_stat), "pvalue": float(mw_p)},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "freshman_cohort_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'freshman_cohort_results.json'}")


if __name__ == "__main__":
    main()
