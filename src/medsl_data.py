import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR


MEDSL_FILE_ID = 13592823
MEDSL_API_BASE = "https://dataverse.harvard.edu/api"
MEDSL_PATH = DATA_DIR / "medsl_house_1976_2024.csv"
MEDSL_CLEAN_PATH = DATA_DIR / "medsl_house_1976_2024_clean.csv"


def request_signed_url():
    payload = {
        "guestbookResponse": {
            "name": "Jacob Crainic",
            "email": "jacobcrainic2008@gmail.com",
            "institution": "Independent Researcher",
            "position": "Researcher",
        }
    }
    r = requests.post(
        f"{MEDSL_API_BASE}/access/datafile/{MEDSL_FILE_ID}?signed=true",
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"]["signedUrl"]


def download_medsl():
    if MEDSL_CLEAN_PATH.exists():
        return MEDSL_CLEAN_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MEDSL_PATH.exists():
        url = request_signed_url()
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(MEDSL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
    with open(MEDSL_PATH, "r", encoding="utf-8") as fin, open(MEDSL_CLEAN_PATH, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i == 0:
                fout.write(line)
                continue
            stripped = line.rstrip("\n").rstrip("\r")
            if stripped.startswith('"') and stripped.endswith('"'):
                stripped = stripped[1:-1]
            stripped = stripped.replace('\\"', '"')
            fout.write(stripped + "\n")
    return MEDSL_CLEAN_PATH


def congress_to_election_year(c):
    return 1786 + 2 * c


def load_house_returns():
    path = download_medsl()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    df = df[df["stage"].str.upper() == "GEN"].copy()
    df = df[df["special"].astype(str).str.lower() != "true"].copy()
    df = df[df["runoff"].astype(str).str.lower() != "true"].copy()
    df["state_po"] = df["state_po"].str.upper()
    df["district"] = pd.to_numeric(df["district"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["district", "candidatevotes", "totalvotes"])
    df["candidatevotes"] = pd.to_numeric(df["candidatevotes"], errors="coerce")
    df = df.dropna(subset=["candidatevotes"])
    df["party_clean"] = df["party"].fillna("").str.upper().str.strip()
    df["is_dem"] = df["party_clean"].str.contains("DEMOCRAT", na=False)
    df["is_rep"] = df["party_clean"].str.contains("REPUBLICAN", na=False)
    return df


def aggregate_district_year(df):
    df = df.copy()
    df["dem_votes"] = np.where(df["is_dem"], df["candidatevotes"], 0)
    df["rep_votes"] = np.where(df["is_rep"], df["candidatevotes"], 0)
    grouped = (
        df.groupby(["year", "state_po", "district"], as_index=False)
        .agg(
            dem_votes=("dem_votes", "sum"),
            rep_votes=("rep_votes", "sum"),
            total_votes=("totalvotes", "max"),
        )
    )
    two_party = grouped["dem_votes"] + grouped["rep_votes"]
    grouped["dem_share"] = np.where(two_party > 0, grouped["dem_votes"] / two_party, np.nan)
    grouped["margin"] = (grouped["dem_share"] - 0.5).abs() * 2
    grouped = grouped[two_party > 0].copy()
    return grouped


def district_competitiveness_panel():
    raw = load_house_returns()
    agg = aggregate_district_year(raw)
    return agg


def attach_competitiveness(member_df, district_panel, n_lookback=2):
    out = member_df.copy()
    election_year = out["congress"].apply(congress_to_election_year)
    out["election_year"] = election_year

    panel = district_panel.copy()
    panel["state_po"] = panel["state_po"].str.upper()
    panel = panel.set_index(["state_po", "district", "year"]).sort_index()

    competitiveness = np.full(len(out), np.nan)
    margin_continuous = np.full(len(out), np.nan)
    dem_lean = np.full(len(out), np.nan)

    for i, row in out.iterrows():
        state = str(row["state"]).upper()
        district = int(row.get("district_code", 0)) if pd.notna(row.get("district_code", 0)) else 0
        target_year = int(row["election_year"])
        margins = []
        leans = []
        for back in range(1, n_lookback + 1):
            yr = target_year - 2 * back
            try:
                rec = panel.loc[(state, district, yr)]
                if isinstance(rec, pd.DataFrame):
                    rec = rec.iloc[0]
                if pd.notna(rec["margin"]):
                    margins.append(float(rec["margin"]))
                    leans.append(float(rec["dem_share"]) - 0.5)
            except KeyError:
                continue
        if margins:
            competitiveness[i] = float(np.mean(margins))
            dem_lean[i] = float(np.mean(leans))
        if not np.isnan(competitiveness[i]):
            margin_continuous[i] = competitiveness[i]

    out["prior_margin"] = competitiveness
    out["prior_dem_lean"] = dem_lean
    return out


if __name__ == "__main__":
    path = download_medsl()
    print(f"MEDSL file at {path}")
    panel = district_competitiveness_panel()
    print(f"Built district panel: {len(panel)} rows, {panel['year'].min()}-{panel['year'].max()}")
    print(panel.head())
    print()
    print(f"Margin distribution percentiles: {panel['margin'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()}")
