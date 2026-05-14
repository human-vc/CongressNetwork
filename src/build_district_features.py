"""Build results/district_features.csv with Polsby-Popper compactness and primary margins.

Required by src/interflex_gerrymandering.R (task #18).

Output schema:
    district_id, congress, state, district_num,
    compactness_pp, primary_margin, general_margin

Strategy:
- Compactness (Polsby-Popper): download Census TIGER cartographic boundary
  shapefiles for each Congress (one file per Congress 100-118), compute
  4*pi*area / perimeter^2 per district. Census provides these at
  https://www2.census.gov/geo/tiger/TIGER<year>/CD/.

- Primary margin: download MEDSL US House primary returns from Harvard Dataverse
  (PERSISTENT_ID:doi:10.7910/DVN/NLTQAD), compute winner-runner-up margin per
  district-cycle. Fall back to general-election margin if a district has no
  contested primary.

- General margin (auxiliary): computed from the existing MEDSL general elections
  in data/medsl_house_1976_2024_clean.csv, used as fallback and as a covariate.

The script is idempotent; cached downloads land in data/raw/tiger/ and
data/raw/medsl_primary/.
"""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
(RAW / "tiger").mkdir(parents=True, exist_ok=True)
(RAW / "medsl_primary").mkdir(parents=True, exist_ok=True)

# Congresses 100-118 → year of the start session (Jan of odd year)
CONGRESS_YEARS = {c: 1987 + (c - 100) * 2 for c in range(100, 119)}

# Map each Congress to the redistricting era shapefile vintage. Census
# publishes cartographic boundary files keyed by year; we use a single file
# per decade (the one valid for most Congresses in that decade).
ERA_SHAPEFILE = {
    "1990s": ("103", 1993, "tl_2010_us_cd111"),  # 1990 redistricting fallback
    "2000s": ("108", 2003, "tl_2010_us_cd108"),  # 2000 redistricting
    "2010s": ("113", 2013, "tl_2020_us_cd116"),  # 2010 redistricting
    "2020s": ("118", 2023, "tl_2024_us_cd118"),  # 2020 redistricting
}


def _http_get(url: str, dest: Path, timeout: int = 60) -> Path:
    """Cached download with a User-Agent (Census blocks anonymous requests)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"  fetch {url}")
    req = Request(url, headers={"User-Agent": "gat-congress-replication/1.0"})
    with urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def _polsby_popper(geom) -> float:
    """4 pi A / P^2 in projected units."""
    a = float(geom.area)
    p = float(geom.length)
    if p <= 0:
        return float("nan")
    return 4.0 * np.pi * a / (p * p)


def fetch_tiger_congress(congress: int) -> Path:
    """Fetch the Census TIGER/Line congressional district shapefile for a Congress.

    Returns the path to the unzipped .shp file. The first call per file downloads
    ~5-20 MB; subsequent calls are cached.
    """
    # Census names files by (congress, year) inconsistently across vintages.
    # The most reliable URL pattern is the 2020 cartographic boundary release
    # for Congresses 116/117/118 and the 2010-vintage files for 111-115.
    cand_urls = [
        f"https://www2.census.gov/geo/tiger/TIGER2024/CD/tl_2024_us_cd{congress}.zip",
        f"https://www2.census.gov/geo/tiger/TIGER2020/CD/tl_2020_us_cd{congress}.zip",
        f"https://www2.census.gov/geo/tiger/TIGER2010/CD/tl_2010_us_cd{congress}.zip",
        f"https://www2.census.gov/geo/tiger/PREVGENZ/cd/cd{congress}shp/cd{congress}_d00_shp.zip",
    ]
    zip_dest = RAW / "tiger" / f"cd{congress}.zip"
    if not zip_dest.exists():
        for u in cand_urls:
            try:
                _http_get(u, zip_dest)
                break
            except Exception as e:
                print(f"    miss {u}: {e}")
                if zip_dest.exists():
                    zip_dest.unlink()
        else:
            raise FileNotFoundError(f"No TIGER shapefile found for Congress {congress}")
    extract_dir = RAW / "tiger" / f"cd{congress}"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_dest) as z:
        z.extractall(extract_dir)
    shp = next(extract_dir.glob("*.shp"), None)
    if shp is None:
        raise FileNotFoundError(f"No .shp inside {zip_dest}")
    return shp


def compute_compactness(congress: int) -> pd.DataFrame:
    """Return DataFrame with state_fips, district_num, compactness_pp for one Congress."""
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas not installed; emitting NaN compactness")
        return pd.DataFrame(columns=["state_fips", "district_num", "compactness_pp", "congress"])

    try:
        shp = fetch_tiger_congress(congress)
    except Exception as e:
        print(f"  Congress {congress}: shapefile unavailable ({e})")
        return pd.DataFrame(columns=["state_fips", "district_num", "compactness_pp", "congress"])

    gdf = gpd.read_file(shp)
    # Project to an equal-area CRS for honest area/perimeter
    gdf = gdf.to_crs("EPSG:5070")  # NAD83 Conus Albers Equal Area

    # Column names vary by vintage. Try a few.
    state_col = next((c for c in ["STATEFP", "STATEFP10", "STATEFP20", "STATE"] if c in gdf.columns), None)
    dist_col = next((c for c in ["CD118FP", "CD117FP", "CD116FP", "CD115FP", "CD114FP",
                                  "CD113FP", "CD112FP", "CD111FP", "CD110FP",
                                  "CDSESSN", "CD"] if c in gdf.columns), None)
    if state_col is None or dist_col is None:
        print(f"  Congress {congress}: unexpected schema {list(gdf.columns)}")
        return pd.DataFrame(columns=["state_fips", "district_num", "compactness_pp", "congress"])

    gdf["compactness_pp"] = gdf.geometry.apply(_polsby_popper)
    gdf["state_fips"] = gdf[state_col].astype(str).str.zfill(2)
    gdf["district_num"] = pd.to_numeric(gdf[dist_col], errors="coerce").astype("Int64")
    gdf["congress"] = congress
    return gdf[["state_fips", "district_num", "compactness_pp", "congress"]].copy()


def fetch_medsl_primary() -> pd.DataFrame:
    """Fetch MEDSL US House primary election returns (1976-2022).

    Harvard Dataverse persistent ID: doi:10.7910/DVN/NLTQAD. The file we want is
    1976-2022-house.tab (TSV); we cache it under data/raw/medsl_primary/.
    """
    cached = RAW / "medsl_primary" / "1976-2022-house-primaries.tsv"
    if cached.exists():
        return pd.read_csv(cached, sep="\t", low_memory=False)

    # Dataverse direct download URL pattern: /api/access/datafile/<file_id>
    # File id 6925884 is the current primary returns TSV per the dataset page.
    url = "https://dataverse.harvard.edu/api/access/datafile/6925884"
    try:
        _http_get(url, cached, timeout=120)
        return pd.read_csv(cached, sep="\t", low_memory=False)
    except Exception as e:
        print(f"MEDSL primary download failed ({e}); falling back to general-election margins only")
        return pd.DataFrame()


def compute_margins() -> pd.DataFrame:
    """Compute primary and general-election margins per (state, district, year).

    Returns DataFrame with: state_po, district_num, year, primary_margin,
    general_margin.
    """
    # General election margins (already in repo)
    gen = pd.read_csv(DATA / "medsl_house_1976_2024_clean.csv", low_memory=False)
    gen = gen[gen["stage"] == "GEN"].copy()
    gen["district_num"] = pd.to_numeric(gen["district"], errors="coerce").astype("Int64")
    gen["share"] = gen["candidatevotes"].astype(float) / gen["totalvotes"].astype(float).replace(0, np.nan)
    gen_top2 = (gen.sort_values("share", ascending=False)
                .groupby(["state_po", "district_num", "year"])
                .head(2)
                .copy())
    gen_top2["rank"] = gen_top2.groupby(["state_po", "district_num", "year"]).cumcount()
    gen_wide = (gen_top2.pivot_table(index=["state_po", "district_num", "year"],
                                       columns="rank", values="share")
                .rename(columns={0: "share1", 1: "share2"})
                .reset_index())
    gen_wide["general_margin"] = (gen_wide["share1"] - gen_wide["share2"]).fillna(gen_wide["share1"] - 0)

    # Primary margins
    pri = fetch_medsl_primary()
    if pri.empty:
        # Fallback: when the MEDSL primary dataset is unreachable, use the
        # general-election margin as a proxy for district-level electoral
        # competitiveness. interflex still runs; the third interaction term is
        # then competitiveness rather than strict primary intensity. Flagged
        # in the output and in the paper's interflex caption.
        print("  primary download unavailable; using general_margin as the competitiveness proxy")
        gen_wide["primary_margin"] = gen_wide["general_margin"]
        gen_wide["primary_margin_is_proxy"] = True
        return gen_wide[["state_po", "district_num", "year", "primary_margin",
                          "general_margin", "primary_margin_is_proxy"]]

    # MEDSL primary tab columns mirror the general one. Rename to be safe.
    cols = {c.lower(): c for c in pri.columns}
    state_col = cols.get("state_po") or cols.get("state")
    dist_col = cols.get("district")
    year_col = cols.get("year")
    cand_col = cols.get("candidatevotes")
    tot_col = cols.get("totalvotes")
    stage_col = cols.get("stage")
    party_col = cols.get("party")
    if None in (state_col, dist_col, year_col, cand_col, tot_col):
        print("MEDSL primary schema unexpected; skipping primary margin computation")
        gen_wide["primary_margin"] = np.nan
        return gen_wide[["state_po", "district_num", "year", "primary_margin", "general_margin"]]

    pri["district_num"] = pd.to_numeric(pri[dist_col], errors="coerce").astype("Int64")
    pri["share"] = pri[cand_col].astype(float) / pri[tot_col].astype(float).replace(0, np.nan)
    if stage_col is not None and "PRI" in pri[stage_col].astype(str).unique():
        pri = pri[pri[stage_col].astype(str) == "PRI"].copy()
    pri_top2 = (pri.sort_values("share", ascending=False)
                .groupby([state_col, "district_num", year_col])
                .head(2)
                .copy())
    pri_top2["rank"] = pri_top2.groupby([state_col, "district_num", year_col]).cumcount()
    pri_wide = (pri_top2.pivot_table(index=[state_col, "district_num", year_col],
                                       columns="rank", values="share")
                .rename(columns={0: "p_share1", 1: "p_share2"})
                .reset_index())
    pri_wide["primary_margin"] = (pri_wide["p_share1"] - pri_wide["p_share2"].fillna(0))
    pri_wide = pri_wide.rename(columns={state_col: "state_po", year_col: "year"})
    out = gen_wide.merge(pri_wide[["state_po", "district_num", "year", "primary_margin"]],
                          on=["state_po", "district_num", "year"], how="left")
    return out[["state_po", "district_num", "year", "primary_margin", "general_margin"]]


# Map two-letter state codes to FIPS for the compactness merge
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


def main():
    print("[1/3] Computing election margins from MEDSL")
    margins = compute_margins()
    margins["state_fips"] = margins["state_po"].map(STATE_FIPS)
    # Year -> congress (general elections in year y elect the Congress that
    # seats in January of y+1)
    margins["congress"] = ((margins["year"] - 1986) // 2 + 100).astype(int)
    print(f"  margins: {len(margins):,} rows")

    print("[2/3] Computing Polsby-Popper compactness per Congress")
    compactness_frames = []
    for c in range(100, 119):
        df = compute_compactness(c)
        if not df.empty:
            compactness_frames.append(df)
    if compactness_frames:
        compactness = pd.concat(compactness_frames, ignore_index=True)
    else:
        compactness = pd.DataFrame(columns=["state_fips", "district_num", "compactness_pp", "congress"])
    print(f"  compactness: {len(compactness):,} rows across {compactness['congress'].nunique() if not compactness.empty else 0} Congresses")

    print("[3/3] Joining and writing results/district_features.csv")
    out = margins.merge(compactness, on=["state_fips", "district_num", "congress"], how="left")
    out["district_id"] = out["state_po"].astype(str) + "-" + out["district_num"].astype(str)
    if "primary_margin_is_proxy" not in out.columns:
        out["primary_margin_is_proxy"] = False
    cols = ["district_id", "congress", "state_po", "district_num",
            "compactness_pp", "primary_margin", "general_margin",
            "primary_margin_is_proxy"]
    out = out[cols].sort_values(["congress", "state_po", "district_num"]).reset_index(drop=True)
    out_path = RESULTS / "district_features.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out):,} rows, {out['compactness_pp'].notna().sum():,} with compactness, {out['primary_margin'].notna().sum():,} with primary margin)")


if __name__ == "__main__":
    main()
