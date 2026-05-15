import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, RESULTS_DIR

CONGRESS_TO_ELECTION_YEAR = {c: 1789 + (c - 1) * 2 + 1 for c in range(100, 120)}

def _normalize(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    s = re.sub(r"[^A-Za-z ,.'-]", " ", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

def _surname(full_name: str) -> str:
    if not isinstance(full_name, str) or not full_name.strip():
        return ""
    if "," in full_name:
        return _normalize(full_name.split(",", 1)[0])
    s = _normalize(full_name)
    if not s:
        return ""
    tokens = [t for t in s.split() if t and t not in _SUFFIXES]
    return tokens[-1] if tokens else ""

def load_members():
    df = pd.read_csv(DATA_DIR / "HSall_members.csv", low_memory=False)
    df = df[df["chamber"].isin(["House", "Senate"])]
    df["icpsr"] = pd.to_numeric(df["icpsr"], errors="coerce")
    df["congress"] = pd.to_numeric(df["congress"], errors="coerce")
    df["born"] = pd.to_numeric(df["born"], errors="coerce")
    df["died"] = pd.to_numeric(df["died"], errors="coerce")
    df["district_code"] = pd.to_numeric(df["district_code"], errors="coerce")
    df["surname"] = df["bioname"].apply(_surname)
    return df

def load_medsl():
    df = pd.read_csv(DATA_DIR / "medsl_house_1976_2024_clean.csv", low_memory=False)
    df = df[df["stage"].str.upper() == "GEN"]
    df = df[df["special"] != True]
    df["candidate_norm"] = df["candidate"].apply(_surname)
    df["candidatevotes"] = pd.to_numeric(df["candidatevotes"], errors="coerce").fillna(0)
    df["totalvotes"] = pd.to_numeric(df["totalvotes"], errors="coerce").fillna(0)
    return df

def load_house_deaths():
    p = DATA_DIR / "house_deaths_1987_2025.csv"
    if not p.exists():
        return pd.DataFrame(columns=["congress", "state", "district", "surname"])
    df = pd.read_csv(p)
    df["surname"] = df["name"].apply(_surname)
    df["congress"] = pd.to_numeric(df["congress"], errors="coerce")
    df["district"] = pd.to_numeric(df["district"], errors="coerce")
    return df[["congress", "state", "district", "surname"]]

def medsl_general_winner(medsl, election_year, state_po, district, surname):
    race = medsl[
        (medsl["year"] == election_year)
        & (medsl["state_po"] == state_po)
        & (medsl["district"] == district)
    ]
    if len(race) == 0:
        return False, False
    winner_row = race.sort_values("candidatevotes", ascending=False).iloc[0]
    return _surname(winner_row["candidate"]) == surname, True

def medsl_appeared_as_candidate(medsl, election_year, state_po, district, surname):
    race = medsl[
        (medsl["year"] == election_year)
        & (medsl["state_po"] == state_po)
        & (medsl["district"] == district)
    ]
    if len(race) == 0:
        return False
    cands = race["candidate_norm"].tolist()
    return any(s and s == surname for s in cands)

def classify(members, medsl, deaths, congresses):
    house_rosters = {
        int(c): set(members[(members.chamber == "House") & (members.congress == c)]["icpsr"].dropna().astype(int))
        for c in congresses
    }
    senate_rosters = {
        int(c): set(members[(members.chamber == "Senate") & (members.congress == c)]["icpsr"].dropna().astype(int))
        for c in congresses
    }
    house_meta = members[members.chamber == "House"].copy()

    deaths_lookup = set()
    for r in deaths.itertuples():
        deaths_lookup.add((int(r.congress) if pd.notna(r.congress) else None, r.state, int(r.district) if pd.notna(r.district) else None, r.surname))

    rows = []
    for c in congresses:
        if c + 1 not in house_rosters and c + 1 not in senate_rosters:
            continue
        cohort = house_meta[house_meta.congress == c]
        next_house = house_rosters.get(c + 1, set())
        next_senate = senate_rosters.get(c + 1, set())
        election_year = CONGRESS_TO_ELECTION_YEAR.get(c)
        for r in cohort.itertuples():
            icpsr = int(r.icpsr)
            in_next = icpsr in next_house
            died_office = False
            term_years = [election_year - 1, election_year] if election_year else []
            if pd.notna(r.died) and term_years and term_years[0] <= int(r.died) <= term_years[1] + 1:
                died_office = True
            if (c, r.state_abbrev, int(r.district_code) if pd.notna(r.district_code) else None, r.surname) in deaths_lookup:
                died_office = True

            ran_higher = (not in_next) and (icpsr in next_senate)

            lost_general = False
            lost_primary = False
            voluntary = False
            if not in_next and not died_office and not ran_higher and election_year is not None and pd.notna(r.district_code):
                appeared = medsl_appeared_as_candidate(
                    medsl, election_year, r.state_abbrev, int(r.district_code), r.surname
                )
                won_gen, race_found = medsl_general_winner(
                    medsl, election_year, r.state_abbrev, int(r.district_code), r.surname
                )
                if appeared and not won_gen:
                    lost_general = True
                elif (not appeared) and race_found:
                    lost_primary = True
                else:
                    voluntary = True

            if in_next:
                dt = "still_serving"
            elif died_office:
                dt = "died_in_office"
            elif ran_higher:
                dt = "ran_for_higher_office"
            elif lost_general:
                dt = "lost_general"
            elif lost_primary or voluntary:
                dt = "did_not_seek_general"
            else:
                dt = "ambiguous"

            rows.append({
                "icpsr": icpsr,
                "congress": int(c),
                "in_next": int(in_next),
                "died_in_office": int(died_office),
                "ran_for_higher_office": int(ran_higher),
                "lost_general": int(lost_general),
                "did_not_seek_general": int(lost_primary or voluntary),
                "departure_type": dt,
            })
    return pd.DataFrame(rows)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data...")
    members = load_members()
    medsl = load_medsl()
    deaths = load_house_deaths()
    congresses = sorted(set(members[members.chamber == "House"]["congress"].dropna().astype(int)) & set(range(100, 117)))
    print(f"  members: {len(members):,}, medsl rows: {len(medsl):,}, deaths: {len(deaths)}, congresses: {congresses[0]}-{congresses[-1]}")

    out = classify(members, medsl, deaths, congresses)
    p = RESULTS_DIR / "departure_types.csv"
    out.to_csv(p, index=False)
    print(f"Wrote {p}: {len(out)} rows")
    print(out["departure_type"].value_counts())
    summary = (
        out.groupby("departure_type")
        .size()
        .reindex(["still_serving", "did_not_seek_general", "lost_general",
                  "died_in_office", "ran_for_higher_office", "ambiguous"], fill_value=0)
        .to_dict()
    )
    import json
    (RESULTS_DIR / "departure_types_summary.json").write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
