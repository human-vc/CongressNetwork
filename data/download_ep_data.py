"""Fetch EP roll-call data: HowTheyVote (EP9-EP10) + Hix-Noury-Roland (EP1-EP6).

EP7-EP8 (2009-2019) require manual download from Cadmus EUI handle 1814/74918
(403 to scripted requests). Save the unzipped CSV/XLS files under DATA_DIR/ep_raw/ep7
and .../ep8 with the same column conventions as Hix files.
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_DIR

EP_RAW_DIR = DATA_DIR / "ep_raw"

HOWTHEYVOTE_BASE = "https://github.com/HowTheyVote/data/releases/latest/download"
HOWTHEYVOTE_FILES = [
    "members.csv.gz",
    "votes.csv.gz",
    "member_votes.csv.gz",
    "groups.csv.gz",
    "group_memberships.csv.gz",
    "countries.csv.gz",
    "last_updated.txt",
]

HIX_BASE = "https://personal.lse.ac.uk/hix/EP%20Data"
HIX_FILES = {
    "mep_info_26Jul11.zip": "mep_info.zip",
    "vote_info_24Jun2010.zip": "vote_info.zip",
    "rcv_ep1.zip": "rcv_ep1.zip",
    "rcv_ep2.zip": "rcv_ep2.zip",
    "rcv_ep3.zip": "rcv_ep3.zip",
    "rcv_ep4.zip": "rcv_ep4.zip",
    "ep5_rcv_11Jul06.zip": "rcv_ep5.zip",
    "ep6_data-9nov2009.ZIP": "rcv_ep6.zip",
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def fetch_howtheyvote() -> None:
    print("HowTheyVote.eu (EP9-EP10):")
    out = EP_RAW_DIR / "howtheyvote"
    for name in HOWTHEYVOTE_FILES:
        download(f"{HOWTHEYVOTE_BASE}/{name}", out / name)


def fetch_hix() -> None:
    print("Hix-Noury-Roland (EP1-EP6):")
    out = EP_RAW_DIR / "hix"
    for remote, local in HIX_FILES.items():
        download(f"{HIX_BASE}/{remote}", out / local)


if __name__ == "__main__":
    EP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch_howtheyvote()
    fetch_hix()
    print("\nEP7-EP8: download manually from https://cadmus.eui.eu/handle/1814/74918")
    print(f"Unzip into {EP_RAW_DIR}/ep7 and {EP_RAW_DIR}/ep8")
