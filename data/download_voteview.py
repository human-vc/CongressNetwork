import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_DIR, VOTEVIEW_URLS

import requests


def download_file(url, dest):
    print(f"Downloading {dest.name}...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in VOTEVIEW_URLS.items():
        dest = DATA_DIR / filename
        if dest.exists():
            print(f"{filename} already exists, skipping")
            continue
        download_file(url, dest)
    print("All Voteview files downloaded.")


if __name__ == "__main__":
    main()
