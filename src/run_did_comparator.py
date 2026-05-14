#!/usr/bin/env python3
"""Python wrapper for did_comparator.R.

Why subprocess and not rpy2:
- rpy2 is faster for tight back-and-forth calls but it makes Python pinned to a
  matching libR ABI, which breaks brittle on macOS / conda. We run R once per
  panel, so the launch overhead is negligible (<1s vs minutes of fit time).
- subprocess gives clean isolation: an R-side segfault never takes Python down,
  the R log is captured verbatim, and we can run multiple panels in parallel
  from Python with concurrent.futures without GIL or rpy2 thread issues.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
R_SCRIPT  = Path(__file__).with_name("did_comparator.R")


def _resolve_rscript() -> str:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise FileNotFoundError(
            "Rscript not on PATH. Install R (`brew install r` on macOS) and "
            "ensure the package deps in did_requirements.txt are installed."
        )
    return rscript


def run_did(panel_csv: Path, outdir: Path, estimator: str = "all",
            timeout_s: int | None = None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [_resolve_rscript(), "--vanilla", str(R_SCRIPT),
           str(panel_csv), str(outdir), estimator]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    (outdir / "rscript.stdout.log").write_text(proc.stdout)
    (outdir / "rscript.stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"did_comparator.R exited {proc.returncode}\n{proc.stderr}")
    results: dict[str, dict] = {}
    for name in ("cdh_dCDH", "bjs_imputation", "fect_LWX", "ddd_triplediff"):
        fp = outdir / f"{name}.json"
        if fp.exists():
            results[name] = json.loads(fp.read_text())
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("panel_csv", type=Path)
    p.add_argument("outdir",    type=Path)
    p.add_argument("--estimator", default="all",
                   choices=["all", "cdh", "bjs", "fect", "ddd"])
    p.add_argument("--timeout-s", type=int, default=None)
    a = p.parse_args(argv)
    results = run_did(a.panel_csv, a.outdir, a.estimator, a.timeout_s)
    print(json.dumps({k: list(v.keys()) for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
