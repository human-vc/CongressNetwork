"""Driver: run the three robustness analyses (NCO/NCE, placebo+LOOCV, BF+DFBETAS).

Use:  python src/run_robustness.py [--skip-r]
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bli_regression import build_panel
from config import RESULTS_DIR


def export_panel():
    panel = build_panel()
    out = RESULTS_DIR / "bli_panel.csv"
    panel.to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-r", action="store_true", help="Skip R BF+DFBETAS")
    args = ap.parse_args()

    print("[1/3] Exporting panel for R + sanity check ...")
    panel_csv = export_panel()
    print(f"      Wrote {panel_csv}")

    print("[2/3] Negative controls (NCO-1..3, NCE-1, NCE-2 perm) ...")
    from negative_controls import main as run_nc
    run_nc()

    print("[3/3] Placebo Congress + LOOCV ...")
    from placebo_loocv import main as run_pl
    run_pl()

    if not args.skip_r:
        r_script = Path(__file__).resolve().parent / "bayes_factor_dfbetas.R"
        out_json = RESULTS_DIR / "bf_dfbetas_results.json"
        print("[3b] Bayes factor + DFBETAS (R subprocess) ...")
        cmd = ["Rscript", "--vanilla", str(r_script),
               str(panel_csv), str(out_json)]
        subprocess.run(cmd, check=True)
    print("Done.")


if __name__ == "__main__":
    main()
