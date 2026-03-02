import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import SEED
import numpy as np
import torch


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 60)
    print("GATCongress Pipeline")
    print("=" * 60)

    print("\n[1/6] Data Pipeline")
    from data_pipeline import main as run_data
    run_data()

    print("\n[2/6] Spectral Analysis")
    from spectral_analysis import main as run_spectral
    run_spectral()

    print("\n[3/6] Model Training")
    from model import main as run_model
    run_model()

    print("\n[4/6] Baselines")
    from baselines import main as run_baselines
    run_baselines()

    print("\n[5/6] Evaluation")
    from evaluate import main as run_evaluate
    run_evaluate()

    print("\n[6/6] BLI Regression")
    from bli_regression import main as run_bli
    run_bli()

    print("\n[7/7] Figures")
    from generate_figures import main as run_figures
    run_figures()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
