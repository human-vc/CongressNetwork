# Running the SBM benchmark on Brev

This script benchmarks the Bridge Legislator Index (BLI) against standard centrality measures on stochastic block models with planted degree-matched bridge nodes. The local M3 Pro version covers scale `medium`; for `large` and `xlarge` use Brev.

## What it tests

For each combination of K main blocs (2, 3, 5, 10), main bloc size m, bridge set size b, intra-bloc density p_in, and inter-bloc density p_out, the script generates planted-bridge SBMs where bridges have degree-matched expected connectivity to non-bridges (Mann-Whitney verified). It then computes BLI plus seven comparator centralities and measures AUC for bridge-vs-non-bridge classification. Reports paired bootstrap CIs and Romano-Wolf stepdown p-values across comparators.

## Brev setup

Spin up a CPU-heavy box (32+ cores is ideal; GPU is not needed). The script is CPU-bound on sparse Lanczos eigenvalue solves with joblib parallelism.

```bash
git clone https://github.com/human-vc/GATCongress.git
cd GATCongress
pip install numpy pandas scipy networkx matplotlib scikit-learn joblib
mkdir -p results paper/paper_figures
```

## Scales

- `smoke`: 6 graphs, runs in seconds. Use to verify install.
- `medium`: ~1,440 graphs across 72 cells × 20 reps. Local M3 Pro: ~3 minutes.
- `large`: ~21,600 graphs across 432 cells × 30 reps. Brev 32-core: ~2-4 hours estimated.
- `xlarge`: ~36,000 graphs across 432 cells × 50 reps with m up to 500 (graph sizes up to n=2050). Brev 64-core: 12-24 hours estimated.

## Run

```bash
python3 src/sbm_benchmark_brev.py --scale large --n-jobs -1
```

For the xlarge scale (recommended for paper):

```bash
nohup python3 src/sbm_benchmark_brev.py --scale xlarge --n-jobs -1 > xlarge.log 2>&1 &
```

To add brute-force BLI calibration for graphs n <= 200 (slower but provides ground-truth comparison):

```bash
python3 src/sbm_benchmark_brev.py --scale medium --brute-force --n-jobs -1
```

## Output

For scale `xlarge`:
- `results/sbm_benchmark_brev_xlarge_full.csv` — per-graph results (~36k rows)
- `results/sbm_benchmark_brev_xlarge_results.json` — paired bootstrap CIs, Romano-Wolf p-values, AUC means by K and n
- `paper/paper_figures/sbm_benchmark_brev_xlarge_forest.pdf` — forest plot of BLI advantage with 95% CIs
- `paper/paper_figures/sbm_benchmark_brev_xlarge_n_scaling.pdf` — AUC vs graph size
- `paper/paper_figures/sbm_benchmark_brev_xlarge_k_scaling.pdf` — AUC vs number of blocks

## Pull results back locally

```bash
brev shell <machine>
cp -r results/sbm_benchmark_brev_xlarge* /local/sync/
cp paper/paper_figures/sbm_benchmark_brev_xlarge* /local/sync/
```

## Notes

- The script uses the first-order BLI approximation as the default since brute-force is O(n^4). Empirical Spearman against brute-force is 0.83-0.88; sufficient for ranking.
- Brute-force BLI is enabled only for n <= 200 with the `--brute-force` flag, used as calibration.
- Each replicate gets a deterministic seed derived from `--seed` (default 42) so the run is reproducible.
- Romano-Wolf stepdown uses 2,000 bootstrap iterations.
