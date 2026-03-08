# CongressNetwork

Spectral analysis of U.S. congressional co-voting networks (100th–118th Congresses).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python data/download_voteview.py
python run_all.py
```

Downloads Voteview data and runs the full pipeline. Results go to `results/`, figures to `results/figures/`.

## Individual stages

```bash
python src/data_pipeline.py              # Build co-voting networks
python src/spectral_analysis.py          # Fiedler values, BLI, SRI, counterfactuals
python src/bli_regression.py             # GEE departure regression
python src/freshman_cohort_analysis.py   # Freshman cohort comparison
python src/vote_filtering.py             # Substantive vs. procedural vote filtering
python src/null_model_analysis.py        # Null model significance tests
python src/recovery_threshold_sensitivity.py  # Threshold sensitivity
python src/counterfactual_sensitivity.py # Counterfactual edge-addition
python src/weighted_spectral.py          # Weighted spectral analysis
python src/generate_figures.py           # Publication figures
```
