# The Geometry of Gridlock

Spectral analysis of U.S. congressional co-voting networks, 100th--118th Congresses (1987--2025). Tracks the collapse of bipartisan cooperation through algebraic connectivity (Fiedler value), introduces a Bridge Legislator Index quantifying individual members' contributions to network cohesion, and demonstrates via GEE regression that bridge position predicts congressional departure beyond ideology, seniority, and party.

## Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `data/download_voteview.py` | Download HSall member and vote CSVs from Voteview |
| 1 | `src/data_pipeline.py` | Build per-congress co-voting networks, adjacency matrices, and defection labels |
| 2 | `src/spectral_analysis.py` | Fiedler values, BLI, SRI, counterfactual removal experiments |
| 3 | `src/model.py` | Train CongressGAT and CongressGCN with multi-task joint optimization |
| 4 | `src/baselines.py` | Train Random Forest and Logistic Regression baselines |
| 5 | `src/evaluate.py` | Unified evaluation with calibrated thresholds and attention analysis |
| 6 | `src/bli_regression.py` | GEE logistic regression: BLI as predictor of congressional departure |
| 7 | `src/generate_figures.py` | Publication-quality figures |

## Quick Start

```bash
pip install -r requirements.txt
python data/download_voteview.py
python run_all.py
```

`run_all.py` executes stages 1--7 in sequence. Results go to `results/`, figures to `paper/paper_figures/`.

## Data

Source: [Voteview](https://voteview.com/) roll-call data (Lewis et al.)

For each congress, the pipeline constructs:
- Binary co-voting adjacency matrix (agreement > 0.5, requiring 20+ shared votes)
- 8-dimensional node features: NOMINATE dim1/dim2, party, participation rate, yea rate, mean agreement, cross-party agreement, within-party agreement
- Defection labels: members voting against their party majority on 10%+ of roll calls

## Spectral Analysis

- **Fiedler value** (algebraic connectivity of the normalized graph Laplacian) tracks network cohesion
- **Structural recovery ratio** formalizes the network's capacity to revert to pre-shock connectivity after political disruptions
- **Bridge Legislator Index (BLI)**: node-removal perturbation measuring each member's contribution to algebraic connectivity
- **Structural Realignment Index (SRI)**: L2 norm of sign-aligned Fiedler vector displacement between consecutive congresses
- **Counterfactual perturbation**: compares Fiedler impact of removing top-BLI vs top-ideology vs random members

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric 2.4+
- See `requirements.txt` for full list
