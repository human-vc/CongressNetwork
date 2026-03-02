# GATCongress

Graph Attention Networks for modeling U.S. Congressional polarization dynamics. Uses Voteview roll-call data to construct co-voting networks, then applies GAT and GCN models to predict legislative defection, identify bridging legislators, and track polarization trends across the 100th-118th Congresses.

## Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `data/download_voteview.py` | Download HSall member and vote CSVs from Voteview |
| 1 | `src/data_pipeline.py` | Build per-congress co-voting networks, adjacency matrices, and defection labels |
| 2 | `src/spectral_analysis.py` | Fiedler values, Bridging Legislator Index (BLI), Structural Realignment Index (SRI), counterfactual removal experiments |
| 3 | `src/model.py` | Train CongressGAT and CongressGCN with multi-task joint optimization |
| 4 | `src/baselines.py` | Train Random Forest and Logistic Regression baselines |
| 5 | `src/evaluate.py` | Unified evaluation with calibrated thresholds, attention analysis, McCarthy 118th case study |
| 6 | `src/bli_regression.py` | GEE logistic regression: does BLI predict congressional departure? |
| 7 | `src/generate_figures.py` | Publication-quality figures |

## Quick Start

```bash
pip install -r requirements.txt
python data/download_voteview.py
python run_all.py
```

`run_all.py` executes stages 1-7 in sequence. Results go to `results/`, figures to `paper/paper_figures/`.

## Models

**CongressGAT**: Two GATConv layers (4 heads), temporal multi-head attention over congress sequences, three prediction heads (defection, coalition, polarization). Joint multi-task training with a single backward pass per epoch across all training congresses.

**CongressGCN**: Same architecture with GCNConv layers instead of GATConv. Serves as the non-attention baseline to isolate the contribution of the attention mechanism.

**Baselines**: Random Forest (200 trees, depth 8) and Logistic Regression, both with balanced class weights.

All models are evaluated using a single threshold calibrated on the 114th Congress validation set.

## Data

Source: [Voteview](https://voteview.com/) roll-call data (Lewis et al.)

For each congress, the pipeline constructs:
- Binary co-voting adjacency matrix (agreement > 0.5, requiring 20+ shared votes)
- 8-dimensional node features: NOMINATE dim1/dim2, party, participation rate, yea rate, mean agreement, cross-party agreement, within-party agreement
- Defection labels: members voting against their party majority on 10%+ of roll calls

## Spectral Analysis

- **Fiedler value** (algebraic connectivity) tracks network cohesion over time
- **BLI**: measures each legislator's contribution to network connectivity via node-removal perturbation
- **SRI**: quantifies structural realignment between consecutive congresses
- **Counterfactuals**: compares Fiedler impact of removing top-BLI vs top-ideology vs random members

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric 2.4+
- See `requirements.txt` for full list
