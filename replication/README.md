# GATCongress Replication Package

Replication materials for **"The Geometry of Gridlock: Tracking Congressional
Polarization Through Spectral Analysis of Co-Voting Networks"** (Crainic, 2026).

## Overview

This repository reproduces every figure, table, theorem verification, and
robustness check in the paper from raw public data. The pipeline takes about
60–75 minutes on a 64-core CPU instance (Brev `c7i.16xlarge` or equivalent),
plus one-time package installation of roughly 10 minutes.

## Data availability and provenance

| Source | What | Access | Cached at |
|---|---|---|---|
| Voteview | U.S. House and Senate roll-call votes, 100th–118th Congresses | `voteview.com/data` (HTTP, public) | `data/HSall_*.csv` |
| MIT Election Lab (MEDSL) | General-election returns 1976–2024 | Harvard Dataverse `doi:10.7910/DVN/IG0UN2` | `data/medsl_house_1976_2024_clean.csv` |
| MEDSL | U.S. House primary returns 1976–2022 | Harvard Dataverse `doi:10.7910/DVN/NLTQAD` | `data/raw/medsl_primary/` |
| HowTheyVote.eu | European Parliament roll-call votes, terms 9–10 (2019–present) | `github.com/HowTheyVote/data` (CC-BY) | `data/raw/ep/` |
| Hix–Noury–Roland archive | European Parliament roll calls, terms 1–6 (1979–2009) | `personal.lse.ac.uk/hix/EP Data/` | `data/raw/ep/` |
| U.S. Census Bureau (TIGER/Line) | Congressional district shapefiles | `www2.census.gov/geo/tiger/` | `data/raw/tiger/` |
| Authors' hand collection | House deaths in office 1987–2025 | Compiled from CRS reports + Wikipedia | `data/house_deaths_1987_2025.csv` |

All sources are public; no restricted-access data are used. The Voteview data
are CC-0 and the MEDSL data are CC-BY-4.0. The European Parliament data are
governed by EU public-sector information rules.

## Computational requirements

### Software

* **R 4.4.2** with packages listed in `renv.lock` (auto-installed by
  `replication/scripts/install_r_packages.R`).
* **Python 3.11** with packages listed in `requirements.txt`. Tested with
  `numpy 2.0`, `scipy 1.13`, `polars 0.20`, `geopandas 0.14`.
* **Tectonic** LaTeX compiler (statically linked, downloaded by the Dockerfile).
* **Quarto** (optional) for `manifest.qmd` rendering.

### Hardware

* **CPU**: 16+ cores recommended; the SBM benchmark scales linearly to 64 cores.
* **RAM**: 64 GB minimum, 128 GB recommended for the EP analysis.
* **Disk**: 20 GB free (Census shapefiles + Voteview CSVs + intermediates).
* **GPU**: not needed.

### Controlled randomness

Every script that uses pseudo-random numbers sets a seed:
* Python: `np.random.seed(7)` and `random.seed(7)` at script entry.
* R: `set.seed(42)` at script entry.

Bootstrap, permutation, and Monte-Carlo replication counts are fixed in the
script headers. Re-running the pipeline twice on identical hardware produces
bit-identical results for all CSV outputs; PDFs may differ in trailing
metadata only.

### Runtime

| Stage | Wall-clock on 64-core box |
|---|---|
| One-time dependency install | 10 min |
| Data download | 5 min |
| Spectral + BLI + theorems | 15 min |
| Causal identification | 10 min |
| R DiD comparators + sensitivity | 20 min |
| Cross-national EP analysis | 8 min |
| Figures + paper compile | 3 min |
| **Total** | **~75 min** |

## How to replicate

### Option 1: Docker (recommended)

```bash
git clone https://github.com/human-vc/GATCongress.git
cd GATCongress
docker build -t gatcongress -f replication/docker/Dockerfile .
docker run --rm -v "$PWD:/work" -w /work gatcongress bash run_all.sh
```

The paper PDF is at `paper/paper.pdf` when finished.

### Option 2: Native install

```bash
git clone https://github.com/human-vc/GATCongress.git
cd GATCongress
uv pip install --system -r requirements.txt
Rscript replication/scripts/install_r_packages.R
bash run_all.sh
```

### Option 3: Run a single stage

Use the Quarto manifest to run any one stage:

```bash
quarto render replication/manifest.qmd --to html
```

## Description of programs/code

| File | Stage | Output |
|---|---|---|
| `src/spectral_analysis.py` | Fiedler trajectory | `results/spectral_results.json` |
| `src/bli_brute_force.py` | Per-member BLI | `results/bli_results.json` |
| `src/bli_regression.py` | GEE departure model | `results/bli_regression_results.json` |
| `src/bli_theorem_verification.py` | Theorems 1–3 symbolic + numeric | `results/bli_theorem_verification.json` |
| `src/bli_proof_verification.py` | Lemma 1 entries to machine precision | `results/bli_proof_verification.json` |
| `src/sbm_concentration_verify.py` | Theorem 4 empirical rate | `results/sbm_concentration_verify_*.json` |
| `src/bli_clt_verify.py` | CLT under SBM | `results/bli_clt_verify_*.json` |
| `src/sbm_benchmark_brev.py` | 17,280-graph AUC benchmark | `results/sbm_benchmark_brev_*.csv` |
| `src/cohort_ddd.py` | Olden-Møen triple difference | `results/cohort_ddd_results.json` |
| `src/staggered_did.py` | Callaway-Sant'Anna staggered DiD | `results/staggered_did_results.json` |
| `src/honest_did.py` | Rambachan-Roth bounds | `results/honest_did_results.json` |
| `src/mediation_analysis.py` | Imai-Keele-Tingley mediation | `results/mediation_results.json` |
| `src/death_in_office.py` | Aronow-Samii exposure mapping | `results/death_in_office_results.json` |
| `src/senate_analysis.py` | Senate Fiedler + bicameral synthetic DiD | `results/senate_analysis_results.json` |
| `src/did_comparator.R` | CHd-DH + BJS + fect + triplediff | `results/did_comparator.json` |
| `src/fect_placebo.R` | Parallel-trends placebo + equivalence | `results/fect_placebo.json` |
| `src/specr_curve.R` | Specification curve over 96 specs | `results/specification_curve.csv` |
| `src/rdit_close_margins.R` | RDiT at close cohort margins | `results/rdit.json` |
| `src/iv_redistricting.R` | IV via redistricting cycles | `results/iv.json` |
| `src/sdid_bicameral.R` | sdid jackknife inference | `results/sdid.json` |
| `src/interflex_gerrymandering.R` | Gerrymandering × primary × BLI | `results/interflex_gerrymandering.json` |
| `src/interference_death.R` | Aronow-Samii via `interference` R pkg | `results/interference_death.json` |
| `src/causalweight_mediation.R` | Doubly robust IPW mediation | `results/causalweight_mediation.json` |
| `src/bayes_factor_dfbetas.R` | Bayes factor + DFBETAS | `results/bayes_factor_dfbetas.json` |
| `src/negative_controls.py` | NCO + NCE tests | `results/negative_controls.json` |
| `src/placebo_loocv.py` | Placebo Congresses + LOOCV | `results/placebo_loocv.json` |
| `src/ep_bli_all_terms.py` | European Parliament BLI replication | `results/ep_bli_all_terms.json` |
| `src/generate_figures.py` | All paper figures | `paper/paper_figures/*.pdf` |
| `src/skeleton_network.py` | Top-k BLI skeleton (TikZ output) | `results/figures/skeleton_tikz/*.tex` |
| `src/alluvial_coalitions.R` | Coalition transition alluvial | `results/figures/alluvial_coalitions.pdf` |

## Instructions to replicators

1. Clone the repository: `git clone https://github.com/human-vc/GATCongress.git`.
2. Build the Docker image: `docker build -t gatcongress -f replication/docker/Dockerfile .`
3. Run the pipeline: `docker run --rm -v "$PWD:/work" -w /work gatcongress bash run_all.sh`.
4. Inspect outputs in `results/`, `paper/paper_figures/`, and `paper/paper.pdf`.
5. Logs from each stage are in `logs/` for debugging.

If a particular stage fails, you can rerun just that stage by invoking the
corresponding script directly; all stages are idempotent and read from cached
intermediate files where available.

## List of tables and programs

See `paper/paper.tex` for the canonical figure and table numbering. The
cross-walk in the table above maps each paper output to its generating script.

## Pipeline DAG

A Mermaid-rendered diagram of the full pipeline is at
`replication/docs/pipeline_dag.mmd`. Render to PNG with `mmdc -i pipeline_dag.mmd -o pipeline_dag.png`.

## Garden of forking paths

`replication/docs/forking_paths.csv` enumerates every researcher degree of
freedom we encountered (co-voting threshold, era cutoffs, treatment of
suspension votes, propensity-score model class, etc.) together with the
alternative we considered and the robustness script that re-runs the
analysis under the alternative.

## License

* Code: MIT License (see `LICENSE-code.txt`).
* Data: CC-BY-4.0 (see `LICENSE-data.txt`); upstream data sources retain their
  own licenses as documented in the "Data availability" table above.

## References

See `paper/references.bib`.

## Acknowledgements

We thank the maintainers of Voteview, MEDSL, HowTheyVote.eu, and the Hix-Noury-
Roland EP archive for making roll-call data publicly available, and the
authors of the R packages `did`, `DIDmultiplegtDYN`, `didimputation`, `fect`,
`triplediff`, `HonestDiD`, `interference`, `causalweight`, `interflex`,
`fixest`, `rdrobust`, `synthdid`, `specr`, and `glmtoolbox` for building the
tools the paper relies on.

## Contact

Jacob Crainic, `jacobcrainic@icloud.com`. Issues at
`github.com/human-vc/GATCongress/issues`.
