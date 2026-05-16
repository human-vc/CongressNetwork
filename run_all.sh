#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR" results results/figures \
         data/raw/tiger data/raw/medsl_primary data/raw/ep

SKIP_INSTALL=0
CORE_ONLY=0
for a in "$@"; do
  case "$a" in
    --skip-install) SKIP_INSTALL=1 ;;
    --core-only)    CORE_ONLY=1 ;;
  esac
done

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

run() {
  local name="$1" cmd="$2"
  echo
  echo "============================================================"
  echo "[$(ts)] START $name"
  echo "============================================================"
  if bash -c "$cmd" 2>&1 | tee "$LOG_DIR/$name.log"; then
    echo "[$(ts)] OK    $name"
  else
    echo "[$(ts)] FAIL  $name (see $LOG_DIR/$name.log)"
    exit 1
  fi
}

run_soft() {
  local name="$1" cmd="$2"
  echo
  echo "------------------------------------------------------------"
  echo "[$(ts)] START $name (soft)"
  echo "------------------------------------------------------------"
  if bash -c "$cmd" 2>&1 | tee "$LOG_DIR/$name.log"; then
    echo "[$(ts)] OK    $name"
  else
    echo "[$(ts)] WARN  $name failed -- continuing (see $LOG_DIR/$name.log)"
  fi
}

if [ "$SKIP_INSTALL" -eq 0 ]; then
  echo
  echo "============================================================"
  echo "[$(ts)] Installing system deps (R 4.4 from CRAN + GDAL/GEOS/PROJ)"
  echo "============================================================"
  if [ -x "$(command -v sudo)" ]; then SUDO=sudo; else SUDO=""; fi

  R_VERSION=""
  if command -v R >/dev/null 2>&1; then
    R_VERSION=$(R --version | head -1 | awk '{print $3}')
  fi

  if [ -z "$R_VERSION" ] || dpkg --compare-versions "$R_VERSION" lt "4.3.0"; then
    echo "[$(ts)] Adding CRAN APT repo for current R (have: ${R_VERSION:-none})"
    $SUDO apt-get update -y 2>&1 | tee "$LOG_DIR/apt_update.log" || true
    $SUDO apt-get install -y --no-install-recommends \
      dirmngr gnupg apt-transport-https ca-certificates software-properties-common wget \
      2>&1 | tee -a "$LOG_DIR/apt_install.log"
    $SUDO mkdir -p /etc/apt/keyrings
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
      | $SUDO gpg --dearmor -o /etc/apt/keyrings/cran.gpg
    UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo jammy)
    echo "deb [signed-by=/etc/apt/keyrings/cran.gpg] https://cloud.r-project.org/bin/linux/ubuntu ${UBUNTU_CODENAME}-cran40/" \
      | $SUDO tee /etc/apt/sources.list.d/cran.list
    $SUDO apt-get update -y 2>&1 | tee -a "$LOG_DIR/apt_update.log" || true

    if [ -n "$R_VERSION" ] && dpkg --compare-versions "$R_VERSION" lt "4.3.0"; then
      $SUDO apt-get purge -y 'r-base*' 'r-cran-*' 2>&1 | tee -a "$LOG_DIR/apt_install.log" || true
      $SUDO rm -rf /usr/local/lib/R/site-library
    fi

    $SUDO apt-get install -y pkgconf 2>&1 | tee -a "$LOG_DIR/apt_install.log" || \
      $SUDO apt-get install -y -o Dpkg::Options::="--force-overwrite" pkgconf 2>&1 | tee -a "$LOG_DIR/apt_install.log"

    $SUDO apt-get install -y --no-install-recommends r-base r-base-dev \
      2>&1 | tee -a "$LOG_DIR/apt_install.log"

    $SUDO apt-get install -y --no-install-recommends \
      libgdal-dev libgeos-dev libproj-dev libudunits2-dev \
      libcurl4-openssl-dev libssl-dev libxml2-dev \
      libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
      libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
      libgsl-dev libv8-dev libsodium-dev libnode-dev libfftw3-dev \
      libnlopt-dev libsuitesparse-dev libcairo2-dev libtbb-dev \
      libglpk-dev libmpfr-dev libmagick++-dev \
      libgl1-mesa-dev libglu1-mesa-dev libx11-dev libxext-dev \
      cmake build-essential gfortran pandoc \
      2>&1 | tee -a "$LOG_DIR/apt_install.log" || true
  fi

  echo
  echo "============================================================"
  echo "[$(ts)] Installing Python deps with pip (into the runtime python3)"
  echo "============================================================"
  pip_install() {
    python3 -m pip install "$@" 2>/tmp/pip_err.log && return 0
    if grep -q "externally-managed\|break-system-packages" /tmp/pip_err.log; then
      python3 -m pip install --break-system-packages "$@"
    else
      cat /tmp/pip_err.log >&2
      return 1
    fi
  }
  pip_install --upgrade pip 2>&1 | tee "$LOG_DIR/pip_bootstrap.log" || true
  pip_install -r requirements.txt 2>&1 | tee "$LOG_DIR/pip_install.log"
  pip_install geopandas shapely fiona pyproj pyogrio \
    2>&1 | tee "$LOG_DIR/pip_geopandas.log" || \
    echo "[$(ts)] WARN geopandas install failed -- compactness term will be NaN"

  echo
  echo "============================================================"
  echo "[$(ts)] Installing R packages (~5 min, PPM binary repo)"
  echo "============================================================"
  $SUDO Rscript replication/scripts/install_r_packages.R 2>&1 | tee "$LOG_DIR/r_install.log"
fi

run "download_voteview"       "python3 data/download_voteview.py"
run "download_medsl"          "python3 src/medsl_data.py"
run "download_ep_data"        "python3 data/download_ep_data.py"
run "data_pipeline"           "python3 src/data_pipeline.py"
run "build_district_features" "python3 src/build_district_features.py"

run "compute_party_unity"     "python3 src/compute_party_unity.py"
run "build_departure_types"   "python3 src/build_departure_types.py"

run "spectral_analysis"  "python3 src/spectral_analysis.py"
run "weighted_spectral"  "python3 src/weighted_spectral.py"
run "bli_regression"     "python3 src/bli_regression.py"
run "null_model"         "python3 src/null_model_analysis.py"

run "bli_theorem_verification" "python3 src/bli_theorem_verification.py"
run "bli_proof_verification"   "python3 src/bli_proof_verification.py"
run "bli_theorem_proofs"       "python3 src/bli_theorem_proofs_verify.py"
run "sbm_benchmark"            "python3 src/sbm_benchmark_brev.py --scale large"
run "sbm_concentration"        "python3 src/sbm_concentration_verify.py --scale large"
run "bli_clt"                  "python3 src/bli_clt_verify.py --scale large"

run "cohort_ddd"          "python3 src/cohort_ddd.py"
run "staggered_did"       "python3 src/staggered_did.py"
run "honest_did"          "python3 src/honest_did.py"
run "mediation_analysis"  "python3 src/mediation_analysis.py"
run "death_in_office"     "python3 src/death_in_office.py"
run "freshman_cohort"     "python3 src/freshman_cohort_analysis.py"
run "senate_pipeline"     "python3 src/senate_pipeline.py"
run "senate_analysis"     "python3 src/senate_analysis.py"

run "counterfactual_sensitivity"     "python3 src/counterfactual_sensitivity.py"
run "recovery_threshold_sensitivity" "python3 src/recovery_threshold_sensitivity.py"
run "sensitivity_sweep"              "python3 src/sensitivity_sweep.py"
run "vote_filtering"                 "python3 src/vote_filtering.py"
run "centrality_comparison"          "python3 src/centrality_comparison.py"
run "negative_controls"              "python3 src/negative_controls.py"
run "placebo_loocv"                  "python3 src/placebo_loocv.py"
run "within_member_robustness"       "python3 src/within_member_robustness.py"

run_soft "ep_bli_all_terms" "python3 src/ep_bli_all_terms.py"

if [ "$CORE_ONLY" -eq 0 ]; then
run_soft "did_comparator"          "Rscript src/did_comparator.R results/cohort_ddd_panel.csv results/did_comparator.json"
run_soft "fect_placebo"            "Rscript src/fect_placebo.R"
run_soft "specr_curve"             "Rscript src/specr_curve.R"
run_soft "rdit_close_margins"      "Rscript src/rdit_close_margins.R results/mediation_panel.csv results/rdit.json"
run_soft "iv_redistricting"        "Rscript src/iv_redistricting.R results/mediation_panel.csv results/iv.json"
run_soft "sdid_bicameral"          "Rscript src/sdid_bicameral.R results/senate_panel.csv results/sdid.json"
run_soft "interflex_gerrymandering" "Rscript src/interflex_gerrymandering.R"
run_soft "interference_death"      "Rscript src/interference_death.R"
run_soft "causalweight_mediation"  "Rscript src/causalweight_mediation.R"
run_soft "bayes_factor_dfbetas"    "Rscript src/bayes_factor_dfbetas.R"
fi

run      "generate_figures"   "python3 src/generate_figures.py"
run      "skeleton_network"   "python3 src/skeleton_network.py"
run      "alluvial_data_prep" "python3 src/alluvial_data_prep.py"
run_soft "alluvial_coalitions" "Rscript src/alluvial_coalitions.R"

run_soft "paper_compile" "cd paper && tectonic paper.tex"

echo
echo "[$(ts)] ALL DONE"
echo "Logs:    $LOG_DIR/"
echo "Results: results/"
echo "Paper:   paper/paper.pdf"
