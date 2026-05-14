#!/usr/bin/env bash
# Orchestration script for the full GATCongress analysis pipeline.
# Designed for a Brev CPU instance (c7i.16xlarge / m7i.16xlarge or similar).
#
# Usage:
#     bash run_all.sh                # full pipeline
#     bash run_all.sh --skip-install # skip dep installs
#     bash run_all.sh --r-only       # just R jobs
#     bash run_all.sh --py-only      # just Python jobs

set -euo pipefail
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR" results

SKIP_INSTALL=0
R_ONLY=0
PY_ONLY=0
for a in "$@"; do
  case "$a" in
    --skip-install) SKIP_INSTALL=1 ;;
    --r-only) R_ONLY=1 ;;
    --py-only) PY_ONLY=1 ;;
  esac
done

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
run() {
  local name="$1" cmd="$2"
  echo
  echo "============================================================"
  echo "[$(ts)] START $name"
  echo "============================================================"
  if /usr/bin/time -v bash -c "$cmd" 2>&1 | tee "$LOG_DIR/$name.log"; then
    echo "[$(ts)] OK    $name"
  else
    echo "[$(ts)] FAIL  $name (see $LOG_DIR/$name.log)"
    exit 1
  fi
}

# 0. Install deps
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

  # Ubuntu's stock R is 4.1; we need R >= 4.3. If R is missing or too old,
  # install from CRAN's apt repo.
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

    # Remove stale R 4.1 + partially-built site library to avoid version conflicts
    if [ -n "$R_VERSION" ] && dpkg --compare-versions "$R_VERSION" lt "4.3.0"; then
      $SUDO apt-get purge -y 'r-base*' 'r-cran-*' 2>&1 | tee -a "$LOG_DIR/apt_install.log" || true
      $SUDO rm -rf /usr/local/lib/R/site-library
    fi

    # CRAN packages declare a Breaks against legacy pkg-config; switch to pkgconf
    $SUDO apt-get install -y pkgconf 2>&1 | tee -a "$LOG_DIR/apt_install.log" || \
      $SUDO apt-get install -y -o Dpkg::Options::="--force-overwrite" pkgconf 2>&1 | tee -a "$LOG_DIR/apt_install.log"

    # Step 1: install R 4.4 from CRAN
    $SUDO apt-get install -y --no-install-recommends r-base r-base-dev \
      2>&1 | tee -a "$LOG_DIR/apt_install.log"

    # Step 2: system libs for the R+Python C/C++ deps (separate call so a failing
    # optional dep doesn't block R itself)
    $SUDO apt-get install -y --no-install-recommends \
      libgdal-dev libgeos-dev libproj-dev libudunits2-dev \
      libcurl4-openssl-dev libssl-dev libxml2-dev \
      libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
      libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
      libgsl-dev libv8-dev libsodium-dev libnode-dev libfftw3-dev \
      cmake build-essential gfortran pandoc \
      2>&1 | tee -a "$LOG_DIR/apt_install.log" || true
  fi

  echo
  echo "============================================================"
  echo "[$(ts)] Installing Python deps with uv"
  echo "============================================================"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  $SUDO env "PATH=$PATH" uv pip install --system --break-system-packages -r requirements.txt 2>&1 | tee "$LOG_DIR/uv_install.log"

  echo
  echo "============================================================"
  echo "[$(ts)] Installing R packages (this takes ~10 min)"
  echo "============================================================"
  $SUDO Rscript replication/scripts/install_r_packages.R 2>&1 | tee "$LOG_DIR/r_install.log"
fi

# 1. Data prep
if [ "$R_ONLY" -eq 0 ]; then
  run "download_voteview"        "python3 data/download_voteview.py"
  run "download_ep_data"         "python3 data/download_ep_data.py"
  run "build_district_features"  "python3 src/build_district_features.py"
fi

# 2. Python compute jobs (parallel-safe in sequence; each script uses joblib internally)
if [ "$R_ONLY" -eq 0 ]; then
  run "sbm_concentration_verify" "python3 src/sbm_concentration_verify.py --scale large"
  run "bli_clt_verify"           "python3 src/bli_clt_verify.py --scale large"
  run "ep_bli_all_terms"         "python3 src/ep_bli_all_terms.py"
  run "negative_controls"        "python3 src/negative_controls.py"
  run "placebo_loocv"            "python3 src/placebo_loocv.py"
  run "skeleton_network"         "python3 src/skeleton_network.py"
  run "alluvial_data_prep"       "python3 src/alluvial_data_prep.py"
fi

# 3. R compute jobs
if [ "$PY_ONLY" -eq 0 ]; then
  run "did_comparator"          "Rscript src/did_comparator.R results/mediation_panel.csv results/did_comparator.json"
  run "fect_placebo"            "Rscript src/fect_placebo.R"
  run "specr_curve"             "Rscript src/specr_curve.R"
  run "rdit_close_margins"      "Rscript src/rdit_close_margins.R results/mediation_panel.csv results/rdit.json"
  run "iv_redistricting"        "Rscript src/iv_redistricting.R results/mediation_panel.csv results/iv.json"
  run "sdid_bicameral"          "Rscript src/sdid_bicameral.R results/senate_panel.csv results/sdid.json"
  run "interflex_gerrymandering" "Rscript src/interflex_gerrymandering.R"
  run "interference_death"      "Rscript src/interference_death.R"
  run "causalweight_mediation"  "Rscript src/causalweight_mediation.R"
  run "bayes_factor_dfbetas"    "Rscript src/bayes_factor_dfbetas.R"
  run "alluvial_coalitions"     "Rscript src/alluvial_coalitions.R"
fi

# 4. Snapshot R env for reproducibility
if [ "$PY_ONLY" -eq 0 ]; then
  run "renv_snapshot" "Rscript -e 'if (!requireNamespace(\"renv\", quietly = TRUE)) install.packages(\"renv\"); renv::init(bare = TRUE); renv::snapshot(prompt = FALSE)'"
fi

# 5. Compile paper
run "paper_compile" "cd paper && tectonic paper.tex"

echo
echo "[$(ts)] ALL DONE"
echo "Logs in $LOG_DIR/"
echo "Results in results/"
echo "Paper at paper/paper.pdf"
