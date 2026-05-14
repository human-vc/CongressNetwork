# Replication container for GATCongress.
# Combines rocker/r-ver (R + tidyverse base) with uv-managed Python and tectonic.
#
# Build:
#     docker build -t gatcongress -f replication/docker/Dockerfile .
# Run:
#     docker run --rm -v "$PWD:/work" -w /work gatcongress bash run_all.sh

FROM rocker/r-ver:4.4.2

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps for geopandas, shapely, pyproj, fiona, brms (Stan), tectonic
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran git curl ca-certificates \
    libcurl4-openssl-dev libssl-dev libxml2-dev \
    libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
    libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
    libgdal-dev libgeos-dev libproj-dev libudunits2-dev libnetcdf-dev \
    libcairo2-dev libsodium-dev libsecret-1-dev \
    cmake pkg-config \
    python3 python3-pip python3-venv \
    libgmp-dev libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

# uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv

# tectonic (LaTeX compiler)
RUN curl --proto '=https' --tlsv1.2 -fsSL https://drop-sh.fullyjustified.net | sh \
    && mv tectonic /usr/local/bin/

WORKDIR /work

# Python deps first (layer cache: only rebuilds when requirements.txt changes)
COPY requirements.txt /work/
RUN uv pip install --system --no-cache -r requirements.txt

# R deps via custom installer (layer cache: only rebuilds when this script changes)
COPY replication/scripts/install_r_packages.R /work/replication/scripts/
RUN Rscript /work/replication/scripts/install_r_packages.R

# Pre-create directories that scripts assume exist
RUN mkdir -p /work/data/raw/tiger /work/data/raw/medsl_primary \
    /work/data/raw/ep /work/results /work/logs

# Default command runs the full pipeline; override with `docker run gatcongress bash`
CMD ["bash", "run_all.sh"]
