#!/usr/bin/env Rscript
# renv_robustness.R
# Pins package versions used by fect_placebo.R and specr_curve.R so the
# robustness pipeline replays cleanly on Brev (64-core) and macOS.
#
#   Rscript src/renv_robustness.R init     # initialise renv + install
#   Rscript src/renv_robustness.R snapshot # update renv.lock
#   Rscript src/renv_robustness.R restore  # restore from renv.lock

if (!requireNamespace("renv", quietly = TRUE))
  install.packages("renv", repos = "https://cloud.r-project.org")

required <- list(
  fect      = "2.4.1",      # Liu-Wang-Xu (2024 AJPS); R >= 4.1.0
  panelView = "1.1.18",
  specr     = "1.0.1",      # masurp/specr inference vignette
  multiverse = "0.6.2",     # mucollective/multiverse, alt to specr
  future    = "1.34.0",
  furrr     = "0.3.1",
  geepack   = "1.3.12",     # GEE working correlations
  data.table = "1.16.0",
  jsonlite  = "1.8.9",
  dplyr     = "1.1.4",
  ggplot2   = "3.5.1",
  broom     = "1.0.7"
)

action <- if (length(commandArgs(TRUE)) >= 1) commandArgs(TRUE)[[1]] else "init"

if (action == "init") {
  renv::init(bare = TRUE)
  for (pkg in names(required)) {
    renv::install(sprintf("%s@%s", pkg, required[[pkg]]))
  }
  renv::install("xuyiqing/fect")  # latest dev, overrides if CRAN lags
  renv::snapshot(prompt = FALSE)
} else if (action == "snapshot") {
  renv::snapshot(prompt = FALSE)
} else if (action == "restore") {
  renv::restore(prompt = FALSE)
} else {
  stop("Unknown action: ", action)
}

cat("[renv_robustness] done\n")
