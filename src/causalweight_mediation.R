#!/usr/bin/env Rscript
# Doubly robust IPW mediation as a comparator to the Imai-Keele-Tingley
# `mediation` ACME decomposition currently used in src/mediation_analysis.py.
# Uses Bodory & Huber's causalweight package (CRAN, version 1.1.4+).
#
# Setup:
#   y  = departed within 2 Congresses (binary)
#   d  = BLI (continuous)              -> medweightcont() with d0=Q1, d1=Q3
#   m  = ideology_distance (centrism)  (continuous)
#   x  = seniority, is_republican, lagged_centrism
#
# Robustness check: medDML() doubly-robust DML mediation with a median-split
# binarised BLI (since medDML requires binary d), reported as a sanity panel.
#
# Usage:
#   Rscript causalweight_mediation.R \
#     results/mediation_panel.csv results/causalweight_mediation_results.json
#     [n_boot] [d0_percentile] [d1_percentile]

suppressPackageStartupMessages({
  if (!requireNamespace("pacman", quietly = TRUE))
    install.packages("pacman", repos = "https://cloud.r-project.org")
  pacman::p_load(causalweight, data.table, jsonlite, future.apply)
})

args        <- commandArgs(trailingOnly = TRUE)
panel_csv   <- if (length(args) >= 1) args[1] else "results/mediation_panel.csv"
out_json    <- if (length(args) >= 2) args[2] else "results/causalweight_mediation_results.json"
n_boot      <- if (length(args) >= 3) as.integer(args[3]) else 1999L
q_lo        <- if (length(args) >= 4) as.numeric(args[4]) else 0.10
q_hi        <- if (length(args) >= 5) as.numeric(args[5]) else 0.90

ERAS <- list(
  early  = c(100, 106),
  middle = c(107, 112),
  late   = c(113, 116),
  pooled = c(100, 116)
)

d <- fread(panel_csv)
d <- d[!is.na(bli) & !is.na(ideology_distance) & !is.na(departed)]
setorder(d, icpsr, congress)
d[, lagged_centrism := shift(centrism, n = 1, type = "lag"), by = icpsr]
d[is.na(lagged_centrism), lagged_centrism := centrism]

run_era <- function(era_name) {
  rng <- ERAS[[era_name]]
  s <- d[congress >= rng[1] & congress <= rng[2]]
  if (nrow(s) < 200) return(NULL)
  y <- as.numeric(s$departed)
  dd <- as.numeric(s$bli)
  m <- as.numeric(s$ideology_distance)
  xmat <- as.matrix(s[, .(seniority = as.numeric(seniority),
                          is_republican = as.numeric(is_republican),
                          lagged_centrism = as.numeric(lagged_centrism))])
  d0 <- as.numeric(quantile(dd, q_lo, na.rm = TRUE))
  d1 <- as.numeric(quantile(dd, q_hi, na.rm = TRUE))
  cw <- tryCatch(
    medweightcont(y = y, d = dd, m = m, x = xmat,
                  d0 = d0, d1 = d1, ATET = FALSE, trim = 0.10,
                  lognorm = FALSE, bw = NULL, boot = n_boot),
    error = function(e) list(error = conditionMessage(e)))
  if (!is.null(cw$error)) return(list(era = era_name, n = nrow(s), error = cw$error))
  # DML robustness panel with binarised treatment.
  d_bin <- as.integer(dd >= median(dd, na.rm = TRUE))
  dml <- tryCatch(
    medDML(y = y, d = d_bin, m = m, x = xmat,
           k = 3, trim = 0.05, multmed = FALSE, normalized = TRUE,
           MLmethod = "lasso"),
    error = function(e) list(error = conditionMessage(e)))
  list(era = era_name, n = nrow(s), d0 = d0, d1 = d1,
       medweightcont = list(results = as.data.frame(cw$results),
                            ntrimmed = cw$ntrimmed),
       medDML = if (is.null(dml$error))
                  list(results = as.data.frame(dml$results),
                       ntrimmed = dml$ntrimmed) else list(error = dml$error))
}

plan(multisession, workers = min(4, max(1, parallel::detectCores() - 1)))
res <- future_lapply(names(ERAS), run_era, future.seed = TRUE)
names(res) <- names(ERAS)
res <- Filter(Negate(is.null), res)

out <- list(method = "Bodory & Huber causalweight: medweightcont + medDML",
            package = "causalweight", n_bootstrap = n_boot,
            d0_quantile = q_lo, d1_quantile = q_hi, by_era = res)
writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE, null = "null"), out_json)
cat("wrote", out_json, "\n")
