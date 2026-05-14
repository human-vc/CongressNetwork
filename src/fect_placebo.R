#!/usr/bin/env Rscript
# fect_placebo.R
# Liu-Wang-Xu (2024 AJPS) Fixed-Effect Counterfactual Estimator: placebo test
# and equivalence (TOST) test for the parallel-trends assumption behind the
# cohort-DDD departure analysis.
#
# Reads:  results/mediation_panel.csv  (icpsr, congress, bli, departed,
#                                       ideology_distance_from_zero, seniority,
#                                       is_republican, ...)
# Writes: results/fect_placebo.json
#
# Usage:
#   Rscript src/fect_placebo.R <panel.csv> <outdir>

suppressPackageStartupMessages({
  if (!requireNamespace("pacman", quietly = TRUE))
    install.packages("pacman", repos = "https://cloud.r-project.org")
  pacman::p_load(data.table, jsonlite, fect, panelView)
})

args      <- commandArgs(trailingOnly = TRUE)
panel_csv <- if (length(args) >= 1) args[[1]] else "results/mediation_panel.csv"
outdir    <- if (length(args) >= 2) args[[2]] else "results"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

panel <- fread(panel_csv)
setnames(panel, "ideology_distance_from_zero", "ideology_distance",
         skip_absent = TRUE)
panel[, member  := as.integer(icpsr)]
panel[, period  := as.integer(congress)]
panel[, bli_hi  := as.integer(bli > median(bli, na.rm = TRUE))]
panel[, outcome := as.numeric(departed)]
panel <- panel[complete.cases(panel[, .(outcome, bli_hi, ideology_distance,
                                        seniority, is_republican)])]

n_cores <- max(1L, parallel::detectCores() - 2L)
cat(sprintf("[fect_placebo] N=%d units=%d T=%d cores=%d\n",
            nrow(panel), uniqueN(panel$member),
            uniqueN(panel$period), n_cores))

fec <- fect(outcome ~ bli_hi + ideology_distance + seniority + is_republican,
            data         = as.data.frame(panel),
            index        = c("member", "period"),
            method       = "fe",
            force        = "two-way",
            se           = TRUE,
            vartype      = "bootstrap",
            nboots       = 1000,
            parallel     = TRUE,
            cores        = n_cores,
            placeboTest  = TRUE,
            placebo.period = c(-2, 0),
            seed         = 20260513)

extract <- function(x) if (is.null(x)) NA_real_ else as.numeric(x)

placebo_row <- if (!is.null(fec$est.placebo)) as.data.frame(fec$est.placebo)
               else data.frame()

out <- list(
  call = list(
    method        = "fe",
    force         = "two-way",
    placebo.period = c(-2, 0),
    nboots        = 1000L,
    n_obs         = nrow(panel),
    n_units       = uniqueN(panel$member),
    n_periods     = uniqueN(panel$period)
  ),
  placebo = list(
    coef    = extract(placebo_row[1, "ATT.placebo"]),
    se      = extract(placebo_row[1, "S.E."]),
    ci_low  = extract(placebo_row[1, "CI.lower"]),
    ci_high = extract(placebo_row[1, "CI.upper"]),
    p_value = extract(placebo_row[1, "p.value"])
  ),
  equivalence = list(
    f_threshold     = if (!is.null(fec$test.out$f.threshold))
                         fec$test.out$f.threshold else 0.5,
    tost_threshold  = if (!is.null(fec$test.out$tost.threshold))
                         fec$test.out$tost.threshold else NA_real_,
    att_bound_lo    = extract(fec$att.bound[1]),
    att_bound_hi    = extract(fec$att.bound[2]),
    tost_p_value    = if (!is.null(fec$test.out$p.tost))
                         extract(fec$test.out$p.tost) else NA_real_,
    f_test_p_value  = if (!is.null(fec$test.out$p.f))
                         extract(fec$test.out$p.f) else NA_real_,
    reject_inequivalence = !is.na(fec$test.out$p.tost) &&
                            fec$test.out$p.tost < 0.05
  ),
  att_avg = list(
    coef    = extract(fec$est.avg[1, "ATT.avg"]),
    se      = extract(fec$est.avg[1, "S.E."]),
    ci_low  = extract(fec$est.avg[1, "CI.lower"]),
    ci_high = extract(fec$est.avg[1, "CI.upper"])
  )
)

json_path <- file.path(outdir, "fect_placebo.json")
write_json(out, json_path, auto_unbox = TRUE, pretty = TRUE, na = "null")
cat(sprintf("[fect_placebo] wrote %s\n", json_path))

if (requireNamespace("ggplot2", quietly = TRUE)) {
  pdf(file.path(outdir, "fect_placebo.pdf"), width = 7, height = 5)
  try(plot(fec, type = "status"), silent = TRUE)
  try(plot(fec), silent = TRUE)
  dev.off()
}
