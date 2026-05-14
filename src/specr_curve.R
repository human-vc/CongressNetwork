#!/usr/bin/env Rscript
# specr_curve.R
# Field-standard specification-curve analysis (Simonsohn et al. 2020) for the
# BLI -> departure relationship. Replaces the home-rolled 96-spec curve with
# the masurp/specr toolchain, using future/furrr for parallel execution.
#
# Reads:  results/mediation_panel.csv
# Writes: results/specr_curve.csv  (one row per specification)
#         results/specr_curve.json (summary stats + metadata)
#         results/specr_curve.pdf  (curve + analytical-choice panels)
#
# Usage:
#   Rscript src/specr_curve.R <panel.csv> <outdir>

suppressPackageStartupMessages({
  if (!requireNamespace("pacman", quietly = TRUE))
    install.packages("pacman", repos = "https://cloud.r-project.org")
  pacman::p_load(data.table, dplyr, jsonlite, specr, future, furrr, ggplot2,
                 geepack, broom)
})

args      <- commandArgs(trailingOnly = TRUE)
panel_csv <- if (length(args) >= 1) args[[1]] else "results/mediation_panel.csv"
outdir    <- if (length(args) >= 2) args[[2]] else "results"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

panel <- fread(panel_csv) |> as.data.frame()
if ("ideology_distance_from_zero" %in% names(panel))
  names(panel)[names(panel) == "ideology_distance_from_zero"] <- "ideology_distance"

panel$departure_2c <- as.integer(panel$departed)
panel$bli_hi_p50  <- as.integer(panel$bli > quantile(panel$bli, 0.50, na.rm=TRUE))
panel$bli_hi_p67  <- as.integer(panel$bli > quantile(panel$bli, 2/3,  na.rm=TRUE))
panel$bli_hi_p75  <- as.integer(panel$bli > quantile(panel$bli, 0.75, na.rm=TRUE))

panel$era <- cut(panel$congress,
                 breaks = c(-Inf, 105, 112, Inf),
                 labels = c("early", "middle", "late"))
panel$era <- as.character(panel$era)

panel$party <- ifelse(panel$is_republican == 1, "rep", "dem")

logit_glm <- function(formula, data) {
  glm(formula = formula, data = data, family = binomial(link = "logit"))
}

probit_glm <- function(formula, data) {
  glm(formula = formula, data = data, family = binomial(link = "probit"))
}

gee_exch <- function(formula, data) {
  data <- data[order(data$icpsr, data$congress), ]
  geepack::geeglm(formula = formula, data = data,
                  id = data$icpsr, family = binomial(link = "logit"),
                  corstr = "exchangeable")
}

gee_ar1 <- function(formula, data) {
  data <- data[order(data$icpsr, data$congress), ]
  geepack::geeglm(formula = formula, data = data,
                  id = data$icpsr, family = binomial(link = "logit"),
                  corstr = "ar1")
}

n_cores <- max(1L, parallel::detectCores() - 2L)
plan(multisession, workers = n_cores)
cat(sprintf("[specr_curve] N=%d cores=%d\n", nrow(panel), n_cores))

specs <- setup(
  data     = panel,
  y        = c("departure_2c"),
  x        = c("bli_hi_p50", "bli_hi_p67", "bli_hi_p75"),
  model    = c("logit_glm", "probit_glm", "gee_exch", "gee_ar1"),
  controls = c("ideology_distance", "seniority", "is_republican"),
  subsets  = list(
    era   = unique(panel$era),
    party = unique(panel$party)
  )
)

cat(sprintf("[specr_curve] running %d specifications...\n", nrow(specs$specs)))
t0 <- Sys.time()
results <- specr(specs, .progress = TRUE)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
plan(sequential)
cat(sprintf("[specr_curve] specr() done in %.1fs\n", elapsed))

est <- as.data.frame(results)
fwrite(est, file.path(outdir, "specr_curve.csv"))

summary_stats <- list(
  n_specs       = nrow(est),
  elapsed_sec   = elapsed,
  cores         = n_cores,
  median_est    = median(est$estimate, na.rm = TRUE),
  q025_est      = quantile(est$estimate, 0.025, na.rm = TRUE),
  q975_est      = quantile(est$estimate, 0.975, na.rm = TRUE),
  share_pos     = mean(est$estimate > 0, na.rm = TRUE),
  share_sig_pos = mean(est$estimate > 0 & est$p.value < 0.05, na.rm = TRUE),
  share_sig_neg = mean(est$estimate < 0 & est$p.value < 0.05, na.rm = TRUE),
  median_p      = median(est$p.value, na.rm = TRUE)
)

write_json(summary_stats, file.path(outdir, "specr_curve.json"),
           auto_unbox = TRUE, pretty = TRUE, na = "null")

pdf(file.path(outdir, "specr_curve.pdf"), width = 11, height = 8)
try(print(plot(results, type = "curve", desc = TRUE)), silent = TRUE)
try(print(plot(results, type = "choices")),            silent = TRUE)
try(print(plot(results, type = "boxplot")),            silent = TRUE)
dev.off()

cat(sprintf("[specr_curve] wrote %s, %s, %s\n",
            file.path(outdir, "specr_curve.csv"),
            file.path(outdir, "specr_curve.json"),
            file.path(outdir, "specr_curve.pdf")))
