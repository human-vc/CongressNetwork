#!/usr/bin/env Rscript
# Binning / marginal-effects diagnostic for the 3-way interaction
#   departure ~ BLI * gerrymander_severity * primary_intensity
# via Hainmueller-Mummolo-Xu 'interflex' (CRAN, version 1.2.6+).
#
# interflex() natively handles ONE moderator X; for a 3-way interaction we
# (i) stratify the panel by Z (primary intensity tercile) and run one
# binning model per stratum, then (ii) fit a kernel diagnostic on the
# pooled data including primary intensity as a Z covariate so its marginal
# variation is absorbed flexibly. This mirrors HMX (PA 2019, Sec. 4.4).
#
# Required inputs (added at the Python data-pipeline stage):
#   results/mediation_panel.csv with district_id, congress, departed, bli,
#   plus a pre-built results/district_features.csv with columns:
#     district_id, congress, compactness_pp (Polsby-Popper, 0-1; lower = worse
#     gerrymander), primary_margin (winning candidate's primary margin;
#     smaller = more intense challenge).
#
# Usage:
#   Rscript interflex_gerrymandering.R results/mediation_panel.csv \
#     results/district_features.csv results/interflex_gerrymandering_results.json
#     [n_boots]

suppressPackageStartupMessages({
  if (!requireNamespace("pacman", quietly = TRUE))
    install.packages("pacman", repos = "https://cloud.r-project.org")
  pacman::p_load(interflex, data.table, jsonlite, ggplot2, parallel)
})

args     <- commandArgs(trailingOnly = TRUE)
panel    <- if (length(args) >= 1) args[1] else "results/mediation_panel.csv"
distfeat <- if (length(args) >= 2) args[2] else "results/district_features.csv"
out_json <- if (length(args) >= 3) args[3] else "results/interflex_gerrymandering_results.json"
nboots   <- if (length(args) >= 4) as.integer(args[4]) else 1000L

d <- fread(panel)
f <- fread(distfeat)
d <- merge(d, f, by = c("district_id", "congress"), all.x = TRUE)
d <- d[!is.na(compactness_pp) & !is.na(primary_margin) & !is.na(bli)]
# Higher gerrymander_severity = worse compactness.
d[, gerrymander := 1 - compactness_pp]
d[, primary_intensity := 1 - primary_margin]
# Z covariates carried through every model.
ctrls <- c("seniority", "is_republican", "ideology_distance")
ctrls <- intersect(ctrls, names(d))

# (i) Stratified binning: split by primary-intensity tercile, then run the
# default 3-bin estimator over gerrymander.
qs <- quantile(d$primary_intensity, c(1/3, 2/3), na.rm = TRUE)
d[, primary_stratum := fifelse(primary_intensity <= qs[1], "low",
                       fifelse(primary_intensity <= qs[2], "mid", "high"))]
n_cores <- max(1, parallel::detectCores() - 1)

fit_stratum <- function(lab) {
  sub <- as.data.frame(d[primary_stratum == lab])
  if (nrow(sub) < 200) return(list(stratum = lab, n = nrow(sub), error = "n<200"))
  out <- tryCatch(
    interflex(estimator = "binning", data = sub,
              Y = "departed", D = "bli", X = "gerrymander",
              Z = ctrls, treat.type = "continuous",
              nbins = 3, vartype = "bootstrap", nboots = nboots,
              parallel = TRUE, cores = n_cores, figure = FALSE),
    error = function(e) list(error = conditionMessage(e)))
  if (!is.null(out$error)) return(list(stratum = lab, n = nrow(sub), error = out$error))
  list(stratum = lab, n = nrow(sub),
       est_binning = out$est.binning,
       wald_test   = out$tests$X.Wald.p,
       bin_cutoffs = out$cutoffs)
}
binning <- lapply(c("low", "mid", "high"), fit_stratum)
names(binning) <- c("low", "mid", "high")

# (ii) Pooled kernel diagnostic: gerrymander as the X moderator with
# primary_intensity threaded through as a Z covariate (full.moderate=TRUE
# lets primary_intensity's effect vary with gerrymander, so the kernel
# absorbs the 3-way interaction nonparametrically).
kernel <- tryCatch(
  interflex(estimator = "kernel", data = as.data.frame(d),
            Y = "departed", D = "bli", X = "gerrymander",
            Z = c(ctrls, "primary_intensity"), treat.type = "continuous",
            full.moderate = TRUE, vartype = "bootstrap", nboots = nboots,
            parallel = TRUE, cores = n_cores, figure = FALSE),
  error = function(e) list(error = conditionMessage(e)))

out <- list(method = "interflex binning (HMX PA 2019) stratified by primary tercile + pooled kernel with full.moderate",
            package = "interflex", n_bootstraps = nboots, n_total = nrow(d),
            stratified_binning = binning,
            pooled_kernel = if (is.null(kernel$error))
                              list(est = kernel$est.kernel, bandwidth = kernel$bw)
                            else list(error = kernel$error))
writeLines(toJSON(out, auto_unbox = TRUE, pretty = TRUE, null = "null", force = TRUE),
           out_json)
cat("wrote", out_json, "\n")
