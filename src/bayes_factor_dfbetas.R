#!/usr/bin/env Rscript
# Bayes factor for BLI + DFBETAS / Cook's D for the GEE.
# Driven from Python via subprocess; reads results/cohort_ddd_panel.csv-like
# panel built from build_panel() (saved to results/bli_panel.csv by the
# Python driver). Writes results/bf_dfbetas_results.json.
#
# Stack: brms + bridgesampling for BF (logistic, bernoulli link).
#        glmtoolbox::glmgee for one-step DFBETAS + Cook's D on cluster-level.

suppressPackageStartupMessages({
  for (pkg in c("brms", "bridgesampling", "glmtoolbox", "jsonlite", "dplyr")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
    library(pkg, character.only = TRUE)
  }
})

args <- commandArgs(trailingOnly = TRUE)
panel_path <- if (length(args) >= 1) args[1] else "results/bli_panel.csv"
out_path   <- if (length(args) >= 2) args[2] else "results/bf_dfbetas_results.json"

cat("Loading panel from", panel_path, "\n")
d <- read.csv(panel_path)
d$icpsr <- as.integer(d$icpsr)
d <- d[order(d$icpsr, d$congress), ]

cat("Fitting GEE (binomial, independence) for DFBETAS / Cook's D ...\n")
fit_gee <- glmgee(
  departed_within_2 ~ bli + ideology_distance + seniority + is_republican,
  id = icpsr, family = binomial(link = "logit"),
  data = d, corstr = "independence"
)

cat("Computing one-step cluster-level DFBETAS for `bli` ...\n")
dfb <- dfbeta(fit_gee, method = "full", coefs = "bli", plot.it = FALSE)
dfb_vec <- as.numeric(dfb)
member_ids <- as.integer(rownames(as.matrix(dfb)))
if (length(member_ids) != length(dfb_vec)) {
  member_ids <- unique(d$icpsr)
}
ord <- order(abs(dfb_vec), decreasing = TRUE)
top_dfb <- head(data.frame(icpsr = member_ids[ord], dfbeta_bli = dfb_vec[ord]), 20)

cat("Computing one-step cluster-level Cook's D ...\n")
cd <- cooks.distance(fit_gee, method = "full", plot.it = FALSE)
cd_vec <- as.numeric(cd)
cd_ids <- as.integer(rownames(as.matrix(cd)))
if (length(cd_ids) != length(cd_vec)) cd_ids <- unique(d$icpsr)
cd_ord <- order(cd_vec, decreasing = TRUE)
top_cd <- head(data.frame(icpsr = cd_ids[cd_ord], cooks_d = cd_vec[cd_ord]), 20)

cat("Fitting brms full + null logistic for Bayes factor (bridge sampling) ...\n")
# normal(0,2.5) weakly informative on logit scale; standardise BLI first.
d$bli_z <- as.numeric(scale(d$bli))

prior_full <- prior(normal(0, 2.5), class = "b")
fit_full <- brm(
  departed_within_2 ~ bli_z + ideology_distance + seniority + is_republican + (1 | icpsr),
  data = d, family = bernoulli(),
  prior = prior_full,
  chains = 4, iter = 6000, warmup = 1500,
  cores = 4, save_pars = save_pars(all = TRUE),
  refresh = 0, seed = 42
)
fit_null <- brm(
  departed_within_2 ~ ideology_distance + seniority + is_republican + (1 | icpsr),
  data = d, family = bernoulli(),
  chains = 4, iter = 6000, warmup = 1500,
  cores = 4, save_pars = save_pars(all = TRUE),
  refresh = 0, seed = 42
)
bf <- bayes_factor(fit_full, fit_null, log = FALSE)
cat("Bayes factor (full vs. null):", bf$bf, "\n")

out <- list(
  bayes_factor_full_vs_null = as.numeric(bf$bf),
  log_bayes_factor = log(as.numeric(bf$bf)),
  prior_full = "normal(0, 2.5) on bli_z",
  top_dfbetas_bli = top_dfb,
  top_cooks_distance = top_cd,
  n_clusters = length(unique(d$icpsr)),
  n_obs = nrow(d)
)
write_json(out, out_path, auto_unbox = TRUE, pretty = TRUE)
cat("Wrote", out_path, "\n")
