## sdid_bicameral.R
## Synthetic DiD for bicameral (House vs Senate) comparison around the
## post-2010 partisan-realignment shock. Arkhangelsky-Athey-Hirshberg-Imbens-Wager
## (2021); inference via Clarke-Pailanir-Athey-Imbens (2024).
##
## Install:
##   install.packages(c("data.table","jsonlite","parallel"))
##   remotes::install_github("synth-inference/synthdid")
## Run:
##   Rscript src/sdid_bicameral.R data/panel.csv out/sdid.json

suppressPackageStartupMessages({
  library(synthdid); library(data.table); library(jsonlite); library(parallel)
})

args      <- commandArgs(trailingOnly = TRUE)
data_path <- ifelse(length(args) >= 1, args[1], "data/panel.csv")
out_path  <- ifelse(length(args) >= 2, args[2], "out/sdid.json")

d <- fread(data_path)
## Required: member_id, congress, chamber ("House"/"Senate"), cp_agree_rate.
## Treatment: Senate (chamber == "Senate") post-2010 (congress >= 112).
d <- d[!is.na(cp_agree_rate)]
d[, unit  := as.character(member_id)]
d[, time  := as.integer(congress)]
d[, treat := as.integer(chamber == "Senate" & congress >= 112)]

## sdid wants a strongly-balanced panel. Aggregate to chamber-Congress
## cell means (this is the canonical bicameral specification; member-level
## is unbalanced because tenure varies).
agg <- d[, .(Y = mean(cp_agree_rate, na.rm = TRUE),
             treat = max(treat)),
         by = .(unit = chamber, time = congress)]
setorder(agg, unit, time)
agg <- as.data.frame(agg[, .(unit, time, Y, treat)])

## panel.matrices expects: unit, time, outcome, treatment (in that order)
setup <- panel.matrices(agg, unit = "unit", time = "time",
                        outcome = "Y", treatment = "treat")

## Pre-treatment window: Congresses 100-111 (1987-2010) -> matching weights
## Post-treatment:        Congresses 112-118 (2011-2024)
tau_sdid <- synthdid_estimate(setup$Y, setup$N0, setup$T0)
tau_sc   <- sc_estimate(setup$Y, setup$N0, setup$T0)
tau_did  <- did_estimate(setup$Y, setup$N0, setup$T0)

## --- Inference. With only one treated unit (Senate) jackknife is
##     undefined; placebo is the recommended method (Clarke et al. 2024). ---
n_treated <- ncol(setup$Y) - setup$N0
se_method <- if (n_treated == 1) "placebo" else "jackknife"

## Parallel placebo inference: each replication is independent.
ncores <- max(1, detectCores() - 2)
if (.Platform$OS.type == "unix") options(mc.cores = ncores)
se <- sqrt(vcov(tau_sdid, method = se_method, replications = 500))

tau_hat  <- as.numeric(tau_sdid)
ci_low   <- tau_hat - 1.96 * se
ci_high  <- tau_hat + 1.96 * se
p_val    <- 2 * (1 - pnorm(abs(tau_hat / se)))

## --- Sanity checks ---
warn <- c()
if (setup$T0 < 5) warn <- c(warn, "Fewer than 5 pre-treatment Congresses; weights unstable")
if (n_treated == 1) warn <- c(warn, "Only one treated unit; jackknife unavailable, using placebo SE")
if (abs(tau_hat) > sd(setup$Y))
  warn <- c(warn, "|tau| exceeds outcome SD; inspect parallel-trends plot")

## --- JSON output ---
out <- list(
  analysis = "Synthetic DiD bicameral (Senate post-2010 vs House)",
  estimate    = tau_hat,
  se          = se,
  ci_lower    = ci_low,
  ci_upper    = ci_high,
  p_value     = p_val,
  se_method   = se_method,
  n_units     = ncol(setup$Y),
  n_control   = setup$N0,
  n_pre       = setup$T0,
  n_post      = nrow(setup$Y) - setup$T0,
  benchmark   = list(sc_estimate = as.numeric(tau_sc),
                     did_estimate = as.numeric(tau_did)),
  unit_weights = attr(tau_sdid, "weights")$omega,
  time_weights = attr(tau_sdid, "weights")$lambda,
  warnings = warn
)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(out, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 6)

## Optional diagnostic plot
try({
  pdf(sub("\\.json$", ".pdf", out_path), width = 7, height = 5)
  print(synthdid_plot(list(sdid = tau_sdid, sc = tau_sc, did = tau_did)))
  dev.off()
}, silent = TRUE)

cat("sdid done. tau_hat =", round(tau_hat, 4), " se =", round(se, 4),
    " (", se_method, ")\n")
