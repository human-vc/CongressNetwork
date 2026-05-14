## rdit_close_margins.R
## RDiT at close cohort margins (Marshall 2024 AJPS template)
## Forcing var: prior-cycle margin (centered at 0). Outcome: cross-party agreement
## rate in first term. Bandwidth: CCT (mserd). Cluster: member.
##
## Install:
##   install.packages(c("rdrobust","data.table","jsonlite","future.apply"))
## Run:
##   Rscript src/rdit_close_margins.R data/panel.csv out/rdit.json

suppressPackageStartupMessages({
  library(rdrobust); library(data.table); library(jsonlite); library(future.apply)
})

args      <- commandArgs(trailingOnly = TRUE)
data_path <- ifelse(length(args) >= 1, args[1], "data/panel.csv")
out_path  <- ifelse(length(args) >= 2, args[2], "out/rdit.json")

d <- fread(data_path)
d <- d[!is.na(margin) & !is.na(cp_agree_first_term) & first_term == 1]
d[, run := margin]                            # forcing variable, centered at 0
d[, treat := as.integer(run >= 0)]            # barely-won = treated

stopifnot(nrow(d) > 200, "member_id" %in% names(d), "congress" %in% names(d))

## --- Pooled RD (Marshall: pool cohorts; identification is at c=0) ---
fit <- rdrobust(
  y         = d$cp_agree_first_term,
  x         = d$run,
  c         = 0,
  p         = 1, q = 2,
  kernel    = "triangular",
  bwselect  = "mserd",                        # CCT MSE-optimal
  vce       = "nn",                           # nearest-neighbor variance
  cluster   = d$member_id,                    # within-member clustering
  all       = TRUE
)

## --- RDiT: dynamic treatment timing. Estimate cohort-specific RDs
##     in parallel across the 19 Congresses, then meta-average. ---
plan(multisession, workers = max(1, parallel::detectCores() - 2))

cohorts  <- sort(unique(d$congress))
per_coh  <- future_lapply(cohorts, function(cc) {
  dc <- d[congress == cc]
  if (nrow(dc) < 60 || sum(dc$run >= 0) < 10 || sum(dc$run < 0) < 10) return(NULL)
  r <- tryCatch(
    rdrobust(dc$cp_agree_first_term, dc$run, c = 0, p = 1,
             kernel = "triangular", bwselect = "mserd", vce = "nn"),
    error = function(e) NULL)
  if (is.null(r)) return(NULL)
  data.table(congress = cc,
             tau   = r$coef["Robust", 1],
             se    = r$se["Robust", 1],
             bw    = r$bws["h", 1],
             n_eff = sum(r$N_h))
}, future.seed = TRUE)

per_coh <- rbindlist(per_coh)
## Inverse-variance meta-average across cohorts
w <- 1 / per_coh$se^2
tau_meta <- sum(w * per_coh$tau) / sum(w)
se_meta  <- sqrt(1 / sum(w))

## --- Sanity checks ---
warn <- c()
if (fit$bws["h", 1] < 0.02) warn <- c(warn, "RD bandwidth < 2pp; recheck data")
if (sum(fit$N_h) < 100)     warn <- c(warn, "effective N < 100; underpowered")
if (any(abs(per_coh$tau) > 0.5))
  warn <- c(warn, "cohort RD > 0.5; outlier cohort, inspect per_coh")

## --- JSON output ---
out <- list(
  analysis = "RDiT close-margin cohort",
  pooled = list(
    estimate     = unname(fit$coef["Robust", 1]),
    se           = unname(fit$se["Robust", 1]),
    ci_lower     = unname(fit$ci["Robust", 1]),
    ci_upper     = unname(fit$ci["Robust", 2]),
    p_value      = unname(fit$pv["Robust", 1]),
    bandwidth_h  = unname(fit$bws["h", 1]),
    bandwidth_b  = unname(fit$bws["b", 1]),
    n_eff_left   = unname(fit$N_h[1]),
    n_eff_right  = unname(fit$N_h[2]),
    kernel       = "triangular",
    bwselect     = "mserd",
    cluster      = "member_id"
  ),
  rdit_meta = list(
    estimate = tau_meta, se = se_meta,
    ci_lower = tau_meta - 1.96 * se_meta,
    ci_upper = tau_meta + 1.96 * se_meta,
    n_cohorts = nrow(per_coh)
  ),
  per_cohort = per_coh,
  warnings   = warn
)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(out, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 6)
cat("RDiT done. tau_hat =", round(fit$coef["Robust",1], 4),
    " h =", round(fit$bws["h",1], 4), "\n")
