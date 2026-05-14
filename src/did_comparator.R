#!/usr/bin/env Rscript
# did_comparator.R
# Runs four staggered-DiD estimators on a cohort-DDD panel and writes ATT(g,t),
# event-study coefficients, pre-period placebos and SEs to JSON.
#
# Expected CSV columns: congress, member_id, district_id, state,
#   cohort_congress, competitive, departure, bli, ideology_distance,
#   seniority, is_republican
#
# Usage:
#   Rscript did_comparator.R <panel.csv> <outdir> [estimator]
#   estimator in {cdh, bjs, fect, ddd, all}  (default: all)

suppressPackageStartupMessages({
  if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman", repos = "https://cloud.r-project.org")
  pacman::p_load(data.table, jsonlite, future, future.apply,
                 DIDmultiplegtDYN, didimputation, fect, triplediff, fixest)
})

args      <- commandArgs(trailingOnly = TRUE)
panel_csv <- args[[1]]
outdir    <- args[[2]]
which_est <- if (length(args) >= 3) args[[3]] else "all"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

panel <- fread(panel_csv)
stopifnot(all(c("congress","member_id","district_id","cohort_congress",
                "competitive","departure","bli","is_republican") %in% names(panel)))
# Treatment indicator: 1 once a unit is at/after its cohort_congress
panel[, D    := as.integer(!is.na(cohort_congress) & congress >= cohort_congress)]
panel[, gvar := ifelse(is.na(cohort_congress), 0L, as.integer(cohort_congress))]
panel[, tvar := as.integer(congress)]
panel[, idn  := as.integer(factor(member_id))]

n_cores <- max(1L, parallel::detectCores() - 1L)
cat(sprintf("[did_comparator] N_obs=%d  N_units=%d  T=%d  cores=%d\n",
            nrow(panel), uniqueN(panel$idn),
            uniqueN(panel$tvar), n_cores))

save_json <- function(obj, name) {
  fp <- file.path(outdir, paste0(name, ".json"))
  write_json(obj, fp, auto_unbox = TRUE, na = "string", null = "null", digits = 8)
  cat(sprintf("[did_comparator] wrote %s\n", fp))
}

safe_run <- function(label, fn) {
  res <- tryCatch(fn(), error = function(e) {
    msg <- conditionMessage(e)
    cat(sprintf("[%s] FAILED: %s\n", label, msg))
    list(error = msg)
  })
  save_json(res, label)
  invisible(res)
}

# ---------------------------------------------------------------------------
# (A) de Chaisemartin & D'Haultfoeuille (DIDmultiplegtDYN)
# ---------------------------------------------------------------------------
run_cdh <- function() {
  # SEs are analytical by default (much faster than bootstrap). Use `by =`
  # for the triple-difference dimension (eligibility = competitive district).
  mod <- did_multiplegt_dyn(
    df        = as.data.frame(panel),
    outcome   = "bli",
    group     = "idn",
    time      = "tvar",
    treatment = "D",
    effects   = 5,                # post-treatment horizons (set <= max gap)
    placebo   = 3,                # pre-treatment placebo periods
    cluster   = "district_id",
    by        = "competitive",    # triple-diff: separate effects by Q
    controls  = c("seniority", "is_republican"),
    graph_off = TRUE,
    ci_level  = 95
  )
  list(
    estimator = "de Chaisemartin-D'Haultfoeuille (DIDmultiplegtDYN)",
    ATE       = as.list(mod$results$ATE),
    Effects   = as.data.frame(mod$results$Effects),
    Placebos  = as.data.frame(mod$results$Placebos),
    N_Effects = mod$results$N_Effects,
    N_Placebos= mod$results$N_Placebos
  )
}

# ---------------------------------------------------------------------------
# (B) Borusyak-Jaravel-Spiess via didimputation
# ---------------------------------------------------------------------------
run_bjs <- function() {
  # Triple-difference via an eligibility-by-event interaction in wtr; easier
  # path: stratify the estimator over competitive==1 vs 0 and difference.
  fit_stratum <- function(stratum_val) {
    d <- panel[competitive == stratum_val]
    if (nrow(d) == 0 || sum(d$D) == 0) return(NULL)
    did_imputation(
      data       = d,
      yname      = "bli",
      gname      = "cohort_congress",
      tname      = "tvar",
      idname     = "idn",
      first_stage= ~ seniority + is_republican | idn + tvar,
      horizon    = TRUE,
      pretrends  = -5:-1,
      cluster_var= "district_id"
    )
  }
  s1 <- fit_stratum(1L)
  s0 <- fit_stratum(0L)
  list(
    estimator = "Borusyak-Jaravel-Spiess (didimputation, stratified)",
    eligible  = if (is.null(s1)) NULL else as.data.frame(s1),
    ineligible= if (is.null(s0)) NULL else as.data.frame(s0)
  )
}

# ---------------------------------------------------------------------------
# (C) Liu-Wang-Xu fect (imputation + placebo + equivalence test)
# ---------------------------------------------------------------------------
run_fect <- function() {
  # macOS note: fect's parallel backend uses future/foreach; explicitly set a
  # multisession plan to avoid forking issues under RStudio / Accelerate BLAS.
  if (.Platform$OS.type != "windows") future::plan(future::multisession, workers = n_cores)
  out <- fect(
    formula      = bli ~ D + seniority + is_republican,
    data         = as.data.frame(panel),
    index        = c("idn", "tvar"),
    method       = "fe",
    force        = "two-way",
    se           = TRUE,
    vartype      = "bootstrap",
    nboots       = 500,           # 200 default is low for n>3k; 500 is a balanced pick
    parallel     = TRUE,
    cores        = n_cores,
    placeboTest  = TRUE,
    placebo.period = c(-2, 0),
    seed         = 20260513
  )
  future::plan(future::sequential)
  list(
    estimator = "Liu-Wang-Xu fect (FE imputation)",
    ATT       = out$att.avg,
    ATT_se    = out$est.att.avg[, "S.E."],
    est_att   = as.data.frame(out$est.att),
    placebo   = out$est.placebo,
    equiv     = out$est.equiv,
    pre_test  = out$pre.test
  )
}

# ---------------------------------------------------------------------------
# (D) Ortiz-Villavicencio & Sant'Anna triplediff
# ---------------------------------------------------------------------------
run_ddd <- function() {
  d <- as.data.frame(panel)
  attgt <- ddd(
    yname        = "bli",
    tname        = "tvar",
    idname       = "idn",
    gname        = "gvar",          # 0 = never-treated
    pname        = "competitive",   # eligibility / Q partition
    xformla      = ~ seniority + is_republican,
    data         = d,
    control_group= "nevertreated",
    base_period  = "universal",
    est_method   = "dr"
  )
  es  <- agg_ddd(attgt, type = "eventstudy")
  agg <- agg_ddd(attgt, type = "simple")
  list(
    estimator    = "Ortiz-Villavicencio & Sant'Anna triplediff (DR)",
    ATTGT        = as.data.frame(process_attgt(attgt)),
    event_study  = as.data.frame(summary(es)$es),
    simple_ATT   = as.list(summary(agg))
  )
}

if (which_est %in% c("cdh","all"))  safe_run("cdh_dCDH",        run_cdh)
if (which_est %in% c("bjs","all"))  safe_run("bjs_imputation",  run_bjs)
if (which_est %in% c("fect","all")) safe_run("fect_LWX",        run_fect)
if (which_est %in% c("ddd","all"))  safe_run("ddd_triplediff",  run_ddd)

cat("[did_comparator] done.\n")
