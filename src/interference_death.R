#!/usr/bin/env Rscript
# Aronow-Samii (2017 AOAS) design-based exposure mapping for the
# death-in-office natural experiment, via the szonszein/interference package
# (Zonszein, Aronow & Samii 2019). Replaces the hand-rolled exposure mapping
# in src/death_in_office.py.
#
# Treatment: an icpsr-Congress unit is "treated" if one of its co-partisan
# state-delegation peers died in that Congress (an exogenous-flag death).
# Adjacency: same-state, same-party co-service in the same Congress (1-hop).
# Outcome: departed within 2 Congresses (binary). Hop=1 yields 4 conditions:
#   d00 (no direct treat, no indirect spillover), d10, d01, d11.
#
# Usage:
#   Rscript interference_death.R \
#     results/mediation_panel.csv data/house_deaths_1987_2025.csv \
#     results/interference_death_results.json [n_permutations]

suppressPackageStartupMessages({
  if (!requireNamespace("pacman", quietly = TRUE))
    install.packages("pacman", repos = "https://cloud.r-project.org")
  pacman::p_load(data.table, jsonlite, Matrix, future, future.apply, igraph)
  if (!requireNamespace("interference", quietly = TRUE)) {
    if (!requireNamespace("devtools", quietly = TRUE))
      install.packages("devtools", repos = "https://cloud.r-project.org")
    devtools::install_github("szonszein/interference", upgrade = "never")
  }
  library(interference)
})

args      <- commandArgs(trailingOnly = TRUE)
panel_csv <- if (length(args) >= 1) args[1] else "results/mediation_panel.csv"
death_csv <- if (length(args) >= 2) args[2] else "data/house_deaths_1987_2025.csv"
out_json  <- if (length(args) >= 3) args[3] else "results/interference_death_results.json"
n_perm    <- if (length(args) >= 4) as.integer(args[4]) else 2000L

panel  <- fread(panel_csv)
deaths <- fread(death_csv)
deaths <- deaths[exogenous_flag == 1L]   # only sudden / exogenous deaths

members <- fread("data/HSall_members.csv")[chamber == "House",
  .(icpsr, congress, state = state_abbrev, party_code, district_code)]
members[, party := fifelse(party_code == 200, "R", fifelse(party_code == 100, "D", "I"))]
panel <- merge(panel, members[, .(icpsr, congress, state, party)],
               by = c("icpsr", "congress"), all.x = TRUE)
panel <- panel[!is.na(state) & party %in% c("R", "D")]

run_congress <- function(cg) {
  ego <- panel[congress == cg]
  if (nrow(ego) < 5) return(NULL)
  N <- nrow(ego)
  # Adjacency: same state + same party, exclude self-loops.
  same_state <- outer(ego$state, ego$state, "==")
  same_party <- outer(ego$party, ego$party, "==")
  adj <- (same_state & same_party) * 1L
  diag(adj) <- 0L
  storage.mode(adj) <- "numeric"
  # Treatment: did this member's state-party block contain an exogenous death?
  d_cg <- deaths[congress == cg]
  tr <- as.integer(paste(ego$state, ego$party) %in% paste(d_cg$state, d_cg$party))
  if (sum(tr) == 0 || sum(tr) == N) return(NULL)
  # Design: replicate the realised propensity by permuting tr within Congress.
  p_hat <- mean(tr)
  R_rep <- min(n_perm, choose(N, sum(tr)))
  pot_tr <- make_tr_vec_permutation(N = N, p = p_hat, R = R_rep, seed = 4224L)
  exposure   <- make_exposure_map_AS(adj, tr, hop = 1)
  prob_exp   <- make_exposure_prob(pot_tr, adj, make_exposure_map_AS,
                                   list(hop = 1))
  est <- tryCatch(
    estimates(obs_exposure = exposure,
              obs_outcome  = as.numeric(ego$departed),
              obs_prob_exposure = prob_exp,
              n_var_permutations = 30L,
              control_condition = "d00"),
    error = function(e) NULL)
  if (is.null(est)) return(NULL)
  list(congress = cg, N = N, n_treated = sum(tr),
       tau_ht = est$tau_ht, var_tau_ht = est$var_tau_ht,
       tau_h  = est$tau_h,  var_tau_h  = est$var_tau_h)
}

plan(multisession, workers = max(1, parallel::detectCores() - 1))
congs <- sort(unique(panel$congress))
res <- future_lapply(congs, run_congress, future.seed = TRUE)
res <- Filter(Negate(is.null), res)

pool_ht <- function(taus, vars) {
  w <- 1 / vars; w <- w / sum(w)
  list(est = sum(w * taus), se = sqrt(1 / sum(1 / vars)))
}
pick <- function(x, k) if (!is.null(x) && k %in% names(x)) as.numeric(x[[k]]) else NA_real_
contrasts <- unique(unlist(lapply(res, function(x) names(x$tau_ht))))
pooled <- lapply(contrasts, function(k) {
  taus <- vapply(res, function(x) pick(x$tau_ht, k), numeric(1))
  vars <- vapply(res, function(x) pick(x$var_tau_ht, k), numeric(1))
  ok <- is.finite(taus) & is.finite(vars) & vars > 0
  if (sum(ok) < 2) return(NULL)
  c(contrast = k, pool_ht(taus[ok], vars[ok]), n_congresses = sum(ok))
})

writeLines(toJSON(list(per_congress = res, pooled = pooled,
                       n_permutations = n_perm, package = "interference",
                       method = "Aronow-Samii 2017 AOAS, hop=1"),
                  auto_unbox = TRUE, pretty = TRUE, null = "null"), out_json)
cat("wrote", out_json, "\n")
