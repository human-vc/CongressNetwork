## iv_redistricting.R
## Panel IV: redistricting cycle (x gerrymander severity) -> competitiveness -> BLI
## fixest::feols for fast 2SLS with member FE; ivDiag for Lal et al. (2023) audits
## including Mellon (2025) "rain rain go away"-style exclusion-restriction sweep.
##
## Install:
##   install.packages(c("fixest","ivDiag","data.table","jsonlite"))
##   # dev: remotes::install_github("apoorvalal/ivDiag")
## Run:
##   Rscript src/iv_redistricting.R data/panel.csv out/iv.json

suppressPackageStartupMessages({
  library(fixest); library(ivDiag); library(data.table); library(jsonlite)
})

args      <- commandArgs(trailingOnly = TRUE)
data_path <- ifelse(length(args) >= 1, args[1], "data/panel.csv")
out_path  <- ifelse(length(args) >= 2, args[2], "out/iv.json")

d <- fread(data_path)
## Required columns: member_id, state, congress, bli, competitiveness,
##   post_redistrict (0/1), gerry_severity (continuous), plus optional audit vars
d <- d[!is.na(bli) & !is.na(competitiveness) & !is.na(post_redistrict)]
d[, iv_main := post_redistrict * gerry_severity]   # interaction instrument

setFixest_nthreads(parallel::detectCores() - 2)    # fixest auto-parallel

## --- First stage diagnostic ---
fs <- feols(competitiveness ~ iv_main + post_redistrict + gerry_severity |
              member_id + congress, data = d, cluster = ~state)

## --- 2SLS with member + congress FE, clustered by state ---
iv <- feols(bli ~ 1 | member_id + congress |
              competitiveness ~ iv_main,
            data = d, cluster = ~state)

## --- State-cluster vs district-cluster comparison (Lal et al. recommend
##     clustering at the level of instrument assignment = state here). ---
iv_dist <- feols(bli ~ 1 | member_id + congress |
                   competitiveness ~ iv_main,
                 data = d, cluster = ~district_id)

eff_fstat <- fitstat(iv, "ivf1", simplify = TRUE)
weak_iv   <- as.numeric(eff_fstat) < 10

## --- Lal et al. (2023) ivDiag audit: AR test, tF, bootstrap, ZFR ---
controls <- c("gerry_severity", "post_redistrict")
diag_out <- tryCatch(
  ivDiag(data = as.data.frame(d), Y = "bli", D = "competitiveness",
         Z = "iv_main", controls = controls,
         FE = c("member_id", "congress"), cl = "state",
         bootstrap = TRUE),
  error = function(e) list(error = conditionMessage(e)))

## --- Mellon "rain rain go away" audit: regress instrument on candidate
##     channel variables. Any |t|>2 flags a potential exclusion violation. ---
audit_vars <- intersect(
  c("incumbent", "seniority", "log_pop", "pct_urban", "pct_college",
    "median_income", "minority_share", "prior_turnout", "ideology"),
  names(d))
audit <- rbindlist(lapply(audit_vars, function(v) {
  f <- as.formula(sprintf("%s ~ iv_main | member_id + congress", v))
  m <- tryCatch(feols(f, data = d, cluster = ~state), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  s <- summary(m)$coeftable["iv_main", , drop = FALSE]
  data.table(variable = v, coef = s[1,1], se = s[1,2],
             t = s[1,3], p = s[1,4], flag = abs(s[1,3]) > 2)
}))

## --- Sanity checks ---
warn <- c()
if (weak_iv) warn <- c(warn, sprintf("Effective F = %.2f < 10; weak instrument", eff_fstat))
if (any(audit$flag, na.rm = TRUE))
  warn <- c(warn, "Exclusion audit: instrument predicts other channels; see audit table")
if (nrow(d) < 1000) warn <- c(warn, "N < 1000; precision-limited")

co <- summary(iv)$coeftable["fit_competitiveness", ]
out <- list(
  analysis = "Panel IV: post-redistrict x gerrymander -> competitiveness -> BLI",
  iv_state_cluster = list(
    estimate = co[1], se = co[2],
    ci_lower = co[1] - 1.96 * co[2], ci_upper = co[1] + 1.96 * co[2],
    p_value  = co[4], first_stage_F = as.numeric(eff_fstat),
    n_obs    = iv$nobs, cluster = "state"),
  iv_district_cluster = list(
    estimate = summary(iv_dist)$coeftable["fit_competitiveness", 1],
    se       = summary(iv_dist)$coeftable["fit_competitiveness", 2]),
  exclusion_audit = audit,
  ivDiag         = diag_out,
  warnings       = warn
)
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_json(out, out_path, pretty = TRUE, auto_unbox = TRUE, digits = 6)
cat("IV done. beta =", round(co[1], 4), " F =", round(as.numeric(eff_fstat), 2), "\n")
