#!/usr/bin/env Rscript
# One-time installation of every R package used by the analysis.
# Run once on Brev before any other R script:
#     Rscript replication/scripts/install_r_packages.R
#
# Uses Posit Public Package Manager (PPM), which serves pre-compiled binary
# R packages for Ubuntu. This avoids the from-source compilation that fails
# under parallel install when dependencies build out of order.

# --- Binary repo: PPM serves Linux binaries when the User-Agent reports the
#     R version + platform. Without this header PPM falls back to source.
options(
  repos = c(PPM = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"),
  HTTPUserAgent = sprintf(
    "R/%s R (%s)",
    getRversion(),
    paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])
  ),
  Ncpus = max(1, parallel::detectCores() - 1)
)

# pak resolves the full dependency DAG and installs in correct order,
# preferring binaries. Far more reliable than bare install.packages().
if (!requireNamespace("pak", quietly = TRUE)) {
  install.packages("pak")
}

cran_pkgs <- c(
  # DiD comparator (task #5)
  "DIDmultiplegtDYN", "didimputation", "fect", "triplediff",
  # Placebo + equivalence + specification curve (#11, #12)
  "specr", "multiverse", "future", "future.apply", "furrr",
  # RDiT + IV + sdid (#13, #14, #15)
  "rdrobust", "rdmulti", "AER", "fixest", "ivDiag", "synthdid",
  # Mediation + interaction (#21, #22, #18)
  "causalweight", "interflex", "mediation",
  # Bayes factor + GEE diagnostics (#24)
  "brms", "bridgesampling", "BayesFactor", "glmtoolbox", "geepack",
  # Figures (#20)
  "ggalluvial", "ggplot2",
  # I/O + utilities
  "jsonlite", "data.table", "tidyverse", "remotes", "pacman", "devtools"
)

github_pkgs <- c(
  "szonszein/interference"
)

# pak handles CRAN + GitHub in one call, resolving the joint dependency graph.
all_refs <- c(cran_pkgs, paste0("github::", github_pkgs))

message("Installing ", length(all_refs), " packages via pak (binary where available)")
pak::pkg_install(all_refs, ask = FALSE, upgrade = FALSE)

# Verify everything loads
check_pkgs <- c(cran_pkgs, "interference")
ok <- vapply(check_pkgs, function(p) requireNamespace(p, quietly = TRUE), logical(1))
cat("\nInstall summary:\n")
print(data.frame(package = check_pkgs, loaded = ok))

if (!all(ok)) {
  failed <- check_pkgs[!ok]
  message("FAILED: ", paste(failed, collapse = ", "))
  message("Retrying failed packages individually from source...")
  for (p in failed) {
    tryCatch(
      pak::pkg_install(p, ask = FALSE, upgrade = FALSE),
      error = function(e) message("  still failing: ", p, " -- ", conditionMessage(e))
    )
  }
  ok2 <- vapply(check_pkgs, function(p) requireNamespace(p, quietly = TRUE), logical(1))
  still_failed <- check_pkgs[!ok2]
  if (length(still_failed) > 0) {
    message("PERSISTENT FAILURES: ", paste(still_failed, collapse = ", "))
    # Non-fatal: the pipeline degrades gracefully (Stan-based Bayes factor is
    # optional; GEE diagnostics work without it). Exit 0 so run_all.sh continues.
    quit(status = 0)
  }
}
message("All ", length(check_pkgs), " R packages installed and load cleanly.")
