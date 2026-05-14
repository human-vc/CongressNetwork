#!/usr/bin/env Rscript
# One-time installation of every R package used by the analysis.
# Run once on Brev before any other R script:
#     Rscript replication/scripts/install_r_packages.R

options(repos = c(CRAN = "https://cloud.r-project.org"))

cran_pkgs <- c(
  # DiD comparator (task #5)
  "DIDmultiplegtDYN",
  "didimputation",
  "fect",
  "triplediff",

  # Placebo + equivalence + specification curve (#11, #12)
  "specr",
  "multiverse",
  "future", "future.apply", "furrr",

  # RDiT + IV + sdid (#13, #14, #15)
  "rdrobust",
  "rdmulti",
  "AER",
  "fixest",
  "ivDiag",
  "synthdid",

  # Mediation + interaction (#21, #22, #18)
  "causalweight",
  "interflex",
  "mediation",

  # Bayes factor + GEE diagnostics (#24)
  "brms",
  "bridgesampling",
  "BayesFactor",
  "glmtoolbox",
  "geepack",

  # Figures (#20)
  "ggalluvial",
  "ggplot2",

  # I/O + utilities
  "jsonlite",
  "data.table",
  "tidyverse",
  "remotes",
  "pacman",
  "devtools"
)

missing <- setdiff(cran_pkgs, rownames(installed.packages()))
if (length(missing) > 0) {
  message("Installing ", length(missing), " CRAN packages: ",
          paste(missing, collapse = ", "))
  install.packages(missing, Ncpus = max(1, parallel::detectCores() - 1))
}

# GitHub-only packages
if (!requireNamespace("interference", quietly = TRUE)) {
  remotes::install_github("szonszein/interference", upgrade = "never")
}

# Verify everything loads
all_pkgs <- c(cran_pkgs, "interference")
ok <- vapply(all_pkgs, function(p) requireNamespace(p, quietly = TRUE), logical(1))
cat("\nInstall summary:\n")
print(data.frame(package = all_pkgs, loaded = ok))

if (!all(ok)) {
  failed <- all_pkgs[!ok]
  message("FAILED: ", paste(failed, collapse = ", "))
  quit(status = 1)
}
message("All ", length(all_pkgs), " R packages installed and load cleanly.")
