#!/usr/bin/env Rscript
# One-time installation of every R package used by the analysis.
# Run once on Brev before any other R script:
#     sudo Rscript replication/scripts/install_r_packages.R
#
# Uses Posit Public Package Manager (PPM), which serves pre-compiled binary
# R packages for Ubuntu jammy. Binaries install in seconds and resolve
# dependencies naturally, avoiding the from-source compile failures seen
# when building under parallel install.

# --- PPM binary repo. The HTTPUserAgent header is what makes PPM serve
#     Linux binaries instead of falling back to source tarballs.
options(
  repos = c(PPM = "https://packagemanager.posit.co/cran/__linux__/jammy/latest"),
  HTTPUserAgent = sprintf(
    "R/%s R (%s)",
    getRversion(),
    paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"])
  ),
  Ncpus = max(1, parallel::detectCores() - 1)
)

# --- Clean slate: earlier failed attempts left a half-built site-library that
#     poisons the dependency solver. Wipe everything except base/recommended.
site_lib <- .Library.site[1]
if (is.na(site_lib)) site_lib <- file.path(R.home(), "site-library")
if (dir.exists(site_lib)) {
  base_pkgs <- rownames(installed.packages(priority = c("base", "recommended")))
  existing <- list.dirs(site_lib, recursive = FALSE, full.names = FALSE)
  to_remove <- setdiff(existing, base_pkgs)
  if (length(to_remove) > 0) {
    message("Wiping ", length(to_remove), " stale packages from ", site_lib)
    unlink(file.path(site_lib, to_remove), recursive = TRUE, force = TRUE)
  }
}

cran_pkgs <- c(
  # bootstrap
  "remotes", "pacman", "devtools",
  # DiD comparator (task #5)
  "DIDmultiplegtDYN", "didimputation", "fect", "triplediff", "HonestDiD",
  # Placebo + equivalence + specification curve (#11, #12)
  "specr", "multiverse", "future", "future.apply", "furrr",
  # RDiT + IV + sdid (#13, #14, #15)
  "rdrobust", "rdmulti", "AER", "fixest", "ivDiag",
  # Mediation + interaction (#21, #22, #18)
  "causalweight", "interflex", "mediation",
  # Bayes factor + GEE diagnostics (#24)
  "brms", "bridgesampling", "BayesFactor", "glmtoolbox", "geepack",
  # Figures (#20)
  "ggalluvial", "ggplot2",
  # I/O + utilities
  "jsonlite", "data.table", "tidyverse"
)

# GitHub-only packages (not on CRAN/PPM)
github_pkgs <- c(
  "synth-inference/synthdid",   # synthetic DiD (task #15)
  "szonszein/interference"      # Aronow-Samii exposure mapping (task #21)
)

message("Installing ", length(cran_pkgs), " CRAN packages from PPM binary repo")
install.packages(cran_pkgs, dependencies = TRUE)

message("Installing ", length(github_pkgs), " GitHub packages")
for (gp in github_pkgs) {
  tryCatch(
    remotes::install_github(gp, upgrade = "never", dependencies = TRUE),
    error = function(e) message("  github install failed: ", gp, " -- ", conditionMessage(e))
  )
}

# Verify everything loads
check_pkgs <- c(
  setdiff(cran_pkgs, c("remotes", "pacman", "devtools")),
  "synthdid", "interference"
)
ok <- vapply(check_pkgs, function(p) requireNamespace(p, quietly = TRUE), logical(1))
cat("\nInstall summary:\n")
print(data.frame(package = check_pkgs, loaded = ok))

failed <- check_pkgs[!ok]
if (length(failed) > 0) {
  message("\nRetrying ", length(failed), " failed packages individually...")
  for (p in failed) {
    is_gh <- p %in% c("synthdid", "interference")
    tryCatch({
      if (is_gh) {
        gp <- github_pkgs[grepl(paste0("/", p, "$"), github_pkgs)]
        remotes::install_github(gp, upgrade = "never", dependencies = TRUE)
      } else {
        install.packages(p, dependencies = TRUE)
      }
    }, error = function(e) message("  still failing: ", p, " -- ", conditionMessage(e)))
  }
  ok2 <- vapply(check_pkgs, function(p) requireNamespace(p, quietly = TRUE), logical(1))
  still_failed <- check_pkgs[!ok2]
  if (length(still_failed) > 0) {
    message("\nPERSISTENT FAILURES: ", paste(still_failed, collapse = ", "))
    # Non-fatal: pipeline degrades gracefully. Exit 0 so run_all.sh continues.
    quit(status = 0)
  }
}
message("\nAll ", length(check_pkgs), " R packages installed and load cleanly.")
