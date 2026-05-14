# Alluvial plot of legislator coalition transitions across Congresses.
# Reads results/coalition_transitions.csv produced by alluvial_data_prep.py
# and writes results/figures/alluvial_coalitions.pdf.

suppressPackageStartupMessages({
  library(ggplot2)
  library(ggalluvial)
  library(dplyr)
})

find_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  if (!is.null(sys.frames()) && length(sys.frames()) > 0) {
    of <- sys.frame(1)$ofile
    if (!is.null(of)) return(dirname(normalizePath(of)))
  }
  getwd()
}
script_dir <- find_script_dir()
root <- normalizePath(file.path(script_dir, ".."))
csv_path  <- file.path(root, "results", "coalition_transitions.csv")
out_path  <- file.path(root, "results", "figures", "alluvial_coalitions.pdf")

dat <- read.csv(csv_path, stringsAsFactors = FALSE)

stratum_levels <- c("Dem-Bridge", "Dem-Other",
                    "Out-of-house",
                    "Rep-Other", "Rep-Bridge")
fill_palette <- c(
  "Dem-Bridge"   = "#2166ac",
  "Dem-Other"    = "#92c5de",
  "Out-of-house" = "#bdbdbd",
  "Rep-Other"    = "#f4a582",
  "Rep-Bridge"   = "#b2182b"
)

dat$coalition <- factor(dat$coalition, levels = stratum_levels)
dat$congress  <- factor(dat$congress,
                        levels = sort(unique(dat$congress)),
                        labels = paste0(sort(unique(dat$congress)), "th"))

# Drop legislators who are Out-of-house in every panel (no information).
keep_ids <- dat %>%
  group_by(icpsr) %>%
  summarise(any_in = any(coalition != "Out-of-house"), .groups = "drop") %>%
  filter(any_in) %>%
  pull(icpsr)
dat <- dat %>% filter(icpsr %in% keep_ids)

p <- ggplot(dat, aes(x = congress, stratum = coalition,
                     alluvium = icpsr, fill = coalition)) +
  geom_flow(aes.bind = "flows", alpha = 0.55, width = 1/6,
            color = "white", linewidth = 0.05) +
  geom_stratum(width = 1/6, color = "black", linewidth = 0.2) +
  scale_fill_manual(values = fill_palette, name = "Coalition") +
  scale_x_discrete(expand = c(0.04, 0.04)) +
  labs(x = "Congress", y = "Number of legislators",
       title = "Coalition transitions across Congresses",
       subtitle = paste("Bridge = top 25% of |BLI| within party.",
                        "Out-of-house = not in chamber that Congress.")) +
  theme_minimal(base_size = 10) +
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor   = element_blank(),
        plot.title         = element_text(face = "bold"),
        plot.subtitle      = element_text(size = 8, color = "grey30"),
        legend.position    = "right")

ggsave(out_path, p, width = 8, height = 5, device = "pdf",
       useDingbats = FALSE)
cat(sprintf("  wrote %s\n", basename(out_path)))
