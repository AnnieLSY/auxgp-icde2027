#!/usr/bin/env Rscript

required <- c(BGLR = "1.1.4", rrBLUP = "4.6.3", truncnorm = "1.0.9", MASS = "7.3.66")
repos <- c(CRAN = "https://cloud.r-project.org")
if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv", repos = repos)
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("run this file with Rscript")
script_path <- sub("^--file=", "", script_arg)
lockfile <- file.path(dirname(normalizePath(script_path)), "renv.lock")
renv::restore(lockfile = lockfile, prompt = FALSE)
for (package in names(required)) {
  actual <- as.character(utils::packageVersion(package))
  if (actual != required[[package]]) stop(sprintf("%s version %s is installed; tested lock requires %s", package, actual, required[[package]]))
}
cat(sprintf("R %s\n", getRversion()))
for (package in names(required)) cat(sprintf("%s %s\n", package, utils::packageVersion(package)))
