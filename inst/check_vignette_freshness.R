## Vignette freshness check: the precomputed vignettes/timestralR.Rmd must
## contain exactly the echoed R code chunks of
## vignettes/timestralR.Rmd.orig. Fails when the .orig is edited without
## re-running vignettes/precompile.R, or when the generated .Rmd is
## edited by hand. Run from the package root:
##   Rscript inst/check_vignette_freshness.R

orig <- file.path("vignettes", "timestralR.Rmd.orig")
gen <- file.path("vignettes", "timestralR.Rmd")
if (!file.exists(orig) || !file.exists(gen))
  stop("run from the package root (vignettes not found)")

.chunks <- function(path, source) {
  lines <- readLines(path, warn = FALSE)
  out <- character(0)
  in_chunk <- FALSE
  keep <- TRUE
  for (ln in lines) {
    if (!in_chunk && grepl("^```\\{r([ ,}]|$)", ln)) {
      in_chunk <- TRUE
      keep <- !grepl("(include|echo)\\s*=\\s*FALSE", ln)
      next
    }
    if (!in_chunk && !source && grepl("^```\\s*\\{?r([ ,}]|$)", ln)) {
      in_chunk <- TRUE
      keep <- TRUE
      next
    }
    if (in_chunk && grepl("^```\\s*$", ln)) { in_chunk <- FALSE; next }
    if (in_chunk && keep) out <- c(out, ln)
  }
  out
}

norm <- function(x) trimws(x[nzchar(trimws(x))], which = "right")
code_orig <- norm(.chunks(orig, source = TRUE))
## knitr collapses chunk output into the same fence, prefixed with "#>"
code_gen <- norm(.chunks(gen, source = FALSE))
code_gen <- code_gen[!grepl("^#>", code_gen)]
if (!identical(code_orig, code_gen))
  stop("vignettes/timestralR.Rmd is stale: its code chunks differ from ",
       "vignettes/timestralR.Rmd.orig; re-run vignettes/precompile.R")
cat("vignette is fresh\n")
