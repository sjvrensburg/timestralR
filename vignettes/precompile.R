## Precompile the vignette: run the model code in timestralR.Rmd.orig and
## write timestralR.Rmd (static code, output and figures) for R CMD build.
## Run from the package root after installing the package:
##   Rscript vignettes/precompile.R
## Needs the 28M checkpoint (downloaded on first use) and the packages the
## vignette uses (ggplot2, patchwork, zoo, tsibble, dplyr, fable).
## Install from a clean src/ first (roxygen2/pkgload leave -O0 objects
## there, which would make the timings in section 8 misleading).
old <- setwd("vignettes")
unlink("figures", recursive = TRUE)
knitr::knit("timestralR.Rmd.orig", "timestralR.Rmd")
setwd(old)
