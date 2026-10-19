## Dev-only parity for synthetic config variants (bias on/off, no gates,
## no asinh, alternate schedules). Generate the variant checkpoints first:
##   .venv/bin/python inst/make_fixtures.py
## Then: Rscript inst/synthetic_parity.R   (from the package root)
library(timestralR)
library(jsonlite)

python <- Sys.getenv("TIMESTRALR_PYTHON", ".venv/bin/python")
y <- scan(file.path("tests", "fixtures", "input_univariate.csv"),
          sep = ",", quiet = TRUE)
for (v in c("bias", "nogate", "noasinh")) {
  dir <- sprintf("/tmp/timestral-synth-%s", v)
  stopifnot(dir.exists(dir))
  m <- timestral_load(dir, max_context = 4096L)
  fc <- timestral_forecast(m, y, h = 24)
  tg <- timestralR:::.as_batch(y)
  write_json(list(target = as.vector(tg$arr), target_dim = dim(tg$arr),
                  h = 24, sym = TRUE),
             "case.json", auto_unbox = TRUE, digits = NA, na = "null", null = "null")
  stopifnot(system2(python, c(file.path("inst", "official_reference.py"),
                              dir, "case.json", "off.txt")) == 0)
  off <- scan("off.txt", quiet = TRUE)
  mr <- max(abs(as.vector(array(fc$quantiles, c(1, dim(fc$quantiles)))) - off)) /
    max(abs(off))
  cat(v, "max_rel:", signif(mr, 3), "\n")
  stopifnot(mr < 1e-6)
}
cat("synthetic variants OK\n")
