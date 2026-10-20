## Dev-only: regenerate the committed official-parity fixture for the 5M
## checkpoint (tests/fixtures/input_5m_univariate.csv and
## golden_5m_univariate.csv), used by the golden test in
## tests/parts/05-full-checkpoint.R whenever the checkpoint under test
## is the 5M one. The input is the `univariate_missing` case of
## inst/parity_official.R; the golden output is produced by the official
## PyTorch package via inst/official_reference.py. Requires Python +
## torch + timestral on the dev machine only (end users never need
## Python). Run from the package root:
##   Rscript inst/make_5m_fixture.R
## TIMESTRALR_MODEL_DIR points at a local 5M checkpoint directory
## (default: the pinned download in the user cache).
library(timestralR)
library(jsonlite)

model_dir <- Sys.getenv("TIMESTRALR_MODEL_DIR")
if (!nzchar(model_dir))
  model_dir <- timestral_download("5M")
python <- Sys.getenv("TIMESTRALR_PYTHON", ".venv/bin/python")
stopifnot(file.exists(file.path(model_dir, "model.safetensors")),
          file.exists(python))

## The univariate_missing input, generated exactly as in
## inst/parity_official.R (same seed and construction).
set.seed(42); n <- 700; tt <- 1:n
y1 <- 50 + 10 * sin(2 * pi * tt / 24) + cumsum(rnorm(n))
y1[c(5:40, 300, 301, 650)] <- NA
h <- 96L

fx <- file.path("tests", "fixtures")
writeLines(formatC(y1, format = "g", digits = 17),
           file.path(fx, "input_5m_univariate.csv"))

tg <- timestralR:::.as_batch(y1)
case <- list(target = as.vector(tg$arr), target_dim = dim(tg$arr), h = h,
             sym = TRUE, per_shot = 32, cap_k = 1e4)
write_json(case, "case.json", auto_unbox = TRUE, digits = NA, na = "null",
           null = "null")
ref_py <- system.file("official_reference.py", package = "timestralR")
if (!nzchar(ref_py)) ref_py <- file.path("inst", "official_reference.py")
stopifnot(system2(python, c(ref_py, model_dir, "case.json", "off.txt")) == 0)
off <- scan("off.txt", quiet = TRUE)
unlink(c("case.json", "off.txt"))
stopifnot(length(off) == h * 21L)
gold <- matrix(off, nrow = h, ncol = 21L)
write.table(matrix(formatC(gold, format = "g", digits = 10), nrow = h),
            file.path(fx, "golden_5m_univariate.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)

## Report the agreement the test will assert (gate in the test: 2e-5,
## the parity gate of inst/parity_official.R).
m <- timestral_load(model_dir, max_context = 4096L)
fc <- timestral_forecast(m, y1, h = h)
rel <- max(abs(fc$quantiles[1, , ] - gold)) / max(abs(gold))
cat(sprintf("5M fixture written; native vs official max_rel = %.3g\n", rel))
