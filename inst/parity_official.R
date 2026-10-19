## Dev-only parity check against the official PyTorch package.
## Requires: a local checkpoint dir, Python + torch + timestral on the dev
## machine only (end users never need Python). Set before running:
##   TIMESTRALR_MODEL_DIR  directory with config.json + model.safetensors
##   TIMESTRALR_PYTHON     python executable (default: .venv/bin/python)
library(timestralR)
library(jsonlite)

model_dir <- Sys.getenv("TIMESTRALR_MODEL_DIR", "/tmp/timestral-5m")
python <- Sys.getenv("TIMESTRALR_PYTHON", ".venv/bin/python")
stopifnot(file.exists(file.path(model_dir, "model.safetensors")),
          file.exists(python))

source_cases <- function() {
  set.seed(42); n <- 700; tt <- 1:n; cases <- list()
  y1 <- 50 + 10 * sin(2 * pi * tt / 24) + cumsum(rnorm(n)); y1[c(5:40, 300, 301, 650)] <- NA
  cases$univariate_missing <- list(y = y1, h = 96)
  cases$negative_trend_nosym <- list(y = -5 + 0.02 * tt + rnorm(n), h = 30, sym = FALSE)
  cases$multivariate <- list(y = cbind(a = 20 + 5 * sin(2 * pi * tt / 48) + rnorm(n),
                                       b = 5 + 2 * cos(2 * pi * tt / 48) + rnorm(n)), h = 64)
  temp <- sin(2 * pi * (1:(n + 96)) / 24); promo <- as.numeric((1:(n + 96)) %% 7 == 0)
  cases$covariates <- list(y = 100 + 20 * temp[1:n] + 30 * promo[1:n] + rnorm(n), h = 96,
                           past = matrix(rnorm(n), ncol = 1), known = cbind(temp, promo))
  cases$long_horizon <- list(y = y1[200:700], h = 1100)
  cases$long_context_cap <- list(y = 10 + sin(2 * pi * (1:5000) / 50) + rnorm(5000, sd = .1),
                                 h = 2100)
  cases
}

cases <- source_cases()
m <- timestral_load(model_dir, max_context = 4096L)
rows <- function(x) if (is.null(x)) NULL else t(if (is.null(dim(x))) matrix(x, ncol = 1) else as.matrix(x))
res <- list()
ref_py <- system.file("official_reference.py", package = "timestralR")
if (!nzchar(ref_py)) ref_py <- file.path("inst", "official_reference.py")
for (nm in names(cases)) {
  cs <- cases[[nm]]
  sym <- if (is.null(cs$sym)) TRUE else cs$sym
  t0 <- Sys.time()
  fc <- timestral_forecast(m, cs$y, cs$h, cs$past, cs$known, symmetric = sym)
  t_r <- as.numeric(Sys.time() - t0, units = "secs")
  tg <- rows(cs$y); pa <- rows(cs$past); kn <- rows(cs$known)
  write_json(list(target = as.vector(tg), target_dim = dim(tg), h = cs$h, sym = sym,
                  past = if (!is.null(pa)) as.vector(pa), past_dim = dim(pa),
                  known = if (!is.null(kn)) as.vector(kn), known_dim = dim(kn)),
             "case.json", auto_unbox = TRUE, digits = NA, na = "null", null = "null")
  stopifnot(system2(python, c(ref_py, model_dir, "case.json", "off.txt")) == 0)
  off <- array(scan("off.txt", quiet = TRUE), dim(fc$quantiles))
  d <- abs(fc$quantiles - off)
  scale <- max(abs(off))
  res[[nm]] <- data.frame(case = nm, max_rel = signif(max(d) / scale, 3),
                          median_rel = signif(max(abs(fc$quantiles[, , "0.50"] - off[, , 11])) / scale, 3),
                          r_secs = round(t_r, 2))
  print(tail(res, 1)[[1]], row.names = FALSE)
}
out <- do.call(rbind, res)
cat("\nWorst max_rel:", max(out$max_rel), "\n")
