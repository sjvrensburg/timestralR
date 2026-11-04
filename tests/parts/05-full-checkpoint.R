## Optional full-checkpoint tests, run only when TIMESTRALR_TEST_MODEL_DIR
## points at a real checkpoint; otherwise this part only messages the skip.
model_dir <- Sys.getenv("TIMESTRALR_TEST_MODEL_DIR")
if (!nzchar(model_dir)) {
  message("skipping full-checkpoint tests: TIMESTRALR_TEST_MODEL_DIR not set")
} else if (!file.exists(file.path(model_dir, "model.safetensors"))) {
  message("skipping full-checkpoint tests: no model.safetensors in ",
          "TIMESTRALR_TEST_MODEL_DIR (", model_dir, ")")
} else {
  mr <- timestral_load(model_dir, max_context = 4096L)
  ## Golden parity against the official PyTorch package, for the 5M
  ## checkpoint only (identified by its config): the committed fixture
  ## tests/fixtures/{input,golden}_5m_univariate.csv was generated once
  ## with inst/make_5m_fixture.R via inst/official_reference.py. The gate
  ## is the parity gate of inst/parity_official.R; the fixture measured
  ## 3.1e-07 when generated. Other sizes keep the plausibility test below.
  cfgj <- jsonlite::fromJSON(file.path(model_dir, "config.json"))
  if (cfgj$d_model == 192L && length(cfgj$block_schedule) == 8L) {
    y5 <- scan(file.path(fx, "input_5m_univariate.csv"), quiet = TRUE)
    gold5 <- as.matrix(read.csv(file.path(fx, "golden_5m_univariate.csv"),
                                header = FALSE))
    fc5 <- timestral_forecast(mr, y5, h = 96)
    stopifnot(identical(dim(fc5$quantiles), c(1L, 96L, 21L)))
    stopifnot(max(abs(fc5$quantiles[1, , ] - gold5)) / max(abs(gold5)) < 2e-5)
    cat("5M official-parity golden test passed\n")
  }
  set.seed(1)
  yy <- 10 + 3 * sin(2 * pi * (1:600) / 24)
  fcr <- timestral_forecast(mr, yy, h = 50)
  stopifnot(identical(dim(fcr$quantiles), c(1L, 50L, 21L)))
  stopifnot(mean(abs(fcr$quantiles[1, , "0.50"] -
                     (10 + 3 * sin(2 * pi * (601:650) / 24)))) < 0.5)
  ## long horizons: on random walks, median feedback collapses the 80% band
  ## past the first block and quantile paths keep it. Thresholds well inside
  ## the measured values on 5M, 28M and 209M.
  set.seed(42)
  walks <- replicate(4, cumsum(rnorm(512 + 2048)), simplify = FALSE)
  ctx <- lapply(walks, function(w) w[1:512])
  fut <- t(vapply(walks, function(w) w[512 + 1:2048], numeric(2048)))
  later <- 1025:2048
  cov80 <- function(q) {
    i <- which.min(abs(mr$config$quantile_knots - 0.1))
    j <- which.min(abs(mr$config$quantile_knots - 0.9))
    mean(fut[, later] >= q[, 1, later, i] & fut[, later] <= q[, 1, later, j])
  }
  ql_med <- timestral_forecast(mr, ctx, h = 2048, clip_nonnegative = FALSE)$quantiles
  ql_pth <- timestral_forecast(mr, ctx, h = 2048, clip_nonnegative = FALSE,
                               rollout = "quantile_paths")$quantiles
  stopifnot(cov80(ql_med) < 0.35, cov80(ql_pth) > 0.5,
            max(abs(ql_med[, , 1:1024, ] - ql_pth[, , 1:1024, ])) < 1e-9)
  cat("full-checkpoint tests passed\n")
}
