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
  cat("full-checkpoint tests passed\n")
}
