## Package tests. Forecast tests need a local checkpoint; they are skipped
## (with a message) unless TIMESTRALR_TEST_MODEL_DIR points at a directory
## holding config.json + model.safetensors.
library(timestralR)

stopifnot(identical(length(timestral_levels()), 21L))
stopifnot(identical(timestral_levels()[11], 0.5))

model_dir <- Sys.getenv("TIMESTRALR_TEST_MODEL_DIR")
if (!nzchar(model_dir) || !file.exists(file.path(model_dir, "model.safetensors"))) {
  message("skipping forecast tests: TIMESTRALR_TEST_MODEL_DIR not set")
} else {
  m <- timestral_load(model_dir)
  set.seed(1)
  y <- 10 + 3 * sin(2 * pi * (1:600) / 24)
  fc <- timestral_forecast(m, y, h = 50)
  stopifnot(identical(dim(fc$quantiles), c(1L, 50L, 21L)))
  stopifnot(all(apply(fc$quantiles[1, , ], 1, function(r) all(diff(r) >= -1e-6))))
  stopifnot(mean(abs(fc$quantiles[1, , "0.50"] -
                     (10 + 3 * sin(2 * pi * (601:650) / 24)))) < 0.5)
  df <- as_forecast_df(fc)
  stopifnot(nrow(df) == 50L, "q50" %in% names(df))

  err <- tryCatch({
    timestral_forecast(m, rnorm(100), 10, known_covariates = matrix(0, 100, 1))
    FALSE
  }, error = function(e) grepl("rows", conditionMessage(e)))
  stopifnot(isTRUE(err))
  cat("forecast tests passed\n")
}
