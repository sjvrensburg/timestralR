## Package tests. Golden-fixture tests always run against the tiny
## random-weight checkpoint in tests/fixtures (official-PyTorch outputs in
## golden_*.csv; regenerate with inst/make_fixtures.py). Full-checkpoint
## tests run only when TIMESTRALR_TEST_MODEL_DIR points at a real checkpoint.
library(timestralR)

stopifnot(identical(length(timestral_levels()), 21L))
stopifnot(identical(timestral_levels()[11], 0.5))

fx <- file.path("fixtures")
if (!dir.exists(fx)) fx <- file.path("tests", "fixtures")
stopifnot(dir.exists(fx))
m <- timestral_load(file.path(fx, "tiny"), max_context = 4096L)

## golden: univariate
y <- scan(file.path(fx, "input_univariate.csv"), sep = ",", quiet = TRUE)
fc <- timestral_forecast(m, y, h = 24)
gold <- as.matrix(read.csv(file.path(fx, "golden_univariate.csv"), header = FALSE))
stopifnot(identical(dim(fc$quantiles), c(1L, 24L, 21L)))
stopifnot(max(abs(fc$quantiles[1, , ] - gold)) < 1e-5)
stopifnot(all(apply(fc$quantiles[1, , ], 1, function(r) all(diff(r) >= -1e-6))))

## golden: covariates
y2 <- scan(file.path(fx, "input_cov_target.csv"), sep = ",", quiet = TRUE)
pa <- scan(file.path(fx, "input_cov_past.csv"), sep = ",", quiet = TRUE)
kn <- scan(file.path(fx, "input_known.csv"), sep = ",", quiet = TRUE)
fc2 <- timestral_forecast(m, y2, h = 24, past_covariates = matrix(pa, ncol = 1),
                          known_covariates = matrix(kn, ncol = 1))
gold2 <- as.matrix(read.csv(file.path(fx, "golden_covariates.csv"), header = FALSE))
stopifnot(max(abs(fc2$quantiles[1, , ] - gold2)) < 1e-5)

## batch consistency: batched series equal one-at-a-time forecasts
yb <- list(y, y2, rev(y))
fcb <- timestral_forecast(m, yb, h = 24)
stopifnot(identical(dim(fcb$quantiles), c(3L, 1L, 24L, 21L)))
for (b in 1:3) {
  f1 <- timestral_forecast(m, yb[[b]], h = 24)
  stopifnot(max(abs(fcb$quantiles[b, 1, , ] - f1$quantiles[1, , ])) < 1e-10)
}
dfb <- as_forecast_df(fcb)
stopifnot(nrow(dfb) == 3 * 24, all(c("series", "q50") %in% names(dfb)))

## accessors
stopifnot(max(abs(timestral_median(fc) - fc$quantiles[, , 11, drop = FALSE])) == 0)
stopifnot(max(abs(timestral_quantile(fc, 0.9) - fc$quantiles[, , 19, drop = FALSE])) == 0)
err <- tryCatch({ timestral_quantile(fc, 0.33); FALSE },
                error = function(e) grepl("not one of", conditionMessage(e)))
stopifnot(isTRUE(err))

## error parity: shape validation and covariate checks
err <- tryCatch({
  timestral_forecast(m, rnorm(100), 10, known_covariates = matrix(0, 100, 1))
  FALSE
}, error = function(e) grepl("rows", conditionMessage(e)))
stopifnot(isTRUE(err))
err <- tryCatch({
  timestral_forecast(m, rnorm(100), 10,
                     categorical_known_covariates = matrix(Inf, 110, 1))
  FALSE
}, error = function(e) grepl("finite codes", conditionMessage(e)))
stopifnot(isTRUE(err))
err <- tryCatch({ timestral_forecast(m, rnorm(100), 0); FALSE },
                error = function(e) grepl("must be >= 1", conditionMessage(e)))
stopifnot(isTRUE(err))

## categorical forecast runs and is finite
codes <- matrix(rep(1:7, length.out = 120), ncol = 1)
fcc <- timestral_forecast(m, y2, h = 24, categorical_past_covariates = codes[1:96, , drop = FALSE],
                          categorical_known_covariates = codes)
stopifnot(all(is.finite(fcc$quantiles)))

## observation mask runs
om <- rep(TRUE, length(y)); om[20:30] <- FALSE
fcm <- timestral_forecast(m, y, h = 24, observation_mask = om)
stopifnot(all(is.finite(fcm$quantiles)))
cat("fixture tests passed\n")

## optional full checkpoint
model_dir <- Sys.getenv("TIMESTRALR_TEST_MODEL_DIR")
if (!nzchar(model_dir) || !file.exists(file.path(model_dir, "model.safetensors"))) {
  message("skipping full-checkpoint tests: TIMESTRALR_TEST_MODEL_DIR not set")
} else {
  mr <- timestral_load(model_dir, max_context = 4096L)
  set.seed(1)
  yy <- 10 + 3 * sin(2 * pi * (1:600) / 24)
  fcr <- timestral_forecast(mr, yy, h = 50)
  stopifnot(identical(dim(fcr$quantiles), c(1L, 50L, 21L)))
  stopifnot(mean(abs(fcr$quantiles[1, , "0.50"] -
                     (10 + 3 * sin(2 * pi * (601:650) / 24)))) < 0.5)
  cat("full-checkpoint tests passed\n")
}
