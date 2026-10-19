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

## native and R backends agree on the fixtures
m_R <- timestral_load(file.path(fx, "tiny"), max_context = 4096L, backend = "R")
fc_R <- timestral_forecast(m_R, y, h = 24)
stopifnot(max(abs(fc_R$quantiles - fc$quantiles)) < 1e-10)
fc2_R <- timestral_forecast(m_R, y2, h = 24, past_covariates = matrix(pa, ncol = 1),
                            known_covariates = matrix(kn, ncol = 1))
stopifnot(max(abs(fc2_R$quantiles - fc2$quantiles)) < 1e-10)

## multi-block decode (native K/V cache across blocks) agrees with R;
## the second variant truncates the context mid-decode, forcing cache misses
m_mb <- timestral_load(file.path(fx, "tiny"), max_context = 4096L,
                       per_shot_horizon_patches = 1L)
m_mb_R <- timestral_load(file.path(fx, "tiny"), max_context = 4096L,
                         per_shot_horizon_patches = 1L, backend = "R")
fc_mb <- timestral_forecast(m_mb, y, h = 48)
fc_mb_R <- timestral_forecast(m_mb_R, y, h = 48)
stopifnot(max(abs(fc_mb_R$quantiles - fc_mb$quantiles)) < 1e-10)
m_tr <- timestral_load(file.path(fx, "tiny"), max_context = 32L,
                       per_shot_horizon_patches = 1L)
m_tr_R <- timestral_load(file.path(fx, "tiny"), max_context = 32L,
                         per_shot_horizon_patches = 1L, backend = "R")
fc_tr <- timestral_forecast(m_tr, y, h = 48)
fc_tr_R <- timestral_forecast(m_tr_R, y, h = 48)
stopifnot(max(abs(fc_tr_R$quantiles - fc_tr$quantiles)) < 1e-10)

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

## ts input: index preserved, accessors return ts, plot runs
y_ts <- stats::ts(y, start = c(2020, 1), frequency = 12)
fc_ts <- timestral_forecast(m, y_ts, h = 24)
stopifnot(isTRUE(all.equal(fc_ts$index, as.numeric(stats::time(y_ts)))))
stopifnot(abs(fc_ts$future_index[1] - (fc_ts$tsp[2] + 1 / 12)) < 1e-8)
med_ts <- timestral_median(fc_ts)
stopifnot(stats::is.ts(med_ts), stats::frequency(med_ts) == 12)
stopifnot(abs(as.numeric(stats::time(med_ts))[1] -
              (fc_ts$tsp[2] + 1 / 12)) < 1e-8)
stopifnot(max(abs(as.numeric(med_ts) -
                  fc_ts$quantiles[1, , "0.50"])) < 1e-12)
grDevices::pdf(NULL)
plot(fc_ts)
plot(fc)
grDevices::dev.off()

## zoo/xts input (optional packages): index and future index preserved,
## misaligned covariate indexes are rejected
if (requireNamespace("zoo", quietly = TRUE)) {
  idx <- as.Date("2021-01-01") + 0:(length(y) - 1)
  yz <- zoo::zoo(y, order.by = idx)
  fc_z <- timestral_forecast(m, yz, h = 12)
  stopifnot(identical(fc_z$index, idx))
  stopifnot(identical(fc_z$future_index, idx[length(idx)] + 1:12))
  dfz <- as_forecast_df(fc_z)
  stopifnot(inherits(dfz$time, "Date"), dfz$time[1] == idx[length(idx)] + 1)
  pz <- zoo::zoo(matrix(seq_along(y), ncol = 1), order.by = idx)
  fc_zc <- timestral_forecast(m, yz, h = 12, past_covariates = pz)
  stopifnot(all(is.finite(fc_zc$quantiles)))
  pz_bad <- zoo::zoo(matrix(seq_along(y), ncol = 1), order.by = idx + 1)
  err <- tryCatch({
    timestral_forecast(m, yz, h = 12, past_covariates = pz_bad); FALSE
  }, error = function(e) grepl("align", conditionMessage(e)))
  stopifnot(isTRUE(err))
  grDevices::pdf(NULL); plot(fc_z); grDevices::dev.off()
}
if (requireNamespace("xts", quietly = TRUE)) {
  idx <- as.Date("2021-01-01") + 0:(length(y) - 1)
  yx <- xts::xts(y, order.by = idx)
  fc_x <- timestral_forecast(m, yx, h = 12)
  stopifnot(identical(fc_x$future_index, idx[length(idx)] + 1:12))
}

## tsibble input/output (optional package)
if (requireNamespace("tsibble", quietly = TRUE)) {
  idx <- as.Date("2021-01-01") + 0:(length(y) - 1)
  d <- tsibble::as_tsibble(
    data.frame(t = idx, value = y), index = t)
  fc_tb <- timestral_forecast(m, d, h = 12)
  fc_ref <- timestral_forecast(m, y, h = 12)
  stopifnot(max(abs(fc_tb$quantiles - fc_ref$quantiles)) < 1e-12)
  stopifnot(identical(fc_tb$index_var, "t"))
  ft <- tsibble::as_tsibble(fc_tb)
  stopifnot(inherits(ft, "tbl_ts"),
            identical(tsibble::index_var(ft), "t"), nrow(ft) == 12L)
  d2 <- tsibble::as_tsibble(
    data.frame(grp = rep(c("a", "b"), each = length(y)), t = rep(idx, 2),
               v1 = c(y, rev(y)), v2 = c(rev(y), y)),
    index = t, key = grp)
  fc_tb2 <- timestral_forecast(m, d2, h = 12)
  stopifnot(identical(dim(fc_tb2$quantiles), c(2L, 2L, 12L, 21L)))
}

## ggplot2 methods (optional package)
if (requireNamespace("ggplot2", quietly = TRUE)) {
  fc_gg <- timestral_forecast(m, y, h = 12)
  p <- ggplot2::autoplot(fc_gg)
  stopifnot(inherits(p, "ggplot"))
  invisible(ggplot2::ggplot_build(p))
  stopifnot(is.list(ggplot2::autolayer(fc_gg)))
}

## fable model + as_fable (optional packages)
if (requireNamespace("fable", quietly = TRUE) &&
    requireNamespace("tsibble", quietly = TRUE)) {
  suppressMessages(library(fable))
  train_ts <- tsibble::as_tsibble(
    data.frame(t = as.Date("2021-01-01") + 0:(length(y) - 1), value = y),
    index = t)
  fit <- fabletools::model(train_ts, tm = TIMESTRAL(value, model = m))
  fcf <- fabletools::forecast(fit, h = 6)
  stopifnot(fabletools::is_fable(fcf), all(is.finite(fcf$.mean)))
  fb <- fabletools::as_fable(timestral_forecast(m, y_ts, h = 6))
  stopifnot(fabletools::is_fable(fb))
}
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
