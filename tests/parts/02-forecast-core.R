## Batching, accessors, error validation, covariates, masks, ts/plot.
## Sourced by tests/timestralR-tests.R; expects `m`, `m_R`, `y`, `y2`, `fc`.
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
## a list of ts sharing one index keeps it; mixed indexes drop it
yb_ts <- lapply(yb, stats::ts, start = c(2000, 1), frequency = 12)
fcb_ts <- timestral_forecast(m, yb_ts, h = 6)
stopifnot(identical(fcb_ts$index, as.numeric(stats::time(yb_ts[[1]]))),
          identical(fcb_ts$tsp, stats::tsp(yb_ts[[1]])),
          max(abs(fcb_ts$quantiles - fcb$quantiles[, , 1:6, , drop = FALSE])) < 1e-12)
yb_ts[[2]] <- stats::ts(yb[[2]], start = c(2001, 1), frequency = 12)
stopifnot(is.null(timestral_forecast(m, yb_ts, h = 6)$index))
## named lists name the series but leave dim() unnamed
fcb_named <- timestral_forecast(m, list(a = y, b = y2), h = 6)
stopifnot(is.null(names(dim(fcb_named$quantiles))),
          identical(dimnames(fcb_named$quantiles)$series, c("a", "b")))

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
