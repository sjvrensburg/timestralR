## Optional Suggests interfaces: zoo/xts, tsibble, ggplot2, fable, forecast.
## Each block skips cleanly when its package is unavailable locally;
## the runner fails in CI when an optional package is missing.
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
  pb <- ggplot2::ggplot_build(p)
  ## bands drawn widest first, legend narrowest first, graded shades
  fills <- unique(pb$data[[1]]$fill)
  stopifnot(length(fills) == 4L,
            identical(p$scales$get_scales("fill")$breaks,
                      c("50%", "80%", "90%", "98%")))
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
## tidyverts regressions: ts index column and class, index named "time",
## logical tsibble masks, implicit gaps, key-matched tsibble covariates,
## multivariate as_fable, variate labels in plots, model handles
if (requireNamespace("tsibble", quietly = TRUE)) {
  y_m <- stats::ts(y, start = c(2020, 1), frequency = 12)
  ft_m <- tsibble::as_tsibble(timestral_forecast(m, y_m, h = 6))
  obs_m <- tsibble::as_tsibble(y_m)
  stopifnot(identical(tsibble::index_var(ft_m), tsibble::index_var(obs_m)),
            identical(class(ft_m$index), class(obs_m$index)),
            ft_m$index[1] == obs_m$index[nrow(obs_m)] + 1)
  idx <- as.Date("2021-01-01") + 0:(length(y) - 1)
  for (nm in c("time", "step")) {
    dn <- data.frame(idx, value = y); names(dn)[1] <- nm
    ftn <- tsibble::as_tsibble(timestral_forecast(
      m, tsibble::as_tsibble(dn, index = nm), h = 4))
    stopifnot(identical(tsibble::index_var(ftn), nm), nrow(ftn) == 4L,
              ftn[[nm]][1] == idx[length(idx)] + 1)
  }
  d <- tsibble::as_tsibble(data.frame(t = idx, value = y), index = t)
  om <- rep(TRUE, length(y)); om[20:30] <- FALSE
  dm <- tsibble::as_tsibble(data.frame(t = idx, value = om), index = t)
  fc_dm <- timestral_forecast(m, d, h = 12, observation_mask = dm)
  stopifnot(max(abs(fc_dm$quantiles -
                    timestral_forecast(m, y, h = 12,
                                       observation_mask = om)$quantiles)) < 1e-12)
  ## a dropped run of rows is filled with NA and masked, not concatenated
  d_gap <- d[-(20:30), ]
  fc_gap <- timestral_forecast(m, d_gap, h = 12)
  stopifnot(length(fc_gap$index) == length(y), all(is.na(fc_gap$observed[1, 1, 20:30])))
  stopifnot(max(abs(fc_gap$quantiles - fc_dm$quantiles)) < 1e-12)
  ## out-of-order rows are put back in index order
  ord <- c(2L, 1L, 3:length(y))
  fc_ord <- timestral_forecast(m, d[ord, ], h = 12)
  stopifnot(max(abs(fc_ord$quantiles - timestral_forecast(m, d, h = 12)$quantiles)) < 1e-12)
  d2 <- tsibble::as_tsibble(
    data.frame(grp = rep(c("a", "b"), each = length(y)), t = rep(idx, 2),
               v1 = c(y, rev(y)), v2 = c(rev(y), y)),
    index = t, key = grp)
  pc <- tsibble::as_tsibble(
    data.frame(grp = rep(c("b", "a"), each = length(y)), t = rep(idx, 2),
               x = c(rep(1, length(y)), seq_along(y))),
    index = t, key = grp)
  fc_pc <- timestral_forecast(m, d2, h = 4, past_covariates = pc)
  fc_pl <- timestral_forecast(m, d2, h = 4,
                              past_covariates = list(seq_along(y), rep(1, length(y))))
  stopifnot(max(abs(fc_pc$quantiles - fc_pl$quantiles)) < 1e-12)
  pc_bad <- tsibble::as_tsibble(
    data.frame(grp = rep(c("b", "c"), each = length(y)), t = rep(idx, 2),
               x = 1), index = t, key = grp)
  err <- tryCatch({
    timestral_forecast(m, d2, h = 4, past_covariates = pc_bad); FALSE
  }, error = function(e) grepl("must match the series", conditionMessage(e)))
  stopifnot(isTRUE(err))
  kc_inf <- tsibble::as_tsibble(
    data.frame(t = as.Date("2021-01-01") + 0:(length(y) + 3),
               k = c(rep(1, length(y) + 3), Inf)), index = t)
  err <- tryCatch({
    timestral_forecast(m, d, h = 4, categorical_known_covariates = kc_inf); FALSE
  }, error = function(e) grepl("finite codes", conditionMessage(e)))
  stopifnot(isTRUE(err))
  fc_v <- timestral_forecast(m, d2, h = 4)
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    gd <- timestralR:::.fc_gg_data(fc_v, 1L, 0.8)
    stopifnot(identical(levels(gd$hist$variate), c("v1", "v2")))
  }
  if (requireNamespace("fabletools", quietly = TRUE) &&
      requireNamespace("distributional", quietly = TRUE)) {
    fb2 <- fabletools::as_fable(fc_v)
    stopifnot(fabletools::is_fable(fb2), nrow(fb2) == 2L * 2L * 4L,
              identical(sort(tsibble::key_vars(fb2)), c("series", "variate")))
    fb_m <- fabletools::as_fable(timestral_forecast(m, y_m, h = 6))
    stopifnot(identical(class(fb_m$index), class(obs_m$index)))
  }
}
if (requireNamespace("fable", quietly = TRUE) &&
    requireNamespace("tsibble", quietly = TRUE)) {
  train_ts <- tsibble::as_tsibble(
    data.frame(t = as.Date("2021-01-01") + 0:(length(y) - 1), value = y),
    index = t)
  fit <- fabletools::model(train_ts, tm = TIMESTRAL(value, model = m))
  mdl <- fit$tm[[1]]$fit
  stopifnot(is.null(mdl$model), is.list(mdl$handle))
  ## the fitted model holds only the handle, not the weights (assert on
  ## mdl itself: the mable also carries fabletools' model definition,
  ## whose serialised size depends on the caller's environment), and it
  ## forecasts after the cache is cleared
  stopifnot(length(serialize(mdl, NULL)) < length(serialize(m$weights, NULL)))
  fcf1 <- fabletools::forecast(fit, h = 6)
  fit2 <- unserialize(serialize(fit, NULL))
  rm(list = ls(timestralR:::.model_cache), envir = timestralR:::.model_cache)
  fcf2 <- fabletools::forecast(fit2, h = 6)
  stopifnot(max(abs(fcf2$.mean - fcf1$.mean)) < 1e-10)
}

## forecast package: timestral_as_forecast() and the forecast() method give
## a "forecast" object whose mean/intervals are the median and knot pairs;
## levels without a knot pair (95) and non-ts forecasts are rejected;
## accuracy() reports test-set measures
if (requireNamespace("forecast", quietly = TRUE)) {
  ap_tr <- window(AirPassengers, end = c(1959, 12))
  ap_te <- window(AirPassengers, start = c(1960, 1))
  fc_ap <- timestral_forecast(m, ap_tr, h = 12)
  fo <- timestral_as_forecast(fc_ap)
  stopifnot(inherits(fo, "forecast"), identical(fo$level, c(80, 98)),
            identical(as.numeric(fo$mean), as.numeric(timestral_median(fc_ap))),
            identical(as.numeric(fo$lower[, "80%"]),
                      as.numeric(timestral_quantile(fc_ap, 0.1))),
            identical(as.numeric(fo$upper[, "98%"]),
                      as.numeric(timestral_quantile(fc_ap, 0.99))),
            isTRUE(all.equal(stats::tsp(fo$mean), stats::tsp(ap_te))),
            isTRUE(all.equal(stats::tsp(fo$x), stats::tsp(ap_tr))),
            all(is.na(fo$fitted)))
  fo2 <- forecast::forecast(m, ap_tr, h = 12)
  stopifnot(max(abs(fo2$mean - fo$mean)) < 1e-10,
            max(abs(fo2$upper - fo$upper)) < 1e-10)
  fo3 <- forecast::forecast(m, ap_tr, h = 12, level = 90, symmetric = FALSE)
  ## `...` reaches timestral_forecast(); repeated native calls agree to
  ## ~1e-13 (K/V cache reuse), not bit-for-bit
  stopifnot(identical(colnames(fo3$lower), "90%"),
            max(abs(fo3$mean - timestral_median(
              timestral_forecast(m, ap_tr, h = 12, symmetric = FALSE)))) < 1e-10)
  acc <- forecast::accuracy(fo, ap_te)
  stopifnot(all(is.finite(acc["Test set", c("ME", "RMSE", "MAE", "MASE")])))
  .expect_error(timestral_as_forecast(fc_ap, level = 95), "no quantile knot")
  .expect_error(timestral_as_forecast(timestral_forecast(m, y, h = 6)),
                "ts time base")
  .expect_error(forecast::forecast(m, y, h = 6), "needs a ts")
  .expect_error(forecast::forecast(m, ap_tr), "needs `h`")
  ## multivariate: pick a variate by name; batch: pick a series
  fc_mv <- timestral_forecast(m, cbind(mdeaths, fdeaths), h = 6)
  fo_mv <- timestral_as_forecast(fc_mv, variate = "fdeaths")
  stopifnot(identical(fo_mv$series, "fdeaths"),
            identical(as.numeric(fo_mv$mean),
                      as.numeric(timestral_median(fc_mv)[, "fdeaths"])))
  .expect_error(timestral_as_forecast(fc_mv, variate = "nope"), "variate must")
  fc_b <- timestral_forecast(m, list(mdeaths, fdeaths), h = 6)
  stopifnot(identical(as.numeric(timestral_as_forecast(fc_b, series = 2L)$mean),
                      as.numeric(fc_b$quantiles[2, 1, , 11])))
  ## with fabletools loaded too, both share generics::forecast and dispatch
  ## on their own classes
  if (requireNamespace("fabletools", quietly = TRUE))
    stopifnot(inherits(fabletools::forecast(m, ap_tr, h = 3), "forecast"))
}
