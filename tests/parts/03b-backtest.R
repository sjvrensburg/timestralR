## Scoring and rolling-origin backtests.
## Sourced by tests/timestralR-tests.R; expects `m`, `y`, `y2`, `pa`, `kn`,
## `.expect_error`.

## a minimal forecast object with the given knots and quantiles
## (series 1, variate 1, steps in rows of `qmat`)
.fake_fc <- function(levels, qmat) {
  qmat <- matrix(qmat, ncol = length(levels))
  structure(list(quantiles = array(qmat, c(1L, dim(qmat)),
                                   list(variate = "V1", step = NULL,
                                        level = format(levels))),
                 levels = levels, h = nrow(qmat), n_series = 1L),
            class = "timestral_forecast")
}

## pinball and CRPS by hand on 3 knots: q = (1, 2, 3), y = 2.5 gives
## rho = (0.15, 0.25, 0.05); pinball = 2 * mean = 0.3, and trapezoid weights
## (0.2, 0.4, 0.2) give crps = 2 * (0.03 + 0.1 + 0.01) = 0.28
sc <- timestral_score(.fake_fc(c(0.1, 0.5, 0.9), c(1, 2, 3)), 2.5, level = 80)
stopifnot(isTRUE(all.equal(sc$pinball, 0.3)), isTRUE(all.equal(sc$crps, 0.28)),
          isTRUE(all.equal(sc$ae, 0.5)), identical(sc$median, 2),
          identical(sc$cover80, TRUE))
## below every quantile: not covered, rho = (1 - tau) * (q - y)
sc <- timestral_score(.fake_fc(c(0.1, 0.5, 0.9), c(1, 2, 3)), 0, level = 80)
stopifnot(isTRUE(all.equal(sc$pinball, 2 * mean(c(0.9, 1, 0.3)))),
          identical(sc$cover80, FALSE))

## degenerate forecast (all quantiles equal): pinball is |q - y| and CRPS
## is (tau_K - tau_1) |q - y| = 0.98 |q - y| over the official knots
lv <- timestral_levels()
sc <- timestral_score(.fake_fc(lv, matrix(5, 2, 21)), c(2, 9))
stopifnot(isTRUE(all.equal(sc$pinball, c(3, 4))),
          isTRUE(all.equal(sc$crps, 0.98 * c(3, 4))),
          identical(names(sc), c("series", "variate", "step", "actual",
                                 "median", "pinball", "crps", "ae",
                                 "cover80", "cover98")))
## a missing actual gives missing scores, not an error
sc <- timestral_score(.fake_fc(lv, matrix(5, 2, 21)), c(NA, 9))
stopifnot(is.na(sc$pinball[1]), is.na(sc$cover80[1]), !is.na(sc$crps[2]))
.expect_error(timestral_score(.fake_fc(lv, matrix(5, 2, 21)), 1:3),
              "actual must have")
.expect_error(timestral_score(.fake_fc(lv, matrix(5, 2, 21)), 1:2, level = 95),
              "no quantile knot")

## backtest: rows, origins, and one origin equal to a direct forecast
bt <- timestral_backtest(m, y, h = 6, origins = c(40, 52))
stopifnot(identical(nrow(bt), 2L * 6L), identical(unique(bt$origin), c(40L, 52L)),
          identical(bt$t, rep(c(40L, 52L), each = 6) + rep(1:6, 2)),
          identical(bt$actual, y[bt$t]), is.null(bt$time))
fc40 <- timestral_forecast(m, y[1:40], h = 6)
sc40 <- timestral_score(fc40, y[41:46])
stopifnot(max(abs(bt$median[1:6] - sc40$median)) < 1e-10,
          max(abs(bt$crps[1:6] - sc40$crps)) < 1e-10)
## default origins: the last five, spaced by h
n_y <- length(y)
bt_d <- timestral_backtest(m, y, h = 6)
stopifnot(identical(unique(bt_d$origin), as.integer(n_y - 6 - 6 * (4:0))))
bt_i <- timestral_backtest(m, y, h = 6, initial = n_y - 6 - 9, step = 3)
stopifnot(identical(unique(bt_i$origin), as.integer(n_y - 6 - c(9, 6, 3, 0))))

## ts input gets a time column; keep_forecasts stores the forecasts
bt_ts <- timestral_backtest(m, AirPassengers, h = 12, origins = 120,
                            keep_forecasts = TRUE)
stopifnot(isTRUE(all.equal(bt_ts$time, as.numeric(time(AirPassengers))[121:132])),
          identical(names(attr(bt_ts, "forecasts")), "120"),
          inherits(attr(bt_ts, "forecasts")[[1]], "timestral_forecast"))

## batches and multivariate series: one row per series x variate x step
bt_b <- timestral_backtest(m, list(a = y, b = rev(y)), h = 4, origins = 50)
stopifnot(identical(nrow(bt_b), 8L), identical(unique(bt_b$series), c("a", "b")),
          identical(bt_b$actual[bt_b$series == "b"], rev(y)[51:54]))
bt_mv <- timestral_backtest(m, cbind(u = y, w = rev(y)), h = 4, origins = 50)
stopifnot(identical(unique(bt_mv$variate), c("u", "w")))

## covariates are cut to each origin: past to 1:o, known to 1:(o + h)
n2 <- length(y2)
o2 <- n2 - 24
bt_c <- timestral_backtest(m, y2, h = 24, origins = o2,
                           past_covariates = matrix(pa, ncol = 1),
                           known_covariates = matrix(kn[1:n2], ncol = 1))
fc_c <- timestral_forecast(m, y2[1:o2], h = 24,
                           past_covariates = matrix(pa[1:o2], ncol = 1),
                           known_covariates = matrix(kn[1:n2], ncol = 1))
stopifnot(max(abs(bt_c$median - timestral_score(fc_c, y2[o2 + 1:24])$median)) < 1e-10)
.expect_error(timestral_backtest(m, y2, h = 24, origins = o2,
                                 known_covariates = matrix(kn[1:(n2 - 1)], ncol = 1)),
              "known_covariates has .* rows")
## observation_mask is cut like the context
msk <- rep(TRUE, length(y)); msk[30:35] <- FALSE
bt_m <- timestral_backtest(m, y, h = 6, origins = 40, observation_mask = msk)
fc_m <- timestral_forecast(m, y[1:40], h = 6, observation_mask = msk[1:40])
stopifnot(max(abs(bt_m$median - timestral_score(fc_m, y[41:46])$median)) < 1e-10)

## validation
.expect_error(timestral_backtest(m, y, h = 6, origins = n_y - 5), "origins must")
.expect_error(timestral_backtest(m, y, h = 6, origins = 0), "origins must")
.expect_error(timestral_backtest(m, y, h = 6, origins = 10.5), "origins must")
.expect_error(timestral_backtest(m, y, h = 6, initial = n_y), "initial must")
.expect_error(timestral_backtest(m, y, h = 6, step = 0), "step must")
.expect_error(timestral_backtest(m, y[1:5], h = 6), "too few")
.expect_error(timestral_backtest(m, y, h = 6, bogus = 1), "unsupported argument")
