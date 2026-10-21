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

## review fixes: NULL arguments are no-ops, covariates are validated by
## timestral_forecast()'s parser, masked actuals are not scored, series
## keep their names, fractional levels, and early origins fail up front
bt_null <- timestral_backtest(m, y, h = 6, origins = 40, past_covariates = NULL,
                              known_covariates = NULL)
stopifnot(max(abs(bt_null$median - bt$median[1:6])) < 1e-10)
.expect_error(timestral_backtest(m, y, h = 6, origins = 40,
                                 categorical_past_covariates = factor(y > 50)),
              "factor")
msk2 <- rep(TRUE, length(y)); msk2[42:43] <- FALSE
bt_m2 <- timestral_backtest(m, y, h = 6, origins = 40, observation_mask = msk2)
stopifnot(identical(is.na(bt_m2$actual), c(FALSE, TRUE, TRUE, FALSE, FALSE, FALSE)),
          identical(is.na(bt_m2$crps), is.na(bt_m2$actual)))
stopifnot(identical(unique(timestral_backtest(m, list(a = y), h = 4,
                                              origins = 50)$series), "a"),
          identical(unique(timestral_backtest(m, y, h = 4, origins = 50)$series), "S1"),
          identical(unique(timestral_backtest(m, list(y, y), h = 4,
                                              origins = 50)$series), c("S1", "S2")))
sc_f <- timestral_score(fc40, y[41:46], level = c(0.8, 0.98))
stopifnot(identical(sc_f$cover80, sc40$cover80), "cover98" %in% names(sc_f))
.expect_error(timestral_score(fc40, y[41:46], level = 95),
              "for the 95% interval")
y_late <- y; y_late[1:20] <- NA
.expect_error(timestral_backtest(m, list(y, y_late), h = 6, origins = c(10, 40)),
              "series 2 of y has no observed value up to origin 10; start the origins at 21")

## K/V cache across origins: origins congruent modulo patch_size reuse
## the shared leading patches (forward and mirrored passes each keep an
## entry); results match cold-cache forecasts and the R backend
set.seed(7)
y_kv <- cbind(cumsum(rnorm(260)) + 40, cumsum(rnorm(260)) + 60)
org_kv <- c(70, 86, 94, 118, 150, 166)   # 16 apart, plus one class of 94
bt_kv <- timestral_backtest(m, y_kv, h = 40, origins = org_kv,
                            keep_forecasts = TRUE)
stopifnot(identical(unique(bt_kv$origin), as.integer(org_kv)))
for (i in seq_along(org_kv)) {
  cold <- timestral_forecast(timestral_load(tiny, max_context = 4096L),
                             y_kv[seq_len(org_kv[i]), ], h = 40)
  stopifnot(max(abs(cold$quantiles -
                    attr(bt_kv, "forecasts")[[i]]$quantiles)) < 1e-10)
}
r_kv <- timestral_forecast(m_R, y_kv[seq_len(166), ], h = 40)
stopifnot(max(abs(r_kv$quantiles - attr(bt_kv, "forecasts")[[6]]$quantiles)) < 1e-10)
## a shorter context sharing a prefix of the cached one, and a third
## sequence evicting an entry, both stay exact
m_kv <- timestral_load(tiny, max_context = 4096L)
for (len in c(200, 120, 136, 260)) {
  for (yy in list(y_kv[seq_len(len), 1], -y_kv[seq_len(len), 2], y_kv[seq_len(len), 2])) {
    got <- timestral_forecast(m_kv, yy, h = 20, symmetric = FALSE)
    ref <- timestral_forecast(timestral_load(tiny, max_context = 4096L), yy,
                              h = 20, symmetric = FALSE)
    stopifnot(max(abs(got$quantiles - ref$quantiles)) < 1e-10)
  }
}

## the cache is actually used: a forecast after one from a congruent
## origin reuses that origin's patches in both the forward and mirrored
## pass, and the backtest's run order makes that happen (70 and 102 are
## congruent modulo 16, 85 is not and would otherwise evict them)
.reused <- function(mod) timestralR:::.ts_cache_stats(mod$session)$patches_reused
.fresh <- function() timestral_load(tiny, max_context = 4096L)
.cost <- function(o) {   # patches a forecast from o reuses on a cold model
  mf <- .fresh(); timestral_forecast(mf, y_kv[seq_len(o), ], h = 40); .reused(mf)
}
ll70 <- ceiling(70 / m$config$patch_size)
m_w <- .fresh()
invisible(timestral_forecast(m_w, y_kv[seq_len(70), ], h = 40))
r0 <- .reused(m_w)
invisible(timestral_forecast(m_w, y_kv[seq_len(102), ], h = 40))
stopifnot(.reused(m_w) - r0 - .cost(102) == 2 * ll70)
m_b <- .fresh()
invisible(timestral_backtest(m_b, y_kv, h = 40, origins = c(70, 85, 102)))
stopifnot(.reused(m_b) - (.cost(70) + .cost(85) + .cost(102)) == 2 * ll70)
