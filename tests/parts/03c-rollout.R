## rollout = "quantile_paths": the default stays median feedback, forecasts
## within one block do not change, and later blocks agree across backends.
## Sourced by tests/timestralR-tests.R; expects `m`, `m_R`, `y`, `y2`, `pa`,
## `kn`, `tiny`.
paths <- "quantile_paths"
## one patch per block, so that a short horizon spans several blocks
m1 <- timestral_load(tiny, max_context = 4096L, per_shot_horizon_patches = 1L)
m1_R <- timestral_load(tiny, max_context = 4096L, per_shot_horizon_patches = 1L,
                       backend = "R")
blk <- m1$config$patch_size   # steps per block
is_monotone <- function(q) {
  d <- dim(q)
  all(apply(q, seq_len(length(d) - 1L), function(v) all(diff(v) >= -1e-9)))
}

## the default is median feedback, also past the first block
stopifnot(identical(formals(timestral_forecast)$rollout,
                    quote(c("median", "quantile_paths"))))
## (repeat forecasts can differ in the last bits: the K/V cache is exact only
## up to rounding, so compare with a tolerance)
stopifnot(max(abs(timestral_forecast(m1, y, h = 96)$quantiles -
                  timestral_forecast(m1, y, h = 96, rollout = "median")$quantiles)) < 1e-10)
ok <- tryCatch({timestral_forecast(m1, y, h = 8, rollout = "sampled"); FALSE},
               error = function(e) TRUE)
stopifnot(ok)

## within one block the two rollouts are the same forward pass
stopifnot(max(abs(timestral_forecast(m1, y, h = blk)$quantiles -
                  timestral_forecast(m1, y, h = blk, rollout = paths)$quantiles)) < 1e-10)

## past it: the first block is untouched and later blocks differ
fm <- timestral_forecast(m1, y, h = 96)
fp <- timestral_forecast(m1, y, h = 96, rollout = paths)
stopifnot(identical(dim(fp$quantiles), c(1L, 96L, 21L)),
          max(abs(fp$quantiles[, seq_len(blk), ] - fm$quantiles[, seq_len(blk), ])) < 1e-10,
          all(is.finite(fp$quantiles)),
          max(abs(fp$quantiles[, -seq_len(blk), ] - fm$quantiles[, -seq_len(blk), ])) > 1e-9,
          is_monotone(fp$quantiles))
## a horizon ending inside a patch is the head of a longer forecast
fp_short <- timestral_forecast(m1, y, h = 50, rollout = paths)
stopifnot(max(abs(fp_short$quantiles - fp$quantiles[, 1:50, , drop = FALSE])) < 1e-10)
## also with no mirrored pass
f_asym <- timestral_forecast(m1, y, h = 96, symmetric = FALSE, rollout = paths)
stopifnot(all(is.finite(f_asym$quantiles)), is_monotone(f_asym$quantiles))

## native and R backends agree on the same paths
rollout_agrees <- function(...) {
  a <- timestral_forecast(m1, ..., rollout = paths)$quantiles
  b <- timestral_forecast(m1_R, ..., rollout = paths)$quantiles
  stopifnot(identical(dim(a), dim(b)), max(abs(a - b)) < 1e-9, is_monotone(a))
}
rollout_agrees(y, h = 96)
rollout_agrees(cbind(y, rev(y)), h = 96)
rollout_agrees(list(y, y2, rev(y)), h = 96)
rollout_agrees(y2, h = 96, past_covariates = matrix(pa, ncol = 1))

## a series in a batch forecasts as it does alone: paths mix within a series
yb3 <- list(y, y2, rev(y))
fb3 <- timestral_forecast(m1, yb3, h = 96, rollout = paths)
for (b in 1:3) {
  f1 <- timestral_forecast(m1, yb3[[b]], h = 96, rollout = paths)
  stopifnot(max(abs(fb3$quantiles[b, 1, , ] - f1$quantiles[1, , ])) < 1e-9)
}

## variates of a series stay linked within a path: as with median feedback,
## the second variate changes the first one's forecast
fmv <- timestral_forecast(m1, cbind(y, rev(y)), h = 96, rollout = paths)
stopifnot(identical(dim(fmv$quantiles), c(2L, 96L, 21L)),
          max(abs(fmv$quantiles[1, , ] - fp$quantiles[1, , ])) > 1e-9)

## known covariates stay real in every block: changing a future covariate
## value inside the last block leaves earlier blocks alone
h_k <- 96L
n_k <- length(y2)
kn_future <- c(kn, rep(kn[length(kn)], max(0L, n_k + h_k - length(kn))))[seq_len(n_k + h_k)]
kn_late <- kn_future
kn_late[n_k + 65:96] <- kn_late[n_k + 65:96] + 100
fk1 <- timestral_forecast(m1, y2, h = h_k, rollout = paths,
                          past_covariates = matrix(pa, ncol = 1),
                          known_covariates = matrix(kn_future, ncol = 1))
fk2 <- timestral_forecast(m1, y2, h = h_k, rollout = paths,
                          past_covariates = matrix(pa, ncol = 1),
                          known_covariates = matrix(kn_late, ncol = 1))
stopifnot(identical(dim(fk1$quantiles), c(1L, 96L, 21L)),
          all(is.finite(fk1$quantiles)), is_monotone(fk1$quantiles),
          max(abs(fk1$quantiles[, 1:64, ] - fk2$quantiles[, 1:64, ])) < 1e-9,
          max(abs(fk1$quantiles[, 65:96, ] - fk2$quantiles[, 65:96, ])) > 1e-9)
fk_R <- timestral_forecast(m1_R, y2, h = h_k, rollout = paths,
                           past_covariates = matrix(pa, ncol = 1),
                           known_covariates = matrix(kn_future, ncol = 1))
stopifnot(max(abs(fk1$quantiles - fk_R$quantiles)) < 1e-9)

## a context cap applies to every path
m1_cap <- timestral_load(tiny, max_context = 64L, per_shot_horizon_patches = 1L)
m1_cap_R <- timestral_load(tiny, max_context = 64L, per_shot_horizon_patches = 1L,
                           backend = "R")
fc_cap <- timestral_forecast(m1_cap, y, h = 160, rollout = paths)
fc_cap_R <- timestral_forecast(m1_cap_R, y, h = 160, rollout = paths)
stopifnot(all(is.finite(fc_cap$quantiles)),
          max(abs(fc_cap$quantiles - fc_cap_R$quantiles)) < 1e-9)

## mixing: masses, knot lookup, mirror symmetry
mass <- timestralR:::.level_mass(timestral_levels())
stopifnot(abs(sum(mass) - 1) < 1e-12, all(mass > 0),
          max(abs(mass - rev(mass))) < 1e-12)
stopifnot(max(abs(timestralR:::.level_mass((1:9) / 10) - 1 / 9)) < 1e-12)
lv <- timestral_levels()
qk <- array(rep(lv, each = 2), c(1L, 2L, length(lv)))
at <- timestralR:::.quantiles_at(qk, lv, c(0.3, 0.325, 0.001, 0.999))
stopifnot(identical(at[1, 1, 1], 0.3), abs(at[1, 1, 2] - 0.325) < 1e-12,
          identical(at[1, 1, 3], lv[1]), identical(at[1, 1, 4], lv[21]))
set.seed(11)
x <- matrix(rnorm(40), 20, 2)
w <- runif(20)
sym_lv <- c(0.1, 0.25, 0.5, 0.75, 0.9)
mq <- timestralR:::.mix_quantiles(x, w, sym_lv)
mq_neg <- timestralR:::.mix_quantiles(-x, w, sym_lv)
stopifnot(max(abs(mq + mq_neg[rev(seq_along(sym_lv)), ])) < 1e-12,
          all(apply(mq, 2L, function(v) all(diff(v) >= 0))))
## equal samples are returned as they are; the extremes do not overshoot
same <- timestralR:::.mix_quantiles(matrix(3, 5, 1), rep(1, 5), sym_lv)
stopifnot(all(same == 3))
tails <- timestralR:::.mix_quantiles(matrix(c(1, 2), 2, 1), c(1, 1), c(0.01, 0.99))
stopifnot(identical(as.vector(tails), c(1, 2)))

## the other forecast interfaces pass it on
stopifnot(max(abs(predict(m1, y, h = 96, rollout = paths)$quantiles - fp$quantiles)) < 1e-10)
bt <- timestral_backtest(m1, y, h = 64, origins = length(y) - 64L, rollout = paths)
bt_med <- timestral_backtest(m1, y, h = 64, origins = length(y) - 64L)
stopifnot(max(abs(bt$pinball[seq_len(blk)] - bt_med$pinball[seq_len(blk)])) < 1e-10,
          max(abs(bt$pinball[-seq_len(blk)] - bt_med$pinball[-seq_len(blk)])) > 1e-10)
