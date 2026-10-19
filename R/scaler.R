## Port of timestral/scaler.py: causal per-patch normalisation with asinh.
## R numerics are float64 throughout, matching the official float64 statistics.

.row_cumsum <- function(x) t(apply(x, 1, cumsum))

.compute_causal_statistics <- function(data, observed, minimum_scale) {
  w <- observed * 1.0
  cum_w <- .row_cumsum(w)
  cum_x <- .row_cumsum(w * data)
  denom <- pmax(cum_w, 1.0)
  means <- cum_x / denom
  shifted <- cbind(0, means[, -ncol(means), drop = FALSE])
  m2 <- .row_cumsum((data - shifted) * (data - means) * w)
  variance <- m2 / pmax(denom - 1.0, 1.0)
  scales <- pmax(sqrt(variance), minimum_scale)
  list(loc = means, scale = scales)
}

.causal_patch_loc_scale <- function(data, observed, patch_size, minimum_scale) {
  st <- .compute_causal_statistics(data, observed, minimum_scale)
  n_patch <- ncol(data) %/% patch_size
  last <- seq(patch_size, ncol(data), by = patch_size)
  list(loc = st$loc[, rep(last, each = patch_size), drop = FALSE][, seq_len(n_patch * patch_size), drop = FALSE],
       scale = st$scale[, rep(last, each = patch_size), drop = FALSE][, seq_len(n_patch * patch_size), drop = FALSE],
       patch_loc = st$loc[, last, drop = FALSE],
       patch_scale = st$scale[, last, drop = FALSE])
}

.apply_affine_asinh <- function(data, loc, scale, observed, use_asinh) {
  scaled <- (data - loc) / pmax(scale, 1.1920929e-07)
  scaled[!observed] <- 0
  if (use_asinh) scaled[observed] <- asinh(scaled[observed])
  scaled
}

.model_scale <- function(model, inputs, observed) {
  cfg <- model$config
  ps <- .causal_patch_loc_scale(inputs, observed, cfg$patch_size, cfg$minimum_scale)
  list(scaled = .apply_affine_asinh(inputs, ps$loc, ps$scale, observed, cfg$use_asinh),
       loc = ps$loc, scale = ps$scale)
}

.instance_norm_stats <- function(values, observed, minimum_scale) {
  n <- pmax(rowSums(observed), 1.0)
  loc <- rowSums(values * observed) / n
  var <- rowSums(((values - loc) * observed)^2) / n
  scale <- pmax(sqrt(var), minimum_scale)
  is_binary <- apply(values == 0 | values == 1 | !observed, 1, all) & rowSums(observed) > 0
  loc[is_binary] <- 0
  scale[is_binary] <- 1
  list(loc = loc, scale = scale, is_binary = is_binary)
}

.known_cov_scaling <- function(model, inputs, observed, scaled, loc, scale, known_rows) {
  if (!any(known_rows)) return(list(scaled = scaled, loc = loc, scale = scale))
  cfg <- model$config
  st <- .instance_norm_stats(inputs, observed, cfg$minimum_scale)
  known_scaled <- (inputs - st$loc) / pmax(st$scale, 1.1920929e-07)
  known_scaled[!observed] <- 0
  if (cfg$use_asinh) {
    squashed <- asinh(known_scaled)
    take <- observed & !st$is_binary
    known_scaled[take] <- squashed[take]
  }
  sel <- known_rows
  scaled[sel, ] <- known_scaled[sel, , drop = FALSE]
  loc_new <- loc
  scale_new <- scale
  loc_new[sel, ] <- st$loc[sel]
  scale_new[sel, ] <- st$scale[sel]
  list(scaled = scaled, loc = loc_new, scale = scale_new)
}

.clamp_nonfinite_q <- function(q) {
  ## q: (R, S, Q); +/-Inf along Q becomes the nearest finite value; all-Inf -> 0
  d <- dim(q)
  for (r in seq_len(d[1])) for (s in seq_len(d[2])) {
    v <- q[r, s, ]
    fin <- is.finite(v)
    if (all(fin)) next
    if (!any(fin)) { q[r, s, ] <- 0; next }
    v[v == Inf] <- max(v[fin])
    v[v == -Inf] <- min(v[fin])
    v[is.infinite(v)] <- 0
    q[r, s, ] <- v
  }
  q
}

.denormalize_quantiles <- function(model, preds, anchor_loc, anchor_scale) {
  x <- preds
  if (model$config$use_asinh) x <- sinh(x)
  ## anchor_* are per-row vectors; R recycles them along the first (row) dim
  out <- x * as.numeric(anchor_scale) + as.numeric(anchor_loc)
  .clamp_nonfinite_q(out)
}

.welford_combine <- function(cnt, mean, m2, cnt_b, mean_b, m2_b) {
  w <- cnt + cnt_b
  w_safe <- pmax(w, .Machine$double.xmin)
  delta <- mean_b - mean
  mean_new <- mean + delta * cnt_b / w_safe
  m2_new <- m2 + m2_b + delta^2 * cnt * cnt_b / w_safe
  zero <- w == 0
  mean_new[zero] <- 0
  m2_new[zero] <- 0
  list(cnt = w, mean = mean_new, m2 = m2_new)
}

.cumulative_welford <- function(cnt, mean, m2) {
  n <- length(cnt)
  d <- 1L
  while (d < n) {
    idx <- seq_len(n - d)
    left <- .welford_combine(
      c(rep(0, d), cnt[idx]), c(rep(0, d), mean[idx]), c(rep(0, d), m2[idx]),
      cnt, mean, m2)
    cnt <- left$cnt; mean <- left$mean; m2 <- left$m2
    d <- d * 2L
  }
  list(cnt = cnt, mean = mean, m2 = m2)
}
