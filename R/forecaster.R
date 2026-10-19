## Port of timestral/forecaster.py (single-series API, B = 1, as in the
## earlier prototype; variates of the series are forecast jointly).

.ffill_rows <- function(inputs, mask) {
  out <- inputs
  for (r in seq_len(nrow(inputs))) {
    idx <- which(mask[r, ])
    if (length(idx) == 0L) { out[r, ] <- 0; next }
    pos <- findInterval(seq_len(ncol(inputs)), idx)
    vals <- c(0, inputs[r, idx])
    out[r, ] <- vals[pos + 1L]
  }
  out
}

.nan_to_num <- function(x) {
  x[is.na(x)] <- 0
  x[x == Inf] <- 3.4028235e38
  x[x == -Inf] <- -3.4028235e38
  x
}

.pad_to_patch <- function(inputs, mask, ps) {
  ctx <- ncol(inputs)
  if (ctx %% ps == 0L) return(list(x = inputs, m = mask))
  pad <- ps - (ctx %% ps)
  list(x = cbind(matrix(0, nrow(inputs), pad), inputs),
       m = cbind(matrix(FALSE, nrow(inputs), pad), mask))
}

.assemble_rows <- function(model, inputs, mask, h, past, known) {
  ps <- model$config$patch_size
  n_tgt <- nrow(inputs)
  horizon <- ceiling(h / ps) * ps
  ctx_rows <- list(.ffill_rows(inputs, mask))
  ctx_masks <- list(mask)
  roles <- rep(0L, n_tgt)
  n_past <- 0L
  if (!is.null(past)) {
    n_past <- nrow(past)
    ctx_masks <- c(ctx_masks, list(!is.na(past)))
    ctx_rows <- c(ctx_rows, list(.nan_to_num(past)))
    roles <- c(roles, rep(1L, n_past))
  }
  n_known <- 0L
  kf_vals <- NULL; kf_obs <- NULL
  if (!is.null(known)) {
    n_known <- nrow(known)
    kc_ctx <- known[, seq_len(ncol(inputs)), drop = FALSE]
    kc_fut <- known[, ncol(inputs) + seq_len(h), drop = FALSE]
    ctx_masks <- c(ctx_masks, list(!is.na(kc_ctx)))
    ctx_rows <- c(ctx_rows, list(.nan_to_num(kc_ctx)))
    roles <- c(roles, rep(2L, n_known))
    kf_obs <- !is.na(kc_fut)
    kf_vals <- .nan_to_num(kc_fut)
  }
  rows <- do.call(rbind, ctx_rows)
  msk <- do.call(rbind, ctx_masks)
  if (ncol(rows) > model$max_context) {
    keep <- (ncol(rows) - model$max_context + 1L):ncol(rows)
    rows <- rows[, keep, drop = FALSE]
    msk <- msk[, keep, drop = FALSE]
  }
  padded <- .pad_to_patch(rows, msk, ps)
  rows <- padded$x; msk <- padded$m
  known_future <- NULL; known_future_mask <- NULL
  if (n_known > 0L) {
    v_all <- nrow(rows)
    known_future <- matrix(0, v_all, horizon)
    known_future_mask <- matrix(FALSE, v_all, horizon)
    k0 <- n_tgt + n_past
    known_future[k0 + seq_len(n_known), seq_len(h)] <- kf_vals
    known_future_mask[k0 + seq_len(n_known), seq_len(h)] <- kf_obs
  }
  list(rows = rows, mask = msk, role_ids = if (n_past + n_known > 0L) roles else NULL,
       known_future = known_future, known_future_mask = known_future_mask,
       n_tgt = n_tgt)
}

.advancing_frames <- function(model, context, mask, sinh_median) {
  ## sinh_median: (R, H, ps) float64; returns per-row loc/scale matrices (R, H)
  ps <- model$config$patch_size
  R <- nrow(context); H <- dim(sinh_median)[2]
  n_ctx <- ncol(context) %/% ps
  locs <- matrix(0, R, H); scales <- matrix(0, R, H)
  clip <- sqrt(3.4028235e38) / 1e3
  for (r in seq_len(R)) {
    vals <- matrix(context[r, ], nrow = ps)   # (ps, n_ctx)
    wts <- matrix(mask[r, ], nrow = ps) * 1.0
    cnt_p <- colSums(wts)
    mean_p <- colSums(vals * wts) / pmax(cnt_p, 1.0)
    m2_p <- colSums(wts * (vals - rep(mean_p, each = ps))^2)
    mean_p[cnt_p == 0] <- 0; m2_p[cnt_p == 0] <- 0
    cum <- .cumulative_welford(c(cnt_p, rep(0, H)), c(mean_p, rep(0, H)),
                               c(m2_p, rep(0, H)))
    i0 <- n_ctx  # cumulative state at last context patch (1-based)
    cnt <- cum$cnt[i0]; mean <- cum$mean[i0]; m2 <- cum$m2[i0]
    offset <- mean
    for (j in seq_len(H)) {
      var <- max(m2 / max(cnt - 1.0, 1.0), 0)
      scale <- max(sqrt(var), model$config$minimum_scale)
      locs[r, j] <- mean; scales[r, j] <- scale
      folded <- sinh_median[r, j, ] * scale + mean - offset
      folded <- pmin(pmax(folded, -clip), clip) + offset
      b_mean <- mean(folded)
      b_m2 <- sum((folded - b_mean)^2)
      nxt <- .welford_combine(cnt, mean, m2, ps, b_mean, b_m2)
      cnt <- nxt$cnt; mean <- nxt$mean; m2 <- nxt$m2
    }
  }
  list(loc = locs, scale = scales)
}

.refine_block <- function(model, quantiles, context, mask, target_rows) {
  ps <- model$config$patch_size
  n_pred <- dim(quantiles)[2] %/% ps
  if (n_pred <= 1L) return(quantiles)
  R <- dim(quantiles)[1]
  sc <- .model_scale(model, context, mask)
  loc0 <- sc$loc[, ncol(context)]
  scale0 <- sc$scale[, ncol(context)]
  med_idx <- which(model$config$quantile_knots == 0.5)
  med <- matrix(quantiles[, , med_idx], nrow = R)  # (R, n_pred*ps)
  sinh_med <- array(0, c(R, n_pred, ps))
  for (r in seq_len(R))
    sinh_med[r, , ] <- t(matrix((med[r, ] - loc0[r]) / scale0[r], nrow = ps))
  fr <- .advancing_frames(model, context, mask, sinh_med)
  loc_j <- fr$loc; scale_j <- fr$scale
  loc_j[, 1] <- loc0; scale_j[, 1] <- scale0
  loc_step <- loc_j[, rep(seq_len(n_pred), each = ps), drop = FALSE]
  scale_step <- scale_j[, rep(seq_len(n_pred), each = ps), drop = FALSE]
  sinh_norm <- (quantiles - loc0) / scale0
  ## scale/loc_step are (R, steps); vector recycling aligns them over Q slices
  refined <- .clamp_nonfinite_q(sinh_norm * as.vector(scale_step) +
                                  as.vector(loc_step))
  if (!is.null(target_rows)) refined[!target_rows, , ] <- quantiles[!target_rows, , ]
  refined
}

.block_ar_decode <- function(model, asm, h) {
  ps <- model$config$patch_size
  context <- asm$rows; mask <- asm$mask
  role_ids <- asm$role_ids
  known_rows <- if (is.null(role_ids)) NULL else role_ids == 2L
  past_rows <- if (is.null(role_ids)) NULL else role_ids == 1L
  target_rows <- if (is.null(role_ids)) NULL else role_ids == 0L
  med_idx <- which(model$config$quantile_knots == 0.5)
  blocks <- list()
  remaining <- ceiling(h / ps)
  offset <- 0L
  while (remaining > 0) {
    block <- min(32L, remaining)
    blen <- block * ps
    kf <- if (!is.null(asm$known_future)) asm$known_future[, offset + seq_len(blen), drop = FALSE] else NULL
    kfm <- if (!is.null(asm$known_future_mask)) asm$known_future_mask[, offset + seq_len(blen), drop = FALSE] else NULL
    q <- .predict_block(model, context, mask, block, role_ids, kf, kfm)
    q <- .refine_block(model, q, context, mask, target_rows)
    blocks[[length(blocks) + 1L]] <- q
    remaining <- remaining - block
    offset <- offset + blen
    if (remaining <= 0) break
    feedback <- matrix(q[, , med_idx], nrow = nrow(context))
    feedback[!is.finite(feedback) | abs(feedback) >= 1e30] <- 0
    feedback_mask <- matrix(TRUE, nrow(feedback), ncol(feedback))
    if (!is.null(role_ids)) {
      if (!is.null(kf)) {
        feedback[known_rows, ] <- kf[known_rows, , drop = FALSE]
        feedback_mask[known_rows, ] <- kfm[known_rows, , drop = FALSE]
      }
      feedback[past_rows, ] <- 0
      feedback_mask[past_rows, ] <- FALSE
    }
    context <- cbind(context, feedback)
    mask <- cbind(mask, feedback_mask)
    if (ncol(context) > model$max_context) {
      keep_n <- max(ps, (model$max_context %/% ps) * ps)
      context <- context[, (ncol(context) - keep_n + 1L):ncol(context), drop = FALSE]
      mask <- mask[, (ncol(mask) - keep_n + 1L):ncol(mask), drop = FALSE]
    }
  }
  total <- sum(vapply(blocks, function(b) dim(b)[2], integer(1)))
  packed <- array(0, c(dim(blocks[[1]])[1], total, dim(blocks[[1]])[3]))
  pos <- 0L
  for (b in blocks) {
    packed[, pos + seq_len(dim(b)[2]), ] <- b
    pos <- pos + dim(b)[2]
  }
  packed[, seq_len(h), , drop = FALSE]
}

.cap_quantiles <- function(model, quantiles, inputs, mask) {
  ## quantiles: (V, H, Q); inputs/mask: assembled target rows (V, T)
  ps <- model$config$patch_size
  V <- dim(quantiles)[1]
  st <- .causal_patch_loc_scale(inputs, mask, ps, model$config$minimum_scale)
  anchor <- st$scale[, ncol(inputs)] * 1e4
  for (v in seq_len(V)) {
    obs <- mask[v, ]
    hi <- if (any(obs)) max(inputs[v, obs]) else 0
    lo <- if (any(obs)) min(inputs[v, obs]) else 0
    quantiles[v, , ] <- pmin(pmax(quantiles[v, , ], lo - anchor[v]), hi + anchor[v])
  }
  quantiles
}

.forecast_raw <- function(model, inputs, mask, h, past, known) {
  asm <- .assemble_rows(model, inputs, mask, h, past, known)
  packed <- .block_ar_decode(model, asm, h)
  q <- packed[seq_len(asm$n_tgt), , , drop = FALSE]
  tgt_rows <- asm$rows[seq_len(asm$n_tgt), , drop = FALSE]
  tgt_mask <- asm$mask[seq_len(asm$n_tgt), , drop = FALSE]
  q <- .cap_quantiles(model, q, tgt_rows, tgt_mask)
  if (any(is.nan(q))) stop("forecast produced NaN values; the input may be degenerate")
  q
}

.as_target_rows <- function(y) {
  tsp_y <- if (stats::is.ts(y)) stats::tsp(y) else NULL
  ym <- if (is.null(dim(y))) matrix(as.numeric(y), ncol = 1) else as.matrix(y)
  list(rows = t(ym), names = colnames(ym), tsp = tsp_y, n = nrow(ym))
}

.as_cov_rows <- function(x, expected_rows, name) {
  if (is.null(x)) return(NULL)
  if (is.data.frame(x)) x <- as.matrix(x)
  if (is.null(dim(x))) x <- matrix(as.numeric(x), ncol = 1)
  if (nrow(x) != expected_rows)
    stop(name, " must have ", expected_rows, " rows (time in rows), got ", nrow(x))
  t(x)
}

#' Forecast one (possibly multivariate) series
#'
#' @param model a `timestral_model`
#' @param y numeric vector, `ts`, or matrix/data frame with time in rows and
#'   one column per target variate (NA = missing). Variates are forecast jointly.
#' @param h horizon in steps
#' @param past_covariates matrix with time in rows (T rows), known over the context only
#' @param known_covariates matrix with T + h rows, known over context and horizon
#' @param symmetric average with the forecast of the mirrored series (package default)
#' @param clip_nonnegative clip at 0 when every observed target value is >= 0
#' @return a `timestral_forecast`: `quantiles` array (variate, horizon, level) plus metadata
#' @export
timestral_forecast <- function(model, y, h, past_covariates = NULL, known_covariates = NULL,
                               symmetric = TRUE, clip_nonnegative = TRUE) {
  stopifnot(inherits(model, "timestral_model"))
  tg <- .as_target_rows(y)
  inputs <- tg$rows
  storage.mode(inputs) <- "double"
  mask <- is.finite(inputs)
  inputs[!mask] <- 0
  past <- .as_cov_rows(past_covariates, tg$n, "past_covariates")
  known <- .as_cov_rows(known_covariates, tg$n + h, "known_covariates")
  if (!is.null(known) && !model$config$covariate_enabled)
    stop("this checkpoint cannot condition on known-future covariates")
  q <- .forecast_raw(model, inputs, mask, as.integer(h), past, known)
  if (isTRUE(symmetric)) {
    neg_past <- if (is.null(past)) NULL else -past
    neg_known <- if (is.null(known)) NULL else -known
    mirrored <- .forecast_raw(model, -inputs, mask, as.integer(h), neg_past, neg_known)
    q <- 0.5 * (q + (-mirrored)[, , rev(seq_len(dim(q)[3])), drop = FALSE])
  }
  if (isTRUE(clip_nonnegative)) {
    for (v in seq_len(nrow(inputs))) {
      obs <- mask[v, ]
      if (any(obs) && all(inputs[v, obs] >= 0)) q[v, , ] <- pmax(q[v, , ], 0)
    }
  }
  vnames <- tg$names
  if (is.null(vnames)) vnames <- paste0("V", seq_len(nrow(inputs)))
  dimnames(q) <- list(variate = vnames, step = seq_len(h),
                      level = format(timestral_levels()))
  structure(list(quantiles = q, levels = timestral_levels(), h = as.integer(h),
                 tsp = tg$tsp, n = tg$n), class = "timestral_forecast")
}

#' @export
print.timestral_forecast <- function(x, ...) {
  cat("<timestral_forecast> ", dim(x$quantiles)[1], " variate(s), h = ", x$h,
      ", 21 quantiles\n", sep = "")
  med <- x$quantiles[, , "0.50", drop = FALSE][, , 1]
  print(utils::head(t(rbind(med)), 6))
  invisible(x)
}

#' Long data frame: variate, step, (time,) and one column per quantile level
#' @param fc a `timestral_forecast`
#' @export
as_forecast_df <- function(fc) {
  d <- dim(fc$quantiles)
  out <- expand.grid(step = seq_len(d[2]), variate = dimnames(fc$quantiles)$variate,
                     stringsAsFactors = FALSE)[, c("variate", "step")]
  mat <- do.call(rbind, lapply(seq_len(d[1]), function(v) fc$quantiles[v, , ]))
  colnames(mat) <- sprintf("q%02d", round(100 * fc$levels))
  out <- cbind(out, mat)
  if (!is.null(fc$tsp)) out$time <- fc$tsp[2] + out$step / fc$tsp[3]
  out
}
