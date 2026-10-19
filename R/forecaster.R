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

## inputs/mask: (B, V, T) arrays; past/known: lists of (C, L) matrices per
## series (NULL entries allowed). Rows are packed series-major, as upstream.
.assemble_rows <- function(model, inputs, mask, h, past, known) {
  ps <- model$config$patch_size
  B <- dim(inputs)[1]; n_tgt <- dim(inputs)[2]; T_len <- dim(inputs)[3]
  horizon <- ceiling(h / ps) * ps
  rows_list <- list(); mask_list <- list()
  roles_all <- integer(0); groups <- integer(0)
  kf_blocks <- list()
  n_past <- if (is.null(past) || is.null(past[[1]])) 0L else nrow(past[[1]])
  n_known <- if (is.null(known) || is.null(known[[1]])) 0L else nrow(known[[1]])
  target_idx <- integer(0)
  for (b in seq_len(B)) {
    ib <- matrix(inputs[b, , ], nrow = n_tgt)
    mb <- matrix(mask[b, , ], nrow = n_tgt)
    ctx_rows <- list(.ffill_rows(ib, mb))
    ctx_masks <- list(mb)
    roles <- rep(0L, n_tgt)
    if (!is.null(past) && !is.null(past[[b]]) && nrow(past[[b]]) > 0) {
      ctx_masks <- c(ctx_masks, list(!is.na(past[[b]])))
      ctx_rows <- c(ctx_rows, list(.nan_to_num(past[[b]])))
      roles <- c(roles, rep(1L, nrow(past[[b]])))
    }
    kf_b <- NULL
    if (!is.null(known) && !is.null(known[[b]]) && nrow(known[[b]]) > 0) {
      kn <- known[[b]]
      kc_ctx <- kn[, seq_len(T_len), drop = FALSE]
      kc_fut <- kn[, T_len + seq_len(h), drop = FALSE]
      ctx_masks <- c(ctx_masks, list(!is.na(kc_ctx)))
      ctx_rows <- c(ctx_rows, list(.nan_to_num(kc_ctx)))
      roles <- c(roles, rep(2L, nrow(kn)))
      kf_b <- list(vals = .nan_to_num(kc_fut), obs = !is.na(kc_fut))
    }
    rows_b <- do.call(rbind, ctx_rows)
    target_idx <- c(target_idx, length(rows_list) * 0L + (sum(vapply(rows_list, NROW, integer(1))) + seq_len(n_tgt)))
    rows_list[[b]] <- rows_b
    mask_list[[b]] <- do.call(rbind, ctx_masks)
    roles_all <- c(roles_all, roles)
    groups <- c(groups, rep(b, nrow(rows_b)))
    kf_blocks[[b]] <- kf_b
  }
  rows <- do.call(rbind, rows_list)
  msk <- do.call(rbind, mask_list)
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
    pos <- 0L
    for (b in seq_len(B)) {
      v_b <- sum(groups == b)
      if (!is.null(kf_blocks[[b]])) {
        k0 <- pos + n_tgt + n_past
        known_future[k0 + seq_len(n_known), seq_len(h)] <- kf_blocks[[b]]$vals
        known_future_mask[k0 + seq_len(n_known), seq_len(h)] <- kf_blocks[[b]]$obs
      }
      pos <- pos + v_b
    }
  }
  list(rows = rows, mask = msk,
       role_ids = if (n_past + n_known > 0L) roles_all else NULL,
       known_future = known_future, known_future_mask = known_future_mask,
       n_tgt = n_tgt, n_series = B, groups = groups, target_idx = target_idx)
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
    block <- min(model$per_shot_horizon_patches, remaining)
    blen <- block * ps
    kf <- if (!is.null(asm$known_future)) asm$known_future[, offset + seq_len(blen), drop = FALSE] else NULL
    kfm <- if (!is.null(asm$known_future_mask)) asm$known_future_mask[, offset + seq_len(blen), drop = FALSE] else NULL
    q <- .predict_block(model, context, mask, block, role_ids, kf, kfm, asm$groups)
    q <- .refine_block(model, q, context, mask, target_rows)
    blocks[[length(blocks) + 1L]] <- q
    remaining <- remaining - block
    offset <- offset + blen
    if (remaining <= 0) break
    feedback <- matrix(q[, , med_idx], nrow = nrow(context))
    feedback[!is.finite(feedback) | abs(feedback) >= model$extreme_value_threshold] <- 0
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
  anchor <- st$scale[, ncol(inputs)] * model$quantile_real_cap_k
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
  q <- packed[asm$target_idx, , , drop = FALSE]
  if (model$quantile_real_cap_k > 0) {
    tgt_rows <- asm$rows[asm$target_idx, , drop = FALSE]
    tgt_mask <- asm$mask[asm$target_idx, , drop = FALSE]
    q <- .cap_quantiles(model, q, tgt_rows, tgt_mask)
  }
  if (any(is.nan(q)))
    stop("forecast produced ", sum(is.nan(q)),
         " NaN values; the input is degenerate")
  dim(q) <- c(asm$n_tgt, asm$n_series, dim(q)[2], dim(q)[3])
  aperm(q, c(2, 1, 3, 4))
}

## Extract a time index from ts/zoo/xts input and strip the input to
## plain values. Returns list(values, index); index is NULL for plain
## inputs. zoo/xts are optional (Suggests): they are only touched when
## the user passes an object of those classes.
.strip_time_index <- function(x) {
  if (stats::is.ts(x))
    return(list(values = x, index = as.numeric(stats::time(x))))
  if (inherits(x, "xts")) {
    if (!requireNamespace("xts", quietly = TRUE))
      stop("xts input requires the xts package")
    return(list(values = zoo::coredata(x), index = zoo::index(x)))
  }
  if (inherits(x, "zoo")) {
    if (!requireNamespace("zoo", quietly = TRUE))
      stop("zoo input requires the zoo package")
    return(list(values = zoo::coredata(x), index = zoo::index(x)))
  }
  list(values = x, index = NULL)
}

## Future index by extending the last observed interval (any index class
## supporting arithmetic: numeric, Date, POSIXct, ts decimal time, ...).
.future_index <- function(index, h) {
  if (is.null(index) || length(index) < 2L) return(NULL)
  n <- length(index)
  index[n] + (index[n] - index[n - 1L]) * seq_len(h)
}

## Normalise y to a (B, V, T) array. Accepted forms: numeric vector (B=V=1),
## ts/mts, zoo/xts, matrix/data frame with time in rows (B=1), 3-D array
## (time, variate, series), or a list of per-series vectors/matrices.
.as_batch <- function(y) {
  tsp_y <- if (stats::is.ts(y)) stats::tsp(y) else NULL
  stripped <- .strip_time_index(y)
  y <- stripped$values
  if (is.list(y) && !is.data.frame(y)) {
    parts <- lapply(y, function(elt) {
      elt <- .strip_time_index(elt)$values
      m <- if (is.null(dim(elt))) matrix(as.numeric(elt), ncol = 1) else as.matrix(elt)
      storage.mode(m) <- "double"
      m
    })
    Ts <- vapply(parts, nrow, integer(1)); Vs <- vapply(parts, ncol, integer(1))
    if (length(unique(Ts)) != 1L || length(unique(Vs)) != 1L)
      stop("all series in a batch must have the same length and variate count")
    arr <- array(0, c(Ts[1], Vs[1], length(parts)))
    for (b in seq_along(parts)) arr[, , b] <- parts[[b]]
    return(list(arr = aperm(arr, c(3, 2, 1)), vnames = colnames(parts[[1]]),
                tsp = tsp_y, index = NULL, T = Ts[1], V = Vs[1],
                B = length(parts), snames = names(y)))
  }
  if (is.array(y) && length(dim(y)) == 3L) {
    storage.mode(y) <- "double"
    return(list(arr = aperm(y, c(3, 2, 1)), vnames = dimnames(y)[[2]],
                tsp = tsp_y, index = NULL, T = dim(y)[1], V = dim(y)[2],
                B = dim(y)[3], snames = dimnames(y)[[3]]))
  }
  ym <- if (is.null(dim(y))) matrix(as.numeric(y), ncol = 1) else as.matrix(y)
  storage.mode(ym) <- "double"
  arr <- aperm(array(ym, c(nrow(ym), ncol(ym), 1)), c(3, 2, 1))
  list(arr = arr, vnames = colnames(ym), tsp = tsp_y, index = stripped$index,
       T = nrow(ym), V = ncol(ym), B = 1L, snames = NULL)
}

## Covariates to a list of (C, L) matrices, one per series. Forms mirror y:
## vector/matrix (time in rows) for B=1, 3-D array (time, covariate, series),
## or a list of per-series matrices.
.as_cov_batch <- function(x, B, expected_len, name, categorical = FALSE,
                          snames = NULL) {
  if (is.null(x)) return(NULL)
  cov_index <- NULL
  if (inherits(x, "tbl_ts")) {
    ## Per-key matrices, matched to the series of y by key value, then
    ## validated like any other list of per-series covariates.
    gr <- .tsibble_groups(x)
    mats <- gr$groups
    if (gr$keyed) {
      if (is.null(snames) || length(mats) != B ||
          !setequal(snames, gr$snames))
        stop(name, " tsibble keys (", paste(gr$snames, collapse = ", "),
             ") must match the series of y (",
             if (is.null(snames)) "unnamed" else paste(snames, collapse = ", "),
             ")")
      mats <- mats[match(snames, gr$snames)]
    }
    x <- mats
    cov_index <- gr$index
  }
  one <- function(elt) {
    st <- .strip_time_index(elt)
    if (is.null(cov_index) && !is.null(st$index)) cov_index <<- st$index
    elt <- st$values
    m <- if (is.null(dim(elt))) matrix(as.numeric(elt), ncol = 1) else as.matrix(elt)
    storage.mode(m) <- "double"
    if (nrow(m) != expected_len)
      stop(name, " must have ", expected_len, " rows (time in rows), got ", nrow(m))
    t(m)
  }
  out <- if (is.list(x) && !is.data.frame(x)) {
    if (length(x) != B) stop(name, " list must have one entry per series (", B, ")")
    lapply(x, one)
  } else if (is.array(x) && length(dim(x)) == 3L) {
    if (dim(x)[3] != B) stop(name, " must have ", B, " series, got ", dim(x)[3])
    lapply(seq_len(B), function(b) one(x[, , b]))
  } else {
    if (B != 1L) stop(name, " must be a list or 3-D array for a batch of ", B)
    list(one(x))
  }
  Cs <- vapply(out, nrow, integer(1))
  if (length(unique(Cs)) != 1L)
    stop(name, " must have the same covariate count for every series")
  if (Cs[1] == 0L) return(NULL)
  if (categorical && any(vapply(out, function(m) any(is.infinite(m)), logical(1))))
    stop(name, " must hold finite codes or NaN; got an infinite value")
  attr(out, "index") <- cov_index
  out
}

.as_obs_mask <- function(observation_mask, batch) {
  if (is.null(observation_mask)) return(NULL)
  fake <- .batch_dispatch(observation_mask, allow_logical = TRUE)
  if (!identical(dim(fake$arr), dim(batch$arr)))
    stop("observation_mask must be shaped like the context")
  out <- fake$arr != 0
  ## NA (e.g. rows added by filling tsibble gaps) marks a missing value
  out[is.na(out)] <- FALSE
  out
}

.combine_cov <- function(numeric_list, encoded_list, B) {
  if (is.null(numeric_list) && is.null(encoded_list)) return(NULL)
  lapply(seq_len(B), function(b) {
    a <- if (is.null(numeric_list)) NULL else numeric_list[[b]]
    e <- if (is.null(encoded_list)) NULL else encoded_list[[b]]
    if (is.null(a)) return(e)
    if (is.null(e)) return(a)
    rbind(a, e)
  })
}

#' Forecast one or more series
#'
#' @param model a `timestral_model`
#' @param y numeric vector, `ts`/`mts`, `zoo`/`xts` (index preserved), a
#'   `tsibble` (keys become the batch, measured variables the jointly
#'   forecast variates), matrix
#'   or data frame with time in rows and one column per target variate, a 3-D
#'   array (time, variate, series), or a list of per-series vectors/matrices
#'   (NA = missing). Variates of a series are forecast jointly; series in a
#'   batch are forecast independently in one call. Covariates may also be
#'   `ts`/`zoo`/`xts` or `tsibble`; when both `y` and a covariate carry a
#'   time index, the covariate index must align with the context (and
#'   future) index of `y`. Tsibble rows are taken in index order, and
#'   implicit gaps in a regular tsibble are filled with missing values; the
#'   keys of a keyed tsibble covariate are matched to the series of `y` by
#'   value.
#' @param h horizon in steps
#' @param observation_mask optional logical shaped like `y` (for a tsibble
#'   `y`, a tsibble with logical measured variables); FALSE or NA marks
#'   additional missing values
#' @param past_covariates known over the context only (T time rows, or the
#'   batch forms described for `y`)
#' @param known_covariates known over context and horizon (T + h time rows)
#' @param categorical_past_covariates category codes known over the context
#'   only; target-encoded per series (used as-is when a series has several
#'   target variates)
#' @param categorical_known_covariates category codes over context and horizon
#' @param symmetric average with the forecast of the mirrored series (package default)
#' @param clip_nonnegative clip at 0 when every observed target value is >= 0
#' @return a `timestral_forecast`: for one series, `quantiles` is a
#'   (variate, horizon, level) array; for a batch, (series, variate, horizon,
#'   level). Also holds the `observed` context values (with NA for missing),
#'   and the context `index` and `future_index` when `y` carried a time
#'   index (`ts`, `zoo`, `xts`); the future index extends the last observed
#'   interval.
#' @export
timestral_forecast <- function(model, y, h, observation_mask = NULL,
                               past_covariates = NULL, known_covariates = NULL,
                               categorical_past_covariates = NULL,
                               categorical_known_covariates = NULL,
                               symmetric = TRUE, clip_nonnegative = TRUE) {
  stopifnot(inherits(model, "timestral_model"))
  if (length(h) != 1 || is.na(h) || h < 1)
    stop("h (prediction_length) must be >= 1, got ", h)
  h <- as.integer(h)
  tg <- .batch_dispatch(y)
  future_idx <- .future_index(tg$index, h)
  inputs <- tg$arr
  mask <- is.finite(inputs)
  om <- .as_obs_mask(observation_mask, tg)
  if (!is.null(om)) mask <- mask & om
  observed_arr <- inputs
  observed_arr[!mask] <- NA_real_
  dimnames(observed_arr) <- list(series = tg$snames, variate = tg$vnames,
                                 time = NULL)
  inputs[!mask] <- 0
  cfg <- model$config
  get_cov <- function(x, len, nm, cat = FALSE) {
    out <- .as_cov_batch(x, tg$B, len, nm, cat, tg$snames)
    if (!is.null(out) && !cfg$covariate_enabled)
      stop("this checkpoint was not trained with covariates; got ", nm)
    idx <- attr(out, "index")
    if (!is.null(out) && !is.null(idx) && !is.null(tg$index)) {
      expected <- if (len == tg$T) tg$index else c(tg$index, future_idx)
      aligned <- length(idx) == length(expected) &&
        isTRUE(all.equal(idx, expected, check.attributes = FALSE,
                         tolerance = 1e-8))
      if (!aligned)
        stop(nm, " time index does not align with the time index of y")
    }
    out
  }
  past_num <- get_cov(past_covariates, tg$T, "past_covariates")
  known_num <- get_cov(known_covariates, tg$T + h, "known_covariates")
  cat_past <- get_cov(categorical_past_covariates, tg$T,
                      "categorical_past_covariates", TRUE)
  cat_known <- get_cov(categorical_known_covariates, tg$T + h,
                       "categorical_known_covariates", TRUE)
  enc_past <- .encode_categorical_batch(model, cat_past, inputs, mask, tg$B, tg$T)
  enc_known <- .encode_categorical_batch(model, cat_known, inputs, mask, tg$B, tg$T)
  ## Encoded categorical rows for the context come from the past encoding;
  ## for known covariates the encoding already spans context + horizon.
  past <- .combine_cov(past_num, enc_past, tg$B)
  known <- .combine_cov(known_num, enc_known, tg$B)
  if (!is.null(known) && !cfg$supports_known_covariates)
    stop("this checkpoint cannot condition on known-future covariates")
  q <- .forecast_raw(model, inputs, mask, h, past, known)
  if (isTRUE(symmetric)) {
    negate <- function(lst) if (is.null(lst)) NULL else lapply(lst, function(m) -m)
    mirrored <- .forecast_raw(model, -inputs, mask, h, negate(past), negate(known))
    q <- 0.5 * (q + (-mirrored)[, , , rev(seq_len(dim(q)[4])), drop = FALSE])
  }
  if (isTRUE(clip_nonnegative)) {
    for (b in seq_len(tg$B)) for (v in seq_len(tg$V)) {
      obs <- mask[b, v, ]
      if (any(obs) && all(inputs[b, v, obs] >= 0)) q[b, v, , ] <- pmax(q[b, v, , ], 0)
    }
  }
  vnames <- tg$vnames
  if (is.null(vnames)) vnames <- paste0("V", seq_len(tg$V))
  snames <- tg$snames
  if (is.null(snames)) snames <- paste0("S", seq_len(tg$B))
  if (tg$B == 1L) {
    q1 <- q[1, , , , drop = FALSE]
    dim(q1) <- dim(q1)[2:4]
    dimnames(q1) <- list(variate = vnames, step = seq_len(h),
                         level = format(timestral_levels()))
    out_q <- q1
  } else {
    dimnames(q) <- list(series = snames, variate = vnames, step = seq_len(h),
                        level = format(timestral_levels()))
    out_q <- q
  }
  structure(list(quantiles = out_q, levels = timestral_levels(), h = h,
                 tsp = tg$tsp, n = tg$T, n_series = tg$B,
                 observed = observed_arr, index = tg$index,
                 index_var = tg$index_var, future_index = future_idx),
            class = "timestral_forecast")
}

.level_slice <- function(fc, idx) {
  if (fc$n_series == 1L) fc$quantiles[, , idx, drop = FALSE]
  else fc$quantiles[, , , idx, drop = FALSE]
}

## Present a single-series level slice as a ts/mts when the input was a
## ts: time in rows, one column per variate, starting after the context.
.maybe_ts <- function(fc, slice) {
  if (fc$n_series != 1L || is.null(fc$tsp)) return(slice)
  nv <- dim(fc$quantiles)[1]
  m <- matrix(slice[, , 1], nrow = nv)
  out <- stats::ts(t(m), start = fc$tsp[2] + 1 / fc$tsp[3],
                   frequency = fc$tsp[3])
  if (nv > 1L) colnames(out) <- dimnames(fc$quantiles)[[1L]]
  out
}

#' Median forecast
#' @param fc a `timestral_forecast`
#' @return The 0.5-quantile slice. For a single series forecast from a
#'   `ts`, a `ts`/`mts` starting after the context; otherwise the same
#'   slice of the quantiles array as before.
#' @export
timestral_median <- function(fc) {
  .maybe_ts(fc, .level_slice(fc, which(fc$levels == 0.5)))
}

#' Forecast at one quantile level
#' @param fc a `timestral_forecast`
#' @param level one of `timestral_levels()`
#' @return The requested quantile slice, as a `ts`/`mts` for a single
#'   series forecast from a `ts`, otherwise an array slice.
#' @export
timestral_quantile <- function(fc, level) {
  if (length(level) != 1 || !level %in% fc$levels)
    stop("level ", level, " is not one of the model's quantile levels")
  .maybe_ts(fc, .level_slice(fc, match(level, fc$levels)))
}

#' @export
print.timestral_forecast <- function(x, ...) {
  d <- dim(x$quantiles)
  if (x$n_series == 1L) {
    cat("<timestral_forecast> ", d[1], " variate(s), h = ", x$h,
        ", 21 quantiles\n", sep = "")
    med <- x$quantiles[, , "0.50", drop = FALSE][, , 1]
    print(utils::head(t(rbind(med)), 6))
  } else {
    cat("<timestral_forecast> ", x$n_series, " series x ", d[2],
        " variate(s), h = ", x$h, ", 21 quantiles\n", sep = "")
  }
  invisible(x)
}

#' Long data frame: variate, step, (time,) and one column per quantile level
#' @param fc a `timestral_forecast`
#' @export
as_forecast_df <- function(fc) {
  q <- fc$quantiles
  if (fc$n_series == 1L) {
    d <- dim(q)
    out <- expand.grid(step = seq_len(d[2]), variate = dimnames(q)$variate,
                       stringsAsFactors = FALSE)[, c("variate", "step")]
    mat <- do.call(rbind, lapply(seq_len(d[1]), function(v) q[v, , ]))
    colnames(mat) <- sprintf("q%02d", round(100 * fc$levels))
    out <- cbind(out, mat)
    if (!is.null(fc$future_index)) out$time <- fc$future_index[out$step]
    else if (!is.null(fc$tsp)) out$time <- fc$tsp[2] + out$step / fc$tsp[3]
    return(out)
  }
  d <- dim(q)
  frames <- lapply(seq_len(d[1]), function(b) {
    out <- expand.grid(step = seq_len(d[3]), variate = dimnames(q)$variate,
                       stringsAsFactors = FALSE)[, c("variate", "step")]
    mat <- do.call(rbind, lapply(seq_len(d[2]), function(v) q[b, v, , ]))
    colnames(mat) <- sprintf("q%02d", round(100 * fc$levels))
    out <- cbind(series = dimnames(q)$series[b], out, mat,
                 stringsAsFactors = FALSE)
    if (!is.null(fc$future_index)) out$time <- fc$future_index[out$step]
    out
  })
  do.call(rbind, frames)
}
