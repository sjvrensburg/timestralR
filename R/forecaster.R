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

## Upper bound on the forecast horizon, in steps. Without it a huge (or
## non-finite) h either hangs the decode loop or overflows to NA when
## coerced to integer; 10,000 steps is far beyond any sane forecast.
.max_horizon <- 10000L

.validate_horizon <- function(h) {
  if (length(h) != 1L || is.logical(h) || !is.numeric(h) || is.complex(h))
    stop("h (prediction_length) must be a single finite number >= 1")
  if (!is.finite(h) || h < 1)
    stop("h (prediction_length) must be >= 1, got ", format(h))
  if (h != floor(h))
    stop("h (prediction_length) must be a whole number of steps, got ", format(h))
  if (h > .max_horizon)
    stop("h (prediction_length) must be <= ", .max_horizon, ", got ", format(h))
  as.integer(h)
}

## Fail closed on non-numeric series data instead of silently coercing
## (as.numeric() turns character/factor/complex/Date input into NA codes
## or nonsense, which used to forecast as if it were real data). Numeric
## (double/integer) vectors, matrices, arrays and data frames with only
## numeric columns are accepted; ts/zoo/xts wrappers have already been
## stripped to their values by the time this is called.
.check_numeric_input <- function(x, name = "y", allow_logical = FALSE) {
  if (allow_logical && is.logical(x)) return(invisible(TRUE))
  if (is.factor(x))
    stop(name, " must be numeric, not a factor; convert it explicitly ",
         "(e.g. as.numeric()) if the factor codes are the data")
  if (is.data.frame(x)) {
    ok_types <- c("double", "integer", if (allow_logical) "logical")
    for (nm in names(x)) {
      col <- x[[nm]]
      if (is.factor(col) || !(typeof(col) %in% ok_types) ||
          inherits(col, c("Date", "POSIXt", "difftime")))
        stop(name, " column '", nm, "' must be numeric, not ",
             paste(class(col), collapse = "/"))
    }
    return(invisible(TRUE))
  }
  if (!(typeof(x) %in% c("double", "integer")) ||
      inherits(x, c("Date", "POSIXt", "difftime")))
    stop(name, " must be numeric (double or integer), not ",
         if (length(class(x)) > 1L || class(x) != typeof(x))
           paste0(paste(class(x), collapse = "/"), " (", typeof(x), ")")
         else typeof(x),
         "; convert it explicitly if the values really are the series")
  invisible(TRUE)
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
  if (length(dim(inputs)) != 3L || any(dim(inputs) == 0L))
    stop("inputs must be a non-empty (series, variate, time) array; ",
         "at least one series, variate and time step is required")
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
    grown <- .append_feedback(model, context, mask, feedback, kf, kfm, role_ids)
    context <- grown$context
    mask <- grown$mask
  }
  .pack_blocks(blocks, h)
}

## Append one decoded block to the context: targets carry `feedback`, known
## covariates their real future, past-only covariates a blank. Capped at
## max_context, on a patch boundary.
.append_feedback <- function(model, context, mask, feedback, kf, kfm, role_ids) {
  ps <- model$config$patch_size
  feedback[!is.finite(feedback) | abs(feedback) >= model$extreme_value_threshold] <- 0
  feedback_mask <- matrix(TRUE, nrow(feedback), ncol(feedback))
  if (!is.null(role_ids)) {
    known_rows <- role_ids == 2L
    past_rows <- role_ids == 1L
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
  list(context = context, mask = mask)
}

.pack_blocks <- function(blocks, h) {
  total <- sum(vapply(blocks, function(b) dim(b)[2], integer(1)))
  packed <- array(0, c(dim(blocks[[1]])[1], total, dim(blocks[[1]])[3]))
  pos <- 0L
  for (b in blocks) {
    packed[, pos + seq_len(dim(b)[2]), ] <- b
    pos <- pos + dim(b)[2]
  }
  packed[, seq_len(h), , drop = FALSE]
}

## The rollout = "quantile_paths" decoder has no upstream counterpart: the
## scheme comes from Chronos-2 (chronos2/pipeline.py). Median feedback
## treats each block's median as observed, so the next block forgets the
## uncertainty built up so far. Here each decile instead seeds its own copy
## of the context, and the next block's forecasts from all copies are mixed
## back into one set of quantiles.
.path_levels <- (1:9) / 10

## Probability mass of each level, (q[i+1] - q[i-1]) / 2 with 0 and 1 at the
## ends, normalised, as in Chronos-2.
.level_mass <- function(levels) {
  mass <- (c(levels[-1], 1) - c(0, levels[-length(levels)])) / 2
  mass / sum(mass)
}

## Quantiles at levels `at` from (R, H, Q) quantiles at `knots`: (R, H, P).
## Linear between knots, flat beyond them; fine-tuned checkpoints need not
## have the deciles as knots.
.quantiles_at <- function(q, knots, at) {
  d <- dim(q)
  if (length(knots) < 2L)
    return(array(q[, , 1L], c(d[1], d[2], length(at))))
  out <- array(0, c(d[1], d[2], length(at)))
  for (k in seq_along(at)) {
    i <- min(max(findInterval(at[k], knots), 1L), length(knots) - 1L)
    w <- min(max((at[k] - knots[i]) / (knots[i + 1L] - knots[i]), 0), 1)
    out[, , k] <- (1 - w) * q[, , i] + w * q[, , i + 1L]
  }
  out
}

## Quantiles at `levels` of the weighted samples in each column of x: a
## (length(levels), ncol(x)) matrix. Each sample sits at the middle of its
## mass (Chronos-2 puts it at the end), so negating the samples reverses
## the result, which suits a model averaged with its mirrored forecast.
.mix_quantiles <- function(x, weights, levels) {
  n <- nrow(x)
  out <- matrix(0, length(levels), ncol(x))
  total <- sum(weights)
  for (s in seq_len(ncol(x))) {
    o <- order(x[, s])
    v <- x[o, s]
    w <- weights[o]
    cw <- (cumsum(w) - w / 2) / total
    i <- findInterval(levels, cw)
    low <- i == 0L
    high <- i == n
    mid <- !low & !high
    j <- i[mid]
    t <- (levels[mid] - cw[j]) / (cw[j + 1L] - cw[j])
    out[low, s] <- v[1L]
    out[high, s] <- v[n]
    out[mid, s] <- (1 - t) * v[j] + t * v[j + 1L]
  }
  out
}

## Same contract as .block_ar_decode(). The first block is the same forward
## pass; each later one costs a forward pass over length(.path_levels) times
## the rows. A path copies a series' whole group of rows, in its own group
## id so that rows mix only within a path, and keeps the packed series-major
## row order. Paths carry the mixed quantiles of their own level from block
## to block, so their number stays fixed rather than multiplying.
.quantile_path_decode <- function(model, asm, h) {
  ps <- model$config$patch_size
  knots <- model$config$quantile_knots
  role_ids <- asm$role_ids
  target_rows <- if (is.null(role_ids)) NULL else role_ids == 0L
  remaining <- ceiling(h / ps)
  slice <- function(x, offset, blen, rows = NULL) {
    if (is.null(x)) return(NULL)
    if (!is.null(rows)) x <- x[rows, , drop = FALSE]
    x[, offset + seq_len(blen), drop = FALSE]
  }
  block <- min(model$per_shot_horizon_patches, remaining)
  blen <- block * ps
  q <- .predict_block(model, asm$rows, asm$mask, block, role_ids,
                      slice(asm$known_future, 0L, blen),
                      slice(asm$known_future_mask, 0L, blen), asm$groups)
  q <- .refine_block(model, q, asm$rows, asm$mask, target_rows)
  blocks <- list(q)
  remaining <- remaining - block
  if (remaining <= 0) return(.pack_blocks(blocks, h))

  P <- length(.path_levels)
  group_rows <- split(seq_along(asm$groups), asm$groups)
  base <- unlist(lapply(group_rows, rep, times = P), use.names = FALSE)
  path <- unlist(lapply(group_rows, function(r) rep(seq_len(P), each = length(r))),
                 use.names = FALSE)
  groups <- (asm$groups[base] - 1L) * P + path
  roles <- if (is.null(role_ids)) NULL else role_ids[base]
  paths_target <- if (is.null(roles)) NULL else roles == 0L
  context <- asm$rows[base, , drop = FALSE]
  mask <- asm$mask[base, , drop = FALSE]
  targets <- if (is.null(target_rows)) seq_len(nrow(asm$rows)) else which(target_rows)
  ## sample order: all paths at the first level, then at the second, ...
  weights <- as.vector(outer(.level_mass(.path_levels), .level_mass(knots)))
  offset <- blen
  while (remaining > 0) {
    prev <- offset - blen
    at <- .quantiles_at(q, knots, .path_levels)
    n <- length(base)
    feedback <- matrix(at[cbind(rep(base, blen), rep(seq_len(blen), each = n),
                                rep(path, blen))], n, blen)
    grown <- .append_feedback(model, context, mask, feedback,
                              slice(asm$known_future, prev, blen, base),
                              slice(asm$known_future_mask, prev, blen, base), roles)
    context <- grown$context
    mask <- grown$mask
    block <- min(model$per_shot_horizon_patches, remaining)
    blen <- block * ps
    qp <- .predict_block(model, context, mask, block, roles,
                         slice(asm$known_future, offset, blen, base),
                         slice(asm$known_future_mask, offset, blen, base), groups)
    qp <- .refine_block(model, qp, context, mask, paths_target)
    ## covariate rows stay 0: they are never read back, as feedback replaces them
    q <- array(0, c(nrow(asm$rows), blen, length(knots)))
    for (r in targets) {
      samples <- aperm(qp[base == r, , , drop = FALSE], c(1L, 3L, 2L))
      dim(samples) <- c(P * length(knots), blen)
      q[r, , ] <- t(.mix_quantiles(samples, weights, knots))
    }
    blocks[[length(blocks) + 1L]] <- q
    remaining <- remaining - block
    offset <- offset + blen
  }
  .pack_blocks(blocks, h)
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

.forecast_raw <- function(model, inputs, mask, h, past, known, rollout = "median") {
  asm <- .assemble_rows(model, inputs, mask, h, past, known)
  packed <- if (rollout == "quantile_paths") .quantile_path_decode(model, asm, h)
    else .block_ar_decode(model, asm, h)
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
.as_batch <- function(y, allow_logical = FALSE) {
  if (is.null(y))
    stop("y must have at least one time step (got NULL)")
  tsp_y <- if (stats::is.ts(y)) stats::tsp(y) else NULL
  stripped <- .strip_time_index(y)
  y <- stripped$values
  if (is.list(y) && !is.data.frame(y)) {
    if (length(y) == 0L)
      stop("y must contain at least one series (got an empty list)")
    parts <- lapply(y, function(elt) {
      elt <- .strip_time_index(elt)$values
      if (is.null(elt))
        stop("every series in y must have at least one time step (got NULL)")
      .check_numeric_input(elt, "y", allow_logical)
      m <- if (is.null(dim(elt))) matrix(as.numeric(elt), ncol = 1) else as.matrix(elt)
      storage.mode(m) <- "double"
      if (nrow(m) == 0L)
        stop("y must have at least one time step (got a series of length 0)")
      if (ncol(m) == 0L)
        stop("y must have at least one variate (got 0 columns)")
      m
    })
    Ts <- unname(vapply(parts, nrow, integer(1)))
    Vs <- unname(vapply(parts, ncol, integer(1)))
    if (length(unique(Vs)) != 1L)
      stop("all series in a batch must have the same variate count")
    ## shorter series are left-padded with NA (missing), so every series
    ## ends at the forecast origin; padding does not change a forecast
    Tm <- max(Ts)
    arr <- array(NA_real_, c(Tm, Vs[1], length(parts)))
    for (b in seq_along(parts)) arr[Tm - Ts[b] + seq_len(Ts[b]), , b] <- parts[[b]]
    ## the time base of the longest series is kept when every series'
    ## time base is its tail (ts: same frequency and end)
    lon <- which.max(Ts)
    tsps <- lapply(y, function(e) if (stats::is.ts(e)) stats::tsp(e))
    tsp <- tsps[[lon]]
    if (!is.null(tsp) && !all(vapply(tsps, function(t)
      !is.null(t) && isTRUE(all.equal(t[2:3], tsp[2:3])), logical(1))))
      tsp <- NULL
    idxs <- lapply(y, function(e) .strip_time_index(e)$index)
    index <- idxs[[lon]]
    if (!is.null(index) && !all(vapply(seq_along(idxs), function(b)
      !is.null(idxs[[b]]) && length(idxs[[b]]) == Ts[b] &&
        identical(idxs[[b]], index[Tm - Ts[b] + seq_len(Ts[b])]), logical(1))))
      index <- NULL
    return(list(arr = aperm(arr, c(3, 2, 1)), vnames = colnames(parts[[1]]),
                tsp = tsp, index = index, T = Tm, V = Vs[1], B = length(parts),
                snames = names(y), pad = Tm - Ts))
  }
  if (is.array(y) && length(dim(y)) == 3L) {
    .check_numeric_input(y, "y", allow_logical)
    if (any(dim(y) == 0L))
      stop("y must have at least one time step, one variate and one series ",
           "(got dimensions ", paste(dim(y), collapse = " x "), ")")
    storage.mode(y) <- "double"
    return(list(arr = aperm(y, c(3, 2, 1)), vnames = dimnames(y)[[2]],
                tsp = tsp_y, index = NULL, T = dim(y)[1], V = dim(y)[2],
                B = dim(y)[3], snames = dimnames(y)[[3]]))
  }
  .check_numeric_input(y, "y", allow_logical)
  ym <- if (is.null(dim(y))) matrix(as.numeric(y), ncol = 1) else as.matrix(y)
  storage.mode(ym) <- "double"
  if (nrow(ym) == 0L)
    stop("y must have at least one time step (got length 0)")
  if (ncol(ym) == 0L)
    stop("y must have at least one variate (got 0 columns)")
  arr <- aperm(array(ym, c(nrow(ym), ncol(ym), 1)), c(3, 2, 1))
  list(arr = arr, vnames = colnames(ym), tsp = tsp_y, index = stripped$index,
       T = nrow(ym), V = ncol(ym), B = 1L, snames = NULL)
}

## Covariates to a list of (C, L) matrices, one per series. Forms mirror y:
## vector/matrix (time in rows) for B=1, 3-D array (time, covariate, series),
## or a list of per-series matrices.
.as_cov_batch <- function(x, B, expected_len, name, categorical = FALSE,
                          snames = NULL, pad = NULL) {
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
  ## pad[b]: NA steps y's series b was left-padded with; its covariates may
  ## have the series' own length and are padded to match
  if (is.null(pad)) pad <- integer(B)
  one <- function(elt, b) {
    st <- .strip_time_index(elt)
    elt <- st$values
    if (is.null(elt))
      stop(name, " must have ", expected_len, " rows (time in rows), got NULL")
    .check_numeric_input(elt, name)
    m <- if (is.null(dim(elt))) matrix(as.numeric(elt), ncol = 1) else as.matrix(elt)
    storage.mode(m) <- "double"
    if (pad[b] > 0L && nrow(m) == expected_len - pad[b]) {
      m <- rbind(matrix(NA_real_, pad[b], ncol(m)), m)
    } else if (nrow(m) != expected_len) {
      stop(name, " must have ", expected_len - pad[b], " rows (time in rows)",
           if (pad[b] > 0L) paste0(" for series ", b, ", or ", expected_len,
                                   " padded like y"),
           ", got ", nrow(m))
    } else if (is.null(cov_index) && !is.null(st$index)) {
      cov_index <<- st$index
    }
    t(m)
  }
  out <- if (is.list(x) && !is.data.frame(x)) {
    if (length(x) != B) stop(name, " list must have one entry per series (", B, ")")
    lapply(seq_len(B), function(b) one(x[[b]], b))
  } else if (is.array(x) && length(dim(x)) == 3L) {
    if (dim(x)[3] != B) stop(name, " must have ", B, " series, got ", dim(x)[3])
    lapply(seq_len(B), function(b) one(x[, , b], b))
  } else {
    if (B != 1L) stop(name, " must be a list or 3-D array for a batch of ", B)
    list(one(x, 1L))
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
#' Zero-shot probabilistic forecast with 21 quantiles per step.
#'
#' @param model a `timestral_model`
#' @param y numeric vector, `ts`/`mts`, `zoo`/`xts` (index preserved), a
#'   `tsibble` (keys become the batch, measured variables the jointly
#'   forecast variates), matrix
#'   or data frame with time in rows and one column per target variate, a 3-D
#'   array (time, variate, series), or a list of per-series vectors/matrices
#'   (NA = missing). The series of a list (or the key groups of a tsibble)
#'   may differ in length: they are taken to end at the same time, the
#'   forecast origin, and shorter ones are left-padded with missing values,
#'   which does not change their forecasts. A list keeps the longest
#'   series' time index when every other series' index is its tail (for
#'   `ts`, the same frequency and end). Variates of a series are forecast jointly; series in a
#'   batch are forecast independently in one call. Covariates may also be
#'   `ts`/`zoo`/`xts` or `tsibble`; when both `y` and a covariate carry a
#'   time index, the covariate index must align with the context (and
#'   future) index of `y`. Tsibble rows are taken in index order, and
#'   implicit gaps in a regular tsibble are filled with missing values; the
#'   keys of a keyed tsibble covariate are matched to the series of `y` by
#'   value.
#' @param h horizon in steps: a single whole number between 1 and 10000
#' @param observation_mask optional logical shaped like `y` (for a tsibble
#'   `y`, a tsibble with logical measured variables); FALSE or NA marks
#'   additional missing values
#' @param past_covariates known over the context only (T time rows, or the
#'   batch forms described for `y`; in a batch of unequal lengths, a
#'   shorter series' covariates may have its own length, and are padded
#'   like it)
#' @param known_covariates known over context and horizon (T + h time rows)
#' @param categorical_past_covariates category codes known over the context
#'   only; target-encoded per series (used as-is when a series has several
#'   target variates)
#' @param categorical_known_covariates category codes over context and horizon
#' @param symmetric average with the forecast of the mirrored series (package default)
#' @param clip_nonnegative clip at 0 when every observed target value is >= 0;
#'   turn it off for a series that may fall below zero later on
#' @param rollout how a horizon longer than one decode block is rolled out:
#'   `"median"` (the default, as upstream) or `"quantile_paths"`. See
#'   the section on long horizons below.
#' @return a `timestral_forecast`: for one series, `quantiles` is a
#'   (variate, horizon, level) array; for a batch, (series, variate, horizon,
#'   level). Also holds the `observed` context values (with NA for missing),
#'   and the context `index` and `future_index` when `y` carried a time
#'   index (`ts`, `zoo`, `xts`); the future index extends the last observed
#'   interval.
#' @section Long horizons:
#' The model decodes up to one block per forward pass, 1024 steps with the
#' official checkpoints and the default `per_shot_horizon_patches`. Beyond
#' that, `rollout = "median"` feeds each block's median back as if it had
#' been observed, so the next block forgets the uncertainty built up so far
#' and the intervals become far too narrow. The medians remain usable, but
#' do not rely on the intervals past the first block.
#'
#' `rollout = "quantile_paths"` keeps the uncertainty: each of the nine
#' deciles of a block seeds its own copy of the context, and the forecasts
#' from all copies are mixed into the next block's quantiles. It widens the
#' intervals without retraining, and mostly leaves the median alone. In
#' simulations at `h = 4096` from 512 steps of context, it brought the
#' coverage of the 80% interval in the last block on a random walk back to
#' 55-76%, depending on the checkpoint (from 1-2%), and cut the pinball loss
#' by 14-25%.
#'
#' Use it for series that wander. On a stationary or nearly deterministic
#' series the intervals become too wide instead, and paths do not correct a
#' median that has missed a trend. Keep the default
#' `per_shot_horizon_patches`: with shorter blocks the intervals collapse
#' again, at a higher cost. Each block after the first costs a forward pass
#' over nine times the rows, which took about 6 to 15 times as long overall
#' in those simulations. Forecasts within one block are the same under both
#' rollouts. This rollout is not part of upstream `timestral`, so it has no
#' official outputs to check against; the scheme comes from Chronos-2.
#' @examples
#' ## the tiny bundled checkpoint runs offline but its forecasts are
#' ## meaningless; use timestral_load("5M") or larger for real ones
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' ## univariate ts: index preserved, fan chart and ts accessors work
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' plot(fc)
#' timestral_median(fc)
#'
#' ## multivariate: columns of y are forecast jointly
#' fc2 <- timestral_forecast(m, cbind(mdeaths, fdeaths), h = 12)
#'
#' ## zoo/xts inputs keep their time index (needs the zoo package)
#' if (requireNamespace("zoo", quietly = TRUE)) {
#'   months <- seq(as.Date("1949-01-01"), by = "month",
#'                 length.out = length(AirPassengers))
#'   yz <- zoo::zoo(as.numeric(AirPassengers), order.by = months)
#'   fc3 <- timestral_forecast(m, yz, h = 24)
#'   head(as_forecast_df(fc3))   # time column holds Dates
#' }
#'
#' ## tsibble input: keys become the batch (needs the tsibble package)
#' if (requireNamespace("tsibble", quietly = TRUE)) {
#'   fc4 <- timestral_forecast(m, tsibble::as_tsibble(AirPassengers), h = 24)
#'   tsibble::as_tsibble(fc4)
#' }
#' @export
timestral_forecast <- function(model, y, h, observation_mask = NULL,
                               past_covariates = NULL, known_covariates = NULL,
                               categorical_past_covariates = NULL,
                               categorical_known_covariates = NULL,
                               symmetric = TRUE, clip_nonnegative = TRUE,
                               rollout = c("median", "quantile_paths")) {
  stopifnot(inherits(model, "timestral_model"))
  rollout <- match.arg(rollout)
  h <- .validate_horizon(h)
  tg <- .batch_dispatch(y)
  future_idx <- .future_index(tg$index, h)
  inputs <- tg$arr
  mask <- is.finite(inputs)
  om <- .as_obs_mask(observation_mask, tg)
  if (!is.null(om)) mask <- mask & om
  for (b in seq_len(tg$B)) {
    if (!any(mask[b, , ]))
      stop("series ", if (is.null(tg$snames)) b else tg$snames[b],
           " of y has no finite observed values; a forecast needs at ",
           "least one finite (non-NA, non-infinite) observation per series")
  }
  observed_arr <- inputs
  observed_arr[!mask] <- NA_real_
  dimnames(observed_arr) <- list(series = tg$snames, variate = tg$vnames,
                                 time = NULL)
  inputs[!mask] <- 0
  cfg <- model$config
  get_cov <- function(x, len, nm, cat = FALSE) {
    out <- .as_cov_batch(x, tg$B, len, nm, cat, tg$snames, tg$pad)
    if (!is.null(out) && !timestral_supports_covariates(model))
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
  if (!is.null(known) && !timestral_supports_known_covariates(model))
    stop("this checkpoint cannot condition on known-future covariates")
  ## quantile levels come from the checkpoint (fine-tuned checkpoints may
  ## use other knots than timestral_levels())
  levels <- cfg$quantile_knots
  if (isTRUE(symmetric) && any(abs(levels + rev(levels) - 1) >= 1e-9))
    stop("symmetric averaging needs quantile levels symmetric about 0.5; ",
         "this checkpoint's levels are not, so use symmetric = FALSE")
  q <- .forecast_raw(model, inputs, mask, h, past, known, rollout)
  if (isTRUE(symmetric)) {
    negate <- function(lst) if (is.null(lst)) NULL else lapply(lst, function(m) -m)
    mirrored <- .forecast_raw(model, -inputs, mask, h, negate(past), negate(known), rollout)
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
                         level = format(levels))
    out_q <- q1
  } else {
    dimnames(q) <- list(series = snames, variate = vnames, step = seq_len(h),
                        level = format(levels))
    out_q <- q
  }
  structure(list(quantiles = out_q, levels = levels, h = h,
                 tsp = tg$tsp, n = tg$T, n_series = tg$B,
                 observed = observed_arr, index = tg$index,
                 index_var = tg$index_var, future_index = future_idx),
            class = "timestral_forecast")
}

#' Forecast with a Timestral model through `predict()`
#'
#' The [stats::predict()] interface to [timestral_forecast()]. Timestral is
#' a zero-shot model, so there is nothing fitted to predict from: `newdata`
#' is the series to forecast (the context), not new predictor values.
#'
#' @param object a `timestral_model` from [timestral_load()]
#' @param newdata the series to forecast; anything [timestral_forecast()]
#'   accepts as `y`
#' @param h horizon in steps (required)
#' @param ... further arguments to [timestral_forecast()], such as
#'   covariates, `observation_mask` or `symmetric`
#' @return A `timestral_forecast`, exactly as from [timestral_forecast()];
#'   use [timestral_median()] or [timestral_quantile()] for point values.
#' @examples
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' fc <- predict(m, AirPassengers, h = 12)
#' timestral_median(fc)
#' @export
predict.timestral_model <- function(object, newdata, h, ...) {
  if (missing(newdata))
    stop("predict() needs `newdata`: the series to forecast")
  if (missing(h))
    stop("predict() needs `h`: the forecast horizon in steps")
  timestral_forecast(object, newdata, h = h, ...)
}

.level_slice <- function(fc, idx) {
  if (fc$n_series == 1L) fc$quantiles[, , idx, drop = FALSE]
  else fc$quantiles[, , , idx, drop = FALSE]
}

## Present a single-series level slice as a ts/mts when the input was a
## ts: time in rows, one column per variate, starting after the context.
## A ts of forecast values starting one step after a context with `tsp`.
.future_ts <- function(x, tsp) {
  stats::ts(x, start = tsp[2] + 1 / tsp[3], frequency = tsp[3])
}

.maybe_ts <- function(fc, slice) {
  if (fc$n_series != 1L || is.null(fc$tsp)) return(slice)
  nv <- dim(fc$quantiles)[1]
  m <- matrix(slice[, , 1], nrow = nv)
  out <- .future_ts(t(m), fc$tsp)
  if (nv > 1L) colnames(out) <- dimnames(fc$quantiles)[[1L]]
  out
}

#' Median forecast
#'
#' @param fc a `timestral_forecast`
#' @return The 0.5-quantile slice. For a single series forecast from a
#'   `ts`, a `ts`/`mts` starting after the context; otherwise the same
#'   slice of the quantiles array as before.
#' @examples
#' ## the tiny bundled checkpoint runs offline but its forecasts are
#' ## meaningless; use timestral_load("5M") or larger for real ones
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' timestral_median(fc)   # a ts, starting Jan 1961
#' @export
timestral_median <- function(fc) {
  .maybe_ts(fc, .level_slice(fc, which(abs(fc$levels - 0.5) < 1e-9)))
}

#' Forecast at one quantile level
#'
#' Forecast at one quantile level; errors if `level` is not one of the
#' model's quantile levels.
#'
#' @param fc a `timestral_forecast`
#' @param level one of the forecast's quantile levels, `fc$levels`
#' @return The requested quantile slice, as a `ts`/`mts` for a single
#'   series forecast from a `ts`, otherwise an array slice.
#' @examples
#' ## the tiny bundled checkpoint runs offline but its forecasts are
#' ## meaningless; use timestral_load("5M") or larger for real ones
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' timestral_quantile(fc, 0.9)
#' @export
timestral_quantile <- function(fc, level) {
  ## levels match to 1e-9: a literal 0.15 and seq()'s 0.15000000000000002
  ## name the same knot
  i <- if (is.numeric(level) && length(level) == 1 && is.finite(level))
    which(abs(fc$levels - level) < 1e-9)
  if (length(i) != 1)
    stop("level ", level, " is not one of the model's quantile levels")
  .maybe_ts(fc, .level_slice(fc, i))
}

#' @export
print.timestral_forecast <- function(x, ...) {
  d <- dim(x$quantiles)
  if (x$n_series == 1L) {
    cat("<timestral_forecast> ", d[1], " variate(s), h = ", x$h,
        ", ", length(x$levels), " quantiles\n", sep = "")
    med <- x$quantiles[, , which(abs(x$levels - 0.5) < 1e-9), drop = FALSE][, , 1]
    print(utils::head(t(rbind(med)), 6))
  } else {
    cat("<timestral_forecast> ", x$n_series, " series x ", d[2],
        " variate(s), h = ", x$h, ", ", length(x$levels), " quantiles\n",
        sep = "")
  }
  invisible(x)
}

#' Forecast as a long data frame
#'
#' Long data frame: variate, step, (time,) and one column per quantile level.
#'
#' @param fc a `timestral_forecast`
#' @return A data frame with variate, step, optional time, and one column
#'   per quantile level (plus a series column for a batch).
#' @examples
#' ## the tiny bundled checkpoint runs offline but its forecasts are
#' ## meaningless; use timestral_load("5M") or larger for real ones
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' head(as_forecast_df(fc))
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
