## Port of timestral/layers.py + model.py (network and predict_block).

.w <- function(model, name) model$weights[[name]]

.lin <- function(x, weight, bias = NULL) {
  y <- tcrossprod(x, weight)
  if (!is.null(bias)) y <- sweep(y, 2, bias, `+`)
  y
}

.silu <- function(x) x / (1 + exp(-x))
.sigmoid <- function(x) stats::plogis(x)
.softplus <- function(x) ifelse(x > 20, x, log1p(exp(pmin(x, 20))))

.residual_block <- function(x, w_hidden, w_output, w_residual,
                            b_hidden = NULL, b_output = NULL, b_residual = NULL) {
  .lin(.silu(.lin(x, w_hidden, b_hidden)), w_output, b_output) +
    .lin(x, w_residual, b_residual)
}

.rms_norm <- function(x, scale, eps) {
  ## x: (..., D) matrix; normalise over columns
  ms <- rowMeans(x^2)
  sweep(x / sqrt(ms + eps), 2, scale, `*`)
}

.rms_norm3 <- function(H, scale, eps) {
  for (r in seq_len(dim(H)[1])) H[r, , ] <- .rms_norm(H[r, , , drop = TRUE], scale, eps)
  H
}

## ---- attention -------------------------------------------------------------

.heads_qkv <- function(qkv, num_heads, head_dim) {
  ## qkv: (S, 3*D) with torch layout (qkv, head_dim, head) per block.
  ## Returns list of 3 lists (q, k, v), each a list of (S, head_dim) per head.
  D <- num_heads * head_dim
  out <- vector("list", 3)
  for (qi in 1:3) {
    block <- qkv[, (qi - 1) * D + seq_len(D), drop = FALSE]
    heads <- vector("list", num_heads)
    for (h in seq_len(num_heads))
      heads[[h]] <- block[, seq(h, D, by = num_heads), drop = FALSE]
    out[[qi]] <- heads
  }
  out
}

.softmax_rows <- function(scores) {
  scores[!is.finite(scores)] <- -1e30
  m <- apply(scores, 1, max)
  e <- exp(scores - m)
  e / rowSums(e)
}

.attn_time <- function(model, H, prefix, is_time, time_mask, cos_sin) {
  cfg <- model$config
  R <- dim(H)[1]; S <- dim(H)[2]; D <- dim(H)[3]
  nh <- cfg$num_heads; hd <- D %/% nh
  Wqkv <- .w(model, paste0(prefix, ".wQKV.weight"))
  Wo <- .w(model, paste0(prefix, ".wO.weight"))
  Wg <- .w(model, paste0(prefix, "._gate.weight"))
  out <- array(0, dim = dim(H))
  for (r in seq_len(R)) {
    x <- H[r, , ]
    qkv <- .heads_qkv(.lin(x, Wqkv, .w(model, paste0(prefix, ".wQKV.bias"))), nh, hd)
    head_outs <- vector("list", nh)
    for (h in seq_len(nh)) {
      q <- qkv[[1]][[h]]; k <- qkv[[2]][[h]]; v <- qkv[[3]][[h]]
      if (is_time && !is.null(cos_sin)) {
        cosr <- cos_sin$cos[r, , ]; sinr <- cos_sin$sin[r, , ]
        rot <- function(z) cbind(-z[, (hd / 2 + 1):hd, drop = FALSE],
                                 z[, seq_len(hd / 2), drop = FALSE])
        q <- q * cosr + rot(q) * sinr
        k <- k * cosr + rot(k) * sinr
      }
      scores <- tcrossprod(q, k) / sqrt(hd)
      if (!is.null(time_mask)) scores[!time_mask[r, , ]] <- -Inf
      head_outs[[h]] <- .softmax_rows(scores) %*% v
    }
    o <- do.call(cbind, head_outs)
    if (!is.null(Wg)) o <- o * .sigmoid(.lin(x, Wg, .w(model, paste0(prefix, "._gate.bias"))))
    out[r, , ] <- .lin(o, Wo, .w(model, paste0(prefix, ".wO.bias")))
  }
  out
}

.attn_variate <- function(model, H, prefix, variate_mask) {
  cfg <- model$config
  R <- dim(H)[1]; S <- dim(H)[2]; D <- dim(H)[3]
  nh <- cfg$num_heads; hd <- D %/% nh
  Wqkv <- .w(model, paste0(prefix, ".wQKV.weight"))
  Wo <- .w(model, paste0(prefix, ".wO.weight"))
  Wg <- .w(model, paste0(prefix, "._gate.weight"))
  out <- array(0, dim = dim(H))
  for (s in seq_len(S)) {
    x <- H[, s, ]  # (R, D): rows of one series at this slot
    if (R == 1L) x <- matrix(x, nrow = 1)
    qkv <- .heads_qkv(.lin(x, Wqkv, .w(model, paste0(prefix, ".wQKV.bias"))), nh, hd)
    head_outs <- vector("list", nh)
    for (h in seq_len(nh)) {
      scores <- tcrossprod(qkv[[1]][[h]], qkv[[2]][[h]]) / sqrt(hd)
      if (!is.null(variate_mask)) scores[!variate_mask[s, , ]] <- -Inf
      head_outs[[h]] <- .softmax_rows(scores) %*% qkv[[3]][[h]]
    }
    o <- do.call(cbind, head_outs)
    if (!is.null(Wg)) o <- o * .sigmoid(.lin(x, Wg, .w(model, paste0(prefix, "._gate.bias"))))
    out[, s, ] <- .lin(o, Wo, .w(model, paste0(prefix, ".wO.bias")))
  }
  out
}

.rope_cos_sin <- function(model, positions) {
  cfg <- model$config
  hd <- cfg$d_model %/% cfg$num_heads
  rope_dim <- as.integer(hd * cfg$rope_partial_factor)
  inv_freq <- numeric(hd / 2)
  idx <- seq(0, rope_dim - 1, by = 2)
  inv_freq[seq_along(idx)] <- 1 / (cfg$rope_theta^(idx / rope_dim))
  R <- nrow(positions); S <- ncol(positions)
  cos_arr <- array(0, c(R, S, hd)); sin_arr <- array(0, c(R, S, hd))
  for (r in seq_len(R)) {
    fr <- outer(positions[r, ], inv_freq)
    emb <- cbind(fr, fr)
    cos_arr[r, , ] <- cos(emb)
    sin_arr[r, , ] <- sin(emb)
  }
  list(cos = cos_arr, sin = sin_arr)
}

.transformer <- function(model, H, time_mask, variate_mask, positions) {
  cfg <- model$config
  cos_sin <- .rope_cos_sin(model, positions)
  axes_of <- list(time = "time", variate = "variate",
                  time_variate = c("time", "variate"),
                  variate_time = c("variate", "time"))
  for (li in seq_along(cfg$block_schedule)) {
    axes <- axes_of[[cfg$block_schedule[li]]]
    for (ai in seq_along(axes)) {
      prefix <- sprintf("transformer.layers.%d.attns.%d", li - 1L, ai - 1L)
      nprefix <- sprintf("transformer.layers.%d.attn_norms.%d", li - 1L, ai - 1L)
      xn <- .rms_norm3(H, .w(model, paste0(nprefix, ".scale")), cfg$norm_eps)
      attn <- if (axes[ai] == "time") {
        .attn_time(model, xn, prefix, TRUE, time_mask, cos_sin)
      } else {
        .attn_variate(model, xn, prefix, variate_mask)
      }
      H <- H + attn
    }
    mp <- sprintf("transformer.layers.%d.mlp", li - 1L)
    xn <- .rms_norm3(H, .w(model, paste0(mp, "_norm.scale")), cfg$norm_eps)
    for (r in seq_len(dim(H)[1])) {
      gu <- .lin(xn[r, , ], .w(model, paste0(mp, ".w12.weight")),
                 .w(model, paste0(mp, ".w12.bias")))
      hd2 <- ncol(gu) / 2
      gate <- gu[, seq_len(hd2), drop = FALSE]
      up <- gu[, hd2 + seq_len(hd2), drop = FALSE]
      H[r, , ] <- H[r, , ] + .lin(gate * .silu(up), .w(model, paste0(mp, ".w3.weight")),
                                  .w(model, paste0(mp, ".w3.bias")))
    }
  }
  .rms_norm3(H, .w(model, "transformer.out_norm.scale"), cfg$norm_eps)
}

## ---- patch embedding, head, predict_block ---------------------------------

.patch_embed <- function(model, scaled, observed) {
  cfg <- model$config
  ps <- cfg$patch_size
  R <- nrow(scaled); Tt <- ncol(scaled); N <- Tt %/% ps
  D <- cfg$d_model
  tokens <- matrix(0, R * N, 2 * ps)
  for (r in seq_len(R)) {
    vals <- matrix(scaled[r, ], nrow = ps)      # (ps, N)
    miss <- matrix(!observed[r, ], nrow = ps) * 1.0
    tokens[(r - 1) * N + seq_len(N), ] <- cbind(t(vals), t(miss))
  }
  out <- .residual_block(
    tokens,
    .w(model, "representation.patch_proj.hidden_layer.weight"),
    .w(model, "representation.patch_proj.output_layer.weight"),
    .w(model, "representation.patch_proj.residual_layer.weight"),
    .w(model, "representation.patch_proj.hidden_layer.bias"),
    .w(model, "representation.patch_proj.output_layer.bias"),
    .w(model, "representation.patch_proj.residual_layer.bias"))
  H <- array(0, c(R, N, D))
  for (r in seq_len(R)) H[r, , ] <- out[(r - 1) * N + seq_len(N), , drop = FALSE]
  H
}

.forecast_tokens <- function(model, R, n_patches) {
  fq <- .w(model, "representation.forecast_query")
  aperm(array(rep(fq, R * n_patches), dim = c(length(fq), R, n_patches)),
        c(2, 3, 1))
}

.monotone_quantiles <- function(model, raw) {
  cfg <- model$config
  levels <- cfg$quantile_knots
  Q <- length(levels)
  m <- which(levels == 0.5)
  widths <- numeric(Q)
  if (m < Q) for (i in (m + 1):Q) widths[i] <- levels[i] - levels[i - 1]
  if (m > 1) for (i in seq_len(m - 1)) widths[i] <- levels[i + 1] - levels[i]
  d <- dim(raw)  # (..., Q) with Q last
  inc <- cfg$quantile_width_factor * sweep(.softplus(raw), length(d), widths, `*`)
  out <- raw
  med <- raw[, , m]
  if (m < Q) {
    acc <- med * 0
    for (k in (m + 1):Q) {
      acc <- acc + inc[, , k]
      out[, , k] <- med + acc
    }
  }
  if (m > 1) {
    acc <- med * 0
    for (k in (m - 1):1) {
      acc <- acc + inc[, , k]
      out[, , k] <- med - acc
    }
  }
  out[, , m] <- med
  out
}

.quantile_head <- function(model, H) {
  cfg <- model$config
  ps <- cfg$patch_size; Q <- length(cfg$quantile_knots)
  R <- dim(H)[1]; N <- dim(H)[2]
  preds <- array(0, c(R, N * ps, Q))
  for (r in seq_len(R)) {
    raw <- .residual_block(
      H[r, , ],
      .w(model, "head.projection.hidden_layer.weight"),
      .w(model, "head.projection.output_layer.weight"),
      .w(model, "head.projection.residual_layer.weight"))
    ## raw: (N, ps*Q) -> (N*ps, Q)
    arr <- array(0, c(N, ps, Q))
    for (n in seq_len(N))
      arr[n, , ] <- matrix(raw[n, ], nrow = ps, ncol = Q, byrow = TRUE)
    arr <- .monotone_quantiles(model, arr)
    for (n in seq_len(N))
      preds[r, (n - 1) * ps + seq_len(ps), ] <- arr[n, , ]
  }
  preds
}

.run_two_stream <- function(model, x_enc, x_dec, time_mask, variate_mask, positions) {
  H <- array(0, c(dim(x_enc)[1], dim(x_enc)[2] + dim(x_dec)[2], dim(x_enc)[3]))
  H[, seq_len(dim(x_enc)[2]), ] <- x_enc
  H[, dim(x_enc)[2] + seq_len(dim(x_dec)[2]), ] <- x_dec
  h <- .transformer(model, H, time_mask, variate_mask, positions)
  .quantile_head(model, h[, dim(x_enc)[2] + seq_len(dim(x_dec)[2]), , drop = FALSE])
}

.predict_context_only <- function(model, inputs, observed, horizon_patches,
                                  groups = NULL) {
  cfg <- model$config
  ps <- cfg$patch_size
  R <- nrow(inputs); Tt <- ncol(inputs)
  ll <- Tt %/% ps; h <- horizon_patches
  sc <- .model_scale(model, inputs, observed)
  x_enc <- .patch_embed(model, sc$scaled, observed)
  x_dec <- .forecast_tokens(model, R, h)
  obs_patch <- matrix(FALSE, R, ll)
  for (r in seq_len(R))
    obs_patch[r, ] <- apply(matrix(observed[r, ], nrow = ps), 2, any)
  if (is.null(groups)) groups <- rep(1L, R)
  key_valid <- .enc_key_validity(obs_patch, cfg$attend_missing_patches)
  time_mask <- .build_forecast_time_mask(ll, h, key_valid)
  variate_mask <- .build_variate_mask(key_valid, h, groups = groups)
  first_obs <- .group_first_observed_patch(obs_patch, groups)
  positions <- .build_forecast_positions(ll, h, first_obs, R)
  preds <- .run_two_stream(model, x_enc, x_dec, time_mask, variate_mask, positions)
  anchor_loc <- sc$loc[, Tt]  # last-patch loc (constant within patch)
  anchor_scale <- sc$scale[, Tt]
  .denormalize_quantiles(model, preds,
                         anchor_loc,
                         anchor_scale)
}

.predict_window <- function(model, full_inputs, full_observed, horizon_patches, role_ids,
                            groups = NULL) {
  cfg <- model$config
  ps <- cfg$patch_size
  R <- nrow(full_inputs); W <- ncol(full_inputs)
  h <- horizon_patches
  n <- W %/% ps; ll <- n - h
  target_rows <- role_ids == 0L
  known_rows <- role_ids == 2L
  sc <- .model_scale(model, full_inputs, full_observed)
  ks <- .known_cov_scaling(model, full_inputs, full_observed, sc$scaled, sc$loc,
                           sc$scale, known_rows)
  x_enc <- .patch_embed(model, ks$scaled, full_observed)
  x_dec <- .forecast_tokens(model, R, n)
  obs_patch <- matrix(FALSE, R, n)
  for (r in seq_len(R))
    obs_patch[r, ] <- apply(matrix(full_observed[r, ], nrow = ps), 2, any)
  if (is.null(groups)) groups <- rep(1L, R)
  key_valid <- .enc_key_validity(obs_patch, cfg$attend_missing_patches)
  time_mask <- .build_two_stream_time_mask(rep(ll, R), n, cfg$block_size,
                                           key_valid, known_rows)
  variate_mask <- .build_variate_mask(key_valid, n, target_rows, known_rows, groups)
  first_obs <- .group_first_observed_patch(obs_patch, groups)
  positions <- .build_two_stream_positions(n, first_obs, R)
  preds_all <- .run_two_stream(model, x_enc, x_dec, time_mask, variate_mask, positions)
  preds <- preds_all[, ll * ps + seq_len(h * ps), , drop = FALSE]
  anchor_loc <- ks$loc[, ll * ps]  # first step of last context patch (patch-constant)
  anchor_scale <- ks$scale[, ll * ps]
  .denormalize_quantiles(model, preds,
                         anchor_loc,
                         anchor_scale)
}

#' Single-block forecast primitive (internal)
.predict_block <- function(model, inputs, observed, horizon_patches,
                           role_ids = NULL, known_future = NULL,
                           known_future_mask = NULL, groups = NULL) {
  if (horizon_patches < 1)
    stop("horizon_patches must be >= 1, got ", horizon_patches)
  if (is.null(role_ids) || !model$config$covariate_enabled || all(role_ids == 0L))
    return(.predict_context_only(model, inputs, observed, horizon_patches, groups))
  if (horizon_patches > model$config$block_size)
    stop("decode block ", horizon_patches, " must be in [1, ",
         model$config$block_size, "] with covariates")
  R <- nrow(inputs)
  horizon_len <- horizon_patches * model$config$patch_size
  if (is.null(known_future)) {
    known_future <- matrix(0, R, horizon_len)
    known_future_mask <- matrix(FALSE, R, horizon_len)
  }
  .predict_window(model, cbind(inputs, known_future),
                  cbind(observed, known_future_mask), horizon_patches, role_ids,
                  groups)
}

