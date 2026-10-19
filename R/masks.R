## Port of timestral/masks.py. All masks are logical, TRUE = may attend.
## Queries index rows, keys index columns.

.enc_key_validity <- function(obs_per_patch, attend_missing) {
  if (!attend_missing) return(obs_per_patch)
  started <- t(apply(obs_per_patch, 1, function(r) cummax(r) > 0))
  obs_per_patch | started
}

.enc_causal_mask <- function(key_valid_r) {
  ll <- length(key_valid_r)
  causal <- lower.tri(matrix(TRUE, ll, ll), diag = TRUE)
  (causal & matrix(key_valid_r, ll, ll, byrow = TRUE)) | diag(ll)
}

.build_forecast_time_mask <- function(n_ctx, horizon, key_valid) {
  R <- nrow(key_valid)
  S <- n_ctx + horizon
  out <- array(FALSE, dim = c(R, S, S))
  dec_dec <- lower.tri(matrix(TRUE, horizon, horizon), diag = TRUE)
  for (r in seq_len(R)) {
    m <- matrix(FALSE, S, S)
    m[seq_len(n_ctx), seq_len(n_ctx)] <- .enc_causal_mask(key_valid[r, ])
    m[n_ctx + seq_len(horizon), seq_len(n_ctx)] <-
      matrix(key_valid[r, ], horizon, n_ctx, byrow = TRUE)
    m[n_ctx + seq_len(horizon), n_ctx + seq_len(horizon)] <- dec_dec
    out[r, , ] <- m
  }
  out
}

.compute_block_layout <- function(s, n_patches, block_size) {
  p <- seq_len(n_patches) - 1L
  active <- p >= s
  offset <- pmax(p - s, 0L)
  anchor <- s + block_size * (offset %/% block_size)
  anchor[!active] <- 0L
  list(anchor = anchor, active = active)
}

.build_two_stream_time_mask <- function(min_context, n_patches, block_size,
                                        key_valid, bidirectional_rows) {
  R <- nrow(key_valid)
  n <- n_patches
  S <- 2L * n
  out <- array(FALSE, dim = c(R, S, S))
  causal <- lower.tri(matrix(TRUE, n, n), diag = TRUE)
  eye <- diag(n)
  for (r in seq_len(R)) {
    lay <- .compute_block_layout(min_context[r], n, block_size)
    anchor <- lay$anchor; active <- lay$active
    enc_enc <- .enc_causal_mask(key_valid[r, ])
    if (!is.null(bidirectional_rows) && bidirectional_rows[r]) {
      enc_bi <- matrix(key_valid[r, ], n, n, byrow = TRUE) | eye
      enc_enc <- enc_bi
    }
    j <- seq_len(n) - 1L
    ## dec_enc[i, k] = active[i] & (j[k] < anchor[i]) & key_valid[k]
    dec_enc <- outer(active, rep(TRUE, n)) &
      outer(anchor, j, function(a, b) b < a) &
      matrix(key_valid[r, ], n, n, byrow = TRUE)
    same_block <- outer(anchor, anchor, `==`)
    dec_dec <- (same_block & causal & outer(active, active)) | eye
    m <- matrix(FALSE, S, S)
    m[seq_len(n), seq_len(n)] <- enc_enc
    m[n + seq_len(n), seq_len(n)] <- dec_enc
    m[n + seq_len(n), n + seq_len(n)] <- dec_dec
    out[r, , ] <- m
  }
  out
}

.variate_firewall <- function(R, target_rows, bidirectional_rows) {
  if (is.null(target_rows) && is.null(bidirectional_rows)) return(NULL)
  forbid <- matrix(FALSE, R, R)
  if (!is.null(target_rows))
    forbid <- forbid | outer(!target_rows, target_rows)
  if (!is.null(bidirectional_rows))
    forbid <- forbid | outer(bidirectional_rows, !bidirectional_rows)
  forbid
}

.build_variate_mask <- function(key_valid, n_dec, target_rows = NULL,
                                bidirectional_rows = NULL, groups = NULL) {
  R <- nrow(key_valid)
  if (R == 1L) return(NULL)
  if (is.null(groups)) groups <- rep(1L, R)
  n_enc <- ncol(key_valid)
  slot_valid <- cbind(key_valid, matrix(TRUE, R, n_dec))
  S <- n_enc + n_dec
  forbid <- .variate_firewall(R, target_rows, bidirectional_rows)
  row_eq <- outer(groups, groups, `==`)
  out <- array(FALSE, dim = c(S, R, R))
  for (s in seq_len(S)) {
    ## rows = queries, cols = keys: key validity varies by column
    m <- matrix(slot_valid[, s], R, R, byrow = TRUE) & row_eq
    if (!is.null(forbid)) m <- m & !forbid
    diag(m) <- TRUE
    out[s, , ] <- m
  }
  out
}

.group_first_observed_patch <- function(obs_per_patch, groups = NULL) {
  if (is.null(groups)) groups <- rep(1L, nrow(obs_per_patch))
  vapply(seq_len(nrow(obs_per_patch)), function(r) {
    any_obs <- apply(obs_per_patch[groups == groups[r], , drop = FALSE], 2, any)
    idx <- which(any_obs)
    if (length(idx) == 0L) 0L else idx[1] - 1L
  }, integer(1))
}

.build_forecast_positions <- function(n_ctx, horizon, first_obs, R) {
  p <- seq_len(n_ctx + horizon) - 1L
  matrix(p, R, n_ctx + horizon, byrow = TRUE) - first_obs
}

.build_two_stream_positions <- function(n_patches, first_obs, R) {
  p <- seq_len(n_patches) - 1L
  base <- matrix(p, R, n_patches, byrow = TRUE) - first_obs
  cbind(base, base)
}
