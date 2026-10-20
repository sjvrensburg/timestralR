## Port of timestral/covariate_encoding.py + the encoding glue in
## forecaster.py. Codes are handled in float64 throughout, as upstream.

.slots <- function(codes, categories, nan_slot, unseen = NULL) {
  ## 1-based slots: categories occupy 1..length(categories); `nan_slot` is the
  ## (1-based) slot for NA codes; `unseen` the slot for unknown future codes.
  idx0 <- vapply(codes, function(cd) {
    if (is.na(cd)) NA_integer_ else sum(categories < cd)
  }, integer(1))
  idx0 <- pmin(idx0, max(length(categories) - 1L, 0L))
  known <- is.finite(codes) & length(categories) > 0
  if (length(categories) > 0)
    known <- known & !is.na(codes) & (categories[idx0 + 1L] == codes)
  known[is.na(known)] <- FALSE
  if (is.null(unseen) && !all(known | is.na(codes)))
    stop("every past code must be in categories")
  slot <- ifelse(is.na(codes), nan_slot,
                 ifelse(known, idx0 + 1L, if (is.null(unseen)) 1L else unseen))
  as.integer(slot)
}

## Smoothed mean-target encoding of one categorical covariate (internal)
.target_encode <- function(target_context, past_codes, future_codes = NULL,
                           smooth = 1.0) {
  target <- as.numeric(target_context)
  past_codes <- as.numeric(past_codes)
  if (length(target) != length(past_codes))
    stop("target and codes must align")
  observed <- is.finite(target)
  target[!observed] <- 0
  item_count <- sum(observed)
  item_mean <- if (item_count > 0) sum(target) / item_count else 0.0
  categories <- sort(unique(past_codes[is.finite(past_codes)]))
  nan_slot <- length(categories)  # 0-based count; slots below are 1-based
  past_slots <- .slots(past_codes, categories, nan_slot + 1L)
  sums <- numeric(nan_slot + 1)
  counts <- numeric(nan_slot + 1)
  for (s in seq_len(nan_slot + 1)) {
    sel <- past_slots == s
    sums[s] <- sum(target[sel] * observed[sel])
    counts[s] <- sum(observed[sel])
  }
  lookup <- c((smooth * item_mean + sums) / (smooth + counts), item_mean)
  encoded_past <- lookup[past_slots]
  if (is.null(future_codes)) return(list(past = encoded_past, future = NULL))
  future_slots <- .slots(as.numeric(future_codes), categories, nan_slot + 1L,
                         unseen = nan_slot + 2L)
  list(past = encoded_past, future = lookup[future_slots])
}

## Encoding target for one series: forward-filled target from its first
## observation on, over the last max_context steps; NA elsewhere, and NULL
## when the series has several target variates.
.encoding_target <- function(model, inputs_b, mask_b) {
  if (nrow(inputs_b) != 1L) return(NULL)
  t <- ncol(inputs_b)
  started <- cummax(mask_b[1, ] * 1L) > 0
  in_window <- seq_len(t) > (t - model$max_context)
  filled <- .ffill_rows(inputs_b, mask_b)[1, ]
  out <- ifelse(started & in_window, filled, NA_real_)
  out
}

## Encode categorical covariates for the whole batch (list per series).
## codes_list: list of (C, L) code matrices (one per series) or NULL.
.encode_categorical_batch <- function(model, codes_list, inputs, mask, n_series,
                                      T_len) {
  if (is.null(codes_list)) return(NULL)
  lapply(seq_len(n_series), function(b) {
    codes <- codes_list[[b]]
    if (is.null(codes) || nrow(codes) == 0L) return(NULL)
    ib <- inputs[b, , , drop = FALSE]; dim(ib) <- dim(ib)[2:3]
    mb <- mask[b, , , drop = FALSE]; dim(mb) <- dim(mb)[2:3]
    target <- .encoding_target(model, ib, mb)
    if (is.null(target)) return(codes)  # several variates: codes as-is
    out <- matrix(NA_real_, nrow(codes), ncol(codes))
    for (j in seq_len(nrow(codes))) {
      row <- codes[j, ]
      future <- if (ncol(codes) > T_len) row[(T_len + 1):ncol(codes)] else NULL
      enc <- .target_encode(target, row[seq_len(T_len)], future)
      past_enc <- enc$past
      both_na <- is.na(target) & is.na(row[seq_len(T_len)])
      past_enc[both_na] <- NA_real_
      out[j, seq_len(T_len)] <- past_enc
      if (!is.null(future)) out[j, (T_len + 1):ncol(codes)] <- enc$future
    }
    out
  })
}
