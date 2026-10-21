## tsibble support (tsibble is a Suggests-only dependency; these helpers
## are only called for tsibble inputs or by the as_tsibble method).

## Make implicit gaps explicit: missing time steps of a regular tsibble
## become NA rows (masked as missing downstream) so the context stays
## aligned with the calendar.
.tsibble_fill_gaps <- function(x) {
  if (tsibble::is_regular(x) && any(tsibble::has_gaps(x)$.gaps))
    x <- tsibble::fill_gaps(x)
  x
}

## Split a tsibble into per-key-group matrices (time in rows, ordered by
## the index). Returns list(index, index_var, groups, meas, snames, keyed);
## groups must end at the same time, and shorter ones are left-padded with
## NA rows. Logical measured variables
## are accepted only when allow_logical (observation masks).
.tsibble_groups <- function(x, allow_logical = FALSE) {
  if (!requireNamespace("tsibble", quietly = TRUE))
    stop("tsibble input requires the tsibble package")
  x <- .tsibble_fill_gaps(x)
  iv <- tsibble::index_var(x)
  meas <- tsibble::measured_vars(x)
  if (length(meas) == 0L)
    stop("tsibble has no measured variables to forecast")
  df <- as.data.frame(x)
  ok <- function(col) is.numeric(col) || (allow_logical && is.logical(col))
  nonnum <- meas[!vapply(df[meas], ok, logical(1))]
  if (length(nonnum))
    stop("tsibble measured variables must be ",
         if (allow_logical) "numeric or logical" else "numeric",
         "; non-numeric: ", paste(nonnum, collapse = ", "))
  kd <- tsibble::key_data(x)
  keys <- tsibble::key_vars(x)
  rows_list <- kd[[".rows"]]
  groups <- list(); idxs <- list(); snames <- character(length(rows_list))
  for (i in seq_along(rows_list)) {
    rows <- rows_list[[i]]
    rows <- rows[order(df[rows, iv])]
    idxs[[i]] <- df[rows, iv]
    groups[[i]] <- as.matrix(df[rows, meas, drop = FALSE])
    snames[i] <- if (length(keys)) {
      paste(vapply(keys, function(k) as.character(kd[[k]][i]), character(1)),
            collapse = "/")
    } else {
      paste0("S", i)
    }
  }
  ## groups may start at different times but must end together: shorter
  ## ones are left-padded with NA rows onto the longest group's index
  lens <- vapply(idxs, length, integer(1))
  index1 <- if (length(idxs)) idxs[[which.max(lens)]]
  for (i in seq_along(groups)) {
    tail_idx <- index1[length(index1) - lens[i] + seq_len(lens[i])]
    if (!isTRUE(all.equal(idxs[[i]], tail_idx)))
      stop("all tsibble key groups must end at the same time, on the same ",
           "time index (they may start at different times)")
    if (lens[i] < length(index1))
      groups[[i]] <- rbind(matrix(NA, length(index1) - lens[i], ncol(groups[[i]]),
                                  dimnames = list(NULL, colnames(groups[[i]]))),
                           groups[[i]])
  }
  list(index = index1, index_var = iv, groups = groups, meas = meas,
       snames = snames, keyed = length(keys) > 0L,
       pad = length(index1) - lens)
}

## .as_batch equivalent for a tsibble: keys become the batch, measured
## variables become jointly forecast variates.
.as_tsibble_batch <- function(y, allow_logical = FALSE) {
  gr <- .tsibble_groups(y, allow_logical)
  mats <- gr$groups
  if (length(mats) == 0L || nrow(mats[[1]]) == 0L)
    stop("y must have at least one time step (got a tsibble with no rows)")
  arr <- array(0, c(nrow(mats[[1]]), length(gr$meas), length(mats)))
  for (b in seq_along(mats)) arr[, , b] <- mats[[b]]
  list(arr = aperm(arr, c(3, 2, 1)), vnames = gr$meas, tsp = NULL,
       index = gr$index, index_var = gr$index_var, T = nrow(mats[[1]]),
       V = length(gr$meas), B = length(mats), snames = gr$snames,
       pad = gr$pad)
}

.batch_dispatch <- function(y, allow_logical = FALSE) {
  if (inherits(y, "tbl_ts")) .as_tsibble_batch(y, allow_logical) else .as_batch(y, allow_logical)
}

## Future time index in tidyverts form. For ts input this matches the
## index that tsibble::as_tsibble() gives a ts (yearmonth, yearquarter,
## ...), so forecasts join with and score against the observed tsibble.
.tidy_future_index <- function(x) {
  if (!is.null(x$tsp)) {
    dummy <- .future_ts(numeric(x$h), x$tsp)
    return(tsibble::as_tsibble(dummy)$index)
  }
  x$future_index
}

## Index column name for tidyverts output: the input tsibble's index, or
## "index" as tsibble::as_tsibble() uses for ts, zoo and xts.
.tidy_index_var <- function(x) {
  if (!is.null(x$index_var)) x$index_var else "index"
}

#' Convert a Timestral forecast to a tsibble
#'
#' The result is a `tsibble` over the forecast horizon with one row per
#' series/variate/step and one column per quantile level (as in
#' [as_forecast_df()]). Registered as an S3 method for
#' `tsibble::as_tsibble()` when the tsibble package is loaded; call it
#' as `tsibble::as_tsibble(fc)`. The index column keeps the name of the
#' input tsibble's index, and is `index` otherwise, as
#' `tsibble::as_tsibble()` names it for `ts`, `zoo` and `xts`. Forecasts
#' from `ts` input use the same index class (`yearmonth`, `yearquarter`,
#' ...) that `tsibble::as_tsibble()` gives the `ts`, so they join with the
#' observed data. Forecasts can also be created directly from a tsibble:
#' in [timestral_forecast()], tsibble keys become the batch and measured
#' variables the jointly forecast variates.
#'
#' @param x a `timestral_forecast` created from indexed input (`ts`,
#'   `zoo`, `xts` or `tsibble`), so a future time index is available
#' @param ... ignored
#' @return a `tsibble` (`tbl_ts`)
#' @examples
#' ## the tiny bundled checkpoint runs offline but its forecasts are
#' ## meaningless; use timestral_load("5M") or larger for real ones
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' if (requireNamespace("tsibble", quietly = TRUE))
#'   tsibble::as_tsibble(fc)
#' @exportS3Method tsibble::as_tsibble
as_tsibble.timestral_forecast <- function(x, ...) {
  if (is.null(x$future_index))
    stop("this forecast has no time index; create it from ts, zoo, xts ",
         "or tsibble input to convert it to a tsibble")
  if (!requireNamespace("tsibble", quietly = TRUE))
    stop("converting to a tsibble requires the tsibble package")
  df <- as_forecast_df(x)
  iv <- .tidy_index_var(x)
  idx <- .tidy_future_index(x)[df$step]
  df$time <- NULL; df$step <- NULL
  df[[iv]] <- idx
  has_series <- "series" %in% names(df)
  multi_variate <- length(unique(df$variate)) > 1L
  keys <- c(if (has_series) "series", if (multi_variate) "variate")
  tsibble::as_tsibble(df, key = if (length(keys)) keys else NULL,
                      index = iv)
}
