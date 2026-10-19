## tsibble support (tsibble is a Suggests-only dependency; these helpers
## are only called for tsibble inputs or by the as_tsibble method).

## Split a tsibble into per-key-group numeric matrices (time in rows).
## Returns list(index, index_var, groups, meas, snames); every group must
## share the same time index.
.tsibble_groups <- function(x) {
  if (!requireNamespace("tsibble", quietly = TRUE))
    stop("tsibble input requires the tsibble package")
  iv <- tsibble::index_var(x)
  meas <- tsibble::measured_vars(x)
  if (length(meas) == 0L)
    stop("tsibble has no measured variables to forecast")
  df <- as.data.frame(x)
  nonnum <- meas[!vapply(df[meas], is.numeric, logical(1))]
  if (length(nonnum))
    stop("tsibble measured variables must be numeric; non-numeric: ",
         paste(nonnum, collapse = ", "))
  kd <- tsibble::key_data(x)
  keys <- tsibble::key_vars(x)
  rows_list <- kd[[".rows"]]
  index1 <- NULL; groups <- list(); snames <- character(length(rows_list))
  for (i in seq_along(rows_list)) {
    rows <- rows_list[[i]]
    idx <- df[rows, iv]
    if (is.null(index1)) {
      index1 <- idx
    } else if (!isTRUE(all.equal(idx, index1))) {
      stop("all tsibble key groups must share the same time index")
    }
    groups[[i]] <- as.matrix(df[rows, meas, drop = FALSE])
    snames[i] <- if (length(keys)) {
      paste(vapply(keys, function(k) as.character(kd[[k]][i]), character(1)),
            collapse = "/")
    } else {
      paste0("S", i)
    }
  }
  list(index = index1, index_var = iv, groups = groups, meas = meas,
       snames = snames)
}

## .as_batch equivalent for a tsibble: keys become the batch, measured
## variables become jointly forecast variates.
.as_tsibble_batch <- function(y) {
  gr <- .tsibble_groups(y)
  mats <- gr$groups
  arr <- array(0, c(nrow(mats[[1]]), length(gr$meas), length(mats)))
  for (b in seq_along(mats)) arr[, , b] <- mats[[b]]
  list(arr = aperm(arr, c(3, 2, 1)), vnames = gr$meas, tsp = NULL,
       index = gr$index, index_var = gr$index_var, T = nrow(mats[[1]]),
       V = length(gr$meas), B = length(mats), snames = gr$snames)
}

.batch_dispatch <- function(y) {
  if (inherits(y, "tbl_ts")) .as_tsibble_batch(y) else .as_batch(y)
}

#' Convert a Timestral forecast to a tsibble
#'
#' The result is a `tsibble` over the forecast horizon with one row per
#' series/variate/step and one column per quantile level (as in
#' [as_forecast_df()]). Registered as an S3 method for
#' `tsibble::as_tsibble()` when the tsibble package is available; call it
#' as `tsibble::as_tsibble(fc)`.
#'
#' @param x a `timestral_forecast` created from indexed input (`ts`,
#'   `zoo`, `xts` or `tsibble`), so a future time index is available
#' @param ... ignored
#' @return a `tsibble` (`tbl_ts`)
#' @examples
#' \dontrun{
#' m <- timestral_load("5M")
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' tsibble::as_tsibble(fc)
#' }
as_tsibble.timestral_forecast <- function(x, ...) {
  if (is.null(x$future_index))
    stop("this forecast has no time index; create it from ts, zoo, xts ",
         "or tsibble input to convert it to a tsibble")
  if (!requireNamespace("tsibble", quietly = TRUE))
    stop("converting to a tsibble requires the tsibble package")
  df <- as_forecast_df(x)
  iv <- if (!is.null(x$index_var)) x$index_var else "time"
  df[[iv]] <- df$time
  df$time <- NULL; df$step <- NULL
  has_series <- "series" %in% names(df)
  multi_variate <- length(unique(df$variate)) > 1L
  keys <- c(if (has_series) "series", if (multi_variate) "variate")
  tsibble::as_tsibble(df, key = if (length(keys)) keys else NULL,
                      index = iv)
}
