## Rolling-origin evaluation: score quantile forecasts against realised
## values, and backtest a model by forecasting from successive origins of
## one series (or batch). Single-threaded by design (see the threading
## constraint in docs/developer-notes.md); users parallelise across
## origins at the R level.

## Validate central-interval levels against the forecast's knots: percent,
## or fractions in (0, 1) as in the forecast package. Returns the levels
## (percent, sorted) and the knot indices of each interval's bounds. Shared
## by timestral_score() and timestral_as_forecast().
.interval_knots <- function(levels, level) {
  if (!is.numeric(level) || !length(level) || any(!is.finite(level)) ||
      any(level <= 0 | level >= 100))
    stop("level must hold interval levels in percent, between 0 and 100")
  if (max(level) < 1) level <- 100 * level
  level <- sort(unique(level))
  knot <- function(p, L) {
    i <- which(abs(levels - p) < 1e-9)
    if (length(i) != 1L)
      stop("no quantile knot at level ", p, " for the ", L, "% interval; ",
           "available levels are in the forecast's $levels")
    i
  }
  list(level = level,
       lo = vapply(level, function(L) knot((1 - L / 100) / 2, L), integer(1)),
       hi = vapply(level, function(L) knot((1 + L / 100) / 2, L), integer(1)))
}

## Trapezoidal weights of the knots for integrating over (tau_1, tau_K).
.trapezoid_weights <- function(tau) {
  K <- length(tau)
  if (K == 1L) return(1)
  d <- diff(tau)
  (c(d, 0) + c(0, d)) / 2
}

#' Score quantile forecasts against realised values
#'
#' Scores a [timestral_forecast()] against the values that were later
#' observed, one row per series, variate and step.
#'
#' The scores, for realised value \eqn{y} and quantile \eqn{q_\tau} at
#' each level \eqn{\tau} of the forecast (`fc$levels`), use the pinball
#' (quantile) loss \eqn{\rho_\tau = (1\{y < q_\tau\} - \tau)(q_\tau - y)}:
#'
#' * `pinball`: \eqn{2\rho_\tau} averaged over the levels, the quantile
#'   loss used in the GluonTS and Chronos benchmarks (before dividing by
#'   the scale of the series). The factor 2 makes it equal the absolute
#'   error for a point forecast.
#' * `crps`: the continuous ranked probability score,
#'   \eqn{2\int_0^1 \rho_\tau\,d\tau}, approximated by the trapezoidal
#'   rule over the knots. The knots are not evenly spaced, so they are
#'   weighted by their spacing rather than averaged. The integral only
#'   covers the knots' range (0.01 to 0.99 for the official checkpoints),
#'   so for a point forecast it gives \eqn{0.98|q - y|} rather than
#'   \eqn{|q - y|}: the tails beyond the outermost knots are ignored.
#' * `ae`: the absolute error of the median.
#' * `cover<L>`: whether \eqn{y} lies in the central `L`% interval, for
#'   each `level`; each level must pair two knots, as in [plot()].
#'
#' Missing actual values give missing scores. Average the rows with
#' [stats::aggregate()] or dplyr to summarise by step, series or origin.
#'
#' @param fc a `timestral_forecast`
#' @param actual the realised values over the horizon, shaped like a
#'   forecast input with `h` time steps: a vector or `ts` (one series, one
#'   variate), a matrix with time in rows (one series), a 3-D array
#'   (time, variate, series), or a list of those per series
#' @param level interval levels in percent for the coverage columns
#'   (fractions in (0, 1) are taken as percentages)
#' @return A data frame with columns `series`, `variate`, `step`,
#'   `actual`, `median`, `pinball`, `crps`, `ae` and one `cover<L>`
#'   column per level.
#' @seealso [timestral_backtest()] to forecast and score from several
#'   origins.
#' @examples
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' train <- window(AirPassengers, end = c(1959, 12))
#' test <- window(AirPassengers, start = c(1960, 1))
#' sc <- timestral_score(timestral_forecast(m, train, h = 12), test)
#' colMeans(sc[, c("pinball", "crps", "ae", "cover80")])
#' @export
timestral_score <- function(fc, actual, level = c(80, 98)) {
  stopifnot(inherits(fc, "timestral_forecast"))
  q <- fc$quantiles
  if (fc$n_series == 1L) q <- array(q, c(1L, dim(q)), list(NULL, dimnames(q)[[1L]],
                                                          NULL, NULL))
  d <- dim(q)
  act <- .as_batch(actual)
  if (!identical(c(act$B, act$V, act$T), d[1:3]))
    stop("actual must have ", d[1], " series x ", d[2], " variate(s) x ",
         d[3], " steps, like the forecast; got ", act$B, " x ", act$V,
         " x ", act$T)
  tau <- fc$levels
  K <- length(tau)
  ik <- .interval_knots(tau, level)
  level <- ik$level
  med <- which(abs(tau - 0.5) < 1e-9)
  y <- act$arr
  yk <- array(y, c(d[1:3], K))
  tk <- array(rep(tau, each = prod(d[1:3])), d)
  rho <- ((yk < q) - tk) * (q - yk)
  w <- .trapezoid_weights(tau)
  ## weighted sums over the level dimension (the last, so a matrix product)
  sum_k <- function(wt) array(matrix(rho, ncol = K) %*% wt, d[1:3])
  level_k <- function(k) array(q[, , , k], d[1:3])
  ## series labels as timestral_forecast() names a batch ("S1", ...)
  snames <- dimnames(q)[[1L]]
  if (is.null(snames)) snames <- paste0("S", seq_len(d[1]))
  vnames <- dimnames(q)[[2L]]
  if (is.null(vnames)) vnames <- paste0("V", seq_len(d[2]))
  ## rows: step fastest, then variate, then series
  flat <- function(a) as.vector(aperm(a, c(3, 2, 1)))
  out <- data.frame(series = rep(snames, each = d[2] * d[3]),
                    variate = rep(rep(vnames, each = d[3]), d[1]),
                    step = rep(seq_len(d[3]), d[1] * d[2]),
                    actual = flat(y), median = flat(level_k(med)),
                    pinball = flat(sum_k(rep(2 / K, K))),
                    crps = flat(sum_k(2 * w)),
                    stringsAsFactors = FALSE)
  out$ae <- abs(out$median - out$actual)
  for (j in seq_along(level)) {
    lo <- flat(level_k(ik$lo[j]))
    hi <- flat(level_k(ik$hi[j]))
    out[[paste0("cover", format(level[j]))]] <-
      out$actual >= lo & out$actual <= hi
  }
  out
}

## Per-series covariates as a list of matrices with time in rows, parsed
## and validated by timestral_forecast()'s own parser at their full length.
.cov_list <- function(x, B, name, snames) {
  if (inherits(x, "tbl_ts"))
    stop("timestral_backtest() does not take tsibble covariates; see ",
         "?timestral_backtest for tsibble data")
  first <- if (is.list(x) && !is.data.frame(x)) x[[1L]] else x
  len <- if (is.array(x) && length(dim(x)) == 3L) dim(x)[1]
    else NROW(.strip_time_index(first)$values)
  parsed <- .as_cov_batch(x, B, len, name,
                          categorical = grepl("^categorical", name),
                          snames = snames)
  lapply(parsed, t)
}

#' Rolling-origin backtest of a Timestral model
#'
#' Forecasts `y` from a sequence of forecast origins, each time using
#' only the data up to the origin as the context, and scores each
#' forecast against the `h` values that follow with [timestral_score()].
#'
#' Origin `o` means the context is time steps `1:o` and the forecast
#' covers steps `o + 1` to `o + h`, so origins run from 1 to `T - h` for a
#' series of length `T`. By default the origins are `seq(initial, T - h,
#' by = step)`; with `initial = NULL` the five last such origins are used
#' (fewer if the series is short). Covariates and `observation_mask` in
#' `...` are given over the whole series and cut to each origin:
#' past-only covariates and the mask to `1:o`, known-future covariates to
#' `1:(o + h)`, so known covariates need at least `T` rows. Steps the
#' mask marks as missing are not scored either (their scores are `NA`).
#' Forecasts are
#' made one origin at a time, batching the series of a list `y` within
#' each origin.
#'
#' For tsibble data, use `tsibble::stretch_tsibble()` with the
#' [TIMESTRAL()] fable model and `fabletools::accuracy()` instead.
#'
#' @section Parallel use:
#' The backtest is single-threaded, like the rest of the package. To run
#' origins in parallel, split them and run one backtest per chunk, then
#' `rbind()` the results. With forked workers (`parallel::mclapply()` on
#' Linux and macOS) the loaded model can be used directly. Socket clusters
#' and `future` workers are new R processes, where a model's native session
#' is invalid, so load the model in each worker from its directory
#' (`m$dir`), or use `backend = "R"`. See the example.
#'
#' @param model a `timestral_model` from [timestral_load()]
#' @param y the series to backtest: a numeric vector, `ts`/`mts`,
#'   `zoo`/`xts`, a matrix or data frame with time in rows, or a list of
#'   equal-length series (forecast as a batch at each origin)
#' @param h forecast horizon in steps
#' @param origins forecast origins as time indices (the last context
#'   step); overrides `initial` and `step`
#' @param initial the first origin; `NULL` for the last five origins
#' @param step spacing of the origins in time steps
#' @param ... arguments to [timestral_forecast()]: `observation_mask`,
#'   covariates (over the whole series), `symmetric`, `clip_nonnegative`
#' @param level interval levels in percent for the coverage columns
#' @param keep_forecasts keep each origin's `timestral_forecast` in the
#'   `"forecasts"` attribute of the result (a list named by origin)
#' @return A data frame with one row per origin, series, variate and step:
#'   `origin`, `series`, `variate`, `step`, `t` (the time index of the
#'   forecast step, `origin + step`), `time` (its time value, when `y` is
#'   a `ts`, `zoo` or `xts`), then the [timestral_score()] columns.
#' @examples
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' bt <- timestral_backtest(m, AirPassengers, h = 12)
#' unique(bt$origin)
#' aggregate(cbind(pinball, crps, ae) ~ step, data = bt, FUN = mean)
#'
#' \donttest{
#' ## origins in parallel on a socket cluster: reload the model per worker
#' cl <- parallel::makeCluster(2)
#' dir <- m$dir
#' origins <- seq(60, 132, by = 12)
#' chunks <- split(origins, seq_along(origins) %% 2)
#' parts <- parallel::parLapply(cl, chunks, function(o, dir) {
#'   m <- timestralR::timestral_load(dir)
#'   timestralR::timestral_backtest(m, AirPassengers, h = 12, origins = o)
#' }, dir = dir)
#' parallel::stopCluster(cl)
#' bt_all <- do.call(rbind, parts)
#' }
#' @export
timestral_backtest <- function(model, y, h, origins = NULL, initial = NULL,
                               step = h, ..., level = c(80, 98),
                               keep_forecasts = FALSE) {
  stopifnot(inherits(model, "timestral_model"))
  if (inherits(y, "tbl_ts"))
    stop("timestral_backtest() does not take tsibbles; use ",
         "tsibble::stretch_tsibble() with the TIMESTRAL() fable model")
  h <- .validate_horizon(h)
  args <- list(...)
  args <- args[!vapply(args, is.null, logical(1))]
  allowed <- c("observation_mask", "past_covariates", "known_covariates",
               "categorical_past_covariates", "categorical_known_covariates",
               "symmetric", "clip_nonnegative")
  if (length(args) && (is.null(names(args)) || any(!nzchar(names(args)))))
    stop("arguments in `...` must be named")
  bad <- setdiff(names(args), allowed)
  if (length(bad))
    stop("unsupported argument(s) in `...`: ", paste(bad, collapse = ", "),
         "; allowed are ", paste(allowed, collapse = ", "))
  tg <- .as_batch(y)
  Tn <- tg$T
  if (Tn - h < 1L)
    stop("y has ", Tn, " time steps, too few for a backtest with h = ", h)
  if (is.null(origins)) {
    if (!is.numeric(step) || length(step) != 1L || !is.finite(step) ||
        step < 1 || step != round(step))
      stop("step must be a single whole number >= 1")
    origins <- if (is.null(initial)) {
      rev(seq(Tn - h, by = -step, length.out = 5L))
    } else {
      if (!is.numeric(initial) || length(initial) != 1L || !is.finite(initial) ||
          initial < 1 || initial > Tn - h || initial != round(initial))
        stop("initial must be a whole number between 1 and T - h = ", Tn - h)
      seq(initial, Tn - h, by = step)
    }
    origins <- origins[origins >= 1]
  }
  if (!is.numeric(origins) || !length(origins) || any(!is.finite(origins)) ||
      any(origins != round(origins)) || any(origins < 1 | origins > Tn - h))
    stop("origins must be whole numbers between 1 and T - h = ", Tn - h,
         " (the last context step of each forecast)")
  origins <- sort(unique(as.integer(origins)))

  ## everything is sliced from the normalised (series, variate, time) array
  arr <- tg$arr
  dimnames(arr) <- list(tg$snames, tg$vnames, NULL)
  as_input <- function(a) aperm(a, c(3, 2, 1))   # (time, variate, series)
  mask <- if (!is.null(args$observation_mask)) {
    mk <- .as_obs_mask(args$observation_mask, tg)
    dimnames(mk) <- dimnames(arr)
    mk
  }
  past_names <- c("past_covariates", "categorical_past_covariates")
  known_names <- c("known_covariates", "categorical_known_covariates")
  cov_names <- intersect(names(args), c(past_names, known_names))
  covs <- stats::setNames(lapply(cov_names, function(nm)
    .cov_list(args[[nm]], tg$B, nm, tg$snames)), cov_names)
  covs <- covs[lengths(covs) > 0L]   # zero-column covariates: none
  for (nm in names(covs)) {
    need <- if (nm %in% known_names) max(origins) + h else max(origins)
    rows <- vapply(covs[[nm]], nrow, integer(1))
    if (any(rows < need))
      stop(nm, " has ", min(rows), " rows but the last origin (",
           max(origins), ") needs ", need,
           if (nm %in% known_names) " (context plus horizon)")
  }
  opts <- args[intersect(names(args), c("symmetric", "clip_nonnegative"))]
  ## check every origin up front, so no completed work is lost to a late
  ## failure: each series needs an observed value by the first origin
  observed <- is.finite(arr)
  if (!is.null(mask)) observed <- observed & mask
  first_obs <- apply(observed, 1L, function(s) {
    hit <- which(colSums(matrix(s, ncol = Tn)) > 0)
    if (length(hit)) hit[1] else NA_integer_
  })
  late <- is.na(first_obs) | first_obs > origins[1]
  if (any(late))
    stop("series ", paste(if (is.null(tg$snames)) which(late) else tg$snames[late],
                          collapse = ", "),
         " of y has no observed value up to origin ", origins[1],
         "; start the origins at ", max(first_obs, na.rm = TRUE),
         " or later")
  ## actuals the mask marks as missing are not scored
  actual_arr <- arr
  if (!is.null(mask)) actual_arr[!mask] <- NA_real_
  slabels <- if (is.null(tg$snames)) paste0("S", seq_len(tg$B)) else tg$snames

  time_of <- if (!is.null(tg$tsp)) {
    function(t) tg$tsp[1] + (t - 1) / tg$tsp[3]
  } else if (!is.null(tg$index)) {
    function(t) tg$index[t]
  }
  forecasts <- if (keep_forecasts) vector("list", length(origins))
  rows <- vector("list", length(origins))
  for (i in seq_along(origins)) {
    o <- origins[i]
    ctx <- arr[, , seq_len(o), drop = FALSE]
    call_args <- c(list(model, as_input(ctx), h = h), opts)
    if (!is.null(mask))
      call_args$observation_mask <- as_input(mask[, , seq_len(o), drop = FALSE])
    for (nm in names(covs)) {
      len <- if (nm %in% known_names) o + h else o
      call_args[[nm]] <- lapply(covs[[nm]], function(x) x[seq_len(len), , drop = FALSE])
    }
    fc <- do.call(timestral_forecast, call_args)
    if (keep_forecasts) forecasts[[i]] <- fc
    sc <- timestral_score(fc, as_input(actual_arr[, , o + seq_len(h), drop = FALSE]),
                          level = level)
    ## a one-series forecast carries no series name; label as the batch
    sc$series <- rep(slabels, each = nrow(sc) / tg$B)
    tt <- o + sc$step
    front <- data.frame(origin = rep(o, nrow(sc)), series = sc$series,
                        variate = sc$variate, step = sc$step, t = tt,
                        stringsAsFactors = FALSE)
    if (!is.null(time_of)) front$time <- time_of(tt)
    rows[[i]] <- cbind(front, sc[, setdiff(names(sc),
                                           c("series", "variate", "step"))])
  }
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  if (keep_forecasts) attr(out, "forecasts") <- stats::setNames(forecasts, origins)
  out
}
