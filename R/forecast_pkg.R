## forecast package interface (forecast is a Suggests-only dependency; the
## forecast() method is registered, via delayed S3method() in NAMESPACE,
## when the forecast package is loaded). A "forecast" object is univariate
## and ts-based, with central prediction intervals at fixed levels; here
## those intervals are read straight off the quantile knots, never
## interpolated between them.

#' Convert a Timestral forecast to a `forecast`-package object
#'
#' Builds an object of class `"forecast"`, so the forecast package's
#' tools (`autoplot()`, `accuracy()`, `print()`, ...) work on a Timestral
#' forecast. A `"forecast"` object holds one univariate series: pick it
#' with `series` and `variate`.
#'
#' The median is the point forecast (`mean`). Each interval level `L`
#' takes its bounds from the quantile knots at `(1 - L/100) / 2` and
#' `(1 + L/100) / 2`, so `L` must pair two knots of the checkpoint: with
#' the official knots ([timestral_levels()]) 80, 90 and 98 work, but the
#' forecast package's usual 95 does not, because there are no 0.025 and
#' 0.975 knots. Quantiles are not interpolated between knots. The default
#' levels are therefore `c(80, 98)`.
#'
#' @param fc a `timestral_forecast` made from `ts` input, so it has a time
#'   base
#' @param level interval levels in percent (fractions in (0, 1) are
#'   taken as percentages, as in the forecast package); each must pair two
#'   quantile knots (see Details)
#' @param series which series of a batch forecast
#' @param variate which variate, by index or name
#' @return An object of class `"forecast"` with `mean`, `lower`, `upper`
#'   (`ts` matrices with one column per level), `level`, `x` (the observed
#'   context, `NA` where missing), `series` and `method = "Timestral"`.
#'   `fitted` and `residuals` are all `NA`: a zero-shot model is not fitted
#'   to the context, so `forecast::accuracy()` reports test-set measures
#'   (training-set rows are `NaN`).
#' @seealso [forecast.timestral_model()] to forecast and convert in one
#'   call.
#' @examples
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' fc <- timestral_forecast(m, AirPassengers, h = 12)
#' if (requireNamespace("forecast", quietly = TRUE))
#'   timestral_as_forecast(fc)
#' @export
timestral_as_forecast <- function(fc, level = c(80, 98), series = 1L,
                                  variate = 1L) {
  stopifnot(inherits(fc, "timestral_forecast"))
  if (is.null(fc$tsp))
    stop("a forecast-package object needs a ts time base; create the ",
         "forecast from ts input")
  if (!is.numeric(level) || !length(level) || any(!is.finite(level)) ||
      any(level <= 0 | level >= 100))
    stop("level must hold interval levels in percent, between 0 and 100")
  ## fractions are percentages, as in the forecast package
  if (max(level) < 1) level <- 100 * level
  level <- sort(unique(level))
  pp <- .fc_plot_parts(fc, series)
  if (is.character(variate)) {
    vi <- match(variate, pp$vnames)
    if (length(variate) != 1L || is.na(vi))
      stop("variate must be one of ", paste(pp$vnames, collapse = ", "))
  } else {
    vi <- variate
    if (length(vi) != 1L || !is.numeric(vi) || !is.finite(vi) || vi < 1L ||
        vi > pp$nv || vi != round(vi))
      stop("variate must be one variate index between 1 and ", pp$nv)
  }
  q <- pp$qmat(vi)
  tsp <- fc$tsp
  fts <- function(v) .future_ts(v, tsp)
  lo <- vapply(level, function(L) q[, pp$knot((1 - L / 100) / 2)],
               numeric(fc$h))
  hi <- vapply(level, function(L) q[, pp$knot((1 + L / 100) / 2)],
               numeric(fc$h))
  lo <- matrix(lo, nrow = fc$h, dimnames = list(NULL, paste0(level, "%")))
  hi <- matrix(hi, nrow = fc$h, dimnames = list(NULL, paste0(level, "%")))
  x <- stats::ts(pp$obs[vi, ], start = tsp[1], frequency = tsp[3])
  none <- x
  none[] <- NA_real_
  ## accuracy() needs fitted/residuals over x; all-NA marks "not fitted"
  structure(list(method = "Timestral", level = level,
                 mean = fts(q[, pp$knot(0.5)]), lower = fts(lo),
                 upper = fts(hi), x = x, fitted = none, residuals = none,
                 series = pp$vnames[vi]),
            class = "forecast")
}

#' Forecast with the forecast package's `forecast()` generic
#'
#' Registered as an S3 method for `forecast::forecast()` when the forecast
#' package is loaded, so `forecast::forecast(m, AirPassengers, h = 24)`
#' forecasts a `ts` with a Timestral model and returns a `"forecast"`
#' object. It calls [timestral_forecast()] and then
#' [timestral_as_forecast()]; see that page for the interval levels.
#'
#' @param object a `timestral_model` from [timestral_load()]
#' @param y the series to forecast, a `ts` (a univariate `ts`, or an `mts`
#'   whose columns are forecast jointly; pick the column with `variate`)
#' @param h horizon in steps (required)
#' @param level interval levels in percent, each pairing two quantile knots
#' @param variate which variate of a multivariate `y` to return
#' @param ... further arguments to [timestral_forecast()], such as
#'   covariates or `symmetric`; other forecast-generic arguments (`fan`,
#'   `lambda`, `biasadj`, ...) are rejected
#' @return An object of class `"forecast"`.
#' @examples
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' if (requireNamespace("forecast", quietly = TRUE))
#'   forecast::forecast(m, AirPassengers, h = 12)
#' @exportS3Method forecast::forecast
forecast.timestral_model <- function(object, y, h, level = c(80, 98),
                                     variate = 1L, ...) {
  if (missing(y)) stop("forecast() needs `y`: the series to forecast")
  if (missing(h)) stop("forecast() needs `h`: the forecast horizon in steps")
  if (!stats::is.ts(y))
    stop("forecast() for a timestral_model needs a ts; use ",
         "timestral_forecast() for other inputs")
  ## forecast-generic arguments (fan, lambda, biasadj, ...) have no meaning
  ## here; name them instead of failing inside timestral_forecast()
  extra <- setdiff(names(list(...)), names(formals(timestral_forecast)))
  if (length(extra))
    stop("unsupported argument(s) for a timestral_model: ",
         paste(extra, collapse = ", "), "; forecast() passes only ",
         "timestral_forecast() arguments through `...`")
  timestral_as_forecast(timestral_forecast(object, y, h = h, ...),
                        level = level, variate = variate)
}
