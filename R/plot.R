#' Fan-chart plot of a Timestral forecast
#'
#' @param x a `timestral_forecast`
#' @param series which series of a batch to plot (default 1)
#' @param levels central predictive intervals to draw as bands; each must
#'   correspond to a pair of available quantile knots (with the default
#'   knots, 0.5, 0.8, 0.9 and 0.98 all do)
#' @param main panel titles; default uses the variate names
#' @param xlab,ylab axis labels
#' @param ... passed to [graphics::plot()]
#' @return `x`, invisibly
#' @examples
#' \dontrun{
#' m <- timestral_load("5M")
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' plot(fc)
#' }
#' @export
plot.timestral_forecast <- function(x, series = 1L,
                                    levels = c(0.5, 0.8, 0.9, 0.98),
                                    main = NULL, xlab = "Time", ylab = NULL,
                                    ...) {
  if (is.null(x$observed))
    stop("this forecast object has no stored history; re-create it with ",
         "timestral_forecast() from this version")
  if (length(series) != 1L || series < 1L || series > x$n_series)
    stop("series must be one series index between 1 and ", x$n_series)
  obs <- matrix(x$observed[series, , ], nrow = dim(x$observed)[2L])
  nv <- nrow(obs)
  vnames <- rownames(obs)
  if (is.null(vnames)) vnames <- paste0("V", seq_len(nv))
  times <- if (!is.null(x$index)) x$index else seq_len(ncol(obs))
  ftimes <- if (!is.null(x$future_index)) {
    x$future_index
  } else if (length(times) >= 2L) {
    times[length(times)] +
      (times[length(times)] - times[length(times) - 1L]) * seq_len(x$h)
  } else {
    times + seq_len(x$h)
  }
  qmat <- function(v) {
    if (x$n_series == 1L) x$quantiles[v, , ]
    else x$quantiles[series, v, , ]
  }
  knot <- function(p) {
    i <- which.min(abs(x$levels - p))
    if (abs(x$levels[i] - p) > 1e-9)
      stop("no quantile knot at level ", p, "; available levels are in ",
           "timestral_levels() (interval levels must pair two knots)")
    i
  }
  ## widest band lightest, narrowest darkest
  band_cols <- character(length(levels))
  band_cols[order(levels)] <- vapply(
    seq(0.5, 0.16, length.out = length(levels)),
    function(a) grDevices::adjustcolor("#3B7EA1", alpha.f = a), character(1))
  old_par <- graphics::par(mfrow = c(nv, 1L))
  on.exit(graphics::par(old_par))
  for (v in seq_len(nv)) {
    q <- matrix(qmat(v), nrow = x$h)
    hist_y <- obs[v, ]
    finite <- is.finite(hist_y)
    a <- max(which(finite))
    anchor_t <- times[a]; anchor_y <- hist_y[a]
    ylim <- range(c(hist_y[finite],
                    q[, knot(0.01)], q[, knot(0.99)]), finite = TRUE)
    graphics::plot(times, hist_y, type = "n", xlim = range(c(times, ftimes)),
                   ylim = ylim, xlab = xlab,
                   ylab = if (is.null(ylab)) vnames[v] else ylab,
                   main = if (is.null(main)) vnames[v] else main, ...)
    for (bi in order(levels, decreasing = TRUE)) {
      p <- levels[bi]
      lo <- q[, knot((1 - p) / 2)]; hi <- q[, knot(1 - (1 - p) / 2)]
      graphics::polygon(c(anchor_t, ftimes, rev(ftimes), anchor_t),
                        c(anchor_y, hi, rev(lo), anchor_y),
                        col = band_cols[bi], border = NA)
    }
    med <- q[, knot(0.5)]
    graphics::lines(c(anchor_t, ftimes), c(anchor_y, med),
                    col = "#173F5F", lwd = 1.6)
    graphics::lines(times, hist_y, col = "black", lwd = 1.2)
  }
  invisible(x)
}
