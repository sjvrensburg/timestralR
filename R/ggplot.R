## ggplot2 plotting (ggplot2 is a Suggests-only dependency; these methods
## are registered for ggplot2::autoplot/autolayer in .onLoad when ggplot2
## is available).
utils::globalVariables(".data")

## Long-format band/median/history data for one series of a forecast.
.fc_gg_data <- function(fc, series, levels) {
  if (is.null(fc$observed))
    stop("this forecast object has no stored history; re-create it with ",
         "timestral_forecast() from this version")
  if (length(series) != 1L || series < 1L || series > fc$n_series)
    stop("series must be one series index between 1 and ", fc$n_series)
  obs <- matrix(fc$observed[series, , ], nrow = dim(fc$observed)[2L])
  nv <- nrow(obs)
  vnames <- rownames(obs)
  if (is.null(vnames)) vnames <- paste0("V", seq_len(nv))
  times <- if (!is.null(fc$index)) fc$index else seq_len(ncol(obs))
  ftimes <- if (!is.null(fc$future_index)) {
    fc$future_index
  } else if (length(times) >= 2L) {
    times[length(times)] +
      (times[length(times)] - times[length(times) - 1L]) * seq_len(fc$h)
  } else {
    times + seq_len(fc$h)
  }
  qmat <- function(v) {
    if (fc$n_series == 1L) matrix(fc$quantiles[v, , ], nrow = fc$h)
    else matrix(fc$quantiles[series, v, , ], nrow = fc$h)
  }
  knot <- function(p) {
    i <- which.min(abs(fc$levels - p))
    if (abs(fc$levels[i] - p) > 1e-9)
      stop("no quantile knot at level ", p, "; interval levels must pair ",
           "two knots from timestral_levels()")
    i
  }
  hist_df <- do.call(rbind, lapply(seq_len(nv), function(v) {
    data.frame(time = times, variate = vnames[v], value = obs[v, ],
               stringsAsFactors = FALSE)
  }))
  band_df <- do.call(rbind, lapply(seq_len(nv), function(v) {
    q <- qmat(v)
    do.call(rbind, lapply(levels, function(p) {
      data.frame(time = ftimes, variate = vnames[v],
                 lo = q[, knot((1 - p) / 2)], hi = q[, knot(1 - (1 - p) / 2)],
                 interval = factor(paste0(round(100 * p), "%"),
                                   levels = paste0(round(100 * sort(levels)), "%")),
                 stringsAsFactors = FALSE)
    }))
  }))
  med_df <- do.call(rbind, lapply(seq_len(nv), function(v) {
    data.frame(time = ftimes, variate = vnames[v],
               value = qmat(v)[, knot(0.5)], stringsAsFactors = FALSE)
  }))
  list(hist = hist_df, bands = band_df, median = med_df, nv = nv)
}

.fc_band_layers <- function(d, fill = "#3B7EA1", colour = "#173F5F") {
  list(
    ggplot2::geom_ribbon(
      data = d$bands,
      ggplot2::aes(x = .data$time, ymin = .data$lo, ymax = .data$hi,
                   fill = .data$interval, group = .data$interval),
      alpha = 0.35, inherit.aes = FALSE),
    ggplot2::geom_line(
      data = d$median,
      ggplot2::aes(x = .data$time, y = .data$value),
      colour = colour, linewidth = 0.7, inherit.aes = FALSE)
  )
}

#' ggplot2 fan chart of a Timestral forecast
#'
#' Registered as S3 methods for `ggplot2::autoplot()` and
#' `ggplot2::autolayer()` when ggplot2 is available. `autoplot` draws the
#' observed history, the median, and nested predictive interval bands;
#' `autolayer` adds just the forecast bands and median to an existing
#' ggplot (for example of the observed series).
#'
#' @param object a `timestral_forecast`
#' @param series which series of a batch to plot (default 1)
#' @param levels central predictive intervals to draw as bands; each must
#'   correspond to a pair of available quantile knots
#' @param ... ignored
#' @return for `autoplot`, a `ggplot`; for `autolayer`, a list of layers
#' @examples
#' \dontrun{
#' m <- timestral_load("5M")
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' ggplot2::autoplot(fc)
#' }
autoplot.timestral_forecast <- function(object, series = 1L,
                                        levels = c(0.5, 0.8, 0.9, 0.98),
                                        ...) {
  d <- .fc_gg_data(object, series, levels)
  p <- ggplot2::ggplot() +
    .fc_band_layers(d) +
    ggplot2::geom_line(
      data = d$hist,
      ggplot2::aes(x = .data$time, y = .data$value),
      colour = "black", linewidth = 0.6, inherit.aes = FALSE) +
    ggplot2::labs(x = "Time", y = "Value", fill = "Interval")
  if (d$nv > 1L)
    p <- p + ggplot2::facet_wrap(~variate, ncol = 1L, scales = "free_y")
  p
}

#' @rdname autoplot.timestral_forecast
autolayer.timestral_forecast <- function(object, series = 1L,
                                         levels = c(0.5, 0.8, 0.9, 0.98),
                                         ...) {
  .fc_band_layers(.fc_gg_data(object, series, levels))
}
