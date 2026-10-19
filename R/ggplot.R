## ggplot2 plotting (ggplot2 is a Suggests-only dependency; these methods
## are registered for ggplot2::autoplot/autolayer, via delayed S3method()
## in NAMESPACE, when ggplot2 is loaded).
utils::globalVariables(".data")

## Long-format band/median/history data for one series of a forecast.
.fc_gg_data <- function(fc, series, levels) {
  pp <- .fc_plot_parts(fc, series)
  obs <- pp$obs; nv <- pp$nv; vnames <- pp$vnames
  times <- pp$times; ftimes <- pp$ftimes
  qmat <- pp$qmat; knot <- pp$knot
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
#' `ggplot2::autolayer()` when ggplot2 is loaded. `autoplot` draws the
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
#' @exportS3Method ggplot2::autoplot
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
#' @exportS3Method ggplot2::autolayer
autolayer.timestral_forecast <- function(object, series = 1L,
                                         levels = c(0.5, 0.8, 0.9, 0.98),
                                         ...) {
  .fc_band_layers(.fc_gg_data(object, series, levels))
}
