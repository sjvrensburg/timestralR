# Score quantile forecasts against realised values

Scores a
[`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md)
against the values that were later observed, one row per series, variate
and step.

## Usage

``` r
timestral_score(fc, actual, level = c(80, 98), period = NULL, scale = NULL)
```

## Arguments

- fc:

  a `timestral_forecast`

- actual:

  the realised values over the horizon, shaped like a forecast input
  with `h` time steps: a vector or `ts` (one series, one variate), a
  matrix with time in rows (one series), a 3-D array (time, variate,
  series), or a list of those per series

- level:

  interval levels in percent for the coverage columns (fractions in
  (0, 1) are taken as percentages)

- period:

  seasonal period \\m\\ of the default scale; `NULL` takes the frequency
  of a `ts` forecast and 1 otherwise. Set it for seasonal `zoo`, `xts`
  or plain input, which carry no period.

- scale:

  `NULL` for the seasonal-naive scale, or your own: one positive number,
  one per series, a series-by-variate matrix, or a function of one
  context (a numeric vector, `NA` where missing) returning one number,
  such as `function(x) mean(abs(x), na.rm = TRUE)`

## Value

A data frame with columns `series`, `variate`, `step`, `actual`,
`median`, `pinball`, `crps`, `ae`, `scale`, `mase`, `scaled_pinball`,
`scaled_crps` and one `cover<L>` column per level.

## Details

The scores, for realised value \\y\\ and quantile \\q\_\tau\\ at each
level \\\tau\\ of the forecast (`fc$levels`), use the pinball (quantile)
loss \\\rho\_\tau = (1\\y \< q\_\tau\\ - \tau)(q\_\tau - y)\\:

- `pinball`: \\2\rho\_\tau\\ averaged over the levels, the quantile loss
  used in the GluonTS and Chronos benchmarks (before dividing by the
  scale of the series). The factor 2 makes it equal the absolute error
  for a point forecast.

- `crps`: the continuous ranked probability score, \\2\int_0^1
  \rho\_\tau\\d\tau\\, approximated by the trapezoidal rule over the
  knots. The knots are not evenly spaced, so they are weighted by their
  spacing rather than averaged. The integral only covers the knots'
  range (0.01 to 0.99 for the official checkpoints), so for a point
  forecast it gives \\0.98\|q - y\|\\ rather than \\\|q - y\|\\: the
  tails beyond the outermost knots are ignored.

- `ae`: the absolute error of the median.

- `cover<L>`: whether \\y\\ lies in the central `L`% interval, for each
  `level`; each level must pair two knots, as in
  [`plot()`](https://rdrr.io/r/graphics/plot.default.html).

The scaled scores divide `ae`, `pinball` and `crps` by a scale of the
series, so they can be averaged across series of different magnitude:
`mase` (averaged, the mean absolute scaled error of Hyndman and Koehler,
2006), `scaled_pinball` and `scaled_crps`; the `scale` column holds the
divisor. By default the scale is the in-sample mean absolute error of
the seasonal naive forecast over the forecast's context,
\\\mathrm{mean}\|y_t - y\_{t-m}\|\\ for the seasonal period \\m\\
(`period`), over pairs of observed values from the first observed value
on. A context with no more than \\m\\ steps uses lag 1, as in the M4
competition. In a backtest each origin is scaled by its own context, so
the scale never uses values after the origin. A scale that is zero (a
constant context) or cannot be computed gives `NA` scaled scores and a
warning.

Missing actual values give missing scores. Average the rows with
[`stats::aggregate()`](https://rdrr.io/r/stats/aggregate.html) or dplyr
to summarise by step, series or origin.

## See also

[`timestral_backtest()`](https://sjvrensburg.github.io/timestralR/reference/timestral_backtest.md)
to forecast and score from several origins.

## Examples

``` r
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
train <- window(AirPassengers, end = c(1959, 12))
test <- window(AirPassengers, start = c(1960, 1))
sc <- timestral_score(timestral_forecast(m, train, h = 12), test)
colMeans(sc[, c("pinball", "crps", "ae", "mase", "scaled_crps", "cover80")])
#>     pinball        crps          ae        mase scaled_crps     cover80 
#> 159.0576796 161.5815572 208.6903368   6.8535414   5.3064551   0.1666667 
```
