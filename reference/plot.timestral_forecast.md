# Fan-chart plot of a Timestral forecast

Draws the observed context, the median forecast, and nested central
predictive interval bands (widest lightest) per variate. Uses the stored
time index when the forecast input carried one.

## Usage

``` r
# S3 method for class 'timestral_forecast'
plot(
  x,
  series = 1L,
  levels = c(0.5, 0.8, 0.9, 0.98),
  main = NULL,
  xlab = "Time",
  ylab = NULL,
  ...
)
```

## Arguments

- x:

  a `timestral_forecast`

- series:

  which series of a batch to plot (default 1)

- levels:

  central predictive intervals to draw as bands; each must correspond to
  a pair of available quantile knots (with the default knots, 0.5, 0.8,
  0.9 and 0.98 all do)

- main:

  panel titles; default uses the variate names

- xlab, ylab:

  axis labels

- ...:

  passed to
  [`graphics::plot()`](https://rdrr.io/r/graphics/plot.default.html)

## Value

`x`, invisibly

## Examples

``` r
## the tiny bundled checkpoint runs offline but its forecasts are
## meaningless; use timestral_load("5M") or larger for real ones
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
fc <- timestral_forecast(m, AirPassengers, h = 24)
plot(fc)
```
