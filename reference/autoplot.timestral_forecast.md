# ggplot2 fan chart of a Timestral forecast

Registered as S3 methods for
[`ggplot2::autoplot()`](https://ggplot2.tidyverse.org/reference/autoplot.html)
and
[`ggplot2::autolayer()`](https://ggplot2.tidyverse.org/reference/autolayer.html)
when ggplot2 is loaded. `autoplot` draws the observed history, the
median, and nested predictive interval bands; `autolayer` adds just the
forecast bands and median to an existing ggplot (for example of the
observed series). Bands are drawn widest first in graded shades of blue,
matching
[`plot.timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/plot.timestral_forecast.md);
add any fill scale, such as
[`ggplot2::scale_fill_brewer()`](https://ggplot2.tidyverse.org/reference/scale_brewer.html),
to change them.

## Usage

``` r
# S3 method for class 'timestral_forecast'
autoplot(object, series = 1L, levels = c(0.5, 0.8, 0.9, 0.98), ...)

# S3 method for class 'timestral_forecast'
autolayer(object, series = 1L, levels = c(0.5, 0.8, 0.9, 0.98), ...)
```

## Arguments

- object:

  a `timestral_forecast`

- series:

  which series of a batch to plot (default 1)

- levels:

  central predictive intervals to draw as bands; each must correspond to
  a pair of available quantile knots

- ...:

  ignored

## Value

for `autoplot`, a `ggplot`; for `autolayer`, a list of layers

## Examples

``` r
## the tiny bundled checkpoint runs offline but its forecasts are
## meaningless; use timestral_load("5M") or larger for real ones
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
fc <- timestral_forecast(m, AirPassengers, h = 24)
if (requireNamespace("ggplot2", quietly = TRUE))
  ggplot2::autoplot(fc)
```
