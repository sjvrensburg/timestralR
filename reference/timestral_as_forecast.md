# Convert a Timestral forecast to a `forecast`-package object

Builds an object of class `"forecast"`, so the forecast package's tools
([`autoplot()`](https://ggplot2.tidyverse.org/reference/autoplot.html),
[`accuracy()`](https://generics.r-lib.org/reference/accuracy.html),
[`print()`](https://rdrr.io/r/base/print.html), ...) work on a Timestral
forecast. A `"forecast"` object holds one univariate series: pick it
with `series` and `variate`.

## Usage

``` r
timestral_as_forecast(fc, level = c(80, 98), series = 1L, variate = 1L)
```

## Arguments

- fc:

  a `timestral_forecast` made from `ts` input, so it has a time base

- level:

  interval levels in percent (fractions in (0, 1) are taken as
  percentages, as in the forecast package); each must pair two quantile
  knots (see Details)

- series:

  which series of a batch forecast

- variate:

  which variate, by index or name

## Value

An object of class `"forecast"` with `mean`, `lower`, `upper` (`ts`
matrices with one column per level), `level`, `x` (the observed context,
`NA` where missing), `series` and `method = "Timestral"`. `fitted` and
`residuals` are all `NA`: a zero-shot model is not fitted to the
context, so
[`forecast::accuracy()`](https://generics.r-lib.org/reference/accuracy.html)
reports test-set measures (training-set rows are `NaN`).

## Details

The median is the point forecast (`mean`). Each interval level `L` takes
its bounds from the quantile knots at `(1 - L/100) / 2` and
`(1 + L/100) / 2`, so `L` must pair two knots of the checkpoint: with
the official knots
([`timestral_levels()`](https://sjvrensburg.github.io/timestralR/reference/timestral_levels.md))
80, 90 and 98 work, but the forecast package's usual 95 does not,
because there are no 0.025 and 0.975 knots. Quantiles are not
interpolated between knots. The default levels are therefore
`c(80, 98)`.

## See also

[`forecast.timestral_model()`](https://sjvrensburg.github.io/timestralR/reference/forecast.timestral_model.md)
to forecast and convert in one call.

## Examples

``` r
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
fc <- timestral_forecast(m, AirPassengers, h = 12)
if (requireNamespace("forecast", quietly = TRUE))
  timestral_as_forecast(fc)
#>          Point Forecast     Lo 80    Hi 80     Lo 98    Hi 98
#> Jan 1961       291.3287 149.92005 426.9252 106.90228 461.7679
#> Feb 1961       360.7115 254.04978 558.2604 231.30467 601.1172
#> Mar 1961       280.6464 136.70915 418.9652 100.46142 474.5209
#> Apr 1961       318.2002 190.45405 438.8481 159.39363 474.5175
#> May 1961       223.7060  60.50897 322.5853   0.00000 342.4975
#> Jun 1961       189.5016   0.00000 334.6955   0.00000 358.1365
#> Jul 1961       350.5766 186.59084 555.8250 156.56557 588.1156
#> Aug 1961       244.7158  75.22611 408.3032  18.60671 447.1394
#> Sep 1961       270.5898 144.60990 384.3951 106.20850 421.0712
#> Oct 1961       313.5198 175.20696 454.9660 136.71976 494.8207
#> Nov 1961       281.9475 149.32027 418.8006 104.65197 464.4149
#> Dec 1961       301.3957 208.00524 414.8535 166.54181 461.7250
```
