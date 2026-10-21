# Forecast with the forecast package's `forecast()` generic

Registered as an S3 method for
[`forecast::forecast()`](https://generics.r-lib.org/reference/forecast.html)
when the forecast package is loaded, so
`forecast::forecast(m, AirPassengers, h = 24)` forecasts a `ts` with a
Timestral model and returns a `"forecast"` object. It calls
[`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md)
and then
[`timestral_as_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_as_forecast.md);
see that page for the interval levels.

## Usage

``` r
# S3 method for class 'timestral_model'
forecast(object, y, h, level = c(80, 98), variate = 1L, ...)
```

## Arguments

- object:

  a `timestral_model` from
  [`timestral_load()`](https://sjvrensburg.github.io/timestralR/reference/timestral_load.md)

- y:

  the series to forecast, a `ts` (a univariate `ts`, or an `mts` whose
  columns are forecast jointly; pick the column with `variate`)

- h:

  horizon in steps (required)

- level:

  interval levels in percent, each pairing two quantile knots

- variate:

  which variate of a multivariate `y` to return

- ...:

  further arguments to
  [`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md),
  such as covariates or `symmetric`; other forecast-generic arguments
  (`fan`, `lambda`, `biasadj`, ...) are rejected

## Value

An object of class `"forecast"`.

## Examples

``` r
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
if (requireNamespace("forecast", quietly = TRUE))
  forecast::forecast(m, AirPassengers, h = 12)
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
