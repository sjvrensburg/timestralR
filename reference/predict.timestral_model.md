# Forecast with a Timestral model through `predict()`

The [`stats::predict()`](https://rdrr.io/r/stats/predict.html) interface
to
[`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md).
Timestral is a zero-shot model, so there is nothing fitted to predict
from: `newdata` is the series to forecast (the context), not new
predictor values.

## Usage

``` r
# S3 method for class 'timestral_model'
predict(object, newdata, h, ...)
```

## Arguments

- object:

  a `timestral_model` from
  [`timestral_load()`](https://sjvrensburg.github.io/timestralR/reference/timestral_load.md)

- newdata:

  the series to forecast; anything
  [`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md)
  accepts as `y`

- h:

  horizon in steps (required)

- ...:

  further arguments to
  [`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md),
  such as covariates, `observation_mask` or `symmetric`

## Value

A `timestral_forecast`, exactly as from
[`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md);
use
[`timestral_median()`](https://sjvrensburg.github.io/timestralR/reference/timestral_median.md)
or
[`timestral_quantile()`](https://sjvrensburg.github.io/timestralR/reference/timestral_quantile.md)
for point values.

## Examples

``` r
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
fc <- predict(m, AirPassengers, h = 12)
timestral_median(fc)
#>           Jan      Feb      Mar      Apr      May      Jun      Jul      Aug
#> 1961 291.3287 360.7115 280.6464 318.2002 223.7060 189.5016 350.5766 244.7158
#>           Sep      Oct      Nov      Dec
#> 1961 270.5898 313.5198 281.9475 301.3957
```
