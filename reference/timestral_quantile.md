# Forecast at one quantile level

Forecast at one quantile level; errors if `level` is not one of the
model's quantile levels.

## Usage

``` r
timestral_quantile(fc, level)
```

## Arguments

- fc:

  a `timestral_forecast`

- level:

  one of the forecast's quantile levels, `fc$levels`

## Value

The requested quantile slice, as a `ts`/`mts` for a single series
forecast from a `ts`, otherwise an array slice.

## Examples

``` r
## the tiny bundled checkpoint runs offline but its forecasts are
## meaningless; use timestral_load("5M") or larger for real ones
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
fc <- timestral_forecast(m, AirPassengers, h = 24)
timestral_quantile(fc, 0.9)
#>           Jan      Feb      Mar      Apr      May      Jun      Jul      Aug
#> 1961 426.9252 558.2604 418.9652 438.8481 322.5853 334.6955 555.8250 408.3032
#> 1962 392.1073 363.8866 367.0136 426.5677 424.3336 555.7488 408.9492 433.7136
#>           Sep      Oct      Nov      Dec
#> 1961 384.3951 454.9660 418.8006 414.8535
#> 1962 323.5285 342.8294 548.1166 402.3553
```
