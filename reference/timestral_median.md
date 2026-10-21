# Median forecast

Median forecast

## Usage

``` r
timestral_median(fc)
```

## Arguments

- fc:

  a `timestral_forecast`

## Value

The 0.5-quantile slice. For a single series forecast from a `ts`, a
`ts`/`mts` starting after the context; otherwise the same slice of the
quantiles array as before.

## Examples

``` r
## the tiny bundled checkpoint runs offline but its forecasts are
## meaningless; use timestral_load("5M") or larger for real ones
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
fc <- timestral_forecast(m, AirPassengers, h = 24)
timestral_median(fc)   # a ts, starting Jan 1961
#>           Jan      Feb      Mar      Apr      May      Jun      Jul      Aug
#> 1961 291.3287 360.7115 280.6464 318.2002 223.7060 189.5016 350.5766 244.7158
#> 1962 260.1235 270.1221 257.1679 300.3938 292.1597 363.0654 276.1783 315.2443
#>           Sep      Oct      Nov      Dec
#> 1961 270.5898 313.5198 281.9475 301.3957
#> 1962 227.9375 200.5240 348.4063 245.5896
```
