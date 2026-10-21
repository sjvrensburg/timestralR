# Convert a Timestral forecast to a fable forecast table

Registered as an S3 method for
[`fabletools::as_fable()`](https://fabletools.tidyverts.org/reference/as-fable.html)
when fabletools is loaded; call it as `fabletools::as_fable(fc)`. The
distribution column is the 21-quantile
[`distributional::dist_quantile()`](https://pkg.mitchelloharawild.com/distributional/reference/dist_quantile.html)
distribution from the forecast. A univariate forecast uses the variate
name as the response; a fable holds a single distribution column, so a
multivariate forecast becomes a `variate` key with the response in a
`value` column. Forecasts from `ts` input use the same index class
(`yearmonth`, `yearquarter`, ...) as
[`tsibble::as_tsibble()`](https://tsibble.tidyverts.org/reference/as-tsibble.html)
gives the `ts`, so they join with and score against the observed data,
and fable accuracy measures and interval extraction apply. See
[`TIMESTRAL()`](https://sjvrensburg.github.io/timestralR/reference/TIMESTRAL.md)
for using Timestral directly as a fable model.

## Usage

``` r
# S3 method for class 'timestral_forecast'
as_fable(x, ...)
```

## Arguments

- x:

  a `timestral_forecast` created from indexed input (`ts`, `zoo`, `xts`
  or `tsibble`), so a future time index is available

- ...:

  ignored

## Value

a `fbl_ts` forecast table

## Examples

``` r
## the tiny bundled checkpoint runs offline but its forecasts are
## meaningless; use timestral_load("5M") or larger for real ones
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
fc <- timestral_forecast(m, AirPassengers, h = 24)
if (requireNamespace("tsibble", quietly = TRUE) &&
    requireNamespace("distributional", quietly = TRUE) &&
    requireNamespace("fabletools", quietly = TRUE))
  fabletools::as_fable(fc)
#> # A fable: 24 x 2 [1M]
#>       index           V1
#>       <mth>       <dist>
#>  1 1961 Jan quantile[21]
#>  2 1961 Feb quantile[21]
#>  3 1961 Mar quantile[21]
#>  4 1961 Apr quantile[21]
#>  5 1961 May quantile[21]
#>  6 1961 Jun quantile[21]
#>  7 1961 Jul quantile[21]
#>  8 1961 Aug quantile[21]
#>  9 1961 Sep quantile[21]
#> 10 1961 Oct quantile[21]
#> # ℹ 14 more rows
```
