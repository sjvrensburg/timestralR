# Convert a Timestral forecast to a tsibble

The result is a `tsibble` over the forecast horizon with one row per
series/variate/step and one column per quantile level (as in
[`as_forecast_df()`](https://sjvrensburg.github.io/timestralR/reference/as_forecast_df.md)).
Registered as an S3 method for
[`tsibble::as_tsibble()`](https://tsibble.tidyverts.org/reference/as-tsibble.html)
when the tsibble package is loaded; call it as
`tsibble::as_tsibble(fc)`. The index column keeps the name of the input
tsibble's index, and is `index` otherwise, as
[`tsibble::as_tsibble()`](https://tsibble.tidyverts.org/reference/as-tsibble.html)
names it for `ts`, `zoo` and `xts`. Forecasts from `ts` input use the
same index class (`yearmonth`, `yearquarter`, ...) that
[`tsibble::as_tsibble()`](https://tsibble.tidyverts.org/reference/as-tsibble.html)
gives the `ts`, so they join with the observed data. Forecasts can also
be created directly from a tsibble: in
[`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md),
tsibble keys become the batch and measured variables the jointly
forecast variates.

## Usage

``` r
# S3 method for class 'timestral_forecast'
as_tsibble(x, ...)
```

## Arguments

- x:

  a `timestral_forecast` created from indexed input (`ts`, `zoo`, `xts`
  or `tsibble`), so a future time index is available

- ...:

  ignored

## Value

a `tsibble` (`tbl_ts`)

## Examples

``` r
## the tiny bundled checkpoint runs offline but its forecasts are
## meaningless; use timestral_load("5M") or larger for real ones
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
fc <- timestral_forecast(m, AirPassengers, h = 24)
if (requireNamespace("tsibble", quietly = TRUE))
  tsibble::as_tsibble(fc)
#> # A tsibble: 24 x 23 [1M]
#>    variate   q01   q05   q10   q15   q20   q25   q30   q35   q40   q45   q50
#>    <chr>   <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#>  1 V1      107.  120.  150.  170.  190.  218.  230.   248.  258.  271.  291.
#>  2 V1      231.  242.  254.  265.  285.  292.  306.   322.  337.  350.  361.
#>  3 V1      100.  120.  137.  162.  175.  197.  218.   242.  262.  274.  281.
#>  4 V1      159.  168.  190.  208.  237.  249.  264.   277.  295.  307.  318.
#>  5 V1        0    28.2  60.5  97.7 118.  140.  161.   180.  197.  210.  224.
#>  6 V1        0     0     0     0    43.4  67.5  87.8  127.  162.  176.  190.
#>  7 V1      157.  174.  187.  210.  228.  241.  267.   284.  317.  331.  351.
#>  8 V1       18.6  43.9  75.2  98.3 127.  156.  173.   186.  200.  220.  245.
#>  9 V1      106.  130.  145.  169.  186.  206.  217.   224.  238.  257.  271.
#> 10 V1      137.  156.  175.  198.  220.  239.  248.   260.  281.  295.  314.
#> # ℹ 14 more rows
#> # ℹ 11 more variables: q55 <dbl>, q60 <dbl>, q65 <dbl>, q70 <dbl>, q75 <dbl>,
#> #   q80 <dbl>, q85 <dbl>, q90 <dbl>, q95 <dbl>, q99 <dbl>, index <mth>
```
