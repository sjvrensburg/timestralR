# Timestral as a fable model

Defines a Timestral zero-shot forecasting model for use with
[`fabletools::model()`](https://fabletools.tidyverts.org/reference/model.html)
on a `tsibble`. The response is univariate (use
[`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md)
directly for jointly forecast multiple variates); keys in the tsibble
are handled by fable as usual, with the model trained (context captured)
independently per key. Requires the fabletools package (and fable for
the usual workflow); forecasts are reported as 21-quantile
distributions.

## Usage

``` r
TIMESTRAL(formula, model, symmetric = TRUE, clip_nonnegative = TRUE, ...)
```

## Arguments

- formula:

  the response formula, as in other fable models, e.g.
  `TIMESTRAL(value)`

- model:

  a `timestral_model` from
  [`timestral_load()`](https://sjvrensburg.github.io/timestralR/reference/timestral_load.md);
  loaded once and reused for every key and forecast. Fitted models store
  only the checkpoint directory and load settings, not the weights, so a
  mable stays small when saved and is reloaded from disk when forecast
  in a new R session or a parallel worker.

- symmetric, clip_nonnegative:

  as in
  [`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md)

- ...:

  passed to the model definition (currently unused)

## Value

a fable model definition, for use inside
[`fabletools::model()`](https://fabletools.tidyverts.org/reference/model.html)

## Examples

``` r
## the tiny bundled checkpoint runs offline but its forecasts are
## meaningless; use timestral_load("5M") or larger for real ones
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
if (requireNamespace("fable", quietly = TRUE)) {
  fit <- tsibble::as_tsibble(AirPassengers) |>
    fabletools::model(timestral = TIMESTRAL(value, model = m))
  fit |> fabletools::forecast(h = 24)
}
#> # A fable: 24 x 4 [1M]
#> # Key:     .model [1]
#>    .model       index        value .mean
#>    <chr>        <mth>       <dist> <dbl>
#>  1 timestral 1961 Jan quantile[21]  288.
#>  2 timestral 1961 Feb quantile[21]  384.
#>  3 timestral 1961 Mar quantile[21]  283.
#>  4 timestral 1961 Apr quantile[21]  314.
#>  5 timestral 1961 May quantile[21]  206.
#>  6 timestral 1961 Jun quantile[21]  177.
#>  7 timestral 1961 Jul quantile[21]  360.
#>  8 timestral 1961 Aug quantile[21]  245.
#>  9 timestral 1961 Sep quantile[21]  269.
#> 10 timestral 1961 Oct quantile[21]  316.
#> # ℹ 14 more rows
```
