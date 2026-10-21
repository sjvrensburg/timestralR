# Which covariates a checkpoint can condition on

`timestral_supports_covariates()` is `TRUE` when the checkpoint was
trained with covariates, so
[`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md)
accepts `past_covariates` and `categorical_past_covariates`.
`timestral_supports_known_covariates()` is `TRUE` when it can also
condition on known-future covariates (`known_covariates` and
`categorical_known_covariates`), which needs a block schedule that
attends across variates. Both mirror the Python forecaster's
`supports_covariates` and `supports_known_covariates` properties; every
official checkpoint supports both.

## Usage

``` r
timestral_supports_covariates(model)

timestral_supports_known_covariates(model)
```

## Arguments

- model:

  a `timestral_model` from
  [`timestral_load()`](https://sjvrensburg.github.io/timestralR/reference/timestral_load.md)

## Value

`TRUE` or `FALSE`.

## Examples

``` r
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
timestral_supports_covariates(m)
#> [1] TRUE
timestral_supports_known_covariates(m)
#> [1] TRUE
```
