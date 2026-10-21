# Package index

## Models

Download, load and inspect Timestral checkpoints.

- [`timestral_load()`](https://sjvrensburg.github.io/timestralR/reference/timestral_load.md)
  : Load a Timestral model
- [`timestral_download()`](https://sjvrensburg.github.io/timestralR/reference/timestral_download.md)
  : Download a Timestral checkpoint
- [`timestral_supports_covariates()`](https://sjvrensburg.github.io/timestralR/reference/timestral_supports_covariates.md)
  [`timestral_supports_known_covariates()`](https://sjvrensburg.github.io/timestralR/reference/timestral_supports_covariates.md)
  : Which covariates a checkpoint can condition on

## Forecasting

- [`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md)
  : Forecast one or more series

- [`predict(`*`<timestral_model>`*`)`](https://sjvrensburg.github.io/timestralR/reference/predict.timestral_model.md)
  :

  Forecast with a Timestral model through
  [`predict()`](https://rdrr.io/r/stats/predict.html)

- [`timestral_median()`](https://sjvrensburg.github.io/timestralR/reference/timestral_median.md)
  : Median forecast

- [`timestral_quantile()`](https://sjvrensburg.github.io/timestralR/reference/timestral_quantile.md)
  : Forecast at one quantile level

- [`timestral_levels()`](https://sjvrensburg.github.io/timestralR/reference/timestral_levels.md)
  : Quantile levels of the official Timestral checkpoints

## Evaluation

Score forecasts and run rolling-origin backtests.

- [`timestral_score()`](https://sjvrensburg.github.io/timestralR/reference/timestral_score.md)
  : Score quantile forecasts against realised values
- [`timestral_backtest()`](https://sjvrensburg.github.io/timestralR/reference/timestral_backtest.md)
  : Rolling-origin backtest of a Timestral model

## Plotting

- [`plot(`*`<timestral_forecast>`*`)`](https://sjvrensburg.github.io/timestralR/reference/plot.timestral_forecast.md)
  : Fan-chart plot of a Timestral forecast
- [`autoplot(`*`<timestral_forecast>`*`)`](https://sjvrensburg.github.io/timestralR/reference/autoplot.timestral_forecast.md)
  [`autolayer(`*`<timestral_forecast>`*`)`](https://sjvrensburg.github.io/timestralR/reference/autoplot.timestral_forecast.md)
  : ggplot2 fan chart of a Timestral forecast

## Interfaces

Conversions and methods for tsibble, fable and the forecast package.

- [`as_forecast_df()`](https://sjvrensburg.github.io/timestralR/reference/as_forecast_df.md)
  : Forecast as a long data frame

- [`as_tsibble(`*`<timestral_forecast>`*`)`](https://sjvrensburg.github.io/timestralR/reference/as_tsibble.timestral_forecast.md)
  : Convert a Timestral forecast to a tsibble

- [`TIMESTRAL()`](https://sjvrensburg.github.io/timestralR/reference/TIMESTRAL.md)
  : Timestral as a fable model

- [`as_fable(`*`<timestral_forecast>`*`)`](https://sjvrensburg.github.io/timestralR/reference/as_fable.timestral_forecast.md)
  : Convert a Timestral forecast to a fable forecast table

- [`forecast(`*`<timestral_model>`*`)`](https://sjvrensburg.github.io/timestralR/reference/forecast.timestral_model.md)
  :

  Forecast with the forecast package's
  [`forecast()`](https://generics.r-lib.org/reference/forecast.html)
  generic

- [`timestral_as_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_as_forecast.md)
  :

  Convert a Timestral forecast to a `forecast`-package object

## Package

- [`timestralR`](https://sjvrensburg.github.io/timestralR/reference/timestralR-package.md)
  [`timestralR-package`](https://sjvrensburg.github.io/timestralR/reference/timestralR-package.md)
  : timestralR: Zero-Shot Probabilistic Forecasting with Timestral
