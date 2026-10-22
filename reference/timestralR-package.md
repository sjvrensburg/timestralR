# timestralR: Zero-Shot Probabilistic Forecasting with Timestral

Runs Mistral AI's Timestral time-series foundation models from R without
Python or PyTorch.
[`timestral_load()`](https://sjvrensburg.github.io/timestralR/reference/timestral_load.md)
downloads the official `safetensors` checkpoints once (or reads a local
checkpoint directory) and
[`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md)
produces 21-quantile probabilistic forecasts, jointly across the
variates of a series and independently across a batch of series. Inputs
may be plain vectors/matrices, `ts`/`mts`, `zoo`/`xts` (time index
preserved) or `tsibble`; numeric and categorical covariates are
supported. See
[`plot.timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/plot.timestral_forecast.md)
for fan charts and
[`as_forecast_df()`](https://sjvrensburg.github.io/timestralR/reference/as_forecast_df.md)
for a tidy long-format output;
[`autoplot.timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/autoplot.timestral_forecast.md),
[`as_tsibble.timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/as_tsibble.timestral_forecast.md),
[`as_fable.timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/as_fable.timestral_forecast.md)
and
[`TIMESTRAL()`](https://sjvrensburg.github.io/timestralR/reference/TIMESTRAL.md)
connect forecasts to ggplot2 and the tidyverts.

## See also

Useful links:

- <https://github.com/sjvrensburg/timestralR>

- <https://sjvrensburg.github.io/timestralR/>

- Report bugs at <https://github.com/sjvrensburg/timestralR/issues>

## Author

**Maintainer**: Stefan Janse van Rensburg <sjansevanrensburg@gmail.com>

Other contributors:

- Mistral AI (authors of the 'timestral' Python package ported in R/ and
  src/) \[copyright holder\]
