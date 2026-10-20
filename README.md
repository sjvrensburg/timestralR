# timestralR

Run Mistral AI's Timestral forecasting models from R — no Python, no PyTorch,
no external runtime. The package downloads the official `safetensors`
checkpoints from Mistral AI's Hugging Face repositories and runs the
transformer forward pass in a native RcppEigen backend (a pure-R reference
implementation is included as a fallback). The forecasting loop —
causal per-patch normalisation, two-stream time/variate attention, block
decoding with median feedback, mirrored averaging, quantile capping and
clipping — is a direct port of the official `timestral` Python package.

## Install

```r
remotes::install_github("sjvrensburg/timestralR")
# or from a clone:
# R CMD INSTALL timestralR   (or devtools::install())
```

Requires R (>= 4.1) with a C++17 compiler; the hard R dependencies are
`jsonlite`, `digest` and `Rcpp`/`RcppEigen`, plus the base packages `utils`,
`tools` and `stats`. `zoo`, `xts`, `tsibble`, `ggplot2`,
`fabletools`, `fable` and `distributional` are optional and only needed
for the matching interfaces. Model weights are downloaded once per user
on first use (21 MB for 5M up to 837 MB for 209M, fp32) and cached.
Weights are Mistral AI's, under their Modified MIT licence; package code
is Apache 2.0, ported from Mistral AI's Apache-2.0 `timestral` package.

## Use

The vignette walks through every feature with plots:
`vignette("timestralR")`.

```r
library(timestralR)
m  <- timestral_load("28M")                  # or a local directory or Hub repo ID
fc <- timestral_forecast(m, AirPassengers, h = 24)
plot(fc)                                     # fan chart: history + median + bands
timestral_median(fc)                         # a ts, starting Jan 1961
timestral_quantile(fc, 0.9)                  # any of fc$levels
fc$quantiles                                 # array: variate x step x 21 levels
head(as_forecast_df(fc))                     # tidy long format (with time column)
```

Inputs can be numeric vectors, `ts`/`mts`, `zoo`/`xts` (the time index
is kept), matrices or data frames with time in rows, or a list of series
forecast as a batch. Columns of a series are forecast jointly; `NA`
marks missing values, and `observation_mask` can mark more.

```r
# multivariate: columns are forecast jointly
fc2 <- timestral_forecast(m, cbind(mdeaths, fdeaths), h = 12)

# a batch of independent series in one call
fcb <- timestral_forecast(m, list(a = mdeaths, b = fdeaths), h = 12)

# covariates, time in rows: past-only over the context, known-future over
# the context plus the horizon (T + h rows)
y <- as.numeric(AirPassengers)
x <- seq_len(length(y) + 12)
fc3 <- timestral_forecast(m, y, h = 12,
                          past_covariates = log(y),
                          known_covariates = cbind(trend = x),
                          categorical_known_covariates = (x - 1) %% 12)
```

### tidyverts and ggplot2

```r
library(tsibble)
library(fable)

# tsibble input: keys become the batch, measured variables the jointly
# forecast variates; rows are taken in index order and implicit gaps are
# filled with NA
fc <- timestral_forecast(m, as_tsibble(AirPassengers), h = 24)
as_tsibble(fc)                 # quantile columns, yearmonth index
as_fable(fc)                   # 21-quantile distributions
ggplot2::autoplot(fc)          # ggplot2 fan chart

# as a fable model, alongside the usual ones
as_tsibble(AirPassengers) |>
  model(timestral = TIMESTRAL(value, model = m), ets = ETS(value)) |>
  forecast(h = 24)
```

Forecasts from `ts` input get the same index class as
`tsibble::as_tsibble()` gives the `ts`, so they join with and score
against the observed data. Fitted `TIMESTRAL()` models keep a handle to
the checkpoint rather than the weights, so mables stay small when saved
and work in parallel workers.

## Validation

Parity against the official PyTorch package (`timestral` 1.0.0, torch
2.14.1 CPU, fp32 weights), twelve cases (missing values, negative trend
without mirroring, multivariate, numeric covariates, horizon 1100,
context 5000 capped to 4096 with horizon 2100, batch of 3, observation
mask, categorical past/known/multivariate, non-default forecaster
knobs), where `max_rel` is max abs difference over all
quantiles/steps/variates relative to the largest official forecast
value. Synthetic config variants (biases, gate-less, no-asinh) also
match to < 2.3e-7; see `docs/developer-notes.md`:

| model | worst max_rel (case) |
|---|---|
| 5M | 2.0e-6 (long horizon) |
| 28M | 2.9e-7 (spot check, univariate) |
| 123M | 9.6e-6 (long context cap) |
| 209M | 2.6e-6 (long context cap) |

Native-backend timings are in `docs/developer-notes.md` (209M: batch of 10
in 2.3 s, long-context worst case in 5.0 s on the development machine).

Reproduce with `inst/parity_official.R` (developer machine only; end
users never need Python). One case is also gated in CI: the 5M
`univariate_missing` forecast is committed as a golden fixture
(`tests/fixtures/{input,golden}_5m_univariate.csv`, generated once by
`inst/make_5m_fixture.R` from the official package) and the weekly
real-checkpoint run asserts the native forecast matches it to the
parity gate above.

## Not yet supported

Nothing in the Python package's inference surface: batched series,
observation masks, numeric and categorical covariates, and all
forecaster/loader options are supported. (Training and GPU execution are
Python-side concerns with no R analogue.)

## Development

Help pages and `NAMESPACE` are generated with roxygen2
(`Rscript -e 'roxygen2::roxygenise()'`); the native backend in `src/` is
split into one header and one implementation file per unit. Developer
notes, including the source layout, the frozen internal interface and
benchmarks, are in `docs/developer-notes.md`.
