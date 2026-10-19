# timestralR

Run Mistral AI's Timestral forecasting models from R — no Python, no PyTorch,
no external runtime. The package downloads the official `safetensors`
checkpoints from Mistral AI's Hugging Face repositories and runs the
transformer forward pass natively in R (on R's BLAS). The forecasting loop —
causal per-patch normalisation, two-stream time/variate attention, block
decoding with median feedback, mirrored averaging, quantile capping and
clipping — is a direct port of the official `timestral` Python package.

## Install

```r
# from a clone:
# R CMD INSTALL timestralR   (or devtools::install())
```

Requires only R and the `jsonlite` package. Model weights are downloaded
once per user on first use (21 MB for 5M up to 837 MB for 209M, fp32) and
cached. Weights are Mistral AI's, under their Modified MIT licence; package
code is Apache 2.0.

## Use

```r
library(timestralR)
m  <- timestral_load("28M")                  # or a local checkpoint directory
fc <- timestral_forecast(m, AirPassengers, h = 24)
fc$quantiles                                 # array: variate x step x 21 levels
head(as_forecast_df(fc))
# multivariate (columns = variates), past-only and known-future covariates
# (time in rows; known covariates need T + h rows):
fc <- timestral_forecast(m, Y, h = 96, past_covariates = P, known_covariates = K)
```

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

| model | worst max_rel (case) | R time, worst case |
|---|---|---|
| 5M | 2.0e-6 (long horizon) | 0.8 s |
| 28M | 2.9e-7 (spot check, univariate) | — |
| 123M | 9.6e-6 (long context cap) | 8.1 s |
| 209M | 2.6e-6 (long context cap) | 13.5 s |

Reproduce with `inst/parity_official.R` (developer machine only; end
users never need Python).

## Not yet supported

Nothing in the Python package's inference surface: batched series,
observation masks, numeric and categorical covariates, and all
forecaster/loader options are supported. (Training and GPU execution are
Python-side concerns with no R analogue.) Developer notes, including the
frozen internal interface for a future native port, are in
`docs/developer-notes.md`.
