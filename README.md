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
2.14.1 CPU, fp32 weights) on the 5M model, six cases (missing values,
negative trend without mirroring, multivariate, covariates, horizon 1100,
context 5000 capped to 4096 with horizon 2100): worst `max_rel` 2.0e-6,
where `max_rel` is max abs difference over all quantiles/steps/variates
relative to the largest official forecast value. A 28M spot check gives
2.9e-7. Reproduce with `inst/parity_official.R` (developer machine only;
end users never need Python).

## Not yet supported

Categorical covariates (target encoding), batching several independent
series in one call (one series, possibly multivariate, per call).
