#' timestralR: Zero-Shot Probabilistic Forecasting with Timestral
#'
#' Runs Mistral AI's Timestral time-series foundation models from R
#' without Python or PyTorch. [timestral_load()] downloads the official
#' `safetensors` checkpoints once (or reads a local checkpoint directory)
#' and [timestral_forecast()] produces 21-quantile probabilistic
#' forecasts, jointly across the variates of a series and independently
#' across a batch of series. Inputs may be plain vectors/matrices,
#' `ts`/`mts`, or `zoo`/`xts` (time index preserved); numeric and
#' categorical covariates are supported. See [plot.timestral_forecast()]
#' for fan charts and [as_forecast_df()] for a tidy long-format output.
#'
#' @docType package
#' @name timestralR-package
#' @aliases timestralR
NULL
