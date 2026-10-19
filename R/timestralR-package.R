#' timestralR: Zero-Shot Probabilistic Forecasting with Timestral
#'
#' Runs Mistral AI's Timestral time-series foundation models from R
#' without Python or PyTorch. [timestral_load()] downloads the official
#' `safetensors` checkpoints once (or reads a local checkpoint directory)
#' and [timestral_forecast()] produces 21-quantile probabilistic
#' forecasts, jointly across the variates of a series and independently
#' across a batch of series. Inputs may be plain vectors/matrices,
#' `ts`/`mts`, `zoo`/`xts` (time index preserved) or `tsibble`; numeric
#' and categorical covariates are supported. See [plot.timestral_forecast()]
#' for fan charts and [as_forecast_df()] for a tidy long-format output;
#' [autoplot.timestral_forecast()], [as_tsibble.timestral_forecast()],
#' [as_fable.timestral_forecast()] and [TIMESTRAL()] connect forecasts to
#' ggplot2 and the tidyverts.
#'
#' @useDynLib timestralR, .registration = TRUE
#' @importFrom Rcpp evalCpp
"_PACKAGE"
