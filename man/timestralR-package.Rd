\docType{package}
\name{timestralR-package}
\alias{timestralR-package}
\alias{timestralR}
\title{timestralR: Zero-Shot Probabilistic Forecasting with Timestral}
\description{
Runs Mistral AI's Timestral time-series foundation models from R without
Python or PyTorch. \code{\link{timestral_load}} downloads the official
\code{safetensors} checkpoints once (or reads a local checkpoint directory)
and \code{\link{timestral_forecast}} produces 21-quantile probabilistic
forecasts, jointly across the variates of a series and independently across
a batch of series. Inputs may be plain vectors/matrices, \code{ts}/\code{mts},
or \code{zoo}/\code{xts} (time index preserved); numeric and categorical
covariates are supported. See \code{\link{plot.timestral_forecast}} for fan
charts and \code{\link{as_forecast_df}} for a tidy long-format output.
}
\author{
Stefan Janse van Rensburg
}
\keyword{package}
