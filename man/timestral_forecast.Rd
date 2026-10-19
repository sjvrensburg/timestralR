\name{timestral_forecast}
\alias{timestral_forecast}
\title{Forecast one (possibly multivariate) series}
\usage{
timestral_forecast(model, y, h, past_covariates = NULL,
  known_covariates = NULL, symmetric = TRUE, clip_nonnegative = TRUE)
}
\arguments{
\item{model}{a \code{timestral_model}}
\item{y}{numeric vector, \code{ts}, or matrix/data frame with time in rows
and one column per target variate (NA = missing). Variates are forecast jointly.}
\item{h}{horizon in steps}
\item{past_covariates}{matrix with time in rows (T rows), known over the context only}
\item{known_covariates}{matrix with T + h rows, known over context and horizon}
\item{symmetric}{average with the forecast of the mirrored series (package default)}
\item{clip_nonnegative}{clip at 0 when every observed target value is >= 0}
}
\value{A \code{timestral_forecast}: \code{quantiles} array
(variate, horizon, level) plus metadata.}
\description{Zero-shot probabilistic forecast with 21 quantiles per step.}
