\name{timestral_forecast}
\alias{timestral_forecast}
\title{Forecast one or more series}
\usage{
timestral_forecast(model, y, h, observation_mask = NULL,
  past_covariates = NULL, known_covariates = NULL,
  categorical_past_covariates = NULL, categorical_known_covariates = NULL,
  symmetric = TRUE, clip_nonnegative = TRUE)
}
\arguments{
\item{model}{a \code{timestral_model}}
\item{y}{numeric vector, \code{ts}, matrix/data frame with time in rows and one
column per target variate, a 3-D array (time, variate, series), or a list of
per-series vectors/matrices (NA = missing). Variates of a series are forecast
jointly; series in a batch are forecast independently in one call.}
\item{h}{horizon in steps}
\item{observation_mask}{optional logical shaped like \code{y}; FALSE marks
additional missing values}
\item{past_covariates}{known over the context only (T time rows, or the batch
forms described for \code{y})}
\item{known_covariates}{known over context and horizon (T + h time rows)}
\item{categorical_past_covariates}{category codes known over the context only;
target-encoded per series (used as-is when a series has several target variates)}
\item{categorical_known_covariates}{category codes over context and horizon}
\item{symmetric}{average with the forecast of the mirrored series (package default)}
\item{clip_nonnegative}{clip at 0 when every observed target value is >= 0}
}
\value{A \code{timestral_forecast}: for one series, \code{quantiles} is a
(variate, horizon, level) array; for a batch, (series, variate, horizon, level).}
\description{Zero-shot probabilistic forecast with 21 quantiles per step.}
