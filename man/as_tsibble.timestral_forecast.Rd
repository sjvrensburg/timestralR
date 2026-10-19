\name{as_tsibble.timestral_forecast}
\alias{as_tsibble.timestral_forecast}
\title{Convert a Timestral forecast to a tsibble}
\usage{
\method{as_tsibble}{timestral_forecast}(x, ...)
}
\arguments{
\item{x}{a \code{timestral_forecast} created from indexed input (\code{ts},
\code{zoo}, \code{xts} or \code{tsibble}), so a future time index is
available}
\item{...}{ignored}
}
\value{A \code{tsibble} (\code{tbl_ts}).}
\description{
Registered as an S3 method for \code{tsibble::as_tsibble()} when the
tsibble package is available; call it as \code{tsibble::as_tsibble(fc)}.
The result is a tsibble over the forecast horizon with one row per
series/variate/step and one column per quantile level (as in
\code{\link{as_forecast_df}}). Forecasts can also be created directly
from a tsibble: in \code{\link{timestral_forecast}}, tsibble keys become
the batch and measured variables the jointly forecast variates.
}
\examples{
\dontrun{
m <- timestral_load("5M")
fc <- timestral_forecast(m, AirPassengers, h = 24)
tsibble::as_tsibble(fc)
}
}
