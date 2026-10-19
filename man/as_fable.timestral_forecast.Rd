\name{as_fable.timestral_forecast}
\alias{as_fable.timestral_forecast}
\title{Convert a Timestral forecast to a fable forecast table}
\usage{
\method{as_fable}{timestral_forecast}(x, ...)
}
\arguments{
\item{x}{a \code{timestral_forecast} created from indexed input (\code{ts},
\code{zoo}, \code{xts} or \code{tsibble}), so a future time index is
available}
\item{...}{ignored}
}
\value{A \code{fbl_ts} forecast table.}
\description{
Registered as an S3 method for \code{fabletools::as_fable()} when
fabletools is available; call it as \code{fabletools::as_fable(fc)}.
The distribution column is the 21-quantile
\code{distributional::dist_quantile()} distribution from the forecast, so
fable accuracy measures and interval extraction apply. A univariate forecast
uses the variate name as the response; a fable holds a single distribution
column, so a multivariate forecast becomes a \code{variate} key with the
response in a \code{value} column. Forecasts from \code{ts} input use the
same index class (\code{yearmonth}, \code{yearquarter}, ...) as
\code{tsibble::as_tsibble()} gives the \code{ts}, so they join with and score
against the observed data. See
\code{\link{TIMESTRAL}} for using Timestral directly as a fable model.
}
\examples{
\dontrun{
m <- timestral_load("5M")
fc <- timestral_forecast(m, AirPassengers, h = 24)
fabletools::as_fable(fc)
}
}
