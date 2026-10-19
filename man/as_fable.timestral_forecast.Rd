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
Each variate becomes a response whose distribution is the 21-quantile
\code{distributional::dist_quantile()} distribution from the forecast,
so fable accuracy measures and interval extraction apply. See
\code{\link{TIMESTRAL}} for using Timestral directly as a fable model.
}
\examples{
\dontrun{
m <- timestral_load("5M")
fc <- timestral_forecast(m, AirPassengers, h = 24)
fabletools::as_fable(fc)
}
}
