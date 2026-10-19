\name{timestral_median}
\alias{timestral_median}
\title{Median forecast}
\usage{
timestral_median(fc)
}
\arguments{
\item{fc}{a \code{timestral_forecast}}
}
\value{The 0.5-quantile slice. For a single series forecast from a
\code{ts}, a \code{ts}/\code{mts} starting after the context; otherwise the
same slice of the quantiles array as before.}
\description{Median forecast.}
\examples{
\dontrun{
m <- timestral_load("5M")
fc <- timestral_forecast(m, AirPassengers, h = 24)
timestral_median(fc)   # a ts, starting Jan 1961
}
}
