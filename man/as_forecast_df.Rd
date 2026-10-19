\name{as_forecast_df}
\alias{as_forecast_df}
\title{Forecast as a long data frame}
\usage{
as_forecast_df(fc)
}
\arguments{
\item{fc}{a \code{timestral_forecast}}
}
\value{A data frame with variate, step, optional time, and one column per
quantile level.}
\description{Long data frame: variate, step, (time,) and one column per quantile level.}
