\name{timestral_quantile}
\alias{timestral_quantile}
\title{Forecast at one quantile level}
\usage{
timestral_quantile(fc, level)
}
\arguments{
\item{fc}{a \code{timestral_forecast}}
\item{level}{one of \code{timestral_levels()}}
}
\value{The slice of the forecast at \code{level}.}
\description{Forecast at one quantile level; errors if \code{level} is not one
of the model's quantile levels.}
