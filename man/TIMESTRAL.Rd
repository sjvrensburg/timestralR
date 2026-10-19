\name{TIMESTRAL}
\alias{TIMESTRAL}
\title{Timestral as a fable model}
\usage{
TIMESTRAL(formula, model, symmetric = TRUE, clip_nonnegative = TRUE, ...)
}
\arguments{
\item{formula}{the response formula, as in other fable models, e.g.
\code{TIMESTRAL(value)}}
\item{model}{a \code{timestral_model} from \code{\link{timestral_load}};
loaded once and reused for every key and forecast. Fitted models store only
the checkpoint directory and load settings, not the weights, so a mable stays
small when saved and is reloaded from disk when forecast in a new R session or
a parallel worker.}
\item{symmetric, clip_nonnegative}{as in \code{\link{timestral_forecast}}}
\item{...}{passed to the model definition (currently unused)}
}
\value{A fable model definition, for use inside \code{fabletools::model()}.}
\description{
Defines a Timestral zero-shot forecasting model for use with
\code{fabletools::model()} on a \code{tsibble}. The response is univariate
(use \code{\link{timestral_forecast}} directly to jointly forecast multiple
variates); keys in the tsibble are handled by fable as usual, with the model
trained (context captured) independently per key. Requires the fabletools
package (and fable for the usual workflow); forecasts are reported as
21-quantile distributions.
}
\examples{
\dontrun{
library(fable)
m <- timestral_load("5M")
fit <- as_tsibble(AirPassengers) |>
  model(timestral = TIMESTRAL(value, model = m))
fit |> forecast(h = 24)
}
}
