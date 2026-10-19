\name{timestral_load}
\alias{timestral_load}
\title{Load a Timestral model}
\usage{
timestral_load(model = "5M", max_context = 4096L)
}
\arguments{
\item{model}{a size ("5M", "28M", "123M", "209M") to download/cache, or a
local directory holding \code{config.json} and \code{model.safetensors}}
\item{max_context}{context cap in steps (models accept up to 32768;
attention cost grows quadratically, so the default stays at 4096)}
}
\value{A \code{timestral_model} object.}
\description{Load a Timestral model from the official Mistral AI checkpoint.}
