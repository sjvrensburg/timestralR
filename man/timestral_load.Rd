\name{timestral_load}
\alias{timestral_load}
\title{Load a Timestral model}
\usage{
timestral_load(model = "5M", max_context = NULL,
  per_shot_horizon_patches = 32L, quantile_real_cap_k = 1e4,
  extreme_value_threshold = 1e30, revision = "main",
  cache_dir = tools::R_user_dir("timestralR", "cache"),
  local_files_only = FALSE)
}
\arguments{
\item{model}{a size ("5M", "28M", "123M", "209M") to download/cache, or a
local directory holding \code{config.json} and \code{model.safetensors}}
\item{max_context}{context cap in steps; NULL (default) uses the trained
context length from the checkpoint, as in the Python package}
\item{per_shot_horizon_patches}{patches decoded per block (capped at the
trained block size)}
\item{quantile_real_cap_k}{clamp forecasts to the observed context range
plus/minus this multiple of the last patch scale; 0 disables}
\item{extreme_value_threshold}{fed-back medians at or above this magnitude
become 0}
\item{revision}{Hub revision when \code{model} is a size}
\item{cache_dir}{checkpoint cache directory when \code{model} is a size}
\item{local_files_only}{only read from the cache, never download}
}
\value{A \code{timestral_model} object.}
\description{Load a Timestral model from the official Mistral AI checkpoint.}
