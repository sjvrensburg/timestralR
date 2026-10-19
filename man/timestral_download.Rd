\name{timestral_download}
\alias{timestral_download}
\title{Download a Timestral checkpoint}
\usage{
timestral_download(size = c("5M", "28M", "123M", "209M"),
  cache_dir = tools::R_user_dir("timestralR", "cache"))
}
\arguments{
\item{size}{one of "5M", "28M", "123M", "209M"}
\item{cache_dir}{where to keep the files}
}
\value{The checkpoint directory (invisibly containing config and weights).}
\description{Downloads \code{config.json} and \code{model.safetensors} from the
official Mistral AI Hugging Face repository (Modified MIT weights licence),
cached per user.}
