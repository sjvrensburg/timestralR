\name{timestral_download}
\alias{timestral_download}
\title{Download a Timestral checkpoint}
\usage{
timestral_download(size = c("5M", "28M", "123M", "209M"),
  revision = "main",
  cache_dir = tools::R_user_dir("timestralR", "cache"),
  local_files_only = FALSE)
}
\arguments{
\item{size}{one of "5M", "28M", "123M", "209M"}
\item{revision}{Hub revision (branch, tag or commit); default "main"}
\item{cache_dir}{where to keep the files}
\item{local_files_only}{only read from the cache, never download}
}
\value{The checkpoint directory.}
\description{Downloads \code{config.json} and \code{model.safetensors} from the
official Mistral AI Hugging Face repository (Modified MIT weights licence),
cached per user. Set \code{HF_TOKEN} for authenticated downloads.}
\examples{
\dontrun{
timestral_download("5M")
}
}
