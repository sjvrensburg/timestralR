## Minimal safetensors reader (F32 tensors, which is all Timestral ships).
## Format: 8-byte little-endian header length, JSON header, raw tensor bytes.
## Implemented natively because the CRAN 'safetensors' package returns torch
## tensors, and this package must not depend on torch.

.read_f32_tensors <- function(path) {
  con <- file(path, "rb")
  on.exit(close(con))
  hdr_len_raw <- readBin(con, "raw", n = 8L)
  if (length(hdr_len_raw) != 8L) stop("not a safetensors file: '", path, "'")
  hdr_len <- sum(as.numeric(hdr_len_raw) * 256^(0:7))
  header <- jsonlite::fromJSON(rawToChar(readBin(con, "raw", n = hdr_len)),
                               simplifyVector = FALSE)
  data_start <- 8 + hdr_len
  out <- list()
  for (nm in names(header)) {
    if (nm == "__metadata__") next
    info <- header[[nm]]
    if (!identical(info$shape, NULL) && info$dtype != "F32")
      stop("unsupported dtype ", info$dtype, " for tensor ", nm,
           " (only F32 is supported)")
    shape <- as.integer(info$shape)
    n <- if (length(shape) == 0L) 1L else prod(shape)
    seek(con, data_start + info$data_offsets[[1]])
    vals <- readBin(con, numeric(), n = n, size = 4L, endian = "little")
    out[[nm]] <- if (length(shape) == 2L) {
      matrix(vals, nrow = shape[1], ncol = shape[2], byrow = TRUE)
    } else if (length(shape) <= 1L) {
      vals
    } else {
      array(vals, dim = rev(shape)) |> aperm(rev(seq_along(shape)))
    }
  }
  out
}

.read_config <- function(path) {
  cfg <- jsonlite::fromJSON(path)
  cfg$quantile_knots <- as.numeric(cfg$quantile_knots)
  cfg$block_schedule <- as.character(cfg$block_schedule)
  cfg
}

#' Quantile levels returned by every Timestral forecast
#' @export
timestral_levels <- function() {
  c(0.01, 0.05, seq(0.1, 0.9, by = 0.05), 0.95, 0.99)
}

.sizes <- c("5M", "28M", "123M", "209M")

#' Download a Timestral checkpoint from the official Mistral AI Hub repo
#'
#' Downloads `config.json` and `model.safetensors` (cached per user). The
#' weights are Mistral AI's, released under their Modified MIT licence.
#' @param size one of "5M", "28M", "123M", "209M"
#' @param cache_dir where to keep the files
#' @export
timestral_download <- function(size = c("5M", "28M", "123M", "209M"),
                               cache_dir = tools::R_user_dir("timestralR", "cache")) {
  size <- match.arg(size)
  dest_dir <- file.path(cache_dir, tolower(size))
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  for (f in c("config.json", "model.safetensors")) {
    dest <- file.path(dest_dir, f)
    if (!file.exists(dest)) {
      url <- sprintf("https://huggingface.co/mistralai/Timestral-%s/resolve/main/%s",
                     size, f)
      old <- options(timeout = max(600, getOption("timeout")))
      on.exit(options(old), add = TRUE)
      tmp <- paste0(dest, ".part")
      utils::download.file(url, tmp, mode = "wb", quiet = TRUE)
      if (!file.rename(tmp, dest)) stop("download failed: ", url)
    }
  }
  dest_dir
}

#' Load a Timestral model
#' @param model a size ("5M", ...) to download/cache, or a local directory
#'   holding `config.json` and `model.safetensors`
#' @param max_context context cap in steps (models accept up to 32768;
#'   attention cost grows quadratically, so the default stays at 4096)
#' @export
timestral_load <- function(model = "5M", max_context = 4096L) {
  dir <- if (model %in% .sizes) {
    timestral_download(model)
  } else if (dir.exists(model)) {
    normalizePath(model)
  } else {
    stop("'model' must be one of ", paste(.sizes, collapse = ", "),
         " or a directory with config.json and model.safetensors")
  }
  cfg <- .read_config(file.path(dir, "config.json"))
  wts <- .read_f32_tensors(file.path(dir, "model.safetensors"))
  structure(list(weights = wts, config = cfg, dir = dir,
                 max_context = as.integer(max_context)),
            class = "timestral_model")
}

#' @export
print.timestral_model <- function(x, ...) {
  cat("<timestral_model>", basename(x$dir), " d_model =", x$config$d_model,
      " layers =", length(x$config$block_schedule),
      " max_context =", x$max_context, "\n")
  invisible(x)
}
