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
  hdr_raw <- readBin(con, "raw", n = hdr_len)
  if (length(hdr_raw) != hdr_len)
    stop("truncated safetensors header in '", path, "'")
  header <- jsonlite::fromJSON(rawToChar(hdr_raw),
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
    if (length(vals) != n)
      stop("truncated tensor ", nm, " in '", path, "': expected ", n,
           " values, read ", length(vals))
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
  .validate_config(jsonlite::fromJSON(path))
}

#' Quantile levels returned by every Timestral forecast
#'
#' @return Numeric vector of 21 levels from 0.01 to 0.99.
#' @examples
#' timestral_levels()
#' @export
timestral_levels <- function() {
  c(0.01, 0.05, seq(0.1, 0.9, by = 0.05), 0.95, 0.99)
}

.sizes <- c("5M", "28M", "123M", "209M")

#' Download a Timestral checkpoint
#'
#' Downloads `config.json` and `model.safetensors` from the official
#' Mistral AI Hugging Face repository (Modified MIT weights licence),
#' cached per user. Set `HF_TOKEN` for authenticated downloads.
#'
#' @param size one of "5M", "28M", "123M", "209M"
#' @param revision Hub revision (branch, tag or commit); default "main"
#' @param cache_dir where to keep the files
#' @param local_files_only only read from the cache, never download
#' @return The checkpoint directory.
#' @examples
#' \dontrun{
#' timestral_download("5M")
#' }
#' @export
timestral_download <- function(size = c("5M", "28M", "123M", "209M"),
                               revision = "main",
                               cache_dir = tools::R_user_dir("timestralR", "cache"),
                               local_files_only = FALSE) {
  size <- match.arg(size)
  dest_dir <- file.path(cache_dir, tolower(size), revision)
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  for (f in c("config.json", "model.safetensors")) {
    dest <- file.path(dest_dir, f)
    if (!file.exists(dest)) {
      if (local_files_only)
        stop("checkpoint not in cache and local_files_only = TRUE: ", dest)
      url <- sprintf("https://huggingface.co/mistralai/Timestral-%s/resolve/%s/%s",
                     size, revision, f)
      old <- options(timeout = max(600, getOption("timeout")))
      on.exit(options(old), add = TRUE)
      tmp <- paste0(dest, ".part")
      token <- Sys.getenv("HF_TOKEN", "")
      headers <- if (nzchar(token)) c(Authorization = paste("Bearer", token)) else NULL
      status <- utils::download.file(url, tmp, mode = "wb", quiet = TRUE,
                                     headers = headers)
      if (status != 0 || !file.rename(tmp, dest)) stop("download failed: ", url)
    }
  }
  dest_dir
}

#' Load a Timestral model
#'
#' Load a Timestral model from the official Mistral AI checkpoint.
#'
#' @param model a size ("5M", "28M", "123M", "209M") to download/cache, or
#'   a local directory holding `config.json` and `model.safetensors`
#' @param max_context context cap in steps; NULL (default) uses the trained
#'   context length from the checkpoint, as in the Python package
#' @param per_shot_horizon_patches patches decoded per block (capped at the
#'   trained block size)
#' @param quantile_real_cap_k clamp forecasts to the observed context range
#'   plus/minus this multiple of the last patch scale; 0 disables
#' @param extreme_value_threshold fed-back medians at or above this magnitude
#'   become 0
#' @param revision Hub revision when `model` is a size
#' @param cache_dir checkpoint cache directory when `model` is a size
#' @param local_files_only only read from the cache, never download
#' @param backend "native" (default) uses the compiled RcppEigen
#'   implementation of the transformer block; "R" uses the pure-R reference
#'   implementation. Results are identical up to floating-point order.
#' @return A `timestral_model` object.
#' @examples
#' \dontrun{
#' m <- timestral_load("5M")
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' plot(fc)
#' }
#' @export
timestral_load <- function(model = "5M", max_context = NULL,
                           per_shot_horizon_patches = 32L,
                           quantile_real_cap_k = 1e4,
                           extreme_value_threshold = 1e30,
                           revision = "main",
                           cache_dir = tools::R_user_dir("timestralR", "cache"),
                           local_files_only = FALSE,
                           backend = c("native", "R")) {
  if (per_shot_horizon_patches < 1)
    stop("per_shot_horizon_patches must be >= 1, got ", per_shot_horizon_patches)
  dir <- if (model %in% .sizes) {
    timestral_download(model, revision = revision, cache_dir = cache_dir,
                       local_files_only = local_files_only)
  } else if (dir.exists(model)) {
    normalizePath(model)
  } else {
    stop("'model' must be one of ", paste(.sizes, collapse = ", "),
         " or a directory with config.json and model.safetensors")
  }
  backend <- match.arg(backend)
  cfg <- .read_config(file.path(dir, "config.json"))
  wts <- .read_f32_tensors(file.path(dir, "model.safetensors"))
  .check_weights(wts, cfg)
  session <- if (backend == "native") {
    .ts_load(wts, cfg[c("patch_size", "d_model", "num_heads", "mlp_hidden_dim",
                        "block_size", "block_schedule", "quantile_knots",
                        "rope_theta", "rope_partial_factor", "norm_eps",
                        "minimum_scale", "quantile_width_factor", "use_asinh",
                        "attend_missing_patches", "covariate_enabled")])
  }
  structure(list(weights = wts, config = cfg, dir = dir, backend = backend,
                 session = session,
                 max_context = if (is.null(max_context)) cfg$context_length else as.integer(max_context),
                 per_shot_horizon_patches = min(as.integer(per_shot_horizon_patches), cfg$block_size),
                 quantile_real_cap_k = quantile_real_cap_k,
                 extreme_value_threshold = extreme_value_threshold),
            class = "timestral_model")
}

#' @export
print.timestral_model <- function(x, ...) {
  cat("<timestral_model>", basename(x$dir), " d_model =", x$config$d_model,
      " layers =", length(x$config$block_schedule),
      " max_context =", x$max_context, "\n")
  invisible(x)
}
