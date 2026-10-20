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

## Pinned upstream commits and SHA-256 of the two checkpoint files, per
## size (Hugging Face LFS oid for model.safetensors). Downloads at the
## pinned revision are verified against these; a cached file that fails
## verification is re-downloaded (or an error when local_files_only).
.pinned_checkpoints <- list(
  "5M" = list(
    revision = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    sha256 = c("config.json" = "b5f27f61513f8fd1a3aa996f3720d3679cb40bce5bcb08eb3dc320ce726e7c2f",
               "model.safetensors" = "7ace108c4987e7f2a99a75a445240019d1d31c2117139d1bd05ffdfd25aca738")),
  "28M" = list(
    revision = "49b26409ac87b5935539a5d8ffe7f397b720db47",
    sha256 = c("config.json" = "0cdf44b8ea981191de53078e0a5ecbbd437046bd939a546d5dbe17868f71281f",
               "model.safetensors" = "dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d")),
  "123M" = list(
    revision = "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    sha256 = c("config.json" = "7bb4c042fb2d2ed6469b6758ba380dfeda3ab32df47690a87aec1527bdae79bf",
               "model.safetensors" = "599fafa73cadbbc800b5a346ca900fa741ee3e5fd22e3b4a4430e151e74f4bc2")),
  "209M" = list(
    revision = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
    sha256 = c("config.json" = "1764139bb6d37ee5ff543f470495c086cfa6c373f7134fc61c7e3d844538bfd8",
               "model.safetensors" = "1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa")))

.sha256_file <- function(path)
  digest::digest(file = path, algo = "sha256", serialize = FALSE)

#' Download a Timestral checkpoint
#'
#' Downloads `config.json` and `model.safetensors` from the official
#' Mistral AI Hugging Face repository (Modified MIT weights licence),
#' cached per user. Set `HF_TOKEN` for authenticated downloads.
#'
#' @param size one of "5M", "28M", "123M", "209M"
#' @param revision Hub revision (branch, tag or commit); default `NULL`
#'   uses the pinned upstream commit for `size`, whose files are verified
#'   against recorded SHA-256 hashes. Other revisions are not verified.
#' @param cache_dir where to keep the files
#' @param local_files_only only read from the cache, never download
#' @return The checkpoint directory.
#' @examples
#' \dontrun{
#' timestral_download("5M")
#' }
#' @export
timestral_download <- function(size = c("5M", "28M", "123M", "209M"),
                               revision = NULL,
                               cache_dir = tools::R_user_dir("timestralR", "cache"),
                               local_files_only = FALSE) {
  size <- match.arg(size)
  pin <- .pinned_checkpoints[[size]]
  if (is.null(revision)) revision <- pin$revision
  pinned <- identical(revision, pin$revision)
  dest_dir <- file.path(cache_dir, tolower(size), revision)
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  for (f in c("config.json", "model.safetensors")) {
    dest <- file.path(dest_dir, f)
    expected <- if (pinned) unname(pin$sha256[f]) else NULL
    if (file.exists(dest) && !is.null(expected) &&
        !identical(.sha256_file(dest), expected)) {
      if (local_files_only)
        stop("cached checkpoint file fails SHA-256 verification: ", dest)
      unlink(dest)
    }
    if (!file.exists(dest)) {
      if (local_files_only)
        stop("checkpoint not in cache and local_files_only = TRUE: ", dest)
      url <- sprintf("https://huggingface.co/mistralai/Timestral-%s/resolve/%s/%s",
                     size, revision, f)
      old <- options(timeout = max(600, getOption("timeout")))
      on.exit(options(old), add = TRUE)
      tmp <- paste0(dest, ".part")
      unlink(tmp)
      token <- Sys.getenv("HF_TOKEN", "")
      headers <- if (nzchar(token)) c(Authorization = paste("Bearer", token)) else NULL
      status <- tryCatch(
        utils::download.file(url, tmp, mode = "wb", quiet = TRUE,
                             headers = headers),
        error = function(e) { unlink(tmp); stop(e) })
      if (status != 0 || !file.exists(tmp)) {
        unlink(tmp)
        stop("download failed: ", url)
      }
      if (!is.null(expected) && !identical(.sha256_file(tmp), expected)) {
        unlink(tmp)
        stop("downloaded checkpoint file fails SHA-256 verification: ", url)
      }
      if (!file.rename(tmp, dest)) {
        unlink(tmp)
        stop("download failed: ", url)
      }
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
#' @param revision Hub revision when `model` is a size; default `NULL`
#'   uses the pinned, hash-verified upstream commit for the size
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
                           revision = NULL,
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
