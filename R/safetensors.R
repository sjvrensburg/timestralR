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

#' Quantile levels of the official Timestral checkpoints
#'
#' The levels every official checkpoint forecasts. A forecast always
#' carries the levels of the checkpoint that made it in `fc$levels`,
#' which can differ for other (for example fine-tuned) checkpoints.
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

## Unique in-progress download file for `dest`, in the same directory so
## the final rename stays on one filesystem. The old fixed
## `paste0(dest, ".part")` name made parallel downloads (the usual
## bootstrap/parLapply pattern) clobber each other's partial files.
.download_tmp <- function(dest) {
  tempfile(pattern = paste0(basename(dest), "-"), tmpdir = dirname(dest),
           fileext = ".part")
}

## Hub repository IDs are "owner/name"; the official checkpoints are
## mistralai/Timestral-<size> and keep their pins when named either way.
.is_repo_id <- function(x) {
  grepl("^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$", x)
}

.official_size <- function(repo) {
  i <- match(tolower(repo), tolower(paste0("mistralai/Timestral-", .sizes)))
  if (is.na(i)) NULL else .sizes[i]
}

.is_commit <- function(x) grepl("^[0-9a-f]{40}$", x)

.hub_headers <- function(token) {
  if (nzchar(token)) c(Authorization = paste("Bearer", token)) else NULL
}

## Branch and tag names move, so a non-commit revision is resolved to its
## commit on the Hub (and recorded under refs/ for offline reuse) rather
## than cached under its name, where it would go stale.
.resolve_revision <- function(repo, revision, repo_dir, token,
                              local_files_only) {
  if (.is_commit(revision)) return(revision)
  ref <- file.path(repo_dir, "refs", utils::URLencode(revision, reserved = TRUE))
  if (!local_files_only) {
    url <- sprintf("https://huggingface.co/api/models/%s/revision/%s", repo,
                   utils::URLencode(revision, reserved = TRUE))
    tmp <- tempfile(fileext = ".json")
    on.exit(unlink(tmp), add = TRUE)
    status <- tryCatch(
      suppressWarnings(utils::download.file(url, tmp, quiet = TRUE,
                                            headers = .hub_headers(token))),
      error = function(e) 1L)
    sha <- if (status == 0 && file.exists(tmp))
      tryCatch(jsonlite::fromJSON(tmp)$sha, error = function(e) NULL)
    if (!is.character(sha) || length(sha) != 1L || !.is_commit(sha))
      stop("could not resolve revision '", revision, "' of '", repo,
           "' on the Hugging Face Hub; check the repository ID and the ",
           "revision, and set `token` (or HF_TOKEN) for private or gated ",
           "repositories")
    dir.create(dirname(ref), recursive = TRUE, showWarnings = FALSE)
    tmp_ref <- .download_tmp(ref)
    writeLines(sha, tmp_ref)
    if (!file.rename(tmp_ref, ref)) unlink(tmp_ref)
    return(sha)
  }
  sha <- if (file.exists(ref)) readLines(ref, n = 1L, warn = FALSE)
  if (length(sha) != 1L || !.is_commit(sha))
    stop("revision '", revision, "' of '", repo, "' is not in the cache ",
         "and local_files_only = TRUE; pass a commit hash or download once ",
         "online")
  sha
}

## Download `url` to `dest` through a unique .part file, verifying the
## SHA-256 when `expected` is given.
.hub_fetch <- function(url, dest, expected, token) {
  old <- options(timeout = max(600, getOption("timeout")))
  on.exit(options(old), add = TRUE)
  tmp <- .download_tmp(dest)
  unlink(tmp)
  status <- tryCatch(
    utils::download.file(url, tmp, mode = "wb", quiet = TRUE,
                         headers = .hub_headers(token)),
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
    ## Another process may have won the race and published dest
    ## while we downloaded; accept it if it is the file we wanted.
    if (file.exists(dest) &&
        (is.null(expected) || identical(.sha256_file(dest), expected))) {
      unlink(tmp)
    } else {
      unlink(tmp)
      stop("download failed: ", url)
    }
  }
  invisible(dest)
}

#' Download a Timestral checkpoint
#'
#' Downloads `config.json` and `model.safetensors` from a Hugging Face
#' Hub repository, cached per user. A size names the official Mistral AI
#' repository (Modified MIT weights licence); any other `"owner/name"`
#' repository holding a Timestral checkpoint in the same format (for
#' example a fine-tuned model) can be given by its ID.
#'
#' @param model one of "5M", "28M", "123M", "209M", or a Hub repository
#'   ID such as `"mistralai/Timestral-5M"`. The official repositories
#'   are pinned whichever way they are named.
#' @param revision Hub revision (branch, tag or commit). The default
#'   `NULL` uses the pinned upstream commit for an official size, whose
#'   files are verified against recorded SHA-256 hashes, and `"main"` for
#'   any other repository. Branch and tag names are resolved to their
#'   current commit on the Hub (the resolution is cached for offline use);
#'   files at revisions other than the pinned ones are not hash-verified.
#' @param cache_dir where to keep the files
#' @param local_files_only only read from the cache, never download
#' @param token Hugging Face access token for private or gated
#'   repositories; defaults to the `HF_TOKEN` environment variable
#' @return The checkpoint directory.
#' @examples
#' \dontrun{
#' timestral_download("5M")
#' timestral_download("mistralai/Timestral-5M")   # the same, pinned
#' }
#' @export
timestral_download <- function(model = "5M", revision = NULL,
                               cache_dir = tools::R_user_dir("timestralR", "cache"),
                               local_files_only = FALSE,
                               token = Sys.getenv("HF_TOKEN", "")) {
  if (!is.character(model) || length(model) != 1L || is.na(model))
    stop("'model' must be a single string")
  size <- if (model %in% .sizes) model else if (.is_repo_id(model)) .official_size(model)
  if (is.null(size) && !.is_repo_id(model))
    stop("'model' must be one of ", paste(.sizes, collapse = ", "),
         " or a Hugging Face repository ID (\"owner/name\")")
  if (!is.null(revision) &&
      (!is.character(revision) || length(revision) != 1L || is.na(revision) ||
       !nzchar(revision) || revision %in% c(".", "..")))
    stop("'revision' must be a branch, tag or commit name")
  if (is.null(token) || is.na(token)) token <- ""
  if (!is.null(size)) {
    pin <- .pinned_checkpoints[[size]]
    repo <- paste0("mistralai/Timestral-", size)
    repo_dir <- file.path(cache_dir, tolower(size))
    if (is.null(revision)) revision <- pin$revision
  } else {
    pin <- NULL
    repo <- model
    repo_dir <- file.path(cache_dir, "hub", sub("/", "--", model, fixed = TRUE))
    if (is.null(revision)) revision <- "main"
  }
  revision <- .resolve_revision(repo, revision, repo_dir, token,
                                local_files_only)
  pinned <- !is.null(pin) && identical(revision, pin$revision)
  dest_dir <- file.path(repo_dir, revision)
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
      .hub_fetch(sprintf("https://huggingface.co/%s/resolve/%s/%s",
                         repo, revision, f), dest, expected, token)
    }
  }
  dest_dir
}

#' Load a Timestral model
#'
#' Load a Timestral model from the official Mistral AI checkpoint, another
#' Hugging Face Hub repository, or a local checkpoint directory.
#'
#' @param model a size ("5M", "28M", "123M", "209M") to download/cache, a
#'   local directory holding `config.json` and `model.safetensors`, or a
#'   Hugging Face repository ID (`"owner/name"`) to download/cache. A local
#'   directory takes precedence over a repository ID of the same name.
#' @param max_context context cap in steps; NULL (default) uses the trained
#'   context length from the checkpoint, as in the Python package
#' @param per_shot_horizon_patches patches decoded per block (capped at the
#'   trained block size)
#' @param quantile_real_cap_k clamp forecasts to the observed context range
#'   plus/minus this multiple of the last patch scale; 0 disables
#' @param extreme_value_threshold fed-back medians at or above this magnitude
#'   become 0
#' @param revision Hub revision when `model` is a size or repository ID;
#'   see [timestral_download()]. The default `NULL` uses the pinned,
#'   hash-verified upstream commit for an official size and `"main"`
#'   otherwise.
#' @param cache_dir checkpoint cache directory for downloaded checkpoints
#' @param local_files_only only read from the cache, never download
#' @param token Hugging Face access token for private or gated
#'   repositories; defaults to the `HF_TOKEN` environment variable
#' @param backend "native" (default) uses the compiled RcppEigen
#'   implementation of the transformer block; "R" uses the pure-R reference
#'   implementation. Results are identical up to floating-point order.
#' @return A `timestral_model` object.
#' @examples
#' ## A tiny random-weight checkpoint ships with the package, so this runs
#' ## offline; its forecasts are meaningless. Use a size such as "5M" or
#' ## "28M" (downloaded once) for real forecasts.
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' plot(fc)
#' \dontrun{
#' m <- timestral_load("5M")
#' m <- timestral_load("mistralai/Timestral-28M")
#' }
#' @export
timestral_load <- function(model = "5M", max_context = NULL,
                           per_shot_horizon_patches = 32L,
                           quantile_real_cap_k = 1e4,
                           extreme_value_threshold = 1e30,
                           revision = NULL,
                           cache_dir = tools::R_user_dir("timestralR", "cache"),
                           local_files_only = FALSE,
                           token = Sys.getenv("HF_TOKEN", ""),
                           backend = c("native", "R")) {
  if (per_shot_horizon_patches < 1)
    stop("per_shot_horizon_patches must be >= 1, got ", per_shot_horizon_patches)
  if (!is.character(model) || length(model) != 1L || is.na(model))
    stop("'model' must be a single string")
  dir <- if (model %in% .sizes) {
    timestral_download(model, revision = revision, cache_dir = cache_dir,
                       local_files_only = local_files_only, token = token)
  } else if (dir.exists(model)) {
    normalizePath(model)
  } else if (.is_repo_id(model)) {
    timestral_download(model, revision = revision, cache_dir = cache_dir,
                       local_files_only = local_files_only, token = token)
  } else {
    stop("'model' must be one of ", paste(.sizes, collapse = ", "),
         ", a directory with config.json and model.safetensors, or a ",
         "Hugging Face repository ID (\"owner/name\")")
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
  cat("  covariates:",
      if (timestral_supports_known_covariates(x)) "past and known-future"
      else if (timestral_supports_covariates(x)) "past only" else "none",
      "\n")
  invisible(x)
}

#' Which covariates a checkpoint can condition on
#'
#' `timestral_supports_covariates()` is `TRUE` when the checkpoint was
#' trained with covariates, so [timestral_forecast()] accepts
#' `past_covariates` and `categorical_past_covariates`.
#' `timestral_supports_known_covariates()` is `TRUE` when it can also
#' condition on known-future covariates (`known_covariates` and
#' `categorical_known_covariates`), which needs a block schedule that
#' attends across variates. Both mirror the Python forecaster's
#' `supports_covariates` and `supports_known_covariates` properties;
#' every official checkpoint supports both.
#'
#' @param model a `timestral_model` from [timestral_load()]
#' @return `TRUE` or `FALSE`.
#' @examples
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' timestral_supports_covariates(m)
#' timestral_supports_known_covariates(m)
#' @export
timestral_supports_covariates <- function(model) {
  stopifnot(inherits(model, "timestral_model"))
  isTRUE(model$config$covariate_enabled)
}

#' @rdname timestral_supports_covariates
#' @export
timestral_supports_known_covariates <- function(model) {
  stopifnot(inherits(model, "timestral_model"))
  isTRUE(model$config$supports_known_covariates)
}
