## Regression tests for past native/config/cache/plot bugs.
## Sourced by tests/timestralR-tests.R; expects `m`, `y`, `fx`.
## regression: rope_dim must not exceed head_dim (native OOB write otherwise)
cfg_rope <- jsonlite::fromJSON(file.path(fx, "tiny", "config.json"))
cfg_rope$rope_partial_factor <- 2.0
err <- tryCatch({ timestralR:::.validate_config(cfg_rope); FALSE },
                error = function(e) grepl("rope_dim", conditionMessage(e)))
stopifnot(isTRUE(err))

## regression: truncated safetensors must error, not recycle short reads
st_src <- file.path(fx, "tiny", "model.safetensors")
st_bytes <- readBin(st_src, "raw", n = file.info(st_src)$size)
st_trunc <- file.path(tempdir(), "truncated.safetensors")
writeBin(st_bytes[seq_len(length(st_bytes) %/% 2)], st_trunc)
err <- tryCatch({ timestralR:::.read_f32_tensors(st_trunc); FALSE },
                error = function(e) grepl("truncated", conditionMessage(e)))
stopifnot(isTRUE(err))

## regression: malformed native call (short groups) must error, not read OOB
err <- tryCatch({
  timestralR:::.ts_predict_block(m$session, matrix(rnorm(64), nrow = 2),
                                 matrix(TRUE, 2, 32), 1L, NULL, NULL, NULL,
                                 groups = 1L)
  FALSE
}, error = function(e) grepl("groups", conditionMessage(e)))
stopifnot(isTRUE(err))

## pinned checkpoints: every size pinned, corrupt cache fails verification
stopifnot(identical(sort(names(timestralR:::.pinned_checkpoints)),
                    sort(timestralR:::.sizes)))
bad_cache <- file.path(tempdir(), "badcache")
bad_dir <- file.path(bad_cache, "5m",
                     timestralR:::.pinned_checkpoints[["5M"]]$revision)
dir.create(bad_dir, recursive = TRUE, showWarnings = FALSE)
writeLines("{}", file.path(bad_dir, "config.json"))
err <- tryCatch({
  timestral_download("5M", cache_dir = bad_cache, local_files_only = TRUE)
  FALSE
}, error = function(e) grepl("SHA-256", conditionMessage(e)))
stopifnot(isTRUE(err))

## regression: an all-missing history is rejected at forecast time
## (it used to return plausible-looking quantiles, and only plot()
## complained); the error text still names the cause
.expect_error <- function(expr, pattern) {
  err <- tryCatch({ force(expr); NULL }, error = function(e) e)
  stopifnot(!is.null(err), grepl(pattern, conditionMessage(err)))
  invisible(TRUE)
}
.expect_error(timestral_forecast(m, rep(NA_real_, length(y)), h = 6),
              "no finite observed")

## regression (audit 1): empty / zero-dimension y errors, never aborts
.expect_error(timestral_forecast(m, numeric(0), h = 4), "at least one time step")
.expect_error(timestral_forecast(m, NULL, h = 4), "at least one time step")
.expect_error(timestral_forecast(m, list(), h = 4), "at least one series")
.expect_error(timestral_forecast(m, matrix(numeric(0), nrow = 0, ncol = 2), h = 4),
              "at least one time step")
.expect_error(timestral_forecast(m, matrix(numeric(0), nrow = 4, ncol = 0), h = 4),
              "at least one variate")
.expect_error(timestral_forecast(m, array(numeric(0), dim = c(0, 1, 1)), h = 4),
              "at least one time step")
.expect_error(timestral_forecast(m_R, numeric(0), h = 4), "at least one time step")
.expect_error(timestral_forecast(m, rep(Inf, 48), h = 4), "no finite observed")
.expect_error(timestral_forecast(m, list(y, rep(NA_real_, length(y))), h = 4),
              "no finite observed")
.expect_error(timestral_forecast(m, y, h = 4,
                                 observation_mask = rep(FALSE, length(y))),
              "no finite observed")

## regression (audit 1): non-numeric y / covariates error, not silent coercion
.expect_error(timestral_forecast(m, c("a", "b", "c"), h = 4), "must be numeric")
.expect_error(timestral_forecast(m, factor(c("a", "b")), h = 4), "must be numeric")
.expect_error(timestral_forecast(m, complex(4, 1, 1), h = 4), "must be numeric")
.expect_error(timestral_forecast(m, as.Date("2020-01-01") + 0:9, h = 4),
              "must be numeric")
.expect_error(timestral_forecast(m, data.frame(a = letters[1:10]), h = 4),
              "must be numeric")
.expect_error(timestral_forecast(m, list(y, letters[1:10]), h = 4), "must be numeric")
.expect_error(timestral_forecast(m, y, h = 4,
                                 past_covariates = matrix(letters[1:96], ncol = 1)),
              "must be numeric")

## regression (audit 1): horizon must be finite, whole, and bounded
.expect_error(timestral_forecast(m, y, h = -1), "must be >= 1")
.expect_error(timestral_forecast(m, y, h = NA), ">= 1")
.expect_error(timestral_forecast(m, y, h = Inf), ">= 1")
.expect_error(timestral_forecast(m, y, h = 1e9), "<= 10000")
.expect_error(timestral_forecast(m, y, h = 2.5), "whole number")
.expect_error(timestral_forecast(m, y, h = "4"), ">= 1")
.expect_error(timestral_forecast(m, y, h = c(1, 2)), ">= 1")

## regression (audit 1+2): the native entry points validate their inputs
.expect_error(
  timestralR:::.ts_predict_block(m$session, matrix(numeric(0), nrow = 0, ncol = 32),
                                 matrix(logical(0), nrow = 0, ncol = 32),
                                 1L, NULL, NULL, NULL, NULL),
  "R >= 1")
.expect_error(
  timestralR:::.ts_predict_block(m$session, matrix(rnorm(8), nrow = 1),
                                 matrix(TRUE, 1, 8), 1L, NULL, NULL, NULL, NULL),
  "patch_size")
.expect_error(
  timestralR:::.ts_predict_block(m$session, matrix(rnorm(24), nrow = 1),
                                 matrix(TRUE, 1, 24), 1L, NULL, NULL, NULL, NULL),
  "multiple of patch_size")
obs_na <- matrix(TRUE, 1, 32); obs_na[1] <- NA
.expect_error(
  timestralR:::.ts_predict_block(m$session, matrix(rnorm(32), nrow = 1),
                                 obs_na, 1L, NULL, NULL, NULL, NULL),
  "must not contain NA")
.expect_error(
  timestralR:::.ts_predict_block(m$session, matrix(rnorm(32), nrow = 1),
                                 matrix(TRUE, 1, 32), 1L, NA_integer_,
                                 NULL, NULL, NULL),
  "must not contain NA")
.expect_error(
  timestralR:::.ts_predict_block(m$session, matrix(rnorm(32), nrow = 1),
                                 matrix(TRUE, 1, 32), 0L, NULL, NULL, NULL, NULL),
  "horizon_patches")
.expect_error(
  timestralR:::.predict_block(m_R, matrix(numeric(0), nrow = 0, ncol = 32),
                              matrix(logical(0), nrow = 0, ncol = 32), 1L),
  "R >= 1")

## regression (audit 2): .ts_load re-validates the config natively
.native_cfg <- function() m$config[c("patch_size", "d_model", "num_heads",
  "mlp_hidden_dim", "block_size", "block_schedule", "quantile_knots",
  "rope_theta", "rope_partial_factor", "norm_eps", "minimum_scale",
  "quantile_width_factor", "use_asinh", "attend_missing_patches",
  "covariate_enabled")]
bad <- .native_cfg(); bad$num_heads <- 0L
.expect_error(timestralR:::.ts_load(m$weights, bad), "num_heads")
bad <- .native_cfg(); bad$patch_size <- 0L
.expect_error(timestralR:::.ts_load(m$weights, bad), "patch_size")
bad <- .native_cfg(); bad$d_model <- 33L
.expect_error(timestralR:::.ts_load(m$weights, bad), "not divisible")
bad <- .native_cfg(); bad$quantile_knots <- bad$quantile_knots[-11]
.expect_error(timestralR:::.ts_load(m$weights, bad), "must contain 0.5")
bad <- .native_cfg(); bad$block_schedule <- "bogus"
.expect_error(timestralR:::.ts_load(m$weights, bad), "unknown layer kind")
bad <- .native_cfg(); bad$num_heads <- NULL
.expect_error(timestralR:::.ts_load(m$weights, bad), "missing required field")
bad <- .native_cfg(); bad$patch_size <- 0L
.expect_error(timestralR:::.validate_config(
  modifyList(jsonlite::fromJSON(file.path(fx, "tiny", "config.json")), bad)),
  "patch_size")

## regression (audit 3): parallel downloads get distinct .part files
dest <- file.path(tempdir(), "ckpt", "model.safetensors")
tmps <- replicate(4, timestralR:::.download_tmp(dest))
stopifnot(length(unique(tmps)) == 4L,
          all(dirname(tmps) == dirname(dest)),
          all(endsWith(tmps, ".part")),
          !any(tmps == paste0(dest, ".part")))
