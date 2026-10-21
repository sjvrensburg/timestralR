## Regression tests for past native/config/cache/plot bugs.
## Sourced by tests/timestralR-tests.R; expects `m`, `y`, `fx`, `tiny`.
## regression: rope_dim must not exceed head_dim (native OOB write otherwise)
cfg_rope <- jsonlite::fromJSON(file.path(tiny, "config.json"))
cfg_rope$rope_partial_factor <- 2.0
err <- tryCatch({ timestralR:::.validate_config(cfg_rope); FALSE },
                error = function(e) grepl("rope_dim", conditionMessage(e)))
stopifnot(isTRUE(err))

## regression: truncated safetensors must error, not recycle short reads
st_src <- file.path(tiny, "model.safetensors")
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
## the official repository ID keeps the size's pin (same corrupt cache)
err <- tryCatch({
  timestral_download("mistralai/Timestral-5M", cache_dir = bad_cache,
                     local_files_only = TRUE)
  FALSE
}, error = function(e) grepl("SHA-256", conditionMessage(e)))
stopifnot(isTRUE(err))

## Hub repository IDs, offline against a pre-filled cache: a branch name
## resolves through refs/, a commit is used directly, and the loaded
## checkpoint forecasts like the local directory it was copied from
hub_cache <- file.path(tempdir(), "hubcache")
hub_sha <- strrep("ab", 20)
hub_dir <- file.path(hub_cache, "hub", "acme--tiny-ts", hub_sha)
dir.create(hub_dir, recursive = TRUE, showWarnings = FALSE)
file.copy(file.path(tiny, c("config.json", "model.safetensors")), hub_dir)
dir.create(file.path(hub_cache, "hub", "acme--tiny-ts", "refs"),
           showWarnings = FALSE)
writeLines(hub_sha, file.path(hub_cache, "hub", "acme--tiny-ts", "refs", "main"))
stopifnot(identical(
  timestral_download("acme/tiny-ts", cache_dir = hub_cache,
                     local_files_only = TRUE), hub_dir))
stopifnot(identical(
  timestral_download("acme/tiny-ts", revision = hub_sha,
                     cache_dir = hub_cache, local_files_only = TRUE), hub_dir))
m_hub <- timestral_load("acme/tiny-ts", max_context = 4096L,
                        cache_dir = hub_cache, local_files_only = TRUE)
## the K/V cache of the warm `m` makes the two agree to ~1e-14, not bit-for-bit
stopifnot(max(abs(timestral_forecast(m_hub, y, h = 24)$quantiles -
                  timestral_forecast(m, y, h = 24)$quantiles)) < 1e-10)
err <- tryCatch({
  timestral_download("acme/tiny-ts", revision = "dev", cache_dir = hub_cache,
                     local_files_only = TRUE)
  FALSE
}, error = function(e) grepl("local_files_only", conditionMessage(e)))
stopifnot(isTRUE(err))
for (bad in list("not a repo", "a/b/c", NA_character_, c("5M", "28M"), 5)) {
  err <- tryCatch({ timestral_download(bad); FALSE },
                  error = function(e) grepl("'model' must", conditionMessage(e)))
  stopifnot(isTRUE(err))
}
err <- tryCatch({
  timestral_download("acme/tiny-ts", revision = "..", cache_dir = hub_cache,
                     local_files_only = TRUE)
  FALSE
}, error = function(e) grepl("'revision' must", conditionMessage(e)))
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
  modifyList(jsonlite::fromJSON(file.path(tiny, "config.json")), bad)),
  "patch_size")

## regression (audit 3): parallel downloads get distinct .part files
dest <- file.path(tempdir(), "ckpt", "model.safetensors")
tmps <- replicate(4, timestralR:::.download_tmp(dest))
stopifnot(length(unique(tmps)) == 4L,
          all(dirname(tmps) == dirname(dest)),
          all(endsWith(tmps, ".part")),
          !any(tmps == paste0(dest, ".part")))

## regression: forecast quantile levels come from the checkpoint's knots,
## not the official timestral_levels() (fine-tuned Hub checkpoints may
## differ); asymmetric knots refuse symmetric averaging
.knot_ckpt <- function(knots) {
  d <- file.path(tempfile("knots-"))
  dir.create(d)
  file.copy(file.path(tiny, "model.safetensors"), d)
  cfg <- jsonlite::fromJSON(file.path(tiny, "config.json"))
  cfg$quantile_knots <- knots
  jsonlite::write_json(cfg, file.path(d, "config.json"), auto_unbox = TRUE,
                       digits = NA)
  d
}
kn_sym <- timestral_levels()
kn_sym[c(1, 21)] <- c(0.02, 0.98)
m_kn <- timestral_load(.knot_ckpt(kn_sym), max_context = 4096L)
fc_kn <- timestral_forecast(m_kn, AirPassengers, h = 6)
stopifnot(isTRUE(all.equal(fc_kn$levels, kn_sym, tolerance = 1e-12)),
          identical(dimnames(fc_kn$quantiles)$level, format(kn_sym)),
          identical(as.numeric(timestral_quantile(fc_kn, 0.02)),
                    as.numeric(fc_kn$quantiles[1, , 1])))
.expect_error(timestral_quantile(fc_kn, 0.01), "quantile levels")
pdf(NULL); plot(fc_kn, levels = c(0.5, 0.8, 0.96)); dev.off()
kn_asym <- timestral_levels()
kn_asym[1] <- 0.02
m_ka <- timestral_load(.knot_ckpt(kn_asym), max_context = 4096L)
.expect_error(timestral_forecast(m_ka, AirPassengers, h = 6),
              "symmetric = FALSE")
stopifnot(identical(timestral_forecast(m_ka, AirPassengers, h = 6,
                                       symmetric = FALSE)$levels[1], 0.02))
## a literal level and timestral_levels()' seq() value pick the same knot
stopifnot(identical(timestral_quantile(fc, 0.15),
                    timestral_quantile(fc, timestral_levels()[4])))

## unequal-length batches: shorter series are left-padded with NA and end
## at the forecast origin; padding leaves each series' forecast unchanged
set.seed(11)
ua <- cumsum(rnorm(150)) + 30; ub <- cumsum(rnorm(90)) + 10
fc_u <- timestral_forecast(m, list(a = ua, b = ub), h = 20)
stopifnot(identical(dim(fc_u$quantiles)[c(1, 3)], c(2L, 20L)),
          all(is.na(fc_u$observed["b", 1, 1:60])),
          max(abs(fc_u$quantiles["b", , , ] -
                  timestral_forecast(m, ub, h = 20)$quantiles[1, , ])) < 1e-10,
          max(abs(fc_u$quantiles["a", , , ] -
                  timestral_forecast(m, ua, h = 20)$quantiles[1, , ])) < 1e-10)
## ts with a common end keep the longest's time base; otherwise none
ts_a <- ts(ua, end = c(2020, 12), frequency = 12)
ts_b <- ts(ub, end = c(2020, 12), frequency = 12)
stopifnot(identical(timestral_forecast(m, list(ts_a, ts_b), h = 6)$tsp,
                    tsp(ts_a)),
          is.null(timestral_forecast(m, list(ts_a, ts(ub, end = c(2020, 11),
                                                      frequency = 12)),
                                     h = 6)$tsp))
## covariates and masks at a series' own length are padded like it
pc_a <- matrix(sin(seq_along(ua)), ncol = 1); pc_b <- matrix(cos(seq_along(ub)), ncol = 1)
fc_pc <- timestral_forecast(m, list(ua, ub), h = 8, past_covariates = list(pc_a, pc_b))
fc_pc_full <- timestral_forecast(m, list(ua, ub), h = 8,
                                 past_covariates = list(pc_a, rbind(matrix(NA, 60, 1), pc_b)))
stopifnot(max(abs(fc_pc$quantiles - fc_pc_full$quantiles)) < 1e-10,
          max(abs(fc_pc$quantiles[2, , , ] -
                  timestral_forecast(m, ub, h = 8, past_covariates = pc_b)$quantiles[1, , ])) < 1e-10)
kc_b <- matrix(seq_len(90 + 8) %% 3, ncol = 1)
fc_kc <- timestral_forecast(m, list(ua, ub), h = 8,
                            known_covariates = list(matrix(seq_len(158) %% 3, ncol = 1), kc_b))
stopifnot(max(abs(fc_kc$quantiles[2, , , ] -
                  timestral_forecast(m, ub, h = 8, known_covariates = kc_b)$quantiles[1, , ])) < 1e-10)
.expect_error(timestral_forecast(m, list(ua, ub), h = 8,
                                 past_covariates = list(pc_a, pc_b[1:50, , drop = FALSE])),
              "for series 2, or 150 padded like y, got 50")
msk_b <- rep(TRUE, 90); msk_b[85:90] <- FALSE
stopifnot(max(abs(timestral_forecast(m, list(ua, ub), h = 8,
                                     observation_mask = list(rep(TRUE, 150), msk_b))$quantiles[2, , , ] -
                  timestral_forecast(m, ub, h = 8, observation_mask = msk_b)$quantiles[1, , ])) < 1e-10)
.expect_error(timestral_forecast(m, list(ua, cbind(ub, ub)), h = 8), "same variate count")
