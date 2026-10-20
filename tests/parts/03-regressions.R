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

## regression: plotting an all-missing history gives a clear error
fc_na <- timestral_forecast(m, rep(NA_real_, length(y)), h = 6)
err <- tryCatch({
  grDevices::pdf(NULL); plot(fc_na); grDevices::dev.off(); FALSE
}, error = function(e) grepl("no finite observed", conditionMessage(e)))
stopifnot(isTRUE(err))
