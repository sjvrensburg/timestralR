# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

timestralR is an R port of Mistral AI's `timestral` Python package: it runs the Timestral forecasting models from R using the official `safetensors` checkpoints, with no Python or torch at runtime.

## Development loop

- The test runner does `library(timestralR)`, so it tests the **installed** package. Run `R CMD INSTALL .` after any change in `R/` or `src/` before `Rscript tests/timestralR-tests.R`.
- No testthat. Tests are plain `stopifnot()` scripts in `tests/parts/0N-*.R`, sourced in order into one shared environment. Part 01 defines the fixture model `m` (native), `m_R` (R backend), inputs `y`/`y2` and forecast `fc`, and later parts use them. To run one part, source 01 first, from `tests/`:
  `cd tests && Rscript -e 'library(timestralR); source("parts/01-golden-parity.R"); source("parts/03-regressions.R")'`
- Optional-interface tests (zoo, xts, tsibble, ggplot2, fable, ...) skip locally when those packages are missing. Under `CI=true` or `TIMESTRALR_REQUIRE_OPTIONAL=true` a missing package is an error instead.
- Real-checkpoint tests: `TIMESTRALR_TEST_MODEL_DIR="$(Rscript -e 'cat(timestralR::timestral_download("5M"))')" Rscript tests/timestralR-tests.R`. With 5M this also asserts the committed official golden fixture (`tests/fixtures/*_5m_univariate.csv`) at a max_rel gate of 2e-5.
- CI also fails if roxygen output is stale (it runs roxygen2 **7.3.1**, then `git diff` on `man NAMESPACE DESCRIPTION R/RcppExports.R src/RcppExports.cpp`) or if the vignette is stale (`Rscript inst/check_vignette_freshness.R`).

## Vignette

Edit `vignettes/timestralR.Rmd.orig`, never `timestralR.Rmd`. The `.Rmd` is precomputed: install the package from a clean `src/` (pkgload/roxygen leave `-O0` objects behind that skew the timing section), then run `Rscript vignettes/precompile.R`. That needs the 28M checkpoint and ggplot2, patchwork, zoo, tsibble, dplyr and fable.

## Architecture

The R files are ports of their Python counterparts, and each file's header comment names the upstream module (`config.py`, `scaler.py`, `masks.py`, `layers.py`/`model.py`, `forecaster.py`, `covariate_encoding.py`). When behaviour is in question, the upstream Python is the reference.

- **Load:** `timestral_load()` (`R/safetensors.R`) downloads the checkpoint into `tools::R_user_dir("timestralR", "cache")`, checking SHA-256 at the pinned revisions (`HF_TOKEN` is honoured), or reads a local directory. It parses the safetensors file with the package's own reader, validates the config (`R/config.R`), and with `backend = "native"` (the default) builds an Eigen session through `.ts_load`. That session is an external pointer.
- **Forecast:** `R/forecaster.R` does the work outside the network: input coercion, packing rows series-major (target, then past-covariate, then known-covariate rows; see `.assemble_rows()`), block decoding with median feedback, mirroring, quantile capping and clipping. Each decode block calls `.predict_block()`, which dispatches on `model$session` to either the native `.ts_predict_block` or the pure-R network in `R/model.R`.
- **Frozen contract:** `docs/developer-notes.md` specifies the `timestral_model` fields and the `.assemble_rows()`/`.predict_block()` shapes that the native backend replaces drop-in. Changing them means updating the parity harness and the golden fixtures.
- **Native backend (`src/`):** float64 throughout. `lin` sends GEMMs of 4e6 flops or more to the system BLAS and runs smaller ones in Eigen; the threshold was benchmark-tuned. The context-only path keeps a two-entry LRU K/V cache (`KVCache` in `session.h`) that reuses the leading patches a new context shares bit-for-bit with a cached one (across decode blocks and across backtest origins), capped at 256 MB in total. The unit-by-unit layout table is in `docs/developer-notes.md`.
- **Optional interfaces** (`R/fable.R`, `R/ggplot.R`, `R/tsibble.R`): S3 methods for generics from Suggests-only packages are registered with `@exportS3Method pkg::generic`, so R registers them lazily. Call those packages only through `pkg::`.

## Parity and performance

- Golden fixtures pin correctness without Python: official PyTorch outputs in `tests/fixtures/` for the tiny random-weight checkpoint in `inst/extdata/tiny/`. That checkpoint is installed with the package so the examples run offline; tests find it with `system.file("extdata", "tiny", package = "timestralR")`. They are regenerated with `inst/make_fixtures.py`, a dev-only script that needs torch and timestral and uses `.venv/`.
- Full parity against official PyTorch: `inst/parity_official.R`, configured with the env vars `TIMESTRALR_MODEL_DIR`, `TIMESTRALR_PYTHON` (default `.venv/bin/python`) and `TIMESTRALR_CASES`. `inst/synthetic_parity.R` covers config variants. Benchmarks are in `inst/benchmark.R`.
- Before any speed work, read the profiling notes in `docs/developer-notes.md`. They list measured dead ends not to retry unchanged (fused bias/softmax, transposed softmax layout) and options that are off the table (fp32/int8, threads, `-march=native`).
