# Developer notes

## Internal interface freeze (handoff for a future C++ port)

The following internal shapes are the contract a native implementation must
replace drop-in. Do not change them without updating the parity harness and
golden fixtures.

- `timestral_load()` returns a `timestral_model`: `weights` (named list of
  matrices/vectors exactly as in the safetensors file, biases present only
  when the config enables them), `config` (validated list from
  `R/config.R`, with derived `head_dim`, `rope_dim`, `num_quantiles`,
  `median_index`, `supports_known_covariates`), `max_context`,
  `per_shot_horizon_patches`, `quantile_real_cap_k`,
  `extreme_value_threshold`.
- Packed layout: rows are series-major — for each series, target rows, then
  past-covariate rows, then known-covariate rows. `.assemble_rows()` returns
  `rows`/`mask` `(B * V_all, T_padded)`, `groups` (series id per row),
  `role_ids` (0 target, 1 past, 2 known; NULL when no covariates),
  `target_idx`, and padded `known_future`/`known_future_mask`.
- `.predict_block(model, inputs, observed, horizon_patches, role_ids,
  known_future, known_future_mask, groups)` returns `(R, horizon_steps, Q)`
  raw-space quantiles for one decode block. This is the function a native
  kernel replaces; everything above it (assembly, refinement, capping,
  mirroring, clipping) stays in R.
- Golden fixtures in `tests/fixtures/` (tiny random checkpoint + official
  outputs) pin this contract without Python or downloads.

## Parity summary (official PyTorch, torch 2.14.1, fp32)

`inst/parity_official.R`, 12 cases (univariate missing, negative trend,
multivariate, numeric covariates, long horizon, long context cap, batch of
3, observation mask, categorical past/known/multivariate, non-default
knobs): worst `max_rel` 2.0e-6 on 5M (gate: < 2e-5). 123M: 9.6e-6, 209M:
2.6e-6 on the original six cases. Synthetic config variants (biases,
no gates, no asinh, alternate schedules) in `inst/synthetic_parity.R`:
< 2.3e-7.

## Benchmarks (this machine, 2 threads, context cap 4096)

`inst/benchmark.R`:

| size | univariate (h=96) | long horizon (h=1100) | context cap (h=2100) | batch of 10 |
|---|---|---|---|---|
| 5M | 0.05 s | 0.20 s | 0.80 s | 0.31 s |
| 209M | 1.15 s | 3.56 s | 13.81 s | 7.82 s |

Profiling note for the port: time is dominated by the per-row/per-head
attention loops and the transformer linears in `R/model.R`; the Welford
refinement (`R/forecaster.R`) and scaler (`R/scaler.R`) are minor. Port
`.predict_block` internals first.
