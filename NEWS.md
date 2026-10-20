# timestralR 0.5.0

- tidyverts support (all via Suggests, no new hard dependencies):
  `tsibble` input to `timestral_forecast()` (keys become the batch,
  measured variables the jointly forecast variates; tsibble covariates
  supported) and a `tsibble::as_tsibble()` method for forecasts;
  `ggplot2::autoplot()`/`autolayer()` fan-chart methods; and a fable
  interface — a `TIMESTRAL()` model definition for
  `fabletools::model()` (univariate response, per-key contexts,
  21-quantile distributional forecasts) plus a
  `fabletools::as_fable()` method converting forecasts to `fbl_ts`.
  S3 methods for these generics are registered conditionally when the
  owning package is loaded.
- Fixes to the tidyverts interfaces: `as_tsibble()`/`as_fable()` keep
  the index column for `ts`, `zoo` and `xts` forecasts (named `index`,
  and for `ts` with the `yearmonth`/`yearquarter`/... class that
  `tsibble::as_tsibble()` uses) and for tsibble indexes named `time` or
  `step`; multivariate forecasts convert to a fable with a `variate` key;
  quantile distributions are built in a way that works across
  distributional versions. Fitted `TIMESTRAL()` models store a handle to
  the checkpoint instead of the weights, so mables stay small when saved
  and forecast in new sessions or parallel workers. Tsibble inputs are
  put in index order and implicit gaps are filled with missing values;
  logical tsibbles work as observation masks; keyed tsibble covariates
  are matched to series by key and validated like other covariates. Plots
  label variates by name.
- Documentation and `NAMESPACE` are generated with roxygen2; methods for
  ggplot2, tsibble and fabletools generics are registered through
  delayed `S3method()` entries instead of `.onLoad` hooks. DESCRIPTION
  now states `Depends: R (>= 4.0.0)` (needed for `tools::R_user_dir()`).
- The native backend is split from one file into per-unit headers and
  implementation files (`session`, `nn_ops`, `scaler`, `masks`,
  `attention`, `transformer`, `head`, `predict`, plus the Rcpp entry
  points); outputs are bit-for-bit unchanged.
- New vignette, `vignette("timestralR")`, covering every feature with
  plots (precomputed from `vignettes/timestralR.Rmd.orig`; rebuild with
  `vignettes/precompile.R`).
- `autoplot()`/`autolayer()` draw opaque bands in graded blues, widest
  first, matching `plot()` (any fill scale still overrides them), and
  facet variates in input order. A list of series that share a time
  index keeps it, and a named list no longer leaks a name into `dim()`.

# timestralR 0.4.0

- R workflow support: forecasts keep the observed history and the time
  index. `ts`/`mts`, `zoo` and `xts` inputs (the latter two via
  Suggests, no new hard dependency) preserve their index; the forecast
  object gains `observed`, `index` and `future_index` (extending the
  last observed interval), `as_forecast_df()` reports real index values
  (e.g. Dates), and covariates with a time index are checked for
  alignment with `y`. `timestral_median()`/`timestral_quantile()` now
  return a `ts`/`mts` for a single-series `ts` forecast.
- New `plot()` method for forecasts: fan chart with the observed
  context, median, and 50/80/90/98% predictive bands.
- Documentation: examples in every help page, a package-level help
  page, and `URL`/`BugReports` in DESCRIPTION.

# timestralR 0.3.1

- Native backend speedups, no change to results (native vs R backend
  still agrees to < 1e-10 on the fixtures; multi-block decode on 209M
  agrees to ~4e-13): context K/V cached across decode blocks, rows
  batched into single GEMMs, fused attention with head-block QKV
  layout and mask-as-bias, and activations kept in one stacked matrix
  through the transformer. The native backend remains
  single-threaded by design (see `docs/developer-notes.md`).
- Speed on this machine (209M): batch of 10 4.6 s -> 2.3 s,
  long-context 6.4 s -> 5.0 s, long horizon 1.7 s -> 1.3 s.

# timestralR 0.3.0

- Native RcppEigen backend (default, `backend = "native"` in
  `timestral_load()`): the single-block transformer primitive runs in
  C++ with Eigen, with large products routed to the system BLAS and
  small ones kept in Eigen (threshold tuned by benchmark; the tuned
  BLAS is not always faster). The pure-R implementation remains as
  `backend = "R"`, the reference and fallback. Results agree to < 1e-10
  on the golden fixtures; official parity is unchanged.
- Speed on this machine: 209M long-context case 13.8 s (R) -> 6.4 s
  (native); 5M 0.80 s -> 0.22 s.

# timestralR 0.2.0

- Complete parity with the Python `timestral` package: batched series
  (list or 3-D array input), `observation_mask`, categorical past/known
  covariates with target encoding, and the forecaster/loader options
  `per_shot_horizon_patches`, `quantile_real_cap_k`,
  `extreme_value_threshold`, `revision`, `cache_dir`, `local_files_only`.
- New accessors `timestral_median()` and `timestral_quantile()`.
- Config validation and strict weight checking at load, matching upstream;
  checkpoints with biases or non-default gates/schedules are supported.
- Breaking: `max_context` now defaults to the checkpoint's trained context
  length (32768), as in Python, instead of 4096. Pass
  `max_context = 4096L` to keep the previous behaviour.
- Golden-fixture tests (tiny checkpoint in `tests/fixtures/`) now run in
  every check, with no Python or downloads required.

# timestralR 0.1.0

- Initial native implementation: official fp32 safetensors checkpoints,
  forward pass and forecasting loop in pure R. Parity vs official PyTorch
  worst `max_rel` 2.0e-6 (5M, six cases).
