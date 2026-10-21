# timestralR (development version)

- `timestral_backtest()` runs a rolling-origin backtest: it forecasts
  from a sequence of origins using only the data up to each one (with
  covariates and the observation mask cut to match), and returns one
  scored row per origin, series, variate and step. `timestral_score()`
  scores any forecast against realised values: absolute error of the
  median, pinball loss averaged over the levels, CRPS approximated from
  the quantiles (trapezoidal over the knots), and interval coverage. Both
  are single-threaded; `?timestral_backtest` shows how to split origins
  across parallel workers. With the native backend, origins that differ
  by a multiple of the patch size (32 for the official checkpoints) reuse
  the K/V cache over the context they share, which roughly halves a
  long-series backtest with such a `step` (1.30 to 0.63 s for 12 origins
  on 28M).

- The native K/V cache keeps two entries (the forward and mirrored passes
  of a symmetric forecast no longer evict each other) and reuses the
  longest run of leading patches a context shares with a cached one,
  rather than only an exact extension of it.

- `timestral_supports_covariates()` and
  `timestral_supports_known_covariates()` report which covariates a
  checkpoint can condition on (the Python forecaster's
  `supports_covariates`/`supports_known_covariates`), and printing a
  model shows the same.
- `predict()` method for `timestral_model`: `predict(m, newdata, h)` is
  `timestral_forecast(m, newdata, h)` and returns the same
  `timestral_forecast`.
- forecast package interface (forecast in Suggests):
  `timestral_as_forecast()` converts a forecast from `ts` input to a
  `"forecast"` object, and `forecast::forecast(m, y, h)` forecasts and
  converts in one call, so `accuracy()`, `autoplot()` and the rest of the
  forecast package work. Interval bounds are read off the quantile
  knots, never interpolated; the default levels are 80% and 98% because
  the official knots have no 2.5%/97.5% quantiles.
- Printing a forecast reports the checkpoint's number of quantile levels
  rather than always 21.

- `timestral_load()` and `timestral_download()` accept any Hugging Face
  Hub repository ID (`"owner/name"`) holding a Timestral checkpoint, for
  example a fine-tuned model, as well as the four sizes. The official
  repositories (`"mistralai/Timestral-5M"`, ...) keep their pinned,
  hash-verified commits however they are named. Branch and tag revisions
  are resolved to their current commit on the Hub and the resolution is
  cached for `local_files_only` use, so a moving branch such as `"main"`
  no longer stays frozen at its first download. Both functions gain a
  `token` argument (default: the `HF_TOKEN` environment variable).
- The first argument of `timestral_download()` is renamed from `size` to
  `model`, matching `timestral_load()`. Positional calls are unaffected;
  calls written as `size = "5M"` must use `model = "5M"`.
- Forecasts take their quantile levels (`fc$levels` and the `level`
  dimnames) from the checkpoint's `quantile_knots` instead of always
  using `timestral_levels()`, so checkpoints with other knots are
  labelled correctly; `symmetric = TRUE` now errors for knots that are
  not symmetric about 0.5, as upstream does, instead of silently
  averaging mismatched levels. `plot()` scales its y axis to the
  outermost knots rather than requiring 0.01 and 0.99.
- Licence field tightened to `Apache License (== 2.0)`, the licence of
  the upstream 'timestral' package the code is ported from, and Mistral
  AI is credited as copyright holder of the ported code.
- The tiny random-weight test checkpoint now ships in `inst/extdata/tiny`,
  so every example except the downloads runs offline under
  `R CMD check`.

# timestralR 0.5.0

- Input validation hardening: `timestral_forecast()` now errors (instead
  of aborting the R session, hanging, or silently coercing) on empty or
  zero-dimension `y`, on a series with no finite observed values, on
  non-numeric `y` or covariates (character, factor, complex, `Date`, ...),
  and on a horizon that is not a single whole number in [1, 10000]. The
  native `.ts_load()` re-validates the checkpoint config (positive
  `patch_size`/`num_heads`/`block_size`, `d_model` divisible by
  `num_heads`, valid `rope_dim`, increasing quantile knots containing
  0.5), and `.ts_predict_block()` validates its dimensions, masks and
  id arguments. Parallel checkpoint downloads no longer collide: each
  download writes to a unique `.part` file and a lost rename race against
  an identical published file is accepted.
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
  now states `Depends: R (>= 4.1.0)` (the code uses the base pipe `|>`;
  `tools::R_user_dir()` needs only 4.0.0), and `SystemRequirements`
  states a C++17 compiler.
- Checkpoint downloads are pinned to recorded upstream commits and
  verified against recorded SHA-256 hashes (new `digest` import);
  `revision = NULL` (the default in `timestral_download()` and
  `timestral_load()`) selects the pinned commit, a cached file that
  fails verification is re-downloaded, and failed downloads no longer
  leave `.part` files behind.
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
