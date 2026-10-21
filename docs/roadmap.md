# Roadmap: R conveniences

All four items are done; follow-ups are listed at the end. None changes the frozen internal
interface in `developer-notes.md` (`.assemble_rows()` / `.predict_block()`),
so none needs new parity fixtures. Everything is built on the public
`timestral_forecast()`.

| # | item | status | new deps |
|---|---|---|---|
| 1 | capability helpers | done | none |
| 2 | `predict.timestral_model()` | done | none |
| 3 | `forecast`-package conversion | done | `forecast` (Suggests) |
| 4 | rolling-origin backtest + scoring | done | none |

## Done (items 1–3)

Decisions taken, for reference when extending them:

- **Capability helpers** (`R/safetensors.R`):
  `timestral_supports_covariates()` and
  `timestral_supports_known_covariates()` read `covariate_enabled` and
  the derived `supports_known_covariates` from the config. The two
  covariate guards in `timestral_forecast()` call them, and
  `print.timestral_model` shows a `covariates:` line. They take only a
  `timestral_model`, not a fitted `TIMESTRAL()` model.
- **`predict()`** (`R/forecaster.R`): `predict(object, newdata, h, ...)`
  forwards to `timestral_forecast()` and returns the `timestral_forecast`
  unchanged. `newdata` is the context series. There is no default `h`.
- **forecast package** (`R/forecast_pkg.R`): the package has no
  `as.forecast()` generic, so there is an exported
  `timestral_as_forecast(fc, level, series, variate)` plus
  `forecast.timestral_model()`, registered with
  `@exportS3Method forecast::forecast`. That generic is
  `generics::forecast`, the same one fabletools uses, so the two methods
  share one generic and dispatch on their own classes. Intervals pair two
  knots via `.fc_plot_parts()$knot`, never interpolating; the default
  levels are `c(80, 98)` because the official knots have no 0.025/0.975.
  Only `ts`-based forecasts convert. `fitted`/`residuals` are all `NA`
  (zero-shot, nothing fitted), which `accuracy()` needs present: it then
  reports test-set measures, and the training-set row is `NaN`.
- Repeated native forecasts of the same input agree to ~1e-13, not bit
  for bit (K/V cache reuse), so tests compare them with a tolerance.

## Done (item 4)

- **API** (`R/backtest.R`): `timestral_score(fc, actual, level)` and
  `timestral_backtest(model, y, h, origins, initial, step, ..., level,
  keep_forecasts)`. Origin `o` is the last context step; origins run
  from 1 to `T - h`. The default is the last five origins spaced by
  `step` (default `h`), or `seq(initial, T - h, by = step)`.
- **Slicing:** `y` is normalised once by `.as_batch()` and every
  context is sliced from that (series, variate, time) array, then passed
  back as a 3-D array. This gives one code path for vectors, `ts`,
  matrices, data frames, zoo/xts and lists. Past covariates and the mask
  are cut to `1:o`, known covariates to `1:(o + h)`. Tsibbles are
  refused with a pointer to `stretch_tsibble()` + `TIMESTRAL()`.
- **Scores:** `pinball` = mean over levels of 2ρ_τ (equals |error| for a
  point forecast); `crps` = 2Σ w_τ ρ_τ with trapezoidal weights over the
  knots, so a point forecast gives (τ_K − τ_1)|error| = 0.98|error|
  (tails ignored, documented); `ae`; `cover<L>` per interval level.
- **Parallel:** single-threaded. Forked workers can use the loaded model;
  socket/`future` workers must reload it from `m$dir`, because the native
  session is invalid after serialisation ("invalid or released Timestral
  session"). The `\donttest` example on `?timestral_backtest` does this
  and matches the sequential result exactly.

## Follow-ups (not scheduled)

- ~~**K/V cache across origins**~~ (done): the native cache keeps two
  entries (forward and mirrored) and reuses the longest shared run of
  leading patches, and the backtest runs origins grouped by
  `origin %% patch_size`, so origins a multiple of the patch size apart
  reuse each other's context (12 origins at step 32 on 28M: 1.30 to
  0.63 s). Not covered: covariate forecasts (the window path has no
  cache) and contexts past `max_context` (RoPE positions shift).
- ~~**Scaled scores**~~ (done): `mase`, `scaled_pinball` and
  `scaled_crps`, divided by the in-sample seasonal-naive MAE of each
  origin's context (`period` from the `ts` frequency, else 1; lag 1 when
  the context is too short, as in M4), or a user `scale` (numbers or a
  function of the context). Matches `forecast::accuracy()`'s MASE.
- **Relative skill (deferred):** scores relative to a baseline forecast
  scored on the same targets (the Chronos/GIFT-Eval style geometric mean
  of ratios), e.g. `timestral_skill(bt, bt_baseline)` joining two scored
  data frames on origin, series, variate and step. Needs a way to score
  non-Timestral quantile forecasts.
- ~~**Unequal-length batches**~~ (done): series are end-aligned and
  left-padded with NA (exact: padding does not change a forecast), in
  lists, tsibble key groups, covariates and masks, and the backtest.
