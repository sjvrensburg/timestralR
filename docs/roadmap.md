# Roadmap: R conveniences

Items 1–3 are done; item 4 is open. None changes the frozen internal
interface in `developer-notes.md` (`.assemble_rows()` / `.predict_block()`),
so none needs new parity fixtures. Everything is built on the public
`timestral_forecast()`.

| # | item | status | new deps |
|---|---|---|---|
| 1 | capability helpers | done | none |
| 2 | `predict.timestral_model()` | done | none |
| 3 | `forecast`-package conversion | done | `forecast` (Suggests) |
| 4 | rolling-origin backtest + scoring | open | none |

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

## 4. Rolling-origin backtest and scoring

**API** (new file `R/backtest.R`):

```r
timestral_backtest(model, y, h, origins = NULL, initial = NULL, step = h,
                   ..., keep_forecasts = FALSE)
timestral_score(fc, actual, metrics = c("pinball", "crps", "mae", "coverage"))
```

- `timestral_backtest` returns a data frame with one row per
  origin × series × variate × step: `origin`, `step`, `actual`, `median`,
  `pinball` and `crps`. It optionally also returns the per-origin
  forecasts in an attribute.
- `timestral_score` works on a single forecast plus its realised values, so
  users can score their own loops.

**Scoring**
- *Pinball:* the mean over the 21 levels τ of `2·(1{y<q_τ} − τ)(q_τ − y)`.
  This is the standard quantile loss.
- *CRPS from quantiles:* the weighted sum of pinball losses over the knots,
  using trapezoidal weights from the knot spacing. The knots aren't evenly
  spaced, so an equal-weight average would be biased. Document that this
  approximates CRPS (it ignores the tails beyond 0.01/0.99), as the GluonTS
  and Chronos benchmarks do.
- `MASE` is out of scope; it needs a seasonal naive baseline.

**Decisions**
- *Origins:* the default is `seq(initial, T - h, by = step)`. Explicit
  `origins` (as time indices) override it. Each origin forecasts from
  `y[1:origin]` (truncated to `max_context` by the existing loop), and
  covariates are sliced the same way. Known covariates need values up to
  `origin + h`, so validate that the input spans `T + h` or ends at `T`
  (in which case the last origin must be `<= T - h`).
- *Batching:* stack all origins of a series into one `timestral_forecast()`
  batch call? That isn't possible directly, because a batch needs equal
  lengths. Recommended: loop over origins and batch across series inside
  each origin. That is simple, and memory per call stays bounded.
- *Threading constraint:* no internal parallelism (AGENTS.md,
  `developer-notes.md`). Document the R-level pattern instead: split the
  origins and run `parallel::mclapply()` or `future.apply` over
  `timestral_backtest(model, y, h, origins = chunk)`. Inside the workers,
  reload with `timestral_load(m$dir)` (the external-pointer session doesn't
  survive serialisation; `R/fable.R:13` `.model_handle()` already solves
  this, so reuse it). Add a `\donttest` example.
- *K/V cache:* origins processed in increasing order extend the context
  bit-for-bit, so the single-entry cache (`developer-notes.md`, "Native
  backend") reuses earlier patches only if the previous call's context is
  a strict prefix in rows, groups and first-observed patch. Two things
  break that: the mirrored pass (`symmetric = TRUE`) interleaves a `-y`
  context, and `max_context` truncation slides the window. Expect little
  cache benefit by default, and don't promise any speed-up. A possible
  later step: an option to run all origins' forward passes before the
  mirrored ones. That would need restructuring `timestral_forecast()`, so
  defer it.
- *Input types:* accept anything `.batch_dispatch()` accepts
  (`R/tsibble.R:71`), but slicing by origin needs a time axis. Start with
  vectors, `ts`, matrices and lists. For tsibble, defer to fable's
  `stretch_tsibble()` + `TIMESTRAL()`, which already works, and say so in
  the docs.

**Tests** (a new `tests/parts/03b-backtest.R`, or append to `02`; sourced
after 01, so it can use `m`/`y`): pinball loss on a hand-computed 3-level
toy case; CRPS equals a known closed form for a degenerate forecast (all
quantiles equal → |q − y|); backtest row count is `origins × h`; with one
origin, the result matches a direct `timestral_forecast()` on
`y[1:origin]`; an origin where known covariates are too short errors.

**Docs/NEWS:** a vignette section ("Evaluating forecasts"; the
vignette already backtests one origin by hand in section 2 and scores
fable forecasts with CRPS in section 7) using the 28M
checkpoint, which needs a precompile re-run; a help page that documents the
CRPS approximation and the parallelisation pattern; a NEWS entry.
