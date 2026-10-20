# Roadmap: capability helpers and R conveniences

Four additions, in implementation order. None changes the frozen internal
interface in `developer-notes.md` (`.assemble_rows()` / `.predict_block()`),
so none needs new parity fixtures. Everything is built on the public
`timestral_forecast()`.

| # | item | depends on | new deps | size |
|---|---|---|---|---|
| 1 | capability helpers | – | none | XS (~20 lines + tests) |
| 2 | `predict.timestral_model()` | – | none | S (~40 lines) |
| 3 | `forecast`-package conversion | – | `forecast` (Suggests) | M (~80 lines) |
| 4 | rolling-origin backtest + scoring | 1 (validation), 2 (optional) | none | L (~200 lines) |

Items 1–3 are independent and can ship in one release (0.6.0). Item 4 is
best on its own branch, because its API needs more review.

## 1. `timestral_supports_covariates()` / `timestral_supports_known_covariates()`

**What's there now.** The flags already exist as derived config fields:
`covariate_enabled` is read from `config.json`, with a default of `TRUE`
(`R/config.R:8`), and `supports_known_covariates` is set at
`R/config.R:83` via `.supports_known_covariates()` (`R/config.R:87`). That
function ports upstream's `TimestralConfig.supports_known_covariates`
property (`timestral/config.py:98`). Upstream exposes both on the forecaster
(`forecaster.py:153`, `:157`). In R they are only used internally, as
`stop()` guards (`R/forecaster.R:561`, `:586`).

**API** (in `R/safetensors.R`, next to `print.timestral_model` at `:238`):

```r
timestral_supports_covariates(model)        # logical(1)
timestral_supports_known_covariates(model)  # logical(1)
```

Both call `stopifnot(inherits(model, "timestral_model"))` and return
`isTRUE(model$config$<field>)`.

**Decisions**
- *Names:* these follow the package's `timestral_` prefix rather than
  upstream's property names. Recommended. Plain `supports_covariates()`
  would be too generic to export.
- *Also extend `print.timestral_model`* with a line such as
  `covariates: past + known` / `past only` / `none`. Recommended, because it
  costs one line.
- *Refactor the two guards* (`R/forecaster.R:561`, `:586`) to call the
  helpers, so there is one source of truth. Recommended.
- *Also accept a fitted `TIMESTRAL()` mable model?* Not now. Users can call
  `.model_from_handle()` themselves.

**Tests** (`tests/parts/02-forecast-core.R`): on `m`, both helpers return
the same value as `m$config`. Building a config with
`covariate_enabled = FALSE` (as `03-regressions.R:131` already does, then
`.validate_config()`) makes both `FALSE`. Passing a non-model errors.

**Docs/NEWS:** roxygen with a runnable example. This depends on the
`inst/extdata` tiny checkpoint landing on the current branch; until then,
use `\dontrun{}`. Add one sentence in the vignette's covariates section and
a NEWS entry.

## 2. `predict()` method for `timestral_model`

**API** (`R/forecaster.R`, after `timestral_forecast()`):

```r
#' @export
predict.timestral_model <- function(object, newdata, h, ...)
  timestral_forecast(object, newdata, h = h, ...)
```

**Decisions**
- *Argument name for the series:* `newdata` follows the `stats::predict`
  convention, although the series is really the context, not new
  predictors. Recommended, with the docs saying so plainly. The alternative,
  `y`, matches `timestral_forecast()` but surprises `predict()` users.
- *Return value:* a `timestral_forecast`, unchanged. Recommended. Returning
  a plain median vector, as many `predict()` methods do, would lose the
  quantiles, and `timestral_median()` already covers that use.
- *`h` default:* none, so a missing `h` errors the same way it does in
  `timestral_forecast()` (`.validate_horizon`, `R/forecaster.R:28`). Don't
  guess a horizon from `frequency()`.
- `stats` is already in Imports, so `S3method(predict, timestral_model)`
  needs no new dependency.

**Tests** (`02-forecast-core.R`): `predict(m, y, h = 24)` is `identical()`
to `timestral_forecast(m, y, h = 24)`, both with and without covariates
passed through `...`.

**Docs/NEWS:** one help page that links to `timestral_forecast()`, and a
NEWS line.

## 3. Conversion to a `forecast`-package object

**API** (new file `R/forecast_pkg.R`):

```r
timestral_as_forecast(fc, level = c(80, 98), series = 1L, variate = 1L)
#' @exportS3Method forecast::forecast
forecast.timestral_model(object, y, h, level = c(80, 98), ...)
```

**Decisions**
- *Generic:* the `forecast` package exports no `as.forecast()` generic.
  Recommended: an exported plain function, `timestral_as_forecast(fc, level,
  series, variate)`. That avoids registering against a generic that doesn't
  exist. Also register `forecast::forecast` for `timestral_model`
  (`forecast.timestral_model(object, y, h, level, ...)`, via
  `@exportS3Method forecast::forecast`). This is the idiomatic entry point:
  `forecast(m, AirPassengers, h = 24)`. Watch for a name clash: fabletools
  also has a `forecast` generic (`R/fable.R:101`). The two are registered
  against different namespaces, so both work, but this needs a test with
  both packages loaded.
- *Object structure:* a list with class `"forecast"`: `mean` (median `ts`),
  `lower`/`upper` (`ts` matrices, one column per level, named `"80%"` and
  so on), `level` (numeric percent), `x` (the observed context as a `ts`,
  from `fc$observed`, which `.fc_plot_parts()` at `R/plot.R:4` already
  extracts), `method = "Timestral"`, and `series`.
- *Interval levels:* a central interval at `L%` needs knots at
  `(1 ± L/100)/2`. The knot check in `.fc_plot_parts()$knot`
  (`R/plot.R:22`) already gives the right error. Reuse it, so 80, 90 and 98
  work and 95 errors. Open question: `forecast`'s default levels are
  `c(80, 95)`, and 95 needs the 0.025/0.975 knots, which are not among the
  21. Recommended: default to `c(80, 98)` (98 uses the 0.01/0.99 knots) and
  document why. Interpolating between knots would be inventing quantiles.
- *Only `ts`-backed input:* the `forecast` package assumes `ts`, so error
  when `fc$tsp` is `NULL`. A univariate slice comes from `series` and
  `variate`, because `forecast` objects are univariate.
- *Knots (fixed):* forecasts now take `fc$levels` from the checkpoint's
  `quantile_knots`, and `timestral_quantile()` matches levels to 1e-9, so
  the conversion can read the interval knots from `fc$levels` directly.
  A checkpoint without the knots for a requested interval should get an
  error naming `fc$levels`, as `plot()` already does.

**Tests** (`04-optional-interfaces.R`, inside
`if (requireNamespace("forecast", quietly = TRUE))`, and add `"forecast"` to
`.optional_pkgs` in `tests/timestralR-tests.R` so CI requires it):
`class(x) == "forecast"`; `x$mean` equals `timestral_median(fc)`; the
`lower[, "80%"]` column equals `timestral_quantile(fc, 0.1)`; level 95
errors; non-`ts` input errors; `forecast::accuracy(x, test)` runs.

**Deps/docs:** add `forecast` to Suggests, add a vignette subsection next to
the fable one (this needs a re-run of `vignettes/precompile.R`), and a NEWS
entry.

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

**Docs/NEWS:** a vignette section ("Evaluating forecasts") using the 28M
checkpoint, which needs a precompile re-run; a help page that documents the
CRPS approximation and the parallelisation pattern; a NEWS entry.

## Out of scope

- Hugging Face repo-ID loading and runnable examples via `inst/extdata`
  (on the current branch, `feat/extdata-examples-hub-repos`).
