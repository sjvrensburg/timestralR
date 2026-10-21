# Rolling-origin backtest of a Timestral model

Forecasts `y` from a sequence of forecast origins, each time using only
the data up to the origin as the context, and scores each forecast
against the `h` values that follow with
[`timestral_score()`](https://sjvrensburg.github.io/timestralR/reference/timestral_score.md).

## Usage

``` r
timestral_backtest(
  model,
  y,
  h,
  origins = NULL,
  initial = NULL,
  step = h,
  ...,
  level = c(80, 98),
  keep_forecasts = FALSE
)
```

## Arguments

- model:

  a `timestral_model` from
  [`timestral_load()`](https://sjvrensburg.github.io/timestralR/reference/timestral_load.md)

- y:

  the series to backtest: a numeric vector, `ts`/`mts`, `zoo`/`xts`, a
  matrix or data frame with time in rows, or a list of equal-length
  series (forecast as a batch at each origin)

- h:

  forecast horizon in steps

- origins:

  forecast origins as time indices (the last context step); overrides
  `initial` and `step`

- initial:

  the first origin; `NULL` for the last five origins

- step:

  spacing of the origins in time steps

- ...:

  arguments to
  [`timestral_forecast()`](https://sjvrensburg.github.io/timestralR/reference/timestral_forecast.md):
  `observation_mask`, covariates (over the whole series), `symmetric`,
  `clip_nonnegative`

- level:

  interval levels in percent for the coverage columns

- keep_forecasts:

  keep each origin's `timestral_forecast` in the `"forecasts"` attribute
  of the result (a list named by origin)

## Value

A data frame with one row per origin, series, variate and step:
`origin`, `series`, `variate`, `step`, `t` (the time index of the
forecast step, `origin + step`), `time` (its time value, when `y` is a
`ts`, `zoo` or `xts`), then the
[`timestral_score()`](https://sjvrensburg.github.io/timestralR/reference/timestral_score.md)
columns.

## Details

Origin `o` means the context is time steps `1:o` and the forecast covers
steps `o + 1` to `o + h`, so origins run from 1 to `T - h` for a series
of length `T`. By default the origins are
`seq(initial, T - h, by = step)`; with `initial = NULL` the five last
such origins are used (fewer if the series is short). Covariates and
`observation_mask` in `...` are given over the whole series and cut to
each origin: past-only covariates and the mask to `1:o`, known-future
covariates to `1:(o + h)`, so known covariates need at least `T` rows.
Steps the mask marks as missing are not scored either (their scores are
`NA`). Forecasts are made one origin at a time, batching the series of a
list `y` within each origin.

For tsibble data, use
[`tsibble::stretch_tsibble()`](https://tsibble.tidyverts.org/reference/stretch_tsibble.html)
with the
[`TIMESTRAL()`](https://sjvrensburg.github.io/timestralR/reference/TIMESTRAL.md)
fable model and
[`fabletools::accuracy()`](https://generics.r-lib.org/reference/accuracy.html)
instead.

## Speed

With the native backend, a forecast reuses the work done on the leading
patches its context shares with an earlier origin's context (a K/V
cache; results are unchanged up to rounding, about `1e-13`). Contexts
are left-padded to whole patches, so two origins share patches only when
they differ by a multiple of the patch size (`m$config$patch_size`, 32
for the official checkpoints): a `step` that is a multiple of it (or
`origins` that mostly are) roughly halves the time on long series. The
cache is not used when covariates are given (an `observation_mask` is
fine), nor once a context is longer than the model's `max_context`,
which slides the window.

## Parallel use

The backtest is single-threaded, like the rest of the package. To run
origins in parallel, split them and run one backtest per chunk, then
[`rbind()`](https://rdrr.io/r/base/cbind.html) the results. With forked
workers
([`parallel::mclapply()`](https://rdrr.io/r/parallel/mclapply.html) on
Linux and macOS) the loaded model can be used directly. Socket clusters
and `future` workers are new R processes, where a model's native session
is invalid, so load the model in each worker from its directory
(`m$dir`), or use `backend = "R"`. See the example.

## Examples

``` r
m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
bt <- timestral_backtest(m, AirPassengers, h = 12)
unique(bt$origin)
#> [1]  84  96 108 120 132
aggregate(cbind(pinball, crps, ae) ~ step, data = bt, FUN = mean)
#>    step   pinball      crps        ae
#> 1     1  67.08972  70.15283 106.26091
#> 2     2  24.85504  25.81827  34.80820
#> 3     3  92.02138  95.27061 137.76800
#> 4     4  86.82393  89.57678 121.67124
#> 5     5 162.23108 162.64167 190.13702
#> 6     6 235.18160 235.38149 276.10486
#> 7     7 170.17754 173.56055 235.70929
#> 8     8 267.16880 267.12980 314.83744
#> 9     9 167.94929 169.04942 204.80626
#> 10   10  78.04683  81.29396 123.80253
#> 11   11  54.37409  56.83019  92.86966
#> 12   12  78.92028  81.83164 117.31144

# \donttest{
## origins in parallel on a socket cluster: reload the model per worker
cl <- parallel::makeCluster(2)
dir <- m$dir
origins <- seq(60, 132, by = 12)
chunks <- split(origins, seq_along(origins) %% 2)
parts <- parallel::parLapply(cl, chunks, function(o, dir) {
  m <- timestralR::timestral_load(dir)
  timestralR::timestral_backtest(m, AirPassengers, h = 12, origins = o)
}, dir = dir)
parallel::stopCluster(cl)
bt_all <- do.call(rbind, parts)
# }
```
