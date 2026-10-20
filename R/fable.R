## fable interface (fabletools/distributional are Suggests-only; the
## TIMESTRAL model definition talks to fabletools at call time, and the
## forecast/model_sum S3 methods are registered (delayed S3method() in
## NAMESPACE) when fabletools is loaded).

## Fitted fable models hold a small handle (checkpoint directory and load
## settings) rather than the timestral_model itself, so a mable does not
## carry a copy of the weights per key and stays valid after saveRDS() or
## in parallel workers, where the native session pointer would be NULL.
## Live models are cached per R process and reloaded from disk on a miss.
.model_cache <- new.env(parent = emptyenv())

.model_handle <- function(model) {
  args <- list(model = model$dir, max_context = model$max_context,
               per_shot_horizon_patches = model$per_shot_horizon_patches,
               quantile_real_cap_k = model$quantile_real_cap_k,
               extreme_value_threshold = model$extreme_value_threshold,
               backend = model$backend)
  list(key = paste(vapply(args, format, character(1), digits = 17),
                   collapse = "|"),
       args = args)
}

.model_from_handle <- function(handle) {
  model <- .model_cache[[handle$key]]
  if (is.null(model)) {
    model <- do.call(timestral_load, handle$args)
    assign(handle$key, model, envir = .model_cache)
  }
  model
}

.train_timestral <- function(.data, handle, symmetric, clip_nonnegative) {
  vars <- tsibble::measured_vars(.data)
  if (length(vars) != 1L)
    stop("the TIMESTRAL fable model supports a single (univariate) ",
         "response; got ", length(vars), ": ", paste(vars, collapse = ", "))
  .data <- .tsibble_fill_gaps(.data)
  y <- as.numeric(.data[[vars]])[order(.data[[tsibble::index_var(.data)]])]
  structure(list(handle = handle, var = vars, context = y,
                 symmetric = symmetric, clip_nonnegative = clip_nonnegative,
                 nobs = sum(is.finite(y))),
            class = "model_timestral")
}

#' Timestral as a fable model
#'
#' Defines a Timestral zero-shot forecasting model for use with
#' `fabletools::model()` on a `tsibble`. The response is univariate (use
#' [timestral_forecast()] directly for jointly forecast multiple
#' variates); keys in the tsibble are handled by fable as usual, with the
#' model trained (context captured) independently per key. Requires the
#' fabletools package (and fable for the usual workflow); forecasts are
#' reported as 21-quantile distributions.
#'
#' @param formula the response formula, as in other fable models, e.g.
#'   `TIMESTRAL(value)`
#' @param model a `timestral_model` from [timestral_load()]; loaded once
#'   and reused for every key and forecast. Fitted models store only the
#'   checkpoint directory and load settings, not the weights, so a mable
#'   stays small when saved and is reloaded from disk when forecast in a
#'   new R session or a parallel worker.
#' @param symmetric,clip_nonnegative as in [timestral_forecast()]
#' @param ... passed to the model definition (currently unused)
#' @return a fable model definition, for use inside `fabletools::model()`
#' @examples
#' ## the tiny bundled checkpoint runs offline but its forecasts are
#' ## meaningless; use timestral_load("5M") or larger for real ones
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' if (requireNamespace("fable", quietly = TRUE)) {
#'   fit <- tsibble::as_tsibble(AirPassengers) |>
#'     fabletools::model(timestral = TIMESTRAL(value, model = m))
#'   fit |> fabletools::forecast(h = 24)
#' }
#' @export
TIMESTRAL <- function(formula, model, symmetric = TRUE,
                      clip_nonnegative = TRUE, ...) {
  if (!requireNamespace("fabletools", quietly = TRUE))
    stop("the TIMESTRAL fable model requires the fabletools package")
  if (missing(model) || !inherits(model, "timestral_model"))
    stop("`model` must be a timestral_model from timestral_load()")
  handle <- .model_handle(model)
  assign(handle$key, model, envir = .model_cache)
  train_fn <- .timestral_train_fn(handle, symmetric, clip_nonnegative)
  ## fabletools keeps this frame as the model environment; drop the
  ## model so it is not serialised into every fitted model
  rm(model)
  cls <- fabletools::new_model_class(
    "timestral", train = train_fn, specials = fabletools::new_specials())
  fabletools::new_model_definition(cls, !!rlang::enquo(formula), ...)
}

## Built in its own frame so the training closure captures only the
## handle, never the model.
.timestral_train_fn <- function(handle, symmetric, clip_nonnegative) {
  force(handle); force(symmetric); force(clip_nonnegative)
  function(.data, specials, ...)
    .train_timestral(.data, handle, symmetric, clip_nonnegative)
}

## Registered for fabletools::forecast when fabletools is loaded.
#' @exportS3Method fabletools::forecast
forecast.model_timestral <- function(object, new_data, specials = NULL,
                                     bootstrap = FALSE, times = 5000, ...) {
  if (!requireNamespace("distributional", quietly = TRUE))
    stop("fable forecasts require the distributional package")
  h <- NROW(new_data)
  fc <- timestral_forecast(.model_from_handle(object$handle),
                           object$context, h = h,
                           symmetric = object$symmetric,
                           clip_nonnegative = object$clip_nonnegative)
  q <- matrix(fc$quantiles[1, , ], nrow = h)
  ## levels passed by position: the argument is `p` in older
  ## distributional releases and `quantile` in newer ones
  out <- distributional::dist_quantile(lapply(seq_len(h), function(i) q[i, ]),
                                       rep(list(fc$levels), h))
  dimnames(out) <- object$var
  out
}

## Registered for fabletools::model_sum when fabletools is loaded.
#' @exportS3Method fabletools::model_sum
model_sum.model_timestral <- function(x) "TIMESTRAL"

#' Convert a Timestral forecast to a fable forecast table
#'
#' Registered as an S3 method for `fabletools::as_fable()` when fabletools
#' is loaded; call it as `fabletools::as_fable(fc)`. The distribution
#' column is the 21-quantile [distributional::dist_quantile()]
#' distribution from the forecast. A univariate forecast uses the variate
#' name as the response; a fable holds a single distribution column, so a
#' multivariate forecast becomes a `variate` key with the response in a
#' `value` column. Forecasts from `ts` input use the same index class
#' (`yearmonth`, `yearquarter`, ...) as `tsibble::as_tsibble()` gives the
#' `ts`, so they join with and score against the observed data, and fable
#' accuracy measures and interval extraction apply. See [TIMESTRAL()] for
#' using Timestral directly as a fable model.
#'
#' @param x a `timestral_forecast` created from indexed input (`ts`,
#'   `zoo`, `xts` or `tsibble`), so a future time index is available
#' @param ... ignored
#' @return a `fbl_ts` forecast table
#' @examples
#' ## the tiny bundled checkpoint runs offline but its forecasts are
#' ## meaningless; use timestral_load("5M") or larger for real ones
#' m <- timestral_load(system.file("extdata", "tiny", package = "timestralR"))
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' if (requireNamespace("tsibble", quietly = TRUE) &&
#'     requireNamespace("distributional", quietly = TRUE) &&
#'     requireNamespace("fabletools", quietly = TRUE))
#'   fabletools::as_fable(fc)
#' @exportS3Method fabletools::as_fable
as_fable.timestral_forecast <- function(x, ...) {
  if (is.null(x$future_index))
    stop("this forecast has no time index; create it from ts, zoo, xts ",
         "or tsibble input to convert it to a fable table")
  for (p in c("tsibble", "distributional", "fabletools"))
    if (!requireNamespace(p, quietly = TRUE))
      stop("converting to a fable table requires the ", p, " package")
  iv <- .tidy_index_var(x)
  fidx <- .tidy_future_index(x)
  B <- x$n_series
  vnames <- dimnames(x$quantiles)$variate
  multi <- length(vnames) > 1L
  response <- if (multi) "value" else vnames
  snames <- dimnames(x$quantiles)$series
  rows <- list()
  for (b in seq_len(B)) for (v in seq_along(vnames)) {
    q <- if (B == 1L) matrix(x$quantiles[v, , ], nrow = x$h)
         else matrix(x$quantiles[b, v, , ], nrow = x$h)
    dfb <- data.frame(row.names = seq_len(x$h))
    if (B > 1L) dfb$series <- snames[b]
    if (multi) dfb$variate <- vnames[v]
    dfb[[iv]] <- fidx
    dist <- distributional::dist_quantile(
      lapply(seq_len(x$h), function(i) q[i, ]),
      rep(list(x$levels), x$h))
    dimnames(dist) <- response
    dfb[[response]] <- dist
    rows[[length(rows) + 1L]] <- dfb
  }
  df <- do.call(rbind, rows)
  keys <- c(if (B > 1L) "series", if (multi) "variate")
  tsib <- tsibble::as_tsibble(df, index = iv,
                              key = if (length(keys)) keys else NULL)
  fabletools::as_fable(tsib, response = response,
                       distribution = tidyselect::all_of(response))
}
