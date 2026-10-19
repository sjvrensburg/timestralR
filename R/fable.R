## fable interface (fabletools/distributional are Suggests-only; the
## TIMESTRAL model definition talks to fabletools at call time, and the
## forecast/model_sum S3 methods are registered in .onLoad when
## fabletools is available).

.train_timestral <- function(.data, model, symmetric, clip_nonnegative) {
  vars <- tsibble::measured_vars(.data)
  if (length(vars) != 1L)
    stop("the TIMESTRAL fable model supports a single (univariate) ",
         "response; got ", length(vars), ": ", paste(vars, collapse = ", "))
  y <- as.numeric(.data[[vars]])
  structure(list(model = model, var = vars, context = y,
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
#' model trained (context captured) independently per key.
#'
#' @param formula the response formula, as in other fable models, e.g.
#'   `TIMESTRAL(value)`
#' @param model a `timestral_model` from [timestral_load()]; loaded once
#'   and reused for every key and forecast
#' @param symmetric,clip_nonnegative as in [timestral_forecast()]
#' @param ... passed to the model definition (currently unused)
#' @return a fable model definition, for use inside `fabletools::model()`
#' @examples
#' \dontrun{
#' library(fable)
#' m <- timestral_load("5M")
#' fit <- as_tsibble(AirPassengers) |>
#'   model(timestral = TIMESTRAL(value, model = m))
#' fit |> forecast(h = 24)
#' }
#' @export
TIMESTRAL <- function(formula, model, symmetric = TRUE,
                      clip_nonnegative = TRUE, ...) {
  if (!requireNamespace("fabletools", quietly = TRUE))
    stop("the TIMESTRAL fable model requires the fabletools package")
  if (missing(model) || !inherits(model, "timestral_model"))
    stop("`model` must be a timestral_model from timestral_load()")
  train_fn <- function(.data, specials, ...)
    .train_timestral(.data, model, symmetric, clip_nonnegative)
  cls <- fabletools::new_model_class(
    "timestral", train = train_fn, specials = fabletools::new_specials())
  fabletools::new_model_definition(cls, !!rlang::enquo(formula), ...)
}

## Registered for fabletools::forecast when fabletools is available.
forecast.model_timestral <- function(object, new_data, specials = NULL,
                                     bootstrap = FALSE, times = 5000, ...) {
  if (!requireNamespace("distributional", quietly = TRUE))
    stop("fable forecasts require the distributional package")
  h <- NROW(new_data)
  fc <- timestral_forecast(object$model, object$context, h = h,
                           symmetric = object$symmetric,
                           clip_nonnegative = object$clip_nonnegative)
  q <- matrix(fc$quantiles[1, , ], nrow = h)
  distributional::dist_quantile(lapply(seq_len(h), function(i) q[i, ]),
                                quantile = rep(list(fc$levels), h))
}

## Registered for fabletools::model_sum when fabletools is available.
model_sum.model_timestral <- function(x) "TIMESTRAL"

#' Convert a Timestral forecast to a fable forecast table
#'
#' Registered as an S3 method for `fabletools::as_fable()` when fabletools
#' is available; call it as `fabletools::as_fable(fc)`. Each variate
#' becomes a response whose distribution is the 21-quantile
#' [distributional::dist_quantile()] distribution from the forecast.
#'
#' @param x a `timestral_forecast` created from indexed input (`ts`,
#'   `zoo`, `xts` or `tsibble`), so a future time index is available
#' @param ... ignored
#' @return a `fbl_ts` forecast table
#' @examples
#' \dontrun{
#' m <- timestral_load("5M")
#' fc <- timestral_forecast(m, AirPassengers, h = 24)
#' fabletools::as_fable(fc)
#' }
as_fable.timestral_forecast <- function(x, ...) {
  if (is.null(x$future_index))
    stop("this forecast has no time index; create it from ts, zoo, xts ",
         "or tsibble input to convert it to a fable table")
  for (p in c("tsibble", "distributional", "fabletools"))
    if (!requireNamespace(p, quietly = TRUE))
      stop("converting to a fable table requires the ", p, " package")
  iv <- if (!is.null(x$index_var)) x$index_var else "index"
  B <- x$n_series
  vnames <- dimnames(x$quantiles)[[if (B == 1L) 1L else 2L]]
  if (is.null(vnames)) vnames <- paste0("V", seq_len(dim(x$quantiles)[2L - (B == 1L)]))
  snames <- dimnames(x$observed)[[1L]]
  rows <- vector("list", B)
  for (b in seq_len(B)) {
    dfb <- data.frame(time = x$future_index, stringsAsFactors = FALSE)
    names(dfb) <- iv
    if (B > 1L) dfb <- cbind(series = snames[b], dfb, stringsAsFactors = FALSE)
    for (v in seq_along(vnames)) {
      q <- if (B == 1L) matrix(x$quantiles[v, , ], nrow = x$h)
           else matrix(x$quantiles[b, v, , ], nrow = x$h)
      dfb[[vnames[v]]] <- distributional::dist_quantile(
        lapply(seq_len(x$h), function(i) q[i, ]),
        quantile = rep(list(x$levels), x$h))
    }
    rows[[b]] <- dfb
  }
  df <- do.call(rbind, rows)
  tsib <- tsibble::as_tsibble(df, index = iv,
                              key = if (B > 1L) "series" else NULL)
  fabletools::as_fable(tsib, response = vnames,
                       distribution = tidyselect::all_of(vnames))
}
