## Conditional S3 registration for Suggests-only generics (ggplot2,
## tsibble, fabletools). Methods are registered when the owning package
## is loaded, or immediately if it already is, so dispatch works
## regardless of library() order without hard dependencies.
.register_s3 <- function(pkg, generic, class, method) {
  register <- function()
    registerS3method(generic, class, method, envir = asNamespace(pkg))
  if (isNamespaceLoaded(pkg)) register()
  setHook(packageEvent(pkg, "onLoad"), function(...) register())
}

.onLoad <- function(libname, pkgname) {
  .register_s3("ggplot2", "autoplot", "timestral_forecast",
               autoplot.timestral_forecast)
  .register_s3("ggplot2", "autolayer", "timestral_forecast",
               autolayer.timestral_forecast)
  .register_s3("tsibble", "as_tsibble", "timestral_forecast",
               as_tsibble.timestral_forecast)
  .register_s3("fabletools", "forecast", "model_timestral",
               forecast.model_timestral)
  .register_s3("fabletools", "model_sum", "model_timestral",
               model_sum.model_timestral)
  .register_s3("fabletools", "as_fable", "timestral_forecast",
               as_fable.timestral_forecast)
}
