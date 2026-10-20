## Optional full-checkpoint tests, run only when TIMESTRALR_TEST_MODEL_DIR
## points at a real checkpoint; otherwise this part only messages the skip.
model_dir <- Sys.getenv("TIMESTRALR_TEST_MODEL_DIR")
if (!nzchar(model_dir)) {
  message("skipping full-checkpoint tests: TIMESTRALR_TEST_MODEL_DIR not set")
} else if (!file.exists(file.path(model_dir, "model.safetensors"))) {
  message("skipping full-checkpoint tests: no model.safetensors in ",
          "TIMESTRALR_TEST_MODEL_DIR (", model_dir, ")")
} else {
  mr <- timestral_load(model_dir, max_context = 4096L)
  set.seed(1)
  yy <- 10 + 3 * sin(2 * pi * (1:600) / 24)
  fcr <- timestral_forecast(mr, yy, h = 50)
  stopifnot(identical(dim(fcr$quantiles), c(1L, 50L, 21L)))
  stopifnot(mean(abs(fcr$quantiles[1, , "0.50"] -
                     (10 + 3 * sin(2 * pi * (601:650) / 24)))) < 0.5)
  cat("full-checkpoint tests passed\n")
}
