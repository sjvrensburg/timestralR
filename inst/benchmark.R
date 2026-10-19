## Dev-only benchmark: worst-case timings per size (context capped at 4096).
## TIMESTRALR_MODEL_DIR selects the checkpoint; run per size.
library(timestralR)
model_dir <- Sys.getenv("TIMESTRALR_MODEL_DIR", "/tmp/timestral-5m")
backend <- Sys.getenv("TIMESTRALR_BACKEND", "native")
cat("backend:", backend, "\n")
m <- timestral_load(model_dir, max_context = 4096L, backend = backend)
set.seed(42); n <- 700; tt <- 1:n
y1 <- 50 + 10 * sin(2 * pi * tt / 24) + cumsum(rnorm(n)); y1[c(5:40, 300, 301, 650)] <- NA
cases <- list(
  univariate_missing = list(y = y1, h = 96),
  long_horizon = list(y = y1[200:700], h = 1100),
  long_context_cap = list(y = 10 + sin(2 * pi * (1:5000) / 50) + rnorm(5000, sd = .1),
                          h = 2100),
  batch10 = list(y = replicate(10, y1, simplify = FALSE), h = 96)
)
for (nm in names(cases)) {
  cs <- cases[[nm]]
  t0 <- Sys.time()
  invisible(timestral_forecast(m, cs$y, cs$h))
  cat(sprintf("%-20s %6.2f s\n", nm,
              as.numeric(Sys.time() - t0, units = "secs")))
}
