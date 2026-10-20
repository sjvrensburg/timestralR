## Package tests, split into parts under tests/parts/ and sourced here in
## order (they share one environment and later parts use earlier objects).
## Golden-fixture tests always run against the tiny random-weight
## checkpoint in tests/fixtures (official-PyTorch outputs in golden_*.csv;
## regenerate with inst/make_fixtures.py). Full-checkpoint tests run only
## when TIMESTRALR_TEST_MODEL_DIR points at a real checkpoint.
##
## Optional-interface parts skip cleanly on local runs without the
## Suggests packages. In CI (CI=true), or when
## TIMESTRALR_REQUIRE_OPTIONAL=true, a missing optional package is an
## error instead: CI installs Suggests, so a skip there would silently
## drop coverage.
library(timestralR)

.test_dir <- if (dir.exists("parts")) "." else file.path("tests")
stopifnot(dir.exists(file.path(.test_dir, "parts")))

.part <- function(name) {
  source(file.path(.test_dir, "parts", name))
  invisible(NULL)
}

.part("01-golden-parity.R")
.part("02-forecast-core.R")
.part("03-regressions.R")

.optional_pkgs <- c("zoo", "xts", "tsibble", "ggplot2", "fable",
                    "fabletools", "distributional")
.missing_optional <- .optional_pkgs[
  !vapply(.optional_pkgs, requireNamespace, logical(1), quietly = TRUE)]
.require_optional <- identical(Sys.getenv("CI"), "true") ||
  identical(Sys.getenv("TIMESTRALR_REQUIRE_OPTIONAL"), "true")
if (.require_optional && length(.missing_optional))
  stop("optional-interface tests are required here but packages are not ",
       "available: ", paste(.missing_optional, collapse = ", "))

.part("04-optional-interfaces.R")
cat("fixture tests passed\n")
if (length(.missing_optional))
  message("skipping optional-interface tests; packages not available: ",
          paste(.missing_optional, collapse = ", "))

.part("05-full-checkpoint.R")
