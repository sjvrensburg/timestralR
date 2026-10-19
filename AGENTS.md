# AGENTS.md

## Commands

- `R CMD INSTALL .` — build the project
- `R CMD check --no-manual timestralR_0.3.0.tar.gz` — run the full test suite (build the tarball first with `R CMD build .`)
- `Rscript tests/timestralR-tests.R` — run the full test suite (forecast tests skip unless `TIMESTRALR_TEST_MODEL_DIR` points at a checkpoint directory)

## Code Map

- `R/` — application source
- `tests/` — automated tests
- `inst/` — project documentation
- `man/` — project documentation

## Conventions

- Keep the forward pass in base R on R's BLAS; do not add a Python, torch, or external-runtime dependency for end users.
- Keep parity checks in `inst/` as developer-only scripts; the package itself must run without Python.
