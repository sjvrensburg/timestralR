## Golden fixtures, backend agreement, multi-block decode, parity fuzz.
## Sourced by tests/timestralR-tests.R; expects the package loaded.
## Defines the checkpoint path `tiny`, the shared fixture model `m`/`m_R`,
## inputs `y`/`y2` and forecast `fc`.
stopifnot(identical(length(timestral_levels()), 21L))
stopifnot(identical(timestral_levels()[11], 0.5))

fx <- file.path("fixtures")
if (!dir.exists(fx)) fx <- file.path("tests", "fixtures")
stopifnot(dir.exists(fx))
## the tiny random-weight checkpoint ships with the package (inst/extdata)
tiny <- system.file("extdata", "tiny", package = "timestralR")
stopifnot(nzchar(tiny))
m <- timestral_load(tiny, max_context = 4096L)

## golden: univariate
y <- scan(file.path(fx, "input_univariate.csv"), sep = ",", quiet = TRUE)
fc <- timestral_forecast(m, y, h = 24)
gold <- as.matrix(read.csv(file.path(fx, "golden_univariate.csv"), header = FALSE))
stopifnot(identical(dim(fc$quantiles), c(1L, 24L, 21L)))
stopifnot(max(abs(fc$quantiles[1, , ] - gold)) < 1e-5)
stopifnot(all(apply(fc$quantiles[1, , ], 1, function(r) all(diff(r) >= -1e-6))))

## golden: covariates
y2 <- scan(file.path(fx, "input_cov_target.csv"), sep = ",", quiet = TRUE)
pa <- scan(file.path(fx, "input_cov_past.csv"), sep = ",", quiet = TRUE)
kn <- scan(file.path(fx, "input_known.csv"), sep = ",", quiet = TRUE)
fc2 <- timestral_forecast(m, y2, h = 24, past_covariates = matrix(pa, ncol = 1),
                          known_covariates = matrix(kn, ncol = 1))
gold2 <- as.matrix(read.csv(file.path(fx, "golden_covariates.csv"), header = FALSE))
stopifnot(max(abs(fc2$quantiles[1, , ] - gold2)) < 1e-5)

## native and R backends agree on the fixtures
m_R <- timestral_load(tiny, max_context = 4096L, backend = "R")
fc_R <- timestral_forecast(m_R, y, h = 24)
stopifnot(max(abs(fc_R$quantiles - fc$quantiles)) < 1e-10)
fc2_R <- timestral_forecast(m_R, y2, h = 24, past_covariates = matrix(pa, ncol = 1),
                            known_covariates = matrix(kn, ncol = 1))
stopifnot(max(abs(fc2_R$quantiles - fc2$quantiles)) < 1e-10)

## multi-block decode (native K/V cache across blocks) agrees with R;
## the second variant truncates the context mid-decode, forcing cache misses
m_mb <- timestral_load(tiny, max_context = 4096L,
                       per_shot_horizon_patches = 1L)
m_mb_R <- timestral_load(tiny, max_context = 4096L,
                         per_shot_horizon_patches = 1L, backend = "R")
fc_mb <- timestral_forecast(m_mb, y, h = 48)
fc_mb_R <- timestral_forecast(m_mb_R, y, h = 48)
stopifnot(max(abs(fc_mb_R$quantiles - fc_mb$quantiles)) < 1e-10)
m_tr <- timestral_load(tiny, max_context = 32L,
                       per_shot_horizon_patches = 1L)
m_tr_R <- timestral_load(tiny, max_context = 32L,
                         per_shot_horizon_patches = 1L, backend = "R")
fc_tr <- timestral_forecast(m_tr, y, h = 48)
fc_tr_R <- timestral_forecast(m_tr_R, y, h = 48)
stopifnot(max(abs(fc_tr_R$quantiles - fc_tr$quantiles)) < 1e-10)

## seeded parity fuzz: native and R backends agree across randomised
## shapes, masks, covariates, batches and context truncation
set.seed(20261020)
.agree <- function(a, b, tol = 1e-8)
  stopifnot(max(abs(a$quantiles - b$quantiles)) < tol)
fz <- cumsum(rnorm(120)); fz[sample.int(120, 15)] <- NA
.agree(timestral_forecast(m, fz, h = 30), timestral_forecast(m_R, fz, h = 30))
fm <- cbind(cumsum(rnorm(90)), cumsum(rnorm(90)))
.agree(timestral_forecast(m, fm, h = 20), timestral_forecast(m_R, fm, h = 20))
fb <- list(cumsum(rnorm(70)), cumsum(rnorm(70)), cumsum(rnorm(70)))
.agree(timestral_forecast(m, fb, h = 18), timestral_forecast(m_R, fb, h = 18))
om_f <- sample(c(TRUE, FALSE), 120, replace = TRUE, prob = c(0.8, 0.2))
.agree(timestral_forecast(m, fz, h = 24, observation_mask = om_f),
       timestral_forecast(m_R, fz, h = 24, observation_mask = om_f))
fp <- matrix(rnorm(96), ncol = 1); fk <- matrix(rnorm(120), ncol = 1)
.agree(timestral_forecast(m, fz[1:96], h = 24,
                          past_covariates = fp, known_covariates = fk),
       timestral_forecast(m_R, fz[1:96], h = 24,
                          past_covariates = fp, known_covariates = fk))
fl <- cumsum(rnorm(300))
m_s <- timestral_load(tiny, max_context = 64L)
m_sR <- timestral_load(tiny, max_context = 64L, backend = "R")
.agree(timestral_forecast(m_s, fl, h = 40),
       timestral_forecast(m_sR, fl, h = 40))
