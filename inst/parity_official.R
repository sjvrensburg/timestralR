## Dev-only parity check against the official PyTorch package.
## Requires: a local checkpoint dir, Python + torch + timestral on the dev
## machine only (end users never need Python). Set before running:
##   TIMESTRALR_MODEL_DIR  directory with config.json + model.safetensors
##   TIMESTRALR_PYTHON     python executable (default: .venv/bin/python)
##   TIMESTRALR_CASES      comma-separated case names (default: all)
library(timestralR)
library(jsonlite)

model_dir <- Sys.getenv("TIMESTRALR_MODEL_DIR", "/tmp/timestral-5m")
python <- Sys.getenv("TIMESTRALR_PYTHON", ".venv/bin/python")
stopifnot(file.exists(file.path(model_dir, "model.safetensors")),
          file.exists(python))

source_cases <- function() {
  set.seed(42); n <- 700; tt <- 1:n; cases <- list()
  y1 <- 50 + 10 * sin(2 * pi * tt / 24) + cumsum(rnorm(n)); y1[c(5:40, 300, 301, 650)] <- NA
  cases$univariate_missing <- list(y = y1, h = 96)
  cases$negative_trend_nosym <- list(y = -5 + 0.02 * tt + rnorm(n), h = 30, sym = FALSE)
  cases$multivariate <- list(y = cbind(a = 20 + 5 * sin(2 * pi * tt / 48) + rnorm(n),
                                       b = 5 + 2 * cos(2 * pi * tt / 48) + rnorm(n)), h = 64)
  temp <- sin(2 * pi * (1:(n + 96)) / 24); promo <- as.numeric((1:(n + 96)) %% 7 == 0)
  cases$covariates <- list(y = 100 + 20 * temp[1:n] + 30 * promo[1:n] + rnorm(n), h = 96,
                           past = matrix(rnorm(n), ncol = 1), known = cbind(temp, promo))
  cases$long_horizon <- list(y = y1[200:700], h = 1100)
  cases$long_context_cap <- list(y = 10 + sin(2 * pi * (1:5000) / 50) + rnorm(5000, sd = .1),
                                 h = 2100)
  ## batch of 3 series (list form), mixed missingness
  mk <- function(off, phase) { z <- off + 5 * sin(2 * pi * tt / 48 + phase) + cumsum(rnorm(n, sd = .3)); z }
  b1 <- mk(30, 0); b1[100:120] <- NA
  b2 <- mk(5, 1); b2[c(1:10, 400)] <- NA
  b3 <- mk(80, 2)
  cases$batch3 <- list(y = list(b1, b2, b3), h = 64)
  ## observation mask hides a finite stretch
  ym <- y1; om <- rep(TRUE, n); om[200:260] <- FALSE
  cases$obs_mask <- list(y = ym, h = 64, obs_mask = om)
  ## categorical covariates: weekly code affects y; unseen code (9) in horizon
  code_past <- rep(1:7, length.out = n)
  eff <- c(0, 5, -3, 2, 8, -5, 1)
  yc <- 40 + eff[code_past] + 3 * sin(2 * pi * tt / 24) + rnorm(n)
  code_known <- c(code_past, rep(1:7, length.out = 96))
  code_known[750] <- 9  # unseen category in the horizon
  cases$categorical_known <- list(y = yc, h = 96,
                                  cat_known = matrix(code_known, ncol = 1))
  cases$categorical_past <- list(y = yc, h = 48,
                                 cat_past = matrix(code_past, ncol = 1))
  ## categorical with multivariate target: codes pass through unencoded
  cases$categorical_multivariate <- list(
    y = cbind(yc, 10 + sin(2 * pi * tt / 24)), h = 48,
    cat_past = matrix(code_past, ncol = 1))
  ## non-default forecaster knobs
  cases$knobs <- list(y = y1, h = 1100, per_shot = 8, cap_k = 100)
  cases
}

## Convert an R-side input (any accepted form) to a flat Fortran-order
## payload + dims in (B, C, L) layout, using the package's own parsers.
payload <- function(arr3) {
  if (is.null(arr3)) return(NULL)
  list(flat = as.vector(arr3), dim = dim(arr3))
}
batch_arr <- function(x, B, T_len) {
  if (is.null(x)) return(NULL)
  lst <- timestralR:::.as_cov_batch(x, B, T_len, "case")
  arr <- array(0, c(B, nrow(lst[[1]]), T_len))
  for (b in seq_len(B)) arr[b, , ] <- lst[[b]]
  arr
}

cases <- source_cases()
only <- Sys.getenv("TIMESTRALR_CASES")
if (nzchar(only)) cases <- cases[strsplit(only, ",")[[1]]]
m <- timestral_load(model_dir, max_context = 4096L)
ref_py <- system.file("official_reference.py", package = "timestralR")
if (!nzchar(ref_py)) ref_py <- file.path("inst", "official_reference.py")
res <- list()
for (nm in names(cases)) {
  cs <- cases[[nm]]
  sym <- if (is.null(cs$sym)) TRUE else cs$sym
  mm <- if (!is.null(cs$per_shot) || !is.null(cs$cap_k)) {
    timestral_load(model_dir, max_context = 4096L,
                   per_shot_horizon_patches = if (is.null(cs$per_shot)) 32L else cs$per_shot,
                   quantile_real_cap_k = if (is.null(cs$cap_k)) 1e4 else cs$cap_k)
  } else m
  t0 <- Sys.time()
  fc <- timestral_forecast(mm, cs$y, cs$h, observation_mask = cs$obs_mask,
                           past_covariates = cs$past, known_covariates = cs$known,
                           categorical_past_covariates = cs$cat_past,
                           categorical_known_covariates = cs$cat_known,
                           symmetric = sym)
  t_r <- as.numeric(Sys.time() - t0, units = "secs")
  tg <- timestralR:::.as_batch(cs$y)
  B <- tg$B; T_len <- tg$T
  case <- list(target = as.vector(tg$arr), target_dim = dim(tg$arr), h = cs$h,
               sym = sym,
               obs_mask = if (is.null(cs$obs_mask)) NULL else as.vector(timestralR:::.as_batch(cs$obs_mask)$arr * 1.0),
               obs_mask_dim = if (is.null(cs$obs_mask)) NULL else dim(tg$arr),
               past = { a <- batch_arr(cs$past, B, T_len); if (is.null(a)) NULL else as.vector(a) },
               past_dim = { a <- batch_arr(cs$past, B, T_len); if (is.null(a)) NULL else dim(a) },
               known = { a <- batch_arr(cs$known, B, T_len + cs$h); if (is.null(a)) NULL else as.vector(a) },
               known_dim = { a <- batch_arr(cs$known, B, T_len + cs$h); if (is.null(a)) NULL else dim(a) },
               cat_past = { a <- batch_arr(cs$cat_past, B, T_len); if (is.null(a)) NULL else as.vector(a) },
               cat_past_dim = { a <- batch_arr(cs$cat_past, B, T_len); if (is.null(a)) NULL else dim(a) },
               cat_known = { a <- batch_arr(cs$cat_known, B, T_len + cs$h); if (is.null(a)) NULL else as.vector(a) },
               cat_known_dim = { a <- batch_arr(cs$cat_known, B, T_len + cs$h); if (is.null(a)) NULL else dim(a) },
               per_shot = if (is.null(cs$per_shot)) 32 else cs$per_shot,
               cap_k = if (is.null(cs$cap_k)) 1e4 else cs$cap_k)
  write_json(case, "case.json", auto_unbox = TRUE, digits = NA, na = "null", null = "null")
  stopifnot(system2(python, c(ref_py, model_dir, "case.json", "off.txt")) == 0)
  off_flat <- scan("off.txt", quiet = TRUE)
  q <- fc$quantiles
  ## fc (B=1) is (V,H,Q); official flat is (B,V,H,Q) Fortran order.
  q_cmp <- if (fc$n_series == 1L) as.vector(array(q, dim = c(1, dim(q)))) else as.vector(q)
  d <- abs(q_cmp - off_flat)
  scale <- max(abs(off_flat))
  res[[nm]] <- data.frame(case = nm, max_rel = signif(max(d) / scale, 3),
                          r_secs = round(t_r, 2))
  print(tail(res, 1)[[1]], row.names = FALSE)
}
out <- do.call(rbind, res)
cat("\nWorst max_rel:", max(out$max_rel), "\n")
## Gate: 2e-5 generally. Multi-block cases (horizon > 1024) feed medians
## back as context, which amplifies the official side's float32 rounding
## (the earlier ONNX fp32 report saw 5.1e-5 on 28M long_context_cap for
## the same reason), so those allow 1.5e-4.
multi <- out$case %in% c("long_horizon", "long_context_cap", "knobs")
fail <- (out$max_rel >= 2e-5 & !multi) | (out$max_rel >= 1.5e-4 & multi)
if (any(fail))
  stop("parity gate failed for: ", paste(out$case[fail], collapse = ", "))
