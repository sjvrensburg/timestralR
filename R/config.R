## Port of timestral/config.py: defaults, validation, derived properties.

.config_defaults <- list(
  rope_theta = 10000.0, rope_partial_factor = 0.5, norm_eps = 1e-6,
  attn_bias = FALSE, mlp_bias = FALSE,
  time_attn_gate = "sdpa_sigmoid", variate_attn_gate = "sdpa_sigmoid",
  minimum_scale = 1e-6, use_asinh = TRUE, stabilize_with_global = FALSE,
  covariate_enabled = TRUE, split_context_horizon_attn = TRUE,
  attend_missing_patches = TRUE, quantile_width_factor = 2.0
)

.config_fields <- c("patch_size", "d_model", "num_heads", "mlp_hidden_dim",
                    "block_schedule", "quantile_knots", "block_size",
                    "context_length", names(.config_defaults))

.layer_kind_axes <- list(time = TRUE, variate = FALSE,
                         time_variate = c(TRUE, FALSE),
                         variate_time = c(FALSE, TRUE))

.validate_config <- function(cfg) {
  unknown <- setdiff(names(cfg), .config_fields)
  if (length(unknown))
    stop("unknown config keys: ", paste(sort(unknown), collapse = ", "))
  for (nm in names(.config_defaults))
    if (is.null(cfg[[nm]])) cfg[[nm]] <- .config_defaults[[nm]]
  for (nm in c("patch_size", "d_model", "num_heads", "mlp_hidden_dim",
               "block_schedule", "quantile_knots", "block_size",
               "context_length")) {
    if (is.null(cfg[[nm]]))
      stop("config is missing required field '", nm, "'")
  }
  for (nm in c("patch_size", "d_model", "num_heads", "mlp_hidden_dim",
               "block_size", "context_length")) {
    v <- cfg[[nm]]
    if (length(v) != 1L || !is.numeric(v) || !is.finite(v) ||
        v != floor(v) || v < 1)
      stop("config field '", nm, "' must be a positive integer, got ",
           paste(format(v), collapse = ", "))
    cfg[[nm]] <- as.integer(v)
  }
  cfg$quantile_knots <- as.numeric(cfg$quantile_knots)
  if (length(cfg$quantile_knots) == 0L || any(!is.finite(cfg$quantile_knots)) ||
      any(cfg$quantile_knots <= 0 | cfg$quantile_knots >= 1))
    stop("config field 'quantile_knots' must hold finite values in (0, 1)")
  cfg$block_schedule <- as.character(cfg$block_schedule)
  if (length(cfg$block_schedule) == 0L)
    stop("config field 'block_schedule' must name at least one layer")
  for (nm in c("rope_theta", "norm_eps", "minimum_scale",
               "quantile_width_factor", "rope_partial_factor")) {
    v <- cfg[[nm]]
    if (length(v) != 1L || !is.numeric(v) || !is.finite(v))
      stop("config field '", nm, "' must be a single finite number")
  }
  if (cfg$rope_theta <= 0) stop("config field 'rope_theta' must be > 0")
  if (cfg$norm_eps <= 0) stop("config field 'norm_eps' must be > 0")
  if (cfg$minimum_scale < 0) stop("config field 'minimum_scale' must be >= 0")
  if (cfg$quantile_width_factor < 0)
    stop("config field 'quantile_width_factor' must be >= 0")
  if (cfg$d_model %% cfg$num_heads != 0)
    stop("d_model=", cfg$d_model, " not divisible by num_heads=", cfg$num_heads)
  hd <- cfg$d_model %/% cfg$num_heads
  rope_dim <- as.integer(hd * cfg$rope_partial_factor)
  if (rope_dim <= 0 || rope_dim %% 2 != 0 || rope_dim > hd)
    stop("rope_partial_factor=", cfg$rope_partial_factor,
         " gives an invalid rope_dim=", rope_dim,
         " (must be positive, even, and <= head_dim=", hd, ")")
  bad_kind <- setdiff(cfg$block_schedule, names(.layer_kind_axes))
  if (length(bad_kind))
    stop("unknown layer kind ", bad_kind[1], "; expected one of ",
         paste(names(.layer_kind_axes), collapse = ", "))
  for (g in c(cfg$time_attn_gate, cfg$variate_attn_gate))
    if (!g %in% c("none", "sdpa_sigmoid"))
      stop("unknown attention gate ", g)
  knots <- cfg$quantile_knots
  if (!0.5 %in% knots) stop("quantile_knots must contain 0.5")
  if (any(diff(knots) <= 0)) stop("quantile_knots must be increasing")
  if (isTRUE(cfg$stabilize_with_global))
    stop("the advancing refine frame needs stabilize_with_global=FALSE")
  cfg$head_dim <- hd
  cfg$rope_dim <- rope_dim
  cfg$num_quantiles <- length(knots)
  cfg$median_index <- which(knots == 0.5)
  cfg$supports_known_covariates <- .supports_known_covariates(cfg)
  cfg
}

.supports_known_covariates <- function(cfg) {
  if (!isTRUE(cfg$covariate_enabled)) return(FALSE)
  ops <- unlist(.layer_kind_axes[cfg$block_schedule], use.names = FALSE)
  for (i in seq_along(ops)) {
    rest <- if (i < length(ops)) ops[(i + 1):length(ops)] else logical(0)
    if (ops[i] && length(rest) && !all(rest)) return(TRUE)
  }
  FALSE
}

## Expected tensor names/shapes from the config (Python strict=True loading).
.expected_tensors <- function(cfg) {
  D <- cfg$d_model; H <- cfg$mlp_hidden_dim; ps <- cfg$patch_size
  Q <- cfg$num_quantiles
  exp <- list()
  add <- function(name, shape) exp[[name]] <<- shape
  add("representation.patch_proj.hidden_layer.weight", c(4 * D, 2 * ps))
  add("representation.patch_proj.hidden_layer.bias", 4 * D)
  add("representation.patch_proj.output_layer.weight", c(D, 4 * D))
  add("representation.patch_proj.output_layer.bias", D)
  add("representation.patch_proj.residual_layer.weight", c(D, 2 * ps))
  add("representation.patch_proj.residual_layer.bias", D)
  add("representation.forecast_query", D)
  add("head.projection.hidden_layer.weight", c(4 * D, D))
  add("head.projection.output_layer.weight", c(ps * Q, 4 * D))
  add("head.projection.residual_layer.weight", c(ps * Q, D))
  add("transformer.out_norm.scale", D)
  for (li in seq_along(cfg$block_schedule)) {
    axes <- .layer_kind_axes[[cfg$block_schedule[li]]]
    lp <- sprintf("transformer.layers.%d", li - 1L)
    for (ai in seq_along(axes)) {
      ap <- sprintf("%s.attns.%d", lp, ai - 1L)
      add(sprintf("%s.attn_norms.%d.scale", lp, ai - 1L), D)
      add(paste0(ap, ".wQKV.weight"), c(3 * D, D))
      add(paste0(ap, ".wO.weight"), c(D, D))
      gate <- if (axes[ai]) cfg$time_attn_gate else cfg$variate_attn_gate
      if (gate == "sdpa_sigmoid") add(paste0(ap, "._gate.weight"), c(D, D))
      if (isTRUE(cfg$attn_bias)) {
        add(paste0(ap, ".wQKV.bias"), 3 * D)
        add(paste0(ap, ".wO.bias"), D)
        if (gate == "sdpa_sigmoid") add(paste0(ap, "._gate.bias"), D)
      }
    }
    add(paste0(lp, ".mlp_norm.scale"), D)
    add(paste0(lp, ".mlp.w12.weight"), c(2 * H, D))
    add(paste0(lp, ".mlp.w3.weight"), c(D, H))
    if (isTRUE(cfg$mlp_bias)) {
      add(paste0(lp, ".mlp.w12.bias"), 2 * H)
      add(paste0(lp, ".mlp.w3.bias"), D)
    }
  }
  exp
}

.check_weights <- function(weights, cfg) {
  exp <- .expected_tensors(cfg)
  missing <- setdiff(names(exp), names(weights))
  extra <- setdiff(names(weights), names(exp))
  if (length(missing))
    stop("missing weight tensors: ", paste(sort(missing), collapse = ", "))
  if (length(extra))
    stop("unexpected weight tensors: ", paste(sort(extra), collapse = ", "))
  for (nm in names(exp)) {
    got <- dim(weights[[nm]])
    if (is.null(got)) got <- length(weights[[nm]])
    if (!identical(as.integer(got), as.integer(exp[[nm]])))
      stop("weight tensor ", nm, " has shape ", paste(got, collapse = "x"),
           ", expected ", paste(exp[[nm]], collapse = "x"))
  }
  invisible(TRUE)
}
