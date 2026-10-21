// R entry points of the native backend: build a session from the weights
// and config (.ts_load) and run one decode block (.ts_predict_block).
// Mirrors R/model.R + R/scaler.R + R/masks.R exactly (float64); the R
// implementation remains the reference and fallback.

#include "predict.h"
#include "session.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace Rcpp;
using timestral::MaskMat;
using timestral::MatrixXd;
using timestral::RowMat;
using timestral::Session;
using timestral::VectorXd;

namespace {

// Mirror of the numeric ceiling enforced in R (.max_horizon).
constexpr int kMaxSteps = 10000;

SEXP config_field(const List& config, const char* name) {
  if (!config.containsElementNamed(name))
    stop("config is missing required field '%s'", name);
  return config[name];
}

int config_int(const List& config, const char* name) {
  SEXP v = config_field(config, name);
  if (!Rf_isNumeric(v) || Rf_length(v) != 1)
    stop("config field '%s' must be a single finite integer", name);
  double d = as<double>(v);
  if (!std::isfinite(d) || d != std::floor(d) ||
      d < std::numeric_limits<int>::min() || d > std::numeric_limits<int>::max())
    stop("config field '%s' must be a single finite integer", name);
  return static_cast<int>(d);
}

double config_double(const List& config, const char* name) {
  SEXP v = config_field(config, name);
  if (!Rf_isNumeric(v) || Rf_length(v) != 1)
    stop("config field '%s' must be a single finite number", name);
  double d = as<double>(v);
  if (!std::isfinite(d))
    stop("config field '%s' must be a single finite number", name);
  return d;
}

bool config_bool(const List& config, const char* name) {
  SEXP v = config_field(config, name);
  if (!Rf_isLogical(v) || Rf_length(v) != 1 || as<LogicalVector>(v)[0] == NA_LOGICAL)
    stop("config field '%s' must be TRUE or FALSE", name);
  return as<bool>(v);
}

// Re-validate everything the native code divides by, indexes with or
// allocates from. R's .validate_config() already checks these on the
// public path, but .ts_load is a separate entry point (reachable via
// :::) and must not trust its caller: a zero num_heads or patch_size
// used to mean division by zero / silent out-of-bounds reads here.
void validate_cfg(const timestral::Cfg& c) {
  if (c.patch_size < 1) stop("config field 'patch_size' must be >= 1, got %d", c.patch_size);
  if (c.d_model < 1) stop("config field 'd_model' must be >= 1, got %d", c.d_model);
  if (c.num_heads < 1) stop("config field 'num_heads' must be >= 1, got %d", c.num_heads);
  if (c.mlp_hidden_dim < 1)
    stop("config field 'mlp_hidden_dim' must be >= 1, got %d", c.mlp_hidden_dim);
  if (c.block_size < 1) stop("config field 'block_size' must be >= 1, got %d", c.block_size);
  if (c.d_model % c.num_heads != 0)
    stop("d_model=%d not divisible by num_heads=%d", c.d_model, c.num_heads);
  const int hd = c.head_dim();
  const int rd = c.rope_dim();
  if (rd <= 0 || rd % 2 != 0 || rd > hd)
    stop("rope_partial_factor gives an invalid rope_dim=%d (must be positive, "
         "even, and <= head_dim=%d)", rd, hd);
  if (c.block_schedule.empty())
    stop("config field 'block_schedule' must name at least one layer");
  for (const std::string& kind : c.block_schedule)
    if (kind != "time" && kind != "variate" && kind != "time_variate" &&
        kind != "variate_time")
      stop("unknown layer kind '%s' in block_schedule", kind.c_str());
  if (c.knots.empty())
    stop("config field 'quantile_knots' must not be empty");
  bool has_median = false;
  for (size_t i = 0; i < c.knots.size(); ++i) {
    double k = c.knots[i];
    if (!std::isfinite(k) || k <= 0.0 || k >= 1.0)
      stop("config field 'quantile_knots' must hold finite values in (0, 1)");
    if (i > 0 && !(k > c.knots[i - 1]))
      stop("config field 'quantile_knots' must be strictly increasing");
    if (k == 0.5) has_median = true;
  }
  if (!has_median) stop("config field 'quantile_knots' must contain 0.5");
  if (!(c.rope_theta > 0)) stop("config field 'rope_theta' must be > 0");
  if (!(c.norm_eps > 0)) stop("config field 'norm_eps' must be > 0");
  if (c.minimum_scale < 0) stop("config field 'minimum_scale' must be >= 0");
  if (c.quantile_width_factor < 0)
    stop("config field 'quantile_width_factor' must be >= 0");
}

}  // namespace

// [[Rcpp::export(.ts_load)]]
SEXP ts_load(List weights, List config) {
  auto sess = std::make_unique<Session>();
  if (weights.size() == 0)
    stop("weights must be a non-empty named list");
  SEXP names_sexp = Rf_getAttrib(weights, R_NamesSymbol);
  if (names_sexp == R_NilValue)
    stop("weights must be a non-empty named list");
  CharacterVector names(names_sexp);
  for (int i = 0; i < weights.size(); ++i) {
    std::string nm = as<std::string>(names[i]);
    if (nm.empty()) stop("every weight must have a non-empty name");
    SEXP elt = weights[i];
    if (!Rf_isNumeric(elt))
      stop("weight '%s' must be numeric", nm.c_str());
    if (Rf_isMatrix(elt)) {
      NumericMatrix m(elt);
      if (m.nrow() < 1 || m.ncol() < 1)
        stop("weight '%s' must be non-empty", nm.c_str());
      Eigen::Map<const MatrixXd> mapped(m.begin(), m.nrow(), m.ncol());
      sess->mats[nm] = RowMat(mapped);
    } else {
      NumericVector v(elt);
      if (v.size() < 1)
        stop("weight '%s' must be non-empty", nm.c_str());
      sess->vecs[nm] = Eigen::Map<const VectorXd>(v.begin(), v.size());
    }
  }
  timestral::Cfg& c = sess->cfg;
  c.patch_size = config_int(config, "patch_size");
  c.d_model = config_int(config, "d_model");
  c.num_heads = config_int(config, "num_heads");
  c.mlp_hidden_dim = config_int(config, "mlp_hidden_dim");
  c.block_size = config_int(config, "block_size");
  {
    SEXP v = config_field(config, "block_schedule");
    if (!Rf_isString(v) || Rf_length(v) < 1)
      stop("config field 'block_schedule' must name at least one layer");
    c.block_schedule = as<std::vector<std::string>>(v);
  }
  {
    SEXP v = config_field(config, "quantile_knots");
    if (!Rf_isNumeric(v) || Rf_length(v) < 1)
      stop("config field 'quantile_knots' must not be empty");
    c.knots = as<std::vector<double>>(v);
  }
  c.rope_theta = config_double(config, "rope_theta");
  c.rope_partial_factor = config_double(config, "rope_partial_factor");
  c.norm_eps = config_double(config, "norm_eps");
  c.minimum_scale = config_double(config, "minimum_scale");
  c.quantile_width_factor = config_double(config, "quantile_width_factor");
  c.use_asinh = config_bool(config, "use_asinh");
  c.attend_missing = config_bool(config, "attend_missing_patches");
  c.covariate_enabled = config_bool(config, "covariate_enabled");
  validate_cfg(c);
  XPtr<Session> ptr(sess.release(), true);
  ptr.attr("class") = "timestral_session";
  return ptr;
}

// [[Rcpp::export(.ts_predict_block)]]
NumericVector ts_predict_block(SEXP session, NumericMatrix inputs, LogicalMatrix observed,
                               int horizon_patches, Nullable<IntegerVector> role_ids,
                               Nullable<NumericMatrix> known_future,
                               Nullable<LogicalMatrix> known_future_mask,
                               Nullable<IntegerVector> groups) {
  XPtr<Session> sess(session);
  if (!sess) stop("invalid or released Timestral session");
  const Session& s = *sess;
  if (s.cfg.patch_size < 1)
    stop("config field 'patch_size' must be >= 1, got %d", s.cfg.patch_size);
  if (horizon_patches < 1) stop("horizon_patches must be >= 1");
  if (horizon_patches > (kMaxSteps + s.cfg.patch_size - 1) / s.cfg.patch_size)
    stop("horizon_patches (%d) would produce more than %d forecast steps",
         horizon_patches, kMaxSteps);
  int R = inputs.nrow(), T = inputs.ncol();
  if (R < 1) stop("inputs must have at least one row (R >= 1), got %d", R);
  if (T < s.cfg.patch_size)
    stop("inputs must have at least patch_size (%d) columns, got %d",
         s.cfg.patch_size, T);
  if (T % s.cfg.patch_size != 0)
    stop("the number of columns of inputs (%d) must be a multiple of "
         "patch_size (%d)", T, s.cfg.patch_size);
  if (observed.nrow() != R || observed.ncol() != T)
    stop("observed must have the same dimensions as inputs");
  Eigen::Map<const MatrixXd> in_map(inputs.begin(), R, T);
  MatrixXd in = in_map;
  MaskMat obs(R, T);
  for (int r = 0; r < R; ++r)
    for (int t = 0; t < T; ++t) {
      if (observed(r, t) == NA_LOGICAL) stop("observed must not contain NA");
      obs(r, t) = observed(r, t) ? 1 : 0;
      if (obs(r, t) && !std::isfinite(in(r, t)))
        stop("inputs must be finite where observed is TRUE");
    }
  std::vector<int> grp(R, 1);
  if (groups.isNotNull()) {
    IntegerVector g(groups.get());
    if (g.size() != R) stop("groups must have one entry per row of inputs");
    for (int r = 0; r < R; ++r) {
      if (g[r] == NA_INTEGER) stop("groups must not contain NA");
      grp[r] = g[r];
    }
  }
  std::vector<MatrixXd> preds;
  bool has_roles = role_ids.isNotNull();
  std::vector<int> roles;
  if (has_roles) {
    IntegerVector ri(role_ids.get());
    if (ri.size() != R) stop("role_ids must have one entry per row of inputs");
    for (int r = 0; r < R; ++r) {
      if (ri[r] == NA_INTEGER) stop("role_ids must not contain NA");
      if (ri[r] < 0 || ri[r] > 2)
        stop("role_ids must have values in {0, 1, 2}, got %d", ri[r]);
    }
    roles.assign(ri.begin(), ri.end());
  }
  bool all_target = !has_roles;
  if (has_roles) {
    all_target = true;
    for (int v : roles) if (v != 0) all_target = false;
  }
  if (all_target || !s.cfg.covariate_enabled) {
    preds = timestral::predict_context_only(s, in, obs, horizon_patches, grp);
  } else {
    if (horizon_patches > s.cfg.block_size)
      stop("decode block must be <= block_size with covariates");
    int horizon_len = horizon_patches * s.cfg.patch_size;
    MatrixXd full_in(R, T + horizon_len), kf(R, horizon_len);
    MaskMat full_obs(R, T + horizon_len), kfm(R, horizon_len);
    full_in.leftCols(T) = in;
    full_obs.leftCols(T) = obs;
    kf.setZero(); kfm.setZero();
    if (known_future.isNotNull()) {
      if (known_future_mask.isNull())
        stop("known_future_mask must be supplied with known_future");
      NumericMatrix k(known_future.get());
      LogicalMatrix km(known_future_mask.get());
      if (k.nrow() != R || k.ncol() != horizon_len ||
          km.nrow() != R || km.ncol() != horizon_len)
        stop("known_future and known_future_mask must be inputs rows x ",
             "horizon steps");
      for (int r = 0; r < R; ++r)
        for (int t = 0; t < horizon_len; ++t) {
          if (km(r, t) == NA_LOGICAL)
            stop("known_future_mask must not contain NA");
          kf(r, t) = k(r, t);
          kfm(r, t) = km(r, t) ? 1 : 0;
          if (kfm(r, t) && !std::isfinite(kf(r, t)))
            stop("known_future must be finite where known_future_mask is TRUE");
        }
    }
    full_in.rightCols(horizon_len) = kf;
    full_obs.rightCols(horizon_len) = kfm;
    preds = timestral::predict_window(s, full_in, full_obs, horizon_patches, roles, grp);
  }
  if (preds.empty() || preds[0].rows() < 1 || preds[0].cols() < 1)
    stop("native predict produced no output");
  int steps = (int)preds[0].rows(), Q = (int)preds[0].cols();
  NumericVector out(R * steps * Q);
  out.attr("dim") = IntegerVector::create(R, steps, Q);
  for (int r = 0; r < R; ++r)
    for (int st = 0; st < steps; ++st)
      for (int q = 0; q < Q; ++q)
        out[r + R * (st + steps * q)] = preds[r](st, q);
  return out;
}

// K/V cache counters of a session: context-only calls that used the
// cache, how many of them reused patches, and the reused patch total.
// [[Rcpp::export(.ts_cache_stats)]]
List ts_cache_stats(SEXP session) {
  XPtr<Session> sess(session);
  if (!sess) stop("invalid or released Timestral session");
  return List::create(_["calls"] = sess->cache_calls, _["hits"] = sess->cache_hits,
                      _["patches_reused"] = sess->cache_patches_reused);
}
