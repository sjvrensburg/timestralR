// R entry points of the native backend: build a session from the weights
// and config (.ts_load) and run one decode block (.ts_predict_block).
// Mirrors R/model.R + R/scaler.R + R/masks.R exactly (float64); the R
// implementation remains the reference and fallback.

#include "predict.h"
#include "session.h"

#include <memory>
#include <string>
#include <vector>

using namespace Rcpp;
using timestral::MaskMat;
using timestral::MatrixXd;
using timestral::RowMat;
using timestral::Session;
using timestral::VectorXd;

// [[Rcpp::export(.ts_load)]]
SEXP ts_load(List weights, List config) {
  auto sess = std::make_unique<Session>();
  CharacterVector names = weights.names();
  for (int i = 0; i < weights.size(); ++i) {
    std::string nm = as<std::string>(names[i]);
    SEXP elt = weights[i];
    if (Rf_isMatrix(elt)) {
      NumericMatrix m(elt);
      Eigen::Map<const MatrixXd> mapped(m.begin(), m.nrow(), m.ncol());
      sess->mats[nm] = RowMat(mapped);
    } else {
      NumericVector v(elt);
      sess->vecs[nm] = Eigen::Map<const VectorXd>(v.begin(), v.size());
    }
  }
  timestral::Cfg& c = sess->cfg;
  c.patch_size = as<int>(config["patch_size"]);
  c.d_model = as<int>(config["d_model"]);
  c.num_heads = as<int>(config["num_heads"]);
  c.mlp_hidden_dim = as<int>(config["mlp_hidden_dim"]);
  c.block_size = as<int>(config["block_size"]);
  c.block_schedule = as<std::vector<std::string>>(config["block_schedule"]);
  c.knots = as<std::vector<double>>(config["quantile_knots"]);
  c.rope_theta = as<double>(config["rope_theta"]);
  c.rope_partial_factor = as<double>(config["rope_partial_factor"]);
  c.norm_eps = as<double>(config["norm_eps"]);
  c.minimum_scale = as<double>(config["minimum_scale"]);
  c.quantile_width_factor = as<double>(config["quantile_width_factor"]);
  c.use_asinh = as<bool>(config["use_asinh"]);
  c.attend_missing = as<bool>(config["attend_missing_patches"]);
  c.covariate_enabled = as<bool>(config["covariate_enabled"]);
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
  if (horizon_patches < 1) stop("horizon_patches must be >= 1");
  int R = inputs.nrow(), T = inputs.ncol();
  if (observed.nrow() != R || observed.ncol() != T)
    stop("observed must have the same dimensions as inputs");
  Eigen::Map<const MatrixXd> in_map(inputs.begin(), R, T);
  MatrixXd in = in_map;
  MaskMat obs(R, T);
  for (int r = 0; r < R; ++r)
    for (int t = 0; t < T; ++t) obs(r, t) = observed(r, t) ? 1 : 0;
  std::vector<int> grp(R, 1);
  if (groups.isNotNull()) {
    IntegerVector g(groups.get());
    if (g.size() != R) stop("groups must have one entry per row of inputs");
    for (int r = 0; r < R; ++r) grp[r] = g[r];
  }
  std::vector<MatrixXd> preds;
  bool has_roles = role_ids.isNotNull();
  std::vector<int> roles;
  if (has_roles) {
    IntegerVector ri(role_ids.get());
    if (ri.size() != R) stop("role_ids must have one entry per row of inputs");
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
          kf(r, t) = k(r, t);
          kfm(r, t) = km(r, t) ? 1 : 0;
        }
    }
    full_in.rightCols(horizon_len) = kf;
    full_obs.rightCols(horizon_len) = kfm;
    preds = timestral::predict_window(s, full_in, full_obs, horizon_patches, roles, grp);
  }
  int steps = (int)preds[0].rows(), Q = (int)preds[0].cols();
  NumericVector out(R * steps * Q);
  out.attr("dim") = IntegerVector::create(R, steps, Q);
  for (int r = 0; r < R; ++r)
    for (int st = 0; st < steps; ++st)
      for (int q = 0; q < Q; ++q)
        out[r + R * (st + steps * q)] = preds[r](st, q);
  return out;
}
