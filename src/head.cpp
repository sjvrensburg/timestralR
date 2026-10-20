#include "head.h"

#include "nn_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace timestral {

namespace {

// Monotone accumulation from raw head projections: raw is (N, ps*Q)
// for one row's decode tokens; returns (N*ps, Q).
MatrixXd monotone_head_from_raw(const Session& s, const MatrixXd& raw) {
  const int ps = s.cfg.patch_size, Q = (int)s.cfg.knots.size();
  int N = raw.rows();
  int m = s.cfg.median_index();
  std::vector<double> widths(Q, 0.0);
  for (int i = m + 1; i < Q; ++i) widths[i] = s.cfg.knots[i] - s.cfg.knots[i - 1];
  for (int i = 0; i < m; ++i) widths[i] = s.cfg.knots[i + 1] - s.cfg.knots[i];
  MatrixXd out(N * ps, Q);
  for (int n = 0; n < N; ++n)
    for (int st = 0; st < ps; ++st) {
      Eigen::VectorXd rq(Q);
      for (int q = 0; q < Q; ++q) rq(q) = raw(n, st * Q + q);
      Eigen::VectorXd inc(Q);
      for (int q = 0; q < Q; ++q) inc(q) = s.cfg.quantile_width_factor * widths[q] * softplus(rq(q));
      Eigen::VectorXd res = rq;
      double acc = 0;
      for (int q = m + 1; q < Q; ++q) { acc += inc(q); res(q) = rq(m) + acc; }
      acc = 0;
      for (int q = m - 1; q >= 0; --q) { acc += inc(q); res(q) = rq(m) - acc; }
      res(m) = rq(m);
      out.row(n * ps + st) = res.transpose();
    }
  return out;
}

void clamp_nonfinite_rows(MatrixXd& q) {
  for (int i = 0; i < q.rows(); ++i) {
    bool any_fin = false; double mx = 0, mn = 0;
    for (int j = 0; j < q.cols(); ++j) if (std::isfinite(q(i, j))) {
      if (!any_fin) { mx = mn = q(i, j); any_fin = true; }
      mx = std::max(mx, q(i, j)); mn = std::min(mn, q(i, j));
    }
    for (int j = 0; j < q.cols(); ++j) {
      if (q(i, j) == std::numeric_limits<double>::infinity()) q(i, j) = any_fin ? mx : 0.0;
      else if (q(i, j) == -std::numeric_limits<double>::infinity()) q(i, j) = any_fin ? mn : 0.0;
      if (std::isinf(q(i, j))) q(i, j) = 0.0;
    }
  }
}

}  // namespace

std::vector<MatrixXd> patch_embed_range(const Session& s, const MatrixXd& scaled,
                                        const MaskMat& observed, int p0, int p1) {
  const int ps = s.cfg.patch_size;
  int R = scaled.rows(), N = p1 - p0;
  if (N <= 0) return std::vector<MatrixXd>(R, MatrixXd(0, s.cfg.d_model));
  MatrixXd tokens(R * N, 2 * ps);
  for (int r = 0; r < R; ++r)
    for (int p = 0; p < N; ++p)
      for (int k = 0; k < ps; ++k) {
        tokens(r * N + p, k) = scaled(r, (p0 + p) * ps + k);
        tokens(r * N + p, ps + k) = observed(r, (p0 + p) * ps + k) ? 0.0 : 1.0;
      }
  MatrixXd out = residual_block(s, tokens,
      "representation.patch_proj.hidden_layer.weight",
      "representation.patch_proj.output_layer.weight",
      "representation.patch_proj.residual_layer.weight");
  std::vector<MatrixXd> H(R, MatrixXd(N, s.cfg.d_model));
  for (int r = 0; r < R; ++r) H[r] = out.block(r * N, 0, N, s.cfg.d_model);
  return H;
}

std::vector<MatrixXd> patch_embed(const Session& s, const MatrixXd& scaled,
                                  const MaskMat& observed) {
  return patch_embed_range(s, scaled, observed, 0, scaled.cols() / s.cfg.patch_size);
}

std::vector<MatrixXd> forecast_tokens(const Session& s, int R, int n_patches) {
  std::vector<MatrixXd> out(R, MatrixXd(n_patches, s.cfg.d_model));
  for (int r = 0; r < R; ++r)
    for (int i = 0; i < n_patches; ++i) out[r].row(i) = s.vec("representation.forecast_query").transpose();
  return out;
}

std::vector<MatrixXd> monotone_head_batch(const Session& s,
                                          const std::vector<MatrixXd>& tokens_dec) {
  int R = (int)tokens_dec.size(), N = (int)tokens_dec[0].rows();
  MatrixXd raw = residual_block(s, stack_rows(tokens_dec),
                                "head.projection.hidden_layer.weight",
                                "head.projection.output_layer.weight",
                                "head.projection.residual_layer.weight");
  std::vector<MatrixXd> out(R);
  for (int r = 0; r < R; ++r)
    out[r] = monotone_head_from_raw(s, raw.block(r * N, 0, N, raw.cols()));
  return out;
}

std::vector<MatrixXd> denormalize(const Session& s, std::vector<MatrixXd> preds,
                                  const VectorXd& anchor_loc, const VectorXd& anchor_scale) {
  for (size_t r = 0; r < preds.size(); ++r) {
    if (s.cfg.use_asinh) preds[r] = preds[r].unaryExpr([](double v) { return std::sinh(v); });
    preds[r] = preds[r] * anchor_scale[r];
    preds[r] = preds[r].array() + anchor_loc[r];
    clamp_nonfinite_rows(preds[r]);
  }
  return preds;
}

}  // namespace timestral
