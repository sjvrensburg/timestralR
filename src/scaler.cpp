#include "scaler.h"

#include <algorithm>
#include <cmath>

namespace timestral {

ScaleOut model_scale(const Session& s, const MatrixXd& inputs, const MaskMat& observed) {
  const int R = inputs.rows(), T = inputs.cols(), ps = s.cfg.patch_size;
  MatrixXd means(R, T), scales(R, T);
  for (int r = 0; r < R; ++r) {
    double cum_w = 0, cum_x = 0, m2 = 0;
    std::vector<double> mean_t(T), scale_t(T);
    double prev_mean = 0;
    for (int t = 0; t < T; ++t) {
      double w = observed(r, t) ? 1.0 : 0.0;
      double x = inputs(r, t);
      cum_w += w;
      cum_x += w * x;
      double denom = std::max(cum_w, 1.0);
      double mean = cum_x / denom;
      double shifted = (t == 0) ? 0.0 : prev_mean;
      m2 += (x - shifted) * (x - mean) * w;
      double var = m2 / std::max(denom - 1.0, 1.0);
      mean_t[t] = mean;
      scale_t[t] = std::max(std::sqrt(var), s.cfg.minimum_scale);
      prev_mean = mean;
    }
    for (int t = 0; t < T; ++t) { means(r, t) = mean_t[t]; scales(r, t) = scale_t[t]; }
  }
  MatrixXd loc(R, T), scale(R, T);
  int npatch = T / ps;
  for (int p = 0; p < npatch; ++p) {
    int last = (p + 1) * ps - 1;
    for (int k = 0; k < ps; ++k) {
      loc.col(p * ps + k) = means.col(last);
      scale.col(p * ps + k) = scales.col(last);
    }
  }
  MatrixXd scaled(R, T);
  for (int r = 0; r < R; ++r)
    for (int t = 0; t < T; ++t) {
      if (!observed(r, t)) { scaled(r, t) = 0.0; continue; }
      double v = (inputs(r, t) - loc(r, t)) / std::max(scale(r, t), 1.1920929e-07);
      scaled(r, t) = s.cfg.use_asinh ? std::asinh(v) : v;
    }
  return {scaled, loc, scale};
}

void known_cov_scaling(const Session& s, const MatrixXd& inputs, const MaskMat& observed,
                       const std::vector<int>& role_ids, ScaleOut& sc) {
  const int R = inputs.rows();
  std::vector<int> known_rows;
  for (int r = 0; r < R; ++r) if (role_ids[r] == 2) known_rows.push_back(r);
  if (known_rows.empty()) return;
  for (int r = 0; r < R; ++r) {
    double n = 0, sum = 0;
    for (int t = 0; t < inputs.cols(); ++t) if (observed(r, t)) { n += 1; sum += inputs(r, t); }
    double nn = std::max(n, 1.0);
    double loc = sum / nn, var = 0;
    for (int t = 0; t < inputs.cols(); ++t) if (observed(r, t)) var += (inputs(r, t) - loc) * (inputs(r, t) - loc);
    var /= nn;
    double scale = std::max(std::sqrt(var), s.cfg.minimum_scale);
    bool is_binary = n > 0;
    for (int t = 0; t < inputs.cols(); ++t)
      if (observed(r, t) && inputs(r, t) != 0.0 && inputs(r, t) != 1.0) is_binary = false;
    if (is_binary) { loc = 0.0; scale = 1.0; }
    bool is_known = role_ids[r] == 2;
    if (!is_known) continue;
    for (int t = 0; t < inputs.cols(); ++t) {
      double v = observed(r, t) ? (inputs(r, t) - loc) / std::max(scale, 1.1920929e-07) : 0.0;
      if (observed(r, t) && s.cfg.use_asinh && !is_binary) v = std::asinh(v);
      sc.scaled(r, t) = v;
      sc.loc(r, t) = loc;
      sc.scale(r, t) = scale;
    }
  }
}

}  // namespace timestral
