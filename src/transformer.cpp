#include "transformer.h"

#include "attention.h"
#include "nn_ops.h"

#include <cmath>
#include <string>
#include <unordered_map>

namespace timestral {

MatrixXd transformer(const Session& s, MatrixXd H,
                     const std::vector<MaskMat>& time_mask,
                     const std::vector<MaskMat>& variate_mask,
                     const MatrixXd& positions,
                     KVCache* cache, int pos0, int n_enc_active, int R) {
  const Cfg& cfg = s.cfg;
  int S = (int)H.rows() / R, hd = cfg.head_dim();
  int S_total = pos0 + S;
  int rope_dim = cfg.rope_dim();
  std::vector<double> inv_freq(hd / 2, 0.0);
  for (int i = 0, idx = 0; idx < rope_dim; idx += 2, ++i)
    inv_freq[i] = 1.0 / std::pow(cfg.rope_theta, (double)idx / rope_dim);
  MatrixXd cos_s(R * S, hd), sin_s(R * S, hd);
  for (int r = 0; r < R; ++r)
    for (int i = 0; i < S; ++i)
      for (int d = 0; d < hd; ++d) {
        double f = positions(r, pos0 + i) * inv_freq[d % (hd / 2)];
        cos_s(r * S + i, d) = std::cos(f);
        sin_s(r * S + i, d) = std::sin(f);
      }
  // Additive attention biases, converted once for all layers/heads.
  std::vector<MatrixXd> time_bias(R);
  for (int r = 0; r < R; ++r)
    time_bias[r] = mask_to_bias(time_mask[r].block(pos0, 0, S, S_total));
  std::vector<MatrixXd> var_bias;
  if (!variate_mask.empty()) {
    var_bias.resize(S_total);
    for (int sl = 0; sl < S_total; ++sl) var_bias[sl] = mask_to_bias(variate_mask[sl]);
  }
  int nh = cfg.num_heads;
  static const std::unordered_map<std::string, std::vector<bool>> axes_of = {
      {"time", {true}}, {"variate", {false}},
      {"time_variate", {true, false}}, {"variate_time", {false, true}}};
  for (size_t li = 0; li < cfg.block_schedule.size(); ++li) {
    auto axes = axes_of.at(cfg.block_schedule[li]);
    for (size_t ai = 0; ai < axes.size(); ++ai) {
      std::string ap = "transformer.layers." + std::to_string(li) + ".attns." + std::to_string(ai);
      std::string np = "transformer.layers." + std::to_string(li) + ".attn_norms." + std::to_string(ai);
      MatrixXd xn = rms_norm(H, s.vec(np + ".scale"), cfg.norm_eps);
      if (axes[ai]) {
        // One QKV GEMM over all rows, repacked to head-block layout;
        // attention itself stays per row/head on block views.
        MatrixXd qkv_s = lin(s, xn, ap + ".wQKV.weight");
        MatrixXd Q, K, V;
        repack_qkv(qkv_s, nh, hd, Q, K, V);
        rope_inplace(Q, cos_s, sin_s, nh, hd);
        rope_inplace(K, cos_s, sin_s, nh, hd);
        MatrixXd o_s(R * S, cfg.d_model);
        attn_time_stacked(s, Q, K, V, time_bias, o_s, cache, (int)li, pos0,
                          n_enc_active, R, S);
        H += attn_finish(s, o_s, xn, ap);
      } else {
        // Variate attention: permute tokens slot-major (s*R + r) so the
        // QKV projection is again one GEMM; attention stays per slot/head.
        MatrixXd xv(S * R, cfg.d_model);
        for (int sl = 0; sl < S; ++sl)
          for (int r = 0; r < R; ++r) xv.row(sl * R + r) = xn.row(r * S + sl);
        MatrixXd qkv_v = lin(s, xv, ap + ".wQKV.weight");
        MatrixXd Q, K, V;
        repack_qkv(qkv_v, nh, hd, Q, K, V);
        MatrixXd o_v(S * R, cfg.d_model);
        attn_variate_stacked(s, Q, K, V, var_bias, o_v, pos0, R, S);
        MatrixXd y = attn_finish(s, o_v, xv, ap);
        for (int sl = 0; sl < S; ++sl)
          for (int r = 0; r < R; ++r) H.row(r * S + sl) += y.row(sl * R + r);
      }
    }
    std::string mp = "transformer.layers." + std::to_string(li) + ".mlp";
    MatrixXd xn = rms_norm(H, s.vec(mp + "_norm.scale"), cfg.norm_eps);
    int hd2 = cfg.mlp_hidden_dim;
    MatrixXd gu = lin(s, xn, mp + ".w12.weight");
    MatrixXd gate = gu.leftCols(hd2);
    MatrixXd up = gu.rightCols(hd2);
    MatrixXd act = gate.cwiseProduct(up.unaryExpr([](double v) { return silu(v); }));
    H += lin(s, act, mp + ".w3.weight");
  }
  return rms_norm(H, s.vec("transformer.out_norm.scale"), cfg.norm_eps);
}

}  // namespace timestral
