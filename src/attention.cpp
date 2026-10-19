#include "attention.h"

#include "nn_ops.h"

#include <cmath>

namespace timestral {

namespace {

// Extract head h of q/k/v (idx 0/1/2) from a (S, 3D) QKV matrix.
MatrixXd head_slice(const MatrixXd& qkv, int idx, int h, int nh, int hd) {
  int S = qkv.rows(), D = nh * hd;
  MatrixXd out(S, hd);
  for (int d = 0; d < hd; ++d)
    out.col(d) = qkv.col(idx * D + d * nh + h);
  return out;
}

}  // namespace

void repack_qkv(const MatrixXd& qkv, int nh, int hd,
                MatrixXd& Q, MatrixXd& K, MatrixXd& V) {
  int M = qkv.rows(), D = nh * hd;
  Q.resize(M, D); K.resize(M, D); V.resize(M, D);
  for (int h = 0; h < nh; ++h) {
    Q.block(0, h * hd, M, hd) = head_slice(qkv, 0, h, nh, hd);
    K.block(0, h * hd, M, hd) = head_slice(qkv, 1, h, nh, hd);
    V.block(0, h * hd, M, hd) = head_slice(qkv, 2, h, nh, hd);
  }
}

void rope_inplace(MatrixXd& Z, const MatrixXd& cos_s, const MatrixXd& sin_s,
                  int nh, int hd) {
  int M = Z.rows();
  for (int h = 0; h < nh; ++h) {
    auto blk = Z.block(0, h * hd, M, hd);
    MatrixXd rot(M, hd);
    rot.leftCols(hd / 2) = -blk.rightCols(hd / 2);
    rot.rightCols(hd / 2) = blk.leftCols(hd / 2);
    blk = blk.cwiseProduct(cos_s) + rot.cwiseProduct(sin_s);
  }
}

void attn_time_stacked(const Session& s, const MatrixXd& Q, const MatrixXd& K,
                       const MatrixXd& V, const std::vector<MatrixXd>& bias,
                       MatrixXd& o_s, KVCache* cache, int layer, int pos0,
                       int n_enc_active, int R, int S) {
  int nh = s.cfg.num_heads, hd = s.cfg.head_dim();
  int S_total = pos0 + S;
  double scale = 1.0 / std::sqrt((double)hd);
  for (int r = 0; r < R; ++r) {
    for (int h = 0; h < nh; ++h) {
      auto q = Q.block(r * S, h * hd, S, hd);
      auto k = K.block(r * S, h * hd, S, hd);
      auto v = V.block(r * S, h * hd, S, hd);
      if (cache && n_enc_active > 0) {
        cache->K[layer][r].block(pos0, h * hd, n_enc_active, hd) =
            K.block(r * S, h * hd, n_enc_active, hd);
        cache->V[layer][r].block(pos0, h * hd, n_enc_active, hd) =
            V.block(r * S, h * hd, n_enc_active, hd);
      }
      if (cache && pos0 > 0) {
        MatrixXd kall(S_total, hd), vall(S_total, hd);
        kall.topRows(pos0) = cache->K[layer][r].block(0, h * hd, pos0, hd);
        kall.bottomRows(S) = k;
        vall.topRows(pos0) = cache->V[layer][r].block(0, h * hd, pos0, hd);
        vall.bottomRows(S) = v;
        MatrixXd scores = (q * kall.transpose()) * scale;
        o_s.block(r * S, h * hd, S, hd) = softmax_bias(scores, &bias[r]) * vall;
      } else {
        MatrixXd scores = (q * k.transpose()) * scale;
        o_s.block(r * S, h * hd, S, hd) = softmax_bias(scores, &bias[r]) * v;
      }
    }
  }
}

void attn_variate_stacked(const Session& s, const MatrixXd& Q, const MatrixXd& K,
                          const MatrixXd& V, const std::vector<MatrixXd>& var_bias,
                          MatrixXd& o_v, int pos0, int R, int S) {
  int nh = s.cfg.num_heads, hd = s.cfg.head_dim();
  double scale = 1.0 / std::sqrt((double)hd);
  for (int sl = 0; sl < S; ++sl) {
    const MatrixXd* bias = var_bias.empty() ? nullptr : &var_bias[pos0 + sl];
    for (int h = 0; h < nh; ++h) {
      auto q = Q.block(sl * R, h * hd, R, hd);
      auto k = K.block(sl * R, h * hd, R, hd);
      auto v = V.block(sl * R, h * hd, R, hd);
      MatrixXd scores = (q * k.transpose()) * scale;
      o_v.block(sl * R, h * hd, R, hd) = softmax_bias(scores, bias) * v;
    }
  }
}

MatrixXd attn_finish(const Session& s, MatrixXd o, const MatrixXd& xn_stacked,
                     const std::string& prefix) {
  if (s.has_mat(prefix + "._gate.weight")) {
    MatrixXd g = lin(s, xn_stacked, prefix + "._gate.weight");
    o = o.cwiseProduct(g.unaryExpr([](double v) { return sigmoid(v); }));
  }
  return lin(s, o, prefix + ".wO.weight");
}

}  // namespace timestral
