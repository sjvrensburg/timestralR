// Native (RcppEigen) implementation of the Timestral single-block forecast
// primitive. Mirrors R/model.R + R/scaler.R + R/masks.R exactly (float64);
// the R implementation remains the reference and fallback.
//
// Layout notes: safetensors weights are (out, in) row-major as in torch.
// They are stored here as RowMajor matrices; activations are default
// (column-major) Eigen matrices mapped zero-copy from R where possible.

#include <RcppEigen.h>
#include <R_ext/BLAS.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace Rcpp;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MaskMat = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

namespace {

struct Cfg {
  int patch_size = 32, d_model = 0, num_heads = 0, mlp_hidden_dim = 0;
  int block_size = 32;
  std::vector<std::string> block_schedule;
  std::vector<double> knots;
  double rope_theta = 10000.0, rope_partial_factor = 0.5, norm_eps = 1e-6;
  double minimum_scale = 1e-6, quantile_width_factor = 2.0;
  bool use_asinh = true, attend_missing = true, covariate_enabled = true;
  int head_dim() const { return d_model / num_heads; }
  int rope_dim() const { return static_cast<int>(head_dim() * rope_partial_factor); }
  int median_index() const {
    for (size_t i = 0; i < knots.size(); ++i) if (knots[i] == 0.5) return (int)i;
    return -1;
  }
};

// Cross-block K/V cache for the context-only decode path. The R decode
// loop (.block_ar_decode) issues one .ts_predict_block call per block with
// the median feedback appended to the context. Because the time mask is
// causal, the scaler is causal, and positions depend only on the first
// observed patch, an earlier context patch's per-layer K/V cannot change
// when later patches are appended; if the new inputs extend the cached
// inputs exactly (same rows/groups/first-observed), the cached K/V are
// reused and only the appended patches plus the new decode tokens are run
// through the transformer. K/V are stored per layer and row as (ll, D) in
// head-block layout, K after RoPE. Single-entry: an interleaved different
// sequence (e.g. the mirrored forecast) simply misses and rebuilds.
struct KVCache {
  bool valid = false;
  int T = 0, ll = 0;
  MatrixXd inputs;
  MaskMat observed;
  std::vector<int> groups;
  std::vector<int> first_obs;
  std::vector<std::vector<MatrixXd>> K, V;  // [layer][row]
};

struct Session {
  Cfg cfg;
  std::unordered_map<std::string, RowMat> mats;
  std::unordered_map<std::string, VectorXd> vecs;
  mutable KVCache cache;
  bool has_mat(const std::string& n) const { return mats.count(n) > 0; }
  bool has_vec(const std::string& n) const { return vecs.count(n) > 0; }
  const RowMat& mat(const std::string& n) const { return mats.at(n); }
  const VectorXd& vec(const std::string& n) const { return vecs.at(n); }
};

inline double silu(double x) { return x / (1.0 + std::exp(-x)); }
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(std::min(x, 20.0))); }

MatrixXd lin(const Session& s, const MatrixXd& X, const std::string& wname) {
  const RowMat& W = s.mat(wname);
  MatrixXd Y;
  // Hybrid GEMM: the system BLAS (FlexiBLAS/OpenBLAS on this machine) wins
  // on large products, Eigen's own kernels win on the many small ones.
  // Y = X * W^T; W's row-major buffer read as column-major is exactly W^T.
  const double flops = 2.0 * X.rows() * X.cols() * W.rows();
  if (flops >= 4.0e6) {
    Y.resize(X.rows(), W.rows());
    int m = (int)X.rows(), n = (int)W.rows(), k = (int)X.cols();
    double alpha = 1.0, beta = 0.0;
    F77_CALL(dgemm)("N", "N", &m, &n, &k, &alpha, X.data(), &m,
                    W.data(), &k, &beta, Y.data(), &m FCONE FCONE);
  } else {
    Y = X * W.transpose();
  }
  std::string bname = wname;
  size_t pos = bname.rfind(".weight");
  if (pos != std::string::npos) {
    bname.replace(pos, 7, ".bias");
    if (s.has_vec(bname)) Y.rowwise() += s.vec(bname).transpose();
  }
  return Y;
}

MatrixXd residual_block(const Session& s, const MatrixXd& X,
                        const std::string& hidden, const std::string& output,
                        const std::string& residual) {
  MatrixXd h = lin(s, X, hidden);
  h = h.unaryExpr([](double v) { return silu(v); });
  return lin(s, h, output) + lin(s, X, residual);
}

MatrixXd rms_norm(const MatrixXd& X, const VectorXd& scale, double eps) {
  MatrixXd Y(X.rows(), X.cols());
  for (int i = 0; i < X.rows(); ++i) {
    double ms = X.row(i).squaredNorm() / X.cols();
    Y.row(i) = (X.row(i) / std::sqrt(ms + eps)).cwiseProduct(scale.transpose());
  }
  return Y;
}

void rms_norm_rows(std::vector<MatrixXd>& H, const VectorXd& scale, double eps) {
  for (auto& m : H) m = rms_norm(m, scale, eps);
}

// ---- scaler (scaler.py) ----------------------------------------------------

struct ScaleOut { MatrixXd scaled, loc, scale; };

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

// known-covariate whole-row scaling; modifies scaled/loc/scale in place
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

// ---- masks (masks.py) -------------------------------------------------------

using MaskM = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

MaskM enc_key_validity(const MaskM& obs_patch, bool attend_missing) {
  int R = obs_patch.rows(), N = obs_patch.cols();
  MaskM out(R, N);
  for (int r = 0; r < R; ++r) {
    int started = 0;
    for (int j = 0; j < N; ++j) {
      started = std::max(started, obs_patch(r, j));
      out(r, j) = obs_patch(r, j) || (attend_missing && started);
    }
  }
  return out;
}

MaskM enc_causal_mask_row(const Eigen::VectorXi& kv) {
  int ll = kv.size();
  MaskM m = MaskM::Zero(ll, ll);
  for (int i = 0; i < ll; ++i)
    for (int j = 0; j < ll; ++j)
      m(i, j) = ((j <= i && kv(j)) || i == j) ? 1 : 0;
  return m;
}

// time mask per row: vector of (S,S)
std::vector<MaskM> forecast_time_mask(int ll, int h, const MaskM& key_valid) {
  int R = key_valid.rows(), S = ll + h;
  std::vector<MaskM> out(R, MaskM::Zero(S, S));
  for (int r = 0; r < R; ++r) {
    out[r].block(0, 0, ll, ll) = enc_causal_mask_row(key_valid.row(r).transpose());
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < ll; ++j) out[r](ll + i, j) = key_valid(r, j);
    for (int i = 0; i < h; ++i)
      for (int j = 0; j <= i; ++j) out[r](ll + i, ll + j) = 1;
  }
  return out;
}

std::vector<MaskM> two_stream_time_mask(const std::vector<int>& min_ctx, int n,
                                        int block_size, const MaskM& key_valid,
                                        const std::vector<int>& role_ids) {
  int R = key_valid.rows(), S = 2 * n;
  std::vector<MaskM> out(R, MaskM::Zero(S, S));
  for (int r = 0; r < R; ++r) {
    int s = min_ctx[r];
    std::vector<int> anchor(n), active(n);
    for (int p = 0; p < n; ++p) {
      active[p] = p >= s;
      int off = std::max(p - s, 0);
      anchor[p] = active[p] ? s + block_size * (off / block_size) : 0;
    }
    MaskM enc = enc_causal_mask_row(key_valid.row(r).transpose());
    bool bidir = !role_ids.empty() && role_ids[r] == 2;
    if (bidir) {
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) enc(i, j) = (key_valid(r, j) || i == j) ? 1 : 0;
    }
    out[r].block(0, 0, n, n) = enc;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        out[r](n + i, j) = (active[i] && j < anchor[i] && key_valid(r, j)) ? 1 : 0;
        out[r](n + i, n + j) = (((anchor[i] == anchor[j]) && j <= i && active[i] && active[j]) || i == j) ? 1 : 0;
      }
  }
  return out;
}

// variate mask per slot: vector of (R,R); empty when R == 1
std::vector<MaskM> variate_mask(const MaskM& key_valid, int n_dec,
                                const std::vector<int>& groups,
                                const std::vector<int>* target_rows,
                                const std::vector<int>* bidir_rows) {
  int R = key_valid.rows(), n_enc = key_valid.cols();
  if (R == 1) return {};
  int S = n_enc + n_dec;
  std::vector<MaskM> out(S, MaskM::Zero(R, R));
  for (int s = 0; s < S; ++s) {
    for (int q = 0; q < R; ++q)
      for (int k = 0; k < R; ++k) {
        int kv = (s < n_enc) ? key_valid(k, s) : 1;
        int m = (groups[q] == groups[k] && kv) ? 1 : 0;
        if (target_rows && !(*target_rows)[q] && (*target_rows)[k]) m = 0;
        if (bidir_rows && (*bidir_rows)[q] && !(*bidir_rows)[k]) m = 0;
        if (q == k) m = 1;
        out[s](q, k) = m;
      }
  }
  return out;
}

std::vector<int> group_first_obs(const MaskM& obs_patch, const std::vector<int>& groups) {
  int R = obs_patch.rows(), N = obs_patch.cols();
  std::vector<int> out(R, 0);
  for (int r = 0; r < R; ++r) {
    for (int j = 0; j < N; ++j) {
      bool any = false;
      for (int k = 0; k < R; ++k) if (groups[k] == groups[r] && obs_patch(k, j)) any = true;
      if (any) { out[r] = j; break; }
      if (j == N - 1) out[r] = 0;
    }
  }
  return out;
}

MaskM obs_per_patch(const MaskMat& observed, int ps) {
  int R = observed.rows(), N = observed.cols() / ps;
  MaskM out = MaskM::Zero(R, N);
  for (int r = 0; r < R; ++r)
    for (int p = 0; p < N; ++p)
      for (int k = 0; k < ps; ++k) if (observed(r, p * ps + k)) out(r, p) = 1;
  return out;
}

// ---- transformer (layers.py) -------------------------------------------------

MatrixXd softmax_rows(MatrixXd scores, const MaskM* mask) {
  for (int i = 0; i < scores.rows(); ++i) {
    double mx = -1e30;
    for (int j = 0; j < scores.cols(); ++j) {
      if (mask && !(*mask)(i, j)) scores(i, j) = -1e30;
      mx = std::max(mx, scores(i, j));
    }
    double sum = 0;
    for (int j = 0; j < scores.cols(); ++j) {
      scores(i, j) = std::exp(scores(i, j) - mx);
      sum += scores(i, j);
    }
    scores.row(i) /= sum;
  }
  return scores;
}

// Extract head h of q/k/v (idx 0/1/2) from a (S, 3D) QKV matrix.
MatrixXd head_slice(const MatrixXd& qkv, int idx, int h, int nh, int hd) {
  int S = qkv.rows(), D = nh * hd;
  MatrixXd out(S, hd);
  for (int d = 0; d < hd; ++d)
    out.col(d) = qkv.col(idx * D + d * nh + h);
  return out;
}

// Time attention for the active tokens of one row. `mask` covers active
// queries x all S_total = pos0 + S keys. When `cache` is set, keys/values
// for the pos0 earlier context tokens come from the cache at this layer,
// and the first n_enc_active active tokens (new context patches) have
// their K/V appended to the cache.
MatrixXd attn_time_row(const Session& s, const MatrixXd& X, const std::string& prefix,
                       const MaskM& mask, const MatrixXd& cosr, const MatrixXd& sinr,
                       KVCache* cache, int layer, int row, int pos0, int n_enc_active) {
  int S = X.rows(), D = X.cols();
  int nh = s.cfg.num_heads, hd = s.cfg.head_dim();
  int S_total = pos0 + S;
  MatrixXd qkv = lin(s, X, prefix + ".wQKV.weight");
  MatrixXd o(S, D);
  for (int h = 0; h < nh; ++h) {
    MatrixXd q = head_slice(qkv, 0, h, nh, hd);
    MatrixXd k = head_slice(qkv, 1, h, nh, hd);
    MatrixXd v = head_slice(qkv, 2, h, nh, hd);
    auto rot = [&](const MatrixXd& z) {
      MatrixXd r(S, hd);
      r.leftCols(hd / 2) = -z.rightCols(hd / 2);
      r.rightCols(hd / 2) = z.leftCols(hd / 2);
      return r;
    };
    q = q.cwiseProduct(cosr) + rot(q).cwiseProduct(sinr);
    k = k.cwiseProduct(cosr) + rot(k).cwiseProduct(sinr);
    if (cache && n_enc_active > 0) {
      cache->K[layer][row].block(pos0, h * hd, n_enc_active, hd) = k.topRows(n_enc_active);
      cache->V[layer][row].block(pos0, h * hd, n_enc_active, hd) = v.topRows(n_enc_active);
    }
    if (cache && pos0 > 0) {
      MatrixXd kall(S_total, hd), vall(S_total, hd);
      kall.topRows(pos0) = cache->K[layer][row].block(0, h * hd, pos0, hd);
      kall.bottomRows(S) = k;
      vall.topRows(pos0) = cache->V[layer][row].block(0, h * hd, pos0, hd);
      vall.bottomRows(S) = v;
      MatrixXd scores = (q * kall.transpose()) / std::sqrt((double)hd);
      o.block(0, h * hd, S, hd) = softmax_rows(scores, &mask) * vall;
    } else {
      MatrixXd scores = (q * k.transpose()) / std::sqrt((double)hd);
      o.block(0, h * hd, S, hd) = softmax_rows(scores, &mask) * v;
    }
  }
  if (s.has_mat(prefix + "._gate.weight")) {
    MatrixXd g = lin(s, X, prefix + "._gate.weight");
    o = o.cwiseProduct(g.unaryExpr([](double v) { return sigmoid(v); }));
  }
  return lin(s, o, prefix + ".wO.weight");
}

MatrixXd attn_variate_slot(const Session& s, const MatrixXd& X,
                           const std::string& prefix, const MaskM* mask) {
  int Rr = X.rows(), D = X.cols();
  int nh = s.cfg.num_heads, hd = s.cfg.head_dim();
  MatrixXd qkv = lin(s, X, prefix + ".wQKV.weight");
  std::vector<MatrixXd> heads;
  for (int h = 0; h < nh; ++h) {
    MatrixXd q = head_slice(qkv, 0, h, nh, hd);
    MatrixXd k = head_slice(qkv, 1, h, nh, hd);
    MatrixXd v = head_slice(qkv, 2, h, nh, hd);
    MatrixXd scores = (q * k.transpose()) / std::sqrt((double)hd);
    heads.push_back(softmax_rows(scores, mask) * v);
  }
  MatrixXd o(Rr, D);
  for (int h = 0; h < nh; ++h) o.block(0, h * hd, Rr, hd) = heads[h];
  if (s.has_mat(prefix + "._gate.weight")) {
    MatrixXd g = lin(s, X, prefix + "._gate.weight");
    o = o.cwiseProduct(g.unaryExpr([](double v) { return sigmoid(v); }));
  }
  return lin(s, o, prefix + ".wO.weight");
}

// H holds the active tokens only: global token indices pos0..pos0+S-1.
// time_mask/variate_mask/positions are indexed by global token index.
// When `cache` is set, the first n_enc_active active tokens are new
// context patches whose K/V are appended to the cache.
std::vector<MatrixXd> transformer(const Session& s, std::vector<MatrixXd> H,
                                  const std::vector<MaskM>& time_mask,
                                  const std::vector<MaskM>& variate_mask,
                                  const MatrixXd& positions,
                                  KVCache* cache, int pos0, int n_enc_active) {
  const Cfg& cfg = s.cfg;
  int R = (int)H.size(), S = (int)H[0].rows(), hd = cfg.head_dim();
  int S_total = pos0 + S;
  int rope_dim = cfg.rope_dim();
  std::vector<double> inv_freq(hd / 2, 0.0);
  for (int i = 0, idx = 0; idx < rope_dim; idx += 2, ++i)
    inv_freq[i] = 1.0 / std::pow(cfg.rope_theta, (double)idx / rope_dim);
  std::vector<MatrixXd> cosr(R, MatrixXd(S, hd)), sinr(R, MatrixXd(S, hd));
  for (int r = 0; r < R; ++r)
    for (int i = 0; i < S; ++i)
      for (int d = 0; d < hd; ++d) {
        double f = positions(r, pos0 + i) * inv_freq[d % (hd / 2)];
        cosr[r](i, d) = std::cos(f);
        sinr[r](i, d) = std::sin(f);
      }
  static const std::unordered_map<std::string, std::vector<bool>> axes_of = {
      {"time", {true}}, {"variate", {false}},
      {"time_variate", {true, false}}, {"variate_time", {false, true}}};
  for (size_t li = 0; li < cfg.block_schedule.size(); ++li) {
    auto axes = axes_of.at(cfg.block_schedule[li]);
    for (size_t ai = 0; ai < axes.size(); ++ai) {
      std::string ap = "transformer.layers." + std::to_string(li) + ".attns." + std::to_string(ai);
      std::string np = "transformer.layers." + std::to_string(li) + ".attn_norms." + std::to_string(ai);
      std::vector<MatrixXd> xn = H;
      rms_norm_rows(xn, s.vec(np + ".scale"), cfg.norm_eps);
      if (axes[ai]) {
        for (int r = 0; r < R; ++r) {
          MaskM sub = time_mask[r].block(pos0, 0, S, S_total);
          H[r] += attn_time_row(s, xn[r], ap, sub, cosr[r], sinr[r],
                                cache, (int)li, r, pos0, n_enc_active);
        }
      } else {
        std::vector<MatrixXd> out(R, MatrixXd::Zero(S, cfg.d_model));
        for (int sl = 0; sl < S; ++sl) {
          MatrixXd x(R, cfg.d_model);
          for (int r = 0; r < R; ++r) x.row(r) = xn[r].row(sl);
          const MaskM* mk = variate_mask.empty() ? nullptr : &variate_mask[pos0 + sl];
          MatrixXd o = attn_variate_slot(s, x, ap, mk);
          for (int r = 0; r < R; ++r) out[r].row(sl) = o.row(r);
        }
        for (int r = 0; r < R; ++r) H[r] += out[r];
      }
    }
    std::string mp = "transformer.layers." + std::to_string(li) + ".mlp";
    std::vector<MatrixXd> xn = H;
    rms_norm_rows(xn, s.vec(mp + "_norm.scale"), cfg.norm_eps);
    int hd2 = cfg.mlp_hidden_dim;
    for (int r = 0; r < R; ++r) {
      MatrixXd gu = lin(s, xn[r], mp + ".w12.weight");
      MatrixXd gate = gu.leftCols(hd2);
      MatrixXd up = gu.rightCols(hd2);
      MatrixXd act = gate.cwiseProduct(up.unaryExpr([](double v) { return silu(v); }));
      H[r] += lin(s, act, mp + ".w3.weight");
    }
  }
  rms_norm_rows(H, s.vec("transformer.out_norm.scale"), cfg.norm_eps);
  return H;
}

// ---- patch embedding, head, predict_block ------------------------------------

// Patch embedding for patches [p0, p1) only (the cache path embeds just
// the appended patches; earlier patches' tokens are already encoded).
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

MatrixXd monotone_head(const Session& s, const MatrixXd& tokens_dec) {
  // tokens_dec: (N, D) -> (N*ps, Q)
  const int ps = s.cfg.patch_size, Q = (int)s.cfg.knots.size();
  int N = tokens_dec.rows();
  MatrixXd raw = residual_block(s, tokens_dec, "head.projection.hidden_layer.weight",
                                "head.projection.output_layer.weight",
                                "head.projection.residual_layer.weight");
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

// x_enc holds encoder tokens starting at global token index pos0 (only
// the appended patches on a cache hit; all of them otherwise).
std::vector<MatrixXd> run_two_stream(const Session& s, const std::vector<MatrixXd>& x_enc,
                                     const std::vector<MatrixXd>& x_dec,
                                     const std::vector<MaskM>& time_mask,
                                     const std::vector<MaskM>& variate_mask,
                                     const MatrixXd& positions,
                                     KVCache* cache, int pos0) {
  int R = (int)x_enc.size(), n_enc = (int)x_enc[0].rows(), n_dec = (int)x_dec[0].rows();
  std::vector<MatrixXd> H(R, MatrixXd(n_enc + n_dec, s.cfg.d_model));
  for (int r = 0; r < R; ++r) {
    H[r].topRows(n_enc) = x_enc[r];
    H[r].bottomRows(n_dec) = x_dec[r];
  }
  H = transformer(s, H, time_mask, variate_mask, positions, cache, pos0, n_enc);
  std::vector<MatrixXd> out(R);
  for (int r = 0; r < R; ++r) out[r] = monotone_head(s, H[r].bottomRows(n_dec));
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

std::vector<MatrixXd> predict_context_only(const Session& s, const MatrixXd& inputs,
                                           const MaskMat& observed, int horizon_patches,
                                           const std::vector<int>& groups) {
  const int ps = s.cfg.patch_size;
  int R = inputs.rows(), T = inputs.cols(), ll = T / ps, h = horizon_patches;
  ScaleOut sc = model_scale(s, inputs, observed);
  MaskM obp = obs_per_patch(observed, ps);
  MaskM kv = enc_key_validity(obp, s.cfg.attend_missing);
  auto tm = forecast_time_mask(ll, h, kv);
  auto vm = variate_mask(kv, h, groups, nullptr, nullptr);
  auto fo = group_first_obs(obp, groups);
  MatrixXd positions(R, ll + h);
  for (int r = 0; r < R; ++r)
    for (int i = 0; i < ll + h; ++i) positions(r, i) = i - fo[r];
  // Cache decision: reuse is exact iff the new context extends the cached
  // one bit-for-bit with the same grouping and first-observed patches.
  KVCache& cache = s.cache;
  const int n_layers = (int)s.cfg.block_schedule.size();
  const double est_bytes = (double)n_layers * R * ll * s.cfg.d_model * 2 * sizeof(double);
  int pos0 = 0;
  KVCache* ctx = nullptr;
  if (est_bytes <= 256.0 * 1024 * 1024) {
    bool hit = cache.valid && cache.ll <= ll && cache.T <= T &&
               cache.inputs.rows() == R && cache.groups == groups &&
               cache.first_obs == fo;
    if (hit)
      for (int r = 0; r < R && hit; ++r)
        for (int t = 0; t < cache.T; ++t)
          if (inputs(r, t) != cache.inputs(r, t) ||
              observed(r, t) != cache.observed(r, t)) { hit = false; break; }
    if (hit) {
      pos0 = cache.ll;
      for (int li = 0; li < n_layers; ++li)
        for (int r = 0; r < R; ++r) {
          MatrixXd nk(ll, s.cfg.d_model), nv(ll, s.cfg.d_model);
          nk.setZero(); nv.setZero();
          nk.topRows(cache.ll) = cache.K[li][r];
          nv.topRows(cache.ll) = cache.V[li][r];
          cache.K[li][r] = nk; cache.V[li][r] = nv;
        }
    } else {
      cache.K.assign(n_layers, std::vector<MatrixXd>(R, MatrixXd::Zero(ll, s.cfg.d_model)));
      cache.V.assign(n_layers, std::vector<MatrixXd>(R, MatrixXd::Zero(ll, s.cfg.d_model)));
    }
    ctx = &cache;
  }
  auto x_enc = patch_embed_range(s, sc.scaled, observed, pos0, ll);
  auto x_dec = forecast_tokens(s, R, h);
  auto preds = run_two_stream(s, x_enc, x_dec, tm, vm, positions, ctx, pos0);
  if (ctx) {
    cache.inputs = inputs; cache.observed = observed;
    cache.groups = groups; cache.first_obs = fo;
    cache.T = T; cache.ll = ll; cache.valid = true;
  }
  VectorXd al(R), as(R);
  for (int r = 0; r < R; ++r) { al(r) = sc.loc(r, T - 1); as(r) = sc.scale(r, T - 1); }
  return denormalize(s, preds, al, as);
}

std::vector<MatrixXd> predict_window(const Session& s, const MatrixXd& full_inputs,
                                     const MaskMat& full_observed, int horizon_patches,
                                     const std::vector<int>& role_ids,
                                     const std::vector<int>& groups) {
  const int ps = s.cfg.patch_size;
  int R = full_inputs.rows(), W = full_observed.cols();
  int h = horizon_patches, n = W / ps, ll = n - h;
  ScaleOut sc = model_scale(s, full_inputs, full_observed);
  known_cov_scaling(s, full_inputs, full_observed, role_ids, sc);
  auto x_enc = patch_embed(s, sc.scaled, full_observed);
  auto x_dec = forecast_tokens(s, R, n);
  MaskM obp = obs_per_patch(full_observed, ps);
  MaskM kv = enc_key_validity(obp, s.cfg.attend_missing);
  std::vector<int> min_ctx(R, ll);
  auto tm = two_stream_time_mask(min_ctx, n, s.cfg.block_size, kv, role_ids);
  std::vector<int> target_rows(R), bidir_rows(R);
  for (int r = 0; r < R; ++r) { target_rows[r] = role_ids[r] == 0; bidir_rows[r] = role_ids[r] == 2; }
  auto vm = variate_mask(kv, n, groups, &target_rows, &bidir_rows);
  auto fo = group_first_obs(obp, groups);
  MatrixXd positions(R, 2 * n);
  for (int r = 0; r < R; ++r)
    for (int i = 0; i < 2 * n; ++i) positions(r, i) = (i % n) - fo[r];
  auto preds_all = run_two_stream(s, x_enc, x_dec, tm, vm, positions, nullptr, 0);
  std::vector<MatrixXd> preds(R);
  for (int r = 0; r < R; ++r) preds[r] = preds_all[r].block(ll * ps, 0, h * ps, (int)s.cfg.knots.size());
  VectorXd al(R), as(R);
  for (int r = 0; r < R; ++r) { al(r) = sc.loc(r, ll * ps); as(r) = sc.scale(r, ll * ps); }
  return denormalize(s, preds, al, as);
}

}  // namespace

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
  Cfg& c = sess->cfg;
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
  Eigen::Map<const MatrixXd> in_map(inputs.begin(), R, T);
  MatrixXd in = in_map;
  MaskMat obs(R, T);
  for (int r = 0; r < R; ++r)
    for (int t = 0; t < T; ++t) obs(r, t) = observed(r, t) ? 1 : 0;
  std::vector<int> grp(R, 1);
  if (groups.isNotNull()) {
    IntegerVector g(groups.get());
    for (int r = 0; r < R; ++r) grp[r] = g[r];
  }
  std::vector<MatrixXd> preds;
  bool has_roles = role_ids.isNotNull();
  std::vector<int> roles;
  if (has_roles) {
    IntegerVector ri(role_ids.get());
    roles.assign(ri.begin(), ri.end());
  }
  bool all_target = !has_roles;
  if (has_roles) {
    all_target = true;
    for (int v : roles) if (v != 0) all_target = false;
  }
  if (all_target || !s.cfg.covariate_enabled) {
    preds = predict_context_only(s, in, obs, horizon_patches, grp);
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
      NumericMatrix k(known_future.get());
      LogicalMatrix km(known_future_mask.get());
      for (int r = 0; r < R; ++r)
        for (int t = 0; t < horizon_len; ++t) {
          kf(r, t) = k(r, t);
          kfm(r, t) = km(r, t) ? 1 : 0;
        }
    }
    full_in.rightCols(horizon_len) = kf;
    full_obs.rightCols(horizon_len) = kfm;
    preds = predict_window(s, full_in, full_obs, horizon_patches, roles, grp);
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
