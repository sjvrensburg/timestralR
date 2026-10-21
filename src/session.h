#ifndef TIMESTRALR_SESSION_H_
#define TIMESTRALR_SESSION_H_

// Core types shared by the native backend: Eigen aliases, the model
// configuration, the cross-block K/V cache, and the session that owns the
// weights. Layout notes: safetensors weights are (out, in) row-major as in
// torch. They are stored here as RowMajor matrices; activations are default
// (column-major) Eigen matrices mapped zero-copy from R where possible.

#include <RcppEigen.h>

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace timestral {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MaskMat = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

struct Cfg {
  int patch_size = 32, d_model = 0, num_heads = 0, mlp_hidden_dim = 0;
  int block_size = 32;
  std::vector<std::string> block_schedule;
  std::vector<double> knots;
  double rope_theta = 10000.0, rope_partial_factor = 0.5, norm_eps = 1e-6;
  double minimum_scale = 1e-6, quantile_width_factor = 2.0;
  bool use_asinh = true, attend_missing = true, covariate_enabled = true;
  int head_dim() const;
  int rope_dim() const;
  int median_index() const;
};

// K/V cache for the context-only decode path. The R decode loop
// (.block_ar_decode) issues one .ts_predict_block call per block with the
// median feedback appended to the context, and a backtest forecasts from
// successive origins whose contexts share their leading patches. Because
// the time mask is causal, the scaler is causal, and positions depend only
// on the first observed patch, a context patch's per-layer K/V depend only
// on that patch and the ones before it; so the K/V of every leading patch
// whose inputs match the cached ones exactly (same rows/groups/
// first-observed) are reused and only the remaining patches plus the new
// decode tokens are run through the transformer. K/V are stored per layer
// and row as (ll, D) in head-block layout, K after RoPE. Two entries,
// least recently used evicted, so the forward and mirrored sequences of a
// symmetric forecast each keep their own entry.
struct KVCache {
  bool valid = false;
  unsigned long last_use = 0;
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
  mutable std::array<KVCache, 2> caches;
  mutable unsigned long cache_clock = 0;
  bool has_mat(const std::string& n) const;
  bool has_vec(const std::string& n) const;
  const RowMat& mat(const std::string& n) const;
  const VectorXd& vec(const std::string& n) const;
};

}  // namespace timestral

#endif  // TIMESTRALR_SESSION_H_
