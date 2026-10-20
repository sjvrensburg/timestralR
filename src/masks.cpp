#include "masks.h"

#include <algorithm>

namespace timestral {

namespace {

MaskMat enc_causal_mask_row(const Eigen::VectorXi& kv) {
  int ll = kv.size();
  MaskMat m = MaskMat::Zero(ll, ll);
  for (int i = 0; i < ll; ++i)
    for (int j = 0; j < ll; ++j)
      m(i, j) = ((j <= i && kv(j)) || i == j) ? 1 : 0;
  return m;
}

}  // namespace

MaskMat enc_key_validity(const MaskMat& obs_patch, bool attend_missing) {
  int R = obs_patch.rows(), N = obs_patch.cols();
  MaskMat out(R, N);
  for (int r = 0; r < R; ++r) {
    int started = 0;
    for (int j = 0; j < N; ++j) {
      started = std::max(started, obs_patch(r, j));
      out(r, j) = obs_patch(r, j) || (attend_missing && started);
    }
  }
  return out;
}

std::vector<MaskMat> forecast_time_mask(int ll, int h, const MaskMat& key_valid) {
  int R = key_valid.rows(), S = ll + h;
  std::vector<MaskMat> out(R, MaskMat::Zero(S, S));
  for (int r = 0; r < R; ++r) {
    out[r].block(0, 0, ll, ll) = enc_causal_mask_row(key_valid.row(r).transpose());
    for (int i = 0; i < h; ++i)
      for (int j = 0; j < ll; ++j) out[r](ll + i, j) = key_valid(r, j);
    for (int i = 0; i < h; ++i)
      for (int j = 0; j <= i; ++j) out[r](ll + i, ll + j) = 1;
  }
  return out;
}

std::vector<MaskMat> two_stream_time_mask(const std::vector<int>& min_ctx, int n,
                                        int block_size, const MaskMat& key_valid,
                                        const std::vector<int>& role_ids) {
  int R = key_valid.rows(), S = 2 * n;
  std::vector<MaskMat> out(R, MaskMat::Zero(S, S));
  for (int r = 0; r < R; ++r) {
    int s = min_ctx[r];
    std::vector<int> anchor(n), active(n);
    for (int p = 0; p < n; ++p) {
      active[p] = p >= s;
      int off = std::max(p - s, 0);
      anchor[p] = active[p] ? s + block_size * (off / block_size) : 0;
    }
    MaskMat enc = enc_causal_mask_row(key_valid.row(r).transpose());
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

std::vector<MaskMat> variate_mask(const MaskMat& key_valid, int n_dec,
                                const std::vector<int>& groups,
                                const std::vector<int>* target_rows,
                                const std::vector<int>* bidir_rows) {
  int R = key_valid.rows(), n_enc = key_valid.cols();
  if (R == 1) return {};
  int S = n_enc + n_dec;
  std::vector<MaskMat> out(S, MaskMat::Zero(R, R));
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

std::vector<int> group_first_obs(const MaskMat& obs_patch, const std::vector<int>& groups) {
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

MaskMat obs_per_patch(const MaskMat& observed, int ps) {
  int R = observed.rows(), N = observed.cols() / ps;
  MaskMat out = MaskMat::Zero(R, N);
  for (int r = 0; r < R; ++r)
    for (int p = 0; p < N; ++p)
      for (int k = 0; k < ps; ++k) if (observed(r, p * ps + k)) out(r, p) = 1;
  return out;
}

}  // namespace timestral
