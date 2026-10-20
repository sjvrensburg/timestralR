#include "predict.h"

#include "head.h"
#include "masks.h"
#include "scaler.h"
#include "transformer.h"

#include <stdexcept>

namespace timestral {

namespace {

// x_enc holds encoder tokens starting at global token index pos0 (only
// the appended patches on a cache hit; all of them otherwise).
std::vector<MatrixXd> run_two_stream(const Session& s, const std::vector<MatrixXd>& x_enc,
                                     const std::vector<MatrixXd>& x_dec,
                                     const std::vector<MaskMat>& time_mask,
                                     const std::vector<MaskMat>& variate_mask,
                                     const MatrixXd& positions,
                                     KVCache* cache, int pos0) {
  int R = (int)x_enc.size(), n_enc = (int)x_enc[0].rows(), n_dec = (int)x_dec[0].rows();
  int S = n_enc + n_dec;
  MatrixXd H(R * S, s.cfg.d_model);
  for (int r = 0; r < R; ++r) {
    H.block(r * S, 0, n_enc, s.cfg.d_model) = x_enc[r];
    H.block(r * S + n_enc, 0, n_dec, s.cfg.d_model) = x_dec[r];
  }
  H = transformer(s, H, time_mask, variate_mask, positions, cache, pos0, n_enc, R);
  std::vector<MatrixXd> dec(R);
  for (int r = 0; r < R; ++r) dec[r] = H.block(r * S + n_enc, 0, n_dec, s.cfg.d_model);
  return monotone_head_batch(s, dec);
}

}  // namespace

std::vector<MatrixXd> predict_context_only(const Session& s, const MatrixXd& inputs,
                                           const MaskMat& observed, int horizon_patches,
                                           const std::vector<int>& groups) {
  const int ps = s.cfg.patch_size;
  int R = inputs.rows(), T = inputs.cols(), ll = T / ps, h = horizon_patches;
  if (ps < 1 || R < 1 || T < ps || T % ps != 0 || h < 1)
    throw std::invalid_argument(
        "predict_context_only needs R >= 1, T a positive multiple of "
        "patch_size, and horizon_patches >= 1");
  ScaleOut sc = model_scale(s, inputs, observed);
  MaskMat obp = obs_per_patch(observed, ps);
  MaskMat kv = enc_key_validity(obp, s.cfg.attend_missing);
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
    // Any mutation or failed run below must not leave stale metadata
    // claiming the cache is valid; it is re-marked only on success.
    const bool was_valid = cache.valid;
    cache.valid = false;
    bool hit = was_valid && cache.ll <= ll && cache.T <= T &&
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
  if (ps < 1 || R < 1 || W < ps || W % ps != 0 || h < 1 || ll < 1 ||
      (int)role_ids.size() != R)
    throw std::invalid_argument(
        "predict_window needs R >= 1, a window that is a positive "
        "multiple of patch_size with at least one context patch, "
        "horizon_patches >= 1, and one role per row");
  ScaleOut sc = model_scale(s, full_inputs, full_observed);
  known_cov_scaling(s, full_inputs, full_observed, role_ids, sc);
  auto x_enc = patch_embed(s, sc.scaled, full_observed);
  auto x_dec = forecast_tokens(s, R, n);
  MaskMat obp = obs_per_patch(full_observed, ps);
  MaskMat kv = enc_key_validity(obp, s.cfg.attend_missing);
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

}  // namespace timestral
