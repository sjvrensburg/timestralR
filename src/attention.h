#ifndef TIMESTRALR_ATTENTION_H_
#define TIMESTRALR_ATTENTION_H_

// Time and variate attention over stacked token matrices, plus the QKV
// repacking and partial RoPE they rely on.

#include "session.h"

#include <string>
#include <vector>

namespace timestral {

// Repack a (M, 3D) QKV projection into separate Q/K/V matrices in
// head-block layout (head h in columns [h*hd, (h+1)*hd)), so per-head
// work below uses contiguous block views instead of strided copies.
void repack_qkv(const MatrixXd& qkv, int nh, int hd,
                MatrixXd& Q, MatrixXd& K, MatrixXd& V);

// Partial RoPE in place on a head-block-layout matrix; cos_s/sin_s are
// (M, hd) and apply identically to every head.
void rope_inplace(MatrixXd& Z, const MatrixXd& cos_s, const MatrixXd& sin_s,
                  int nh, int hd);

// Time attention over stacked rows. Q/K/V are (R*S, D) head-block
// layout, already RoPE'd; token (r, s) is at row r*S + s. bias[r] is
// (S, S_total) for active queries. When `cache` is set, keys/values for
// the pos0 earlier context tokens come from the cache at this layer, and
// the first n_enc_active active tokens have their K/V appended to it.
// Writes the pre-gate, pre-wO output into o_s (R*S, D).
void attn_time_stacked(const Session& s, const MatrixXd& Q, const MatrixXd& K,
                       const MatrixXd& V, const std::vector<MatrixXd>& bias,
                       MatrixXd& o_s, KVCache* cache, int layer, int pos0,
                       int n_enc_active, int R, int S);

// Variate attention over slot-major stacked tokens (slot s at rows
// [s*R, (s+1)*R)). var_bias[slot] is (R, R) or var_bias is empty.
// Writes the pre-gate, pre-wO output into o_v (S*R, D).
void attn_variate_stacked(const Session& s, const MatrixXd& Q, const MatrixXd& K,
                          const MatrixXd& V, const std::vector<MatrixXd>& var_bias,
                          MatrixXd& o_v, int pos0, int R, int S);

// Apply the optional sigmoid gate and the wO projection to stacked
// attention outputs (rows in whatever token order the caller used).
MatrixXd attn_finish(const Session& s, MatrixXd o, const MatrixXd& xn_stacked,
                     const std::string& prefix);

}  // namespace timestral

#endif  // TIMESTRALR_ATTENTION_H_
