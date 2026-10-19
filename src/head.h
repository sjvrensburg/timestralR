#ifndef TIMESTRALR_HEAD_H_
#define TIMESTRALR_HEAD_H_

// Patch embedding and forecast query tokens on the way in; the monotone
// quantile head and denormalisation on the way out.

#include "session.h"

#include <vector>

namespace timestral {

// Patch embedding for patches [p0, p1) only (the cache path embeds just
// the appended patches; earlier patches' tokens are already encoded).
std::vector<MatrixXd> patch_embed_range(const Session& s, const MatrixXd& scaled,
                                        const MaskMat& observed, int p0, int p1);

std::vector<MatrixXd> patch_embed(const Session& s, const MatrixXd& scaled,
                                  const MaskMat& observed);

// n_patches copies of the learned forecast query token per row.
std::vector<MatrixXd> forecast_tokens(const Session& s, int R, int n_patches);

// Monotone quantile head for all rows at once: the projection is one
// batched GEMM over stacked (R*N, D) tokens; the monotone accumulation
// then runs per row on its (N*ps, Q) block.
std::vector<MatrixXd> monotone_head_batch(const Session& s,
                                          const std::vector<MatrixXd>& tokens_dec);

// Back to raw space: sinh (when asinh-scaled), times the anchor scale plus
// the anchor location per row; non-finite values are clamped.
std::vector<MatrixXd> denormalize(const Session& s, std::vector<MatrixXd> preds,
                                  const VectorXd& anchor_loc, const VectorXd& anchor_scale);

}  // namespace timestral

#endif  // TIMESTRALR_HEAD_H_
