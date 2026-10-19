#ifndef TIMESTRALR_PREDICT_H_
#define TIMESTRALR_PREDICT_H_

// Single-block forecast primitive: the context-only path (with the
// cross-block K/V cache) and the covariate window path. Both return one
// (steps, Q) raw-space quantile matrix per row.

#include "session.h"

#include <vector>

namespace timestral {

std::vector<MatrixXd> predict_context_only(const Session& s, const MatrixXd& inputs,
                                           const MaskMat& observed, int horizon_patches,
                                           const std::vector<int>& groups);

std::vector<MatrixXd> predict_window(const Session& s, const MatrixXd& full_inputs,
                                     const MaskMat& full_observed, int horizon_patches,
                                     const std::vector<int>& role_ids,
                                     const std::vector<int>& groups);

}  // namespace timestral

#endif  // TIMESTRALR_PREDICT_H_
