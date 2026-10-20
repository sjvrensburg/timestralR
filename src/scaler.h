#ifndef TIMESTRALR_SCALER_H_
#define TIMESTRALR_SCALER_H_

// Causal per-patch normalisation (scaler.py) and the whole-row scaling of
// known covariates.

#include "session.h"

#include <vector>

namespace timestral {

struct ScaleOut { MatrixXd scaled, loc, scale; };

ScaleOut model_scale(const Session& s, const MatrixXd& inputs, const MaskMat& observed);

// known-covariate whole-row scaling; modifies scaled/loc/scale in place
void known_cov_scaling(const Session& s, const MatrixXd& inputs, const MaskMat& observed,
                       const std::vector<int>& role_ids, ScaleOut& sc);

}  // namespace timestral

#endif  // TIMESTRALR_SCALER_H_
