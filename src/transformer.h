#ifndef TIMESTRALR_TRANSFORMER_H_
#define TIMESTRALR_TRANSFORMER_H_

// The interleaved time/variate transformer stack (layers.py).

#include "session.h"

#include <vector>

namespace timestral {

// H holds the active tokens only: global token indices pos0..pos0+S-1.
// time_mask/variate_mask/positions are indexed by global token index.
// When `cache` is set, the first n_enc_active active tokens are new
// context patches whose K/V are appended to the cache.
// Activations live in one stacked (R*S, D) matrix for the whole
// transformer (token (r, s) at row r*S + s), so norms and projections
// need no per-sublayer stacking copies.
MatrixXd transformer(const Session& s, MatrixXd H,
                     const std::vector<MaskMat>& time_mask,
                     const std::vector<MaskMat>& variate_mask,
                     const MatrixXd& positions,
                     KVCache* cache, int pos0, int n_enc_active, int R);

}  // namespace timestral

#endif  // TIMESTRALR_TRANSFORMER_H_
