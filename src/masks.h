#ifndef TIMESTRALR_MASKS_H_
#define TIMESTRALR_MASKS_H_

// Attention masks (masks.py) and patch-level observation helpers. Masks
// are integer matrices, 1 = may attend.

#include "session.h"

#include <vector>

namespace timestral {

MaskMat enc_key_validity(const MaskMat& obs_patch, bool attend_missing);

// time mask per row: vector of (S,S)
std::vector<MaskMat> forecast_time_mask(int ll, int h, const MaskMat& key_valid);

std::vector<MaskMat> two_stream_time_mask(const std::vector<int>& min_ctx, int n,
                                        int block_size, const MaskMat& key_valid,
                                        const std::vector<int>& role_ids);

// variate mask per slot: vector of (R,R); empty when R == 1
std::vector<MaskMat> variate_mask(const MaskMat& key_valid, int n_dec,
                                const std::vector<int>& groups,
                                const std::vector<int>* target_rows,
                                const std::vector<int>* bidir_rows);

// First observed patch per row, over all rows of the same group.
std::vector<int> group_first_obs(const MaskMat& obs_patch, const std::vector<int>& groups);

// 1 where a patch holds at least one observed step.
MaskMat obs_per_patch(const MaskMat& observed, int ps);

}  // namespace timestral

#endif  // TIMESTRALR_MASKS_H_
