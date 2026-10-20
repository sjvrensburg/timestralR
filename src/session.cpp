#include "session.h"

namespace timestral {

int Cfg::head_dim() const { return d_model / num_heads; }

int Cfg::rope_dim() const { return static_cast<int>(head_dim() * rope_partial_factor); }

int Cfg::median_index() const {
  for (size_t i = 0; i < knots.size(); ++i) if (knots[i] == 0.5) return (int)i;
  return -1;
}

bool Session::has_mat(const std::string& n) const { return mats.count(n) > 0; }
bool Session::has_vec(const std::string& n) const { return vecs.count(n) > 0; }
const RowMat& Session::mat(const std::string& n) const { return mats.at(n); }
const VectorXd& Session::vec(const std::string& n) const { return vecs.at(n); }

}  // namespace timestral
