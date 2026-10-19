#ifndef TIMESTRALR_NN_OPS_H_
#define TIMESTRALR_NN_OPS_H_

// Dense building blocks: activations, linear layers (hybrid Eigen/BLAS
// GEMM), residual MLP blocks, RMS norm, and masked row softmax.

#include "session.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace timestral {

// Scalar activations stay inline: they run inside Eigen unaryExpr loops.
inline double silu(double x) { return x / (1.0 + std::exp(-x)); }
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(std::min(x, 20.0))); }

// Y = X * W^T (+ bias when the session holds "<name>.bias"), for the
// weight matrix `wname`.
MatrixXd lin(const Session& s, const MatrixXd& X, const std::string& wname);

MatrixXd residual_block(const Session& s, const MatrixXd& X,
                        const std::string& hidden, const std::string& output,
                        const std::string& residual);

MatrixXd rms_norm(const MatrixXd& X, const VectorXd& scale, double eps);

// Convert an integer mask to an additive attention bias (0 / -1e30),
// once per call instead of per head per layer.
MatrixXd mask_to_bias(const MaskMat& m);

// Row softmax with an optional additive bias, fused: bias added in one
// vectorised pass, then a single exp/sum pass per row.
MatrixXd softmax_bias(MatrixXd scores, const MatrixXd* bias);

// Stack per-row (S, D) matrices into one (R*S, D) matrix, token (r, s)
// at row r*S + s, so projections run as a single GEMM over all rows.
MatrixXd stack_rows(const std::vector<MatrixXd>& H);

}  // namespace timestral

#endif  // TIMESTRALR_NN_OPS_H_
