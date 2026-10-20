#include "nn_ops.h"

#include <R_ext/BLAS.h>

#include <cmath>

namespace timestral {

MatrixXd lin(const Session& s, const MatrixXd& X, const std::string& wname) {
  const RowMat& W = s.mat(wname);
  MatrixXd Y;
  // Hybrid GEMM: the system BLAS (FlexiBLAS/OpenBLAS on this machine) wins
  // on large products, Eigen's own kernels win on the many small ones.
  // Y = X * W^T; W's row-major buffer read as column-major is exactly W^T.
  const double flops = 2.0 * X.rows() * X.cols() * W.rows();
  if (flops >= 4.0e6) {
    Y.resize(X.rows(), W.rows());
    int m = (int)X.rows(), n = (int)W.rows(), k = (int)X.cols();
    double alpha = 1.0, beta = 0.0;
    F77_CALL(dgemm)("N", "N", &m, &n, &k, &alpha, X.data(), &m,
                    W.data(), &k, &beta, Y.data(), &m FCONE FCONE);
  } else {
    Y = X * W.transpose();
  }
  std::string bname = wname;
  size_t pos = bname.rfind(".weight");
  if (pos != std::string::npos) {
    bname.replace(pos, 7, ".bias");
    if (s.has_vec(bname)) Y.rowwise() += s.vec(bname).transpose();
  }
  return Y;
}

MatrixXd residual_block(const Session& s, const MatrixXd& X,
                        const std::string& hidden, const std::string& output,
                        const std::string& residual) {
  MatrixXd h = lin(s, X, hidden);
  h = h.unaryExpr([](double v) { return silu(v); });
  return lin(s, h, output) + lin(s, X, residual);
}

MatrixXd rms_norm(const MatrixXd& X, const VectorXd& scale, double eps) {
  MatrixXd Y = X;
  for (int i = 0; i < Y.rows(); ++i) {
    double ms = Y.row(i).squaredNorm() / Y.cols();
    Y.row(i) = (Y.row(i) / std::sqrt(ms + eps)).cwiseProduct(scale.transpose());
  }
  return Y;
}

MatrixXd mask_to_bias(const MaskMat& m) {
  MatrixXd b(m.rows(), m.cols());
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j) b(i, j) = m(i, j) ? 0.0 : -1e30;
  return b;
}

MatrixXd softmax_bias(MatrixXd scores, const MatrixXd* bias) {
  if (bias) scores += *bias;
  // Mirror R .softmax_rows: nonfinite scores become -1e30 before the max.
  for (int i = 0; i < scores.rows(); ++i)
    for (int j = 0; j < scores.cols(); ++j)
      if (!std::isfinite(scores(i, j))) scores(i, j) = -1e30;
  for (int i = 0; i < scores.rows(); ++i) {
    double mx = scores.row(i).maxCoeff();
    double sum = 0;
    for (int j = 0; j < scores.cols(); ++j) {
      scores(i, j) = std::exp(scores(i, j) - mx);
      sum += scores(i, j);
    }
    scores.row(i) *= (1.0 / sum);
  }
  return scores;
}

MatrixXd stack_rows(const std::vector<MatrixXd>& H) {
  int R = (int)H.size(), S = (int)H[0].rows(), D = (int)H[0].cols();
  MatrixXd out(R * S, D);
  for (int r = 0; r < R; ++r) out.block(r * S, 0, S, D) = H[r];
  return out;
}

}  // namespace timestral
