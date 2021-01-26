/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/math/math.h"

#include <glog/logging.h>
#include <Eigen/SVD>

namespace sia {

const Eigen::VectorXd slice(const Eigen::VectorXd& x,
                            const std::vector<std::size_t>& indices) {
  Eigen::VectorXd y = Eigen::VectorXd::Zero(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    y(i) = x(indices.at(i));
  }
  return y;
}

const Eigen::MatrixXd slice(const Eigen::MatrixXd& X,
                            const std::vector<std::size_t>& rows,
                            const std::vector<std::size_t>& cols) {
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(rows.size(), cols.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < cols.size(); ++j) {
      Y(i, j) = X(rows.at(i), cols.at(j));
    }
  }
  return Y;
}

bool llt(const Eigen::MatrixXd& A, Eigen::MatrixXd& L) {
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  L = llt.matrixL();
  if (llt.info() != Eigen::ComputationInfo::Success) {
    LOG(WARNING) << "LLT decomposition of matrix A = " << A << " failed with "
                 << llt.info();
    return false;
  }
  return true;
}

bool ldltSqrt(const Eigen::MatrixXd& A, Eigen::MatrixXd& M) {
  Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
  const Eigen::MatrixXd P = ldlt.transpositionsP() * I;
  const Eigen::MatrixXd L = ldlt.matrixL();
  const Eigen::VectorXd D = ldlt.vectorD();
  M = P.transpose() * L * D.array().sqrt().matrix().asDiagonal();
  if (ldlt.info() != Eigen::ComputationInfo::Success) {
    LOG(WARNING) << "LDLT decomposition of matrix A = " << A << " failed with "
                 << ldlt.info();
    return false;
  }
  return true;
}

bool svd(const Eigen::MatrixXd& A,
         Eigen::MatrixXd& U,
         Eigen::VectorXd& S,
         Eigen::MatrixXd& V,
         double tolerance) {
  bool result = true;

  // Compute the SVD of A
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd& singular_values = svd.singularValues();

  // Check for singular condition
  for (int i = 0; i < singular_values.size(); ++i) {
    if (singular_values(i) <= tolerance) {
      LOG(WARNING) << "Singular value is less than tolerance";
      result = false;
    }
  }

  // Return SVD matrices
  U = svd.matrixU();
  S = singular_values.array();
  V = svd.matrixV();
  return result;
}

const Eigen::MatrixXd svdInverse(const Eigen::MatrixXd& U,
                                 const Eigen::VectorXd& S,
                                 const Eigen::MatrixXd& V) {
  return V * S.array().inverse().matrix().asDiagonal() * U.transpose();
}

bool svdInverse(const Eigen::MatrixXd& A,
                Eigen::MatrixXd& Ainv,
                double tolerance) {
  Eigen::MatrixXd U, V;
  Eigen::VectorXd S;
  bool result = svd(A, U, S, V, tolerance);

  // Compute the generalized inverse using SVD
  Ainv = svdInverse(U, S, V);
  return result;
}

const Eigen::MatrixXd lltSqrt(const Eigen::MatrixXd& A) {
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  return llt.matrixL();
}

const Eigen::VectorXd rk4(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> dynamical_system,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u,
    double dt) {
  Eigen::VectorXd f1 = dynamical_system(x, u);
  Eigen::VectorXd f2 = dynamical_system(x + dt * f1 / 2, u);
  Eigen::VectorXd f3 = dynamical_system(x + dt * f2 / 2, u);
  Eigen::VectorXd f4 = dynamical_system(x + dt * f3, u);
  return x + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4);
}

}  // namespace sia
