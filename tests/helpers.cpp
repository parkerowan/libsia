/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "tests/helpers.h"

sia::LinearGaussianDynamics createTestDynamics() {
  Eigen::Matrix<double, 1, 1> F, G, Q;
  F << 0.9;
  G << 0.1;
  Q << 0.1;
  return sia::LinearGaussianDynamics(F, G, Q);
}

sia::LinearGaussianDynamics createIntegratorDynamics() {
  Eigen::Matrix<double, 1, 1> F, G, Q;
  F << 0;
  G << 1;
  Q << 0.1;
  return sia::LinearGaussianDynamics(F, G, Q);
}

sia::LinearGaussianMeasurement createTestMeasurement() {
  Eigen::Matrix<double, 1, 1> H, R;
  H << 1;
  R << 0.01;
  return sia::LinearGaussianMeasurement(H, R);
}

sia::QuadraticCost createTestCost() {
  Eigen::Matrix<double, 1, 1> Q, Qf, R;
  Q << 1;
  Qf << 10;
  R << 0.1;
  return sia::QuadraticCost(Qf, Q, R);
}

Eigen::MatrixXd createPositiveDefiniteMatrix(std::size_t n) {
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(n, n);

  // Enforce the matrix to symmetric
  X = X.selfadjointView<Eigen::Upper>();

  // since A(i,j) < 1 by construction and a symmetric diagonally dominant matrix
  // is symmetric positive definite, which can be ensured by adding nI
  return X + double(n) * Eigen::MatrixXd::Identity(n, n);
}
