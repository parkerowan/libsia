/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>

TEST(Math, infinity) {
  EXPECT_DOUBLE_EQ(log(0), -INFINITY);
}

TEST(Math, sliceVector) {
  std::vector<std::size_t> indices{0, 2};
  Eigen::Vector3d x(0, 1, 2);
  Eigen::VectorXd y = sia::slice(x, indices);

  EXPECT_EQ(y.size(), 2);
  EXPECT_DOUBLE_EQ(y(0), 0);
  EXPECT_DOUBLE_EQ(y(1), 2);
}

TEST(Math, sliceMatrix) {
  std::vector<std::size_t> rows{0, 2};
  std::vector<std::size_t> cols{1, 2};
  Eigen::Matrix3d X;
  X << 0, 1, 2, 3, 4, 5, 6, 7, 8;
  Eigen::MatrixXd Y = sia::slice(X, rows, cols);

  EXPECT_EQ(Y.rows(), 2);
  EXPECT_EQ(Y.cols(), 2);
  EXPECT_DOUBLE_EQ(Y(0, 0), 1);
  EXPECT_DOUBLE_EQ(Y(0, 1), 2);
  EXPECT_DOUBLE_EQ(Y(1, 0), 7);
  EXPECT_DOUBLE_EQ(Y(1, 1), 8);
}

TEST(Math, svd) {
  Eigen::MatrixXd A(2, 3);
  A << 0, 1, 2, 3, 4, 5;
  Eigen::MatrixXd U, V;
  Eigen::VectorXd S;
  EXPECT_TRUE(sia::svd(A, U, S, V));

  // Expect the SVD to reconstruct matrix A
  Eigen::MatrixXd Abar = U * S.asDiagonal() * V.transpose();
  EXPECT_TRUE(Abar.isApprox(A));
}

TEST(Math, svdInverse) {
  Eigen::MatrixXd A(2, 3);
  A << 0, 1, 2, 3, 4, 5;
  Eigen::MatrixXd Ainv;
  EXPECT_TRUE(sia::svdInverse(A, Ainv));
  ASSERT_EQ(Ainv.rows(), 3);
  ASSERT_EQ(Ainv.cols(), 2);

  // Expect A by its inverse to be identity matrix
  Eigen::MatrixXd AAinv = A * Ainv;
  ASSERT_EQ(AAinv.rows(), 2);
  ASSERT_EQ(AAinv.cols(), 2);
  EXPECT_TRUE(Eigen::Matrix2d::Identity().isApprox(AAinv));
}

TEST(Math, rk4) {
  double a = 10.0;
  auto system = [a](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    return a * (u - x);
  };

  Eigen::VectorXd x(1), u(1);
  x << 0;
  u << 10;
  double dt = 0.1;

  // Integrate one time step
  Eigen::VectorXd xkp1num = sia::rk4(system, x, u, dt);
  ASSERT_EQ(xkp1num.size(), 1);

  // Exact value
  Eigen::VectorXd xpk1exact = (1 - exp(-dt * a)) * u;
  ASSERT_EQ(xpk1exact.size(), 1);

  // Should this be pow(dt, 4) for Runge Kutta?
  EXPECT_NEAR(xkp1num(0), xpk1exact(0), pow(dt, 1));
}
