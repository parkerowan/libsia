/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>

TEST(Math, Infinity) {
  EXPECT_DOUBLE_EQ(log(0), -INFINITY);
}

TEST(Math, SliceVector) {
  std::vector<std::size_t> indices{0, 2};
  Eigen::Vector3d x(0, 1, 2);
  Eigen::VectorXd y = sia::slice(x, indices);

  EXPECT_EQ(y.size(), 2);
  EXPECT_DOUBLE_EQ(y(0), 0);
  EXPECT_DOUBLE_EQ(y(1), 2);
}

TEST(Math, SliceMatrix) {
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

TEST(Math, Svd) {
  Eigen::MatrixXd A(2, 2);
  A << 0, 1, 2, 3;
  Eigen::MatrixXd U, V;
  Eigen::VectorXd S;
  EXPECT_TRUE(sia::svd(A, U, S, V));

  // Expect the SVD to reconstruct matrix A
  Eigen::MatrixXd Abar = U * S.asDiagonal() * V.transpose();
  EXPECT_TRUE(Abar.isApprox(A));

  // Compute the inverse from SVD
  Eigen::MatrixXd AAinv = A * sia::svdInverse(U, S, V);
  EXPECT_TRUE(Eigen::Matrix2d::Identity().isApprox(AAinv));

  // Compute the LLT decomposition
  A << 2, 0.1, 0.1, 1;
  Eigen::MatrixXd L;
  EXPECT_TRUE(sia::llt(A, L));
  EXPECT_TRUE(A.isApprox(L * L.transpose()));

  // Compute sqrt using the LDLT composition
  Eigen::MatrixXd M;
  EXPECT_TRUE(sia::ldltSqrt(A, M));
  EXPECT_TRUE(A.isApprox(M * M.transpose()));

  A << 2.1, 0, 0, 0;  // positive semi-definite
  EXPECT_TRUE(sia::ldltSqrt(A, M));
  EXPECT_TRUE(A.isApprox(M * M.transpose()));
}

TEST(Math, SvdInverse) {
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

TEST(Math, Rk4) {
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

TEST(Math, FxScalarDerivatives) {
  Eigen::MatrixXd A(2, 2);
  A << 0.1, 0.2, 0.3, 0.4;

  Eigen::VectorXd x(2);
  x << 1, 2;

  auto f = [A](const Eigen::VectorXd& x) {
    const Eigen::VectorXd Ax = A * x;
    return x.dot(Ax);
  };

  // error
  Eigen::VectorXd e = Eigen::VectorXd::Zero(2);
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(2, 2);

  // df/dx
  Eigen::VectorXd dfdx_numerical = sia::dfdx(f, x);
  Eigen::VectorXd dfdx_analytic = x.transpose() * (A + A.transpose());
  e = dfdx_numerical - dfdx_analytic;
  EXPECT_NEAR(e.dot(e), 0.0, 1.0e-8);

  // d2f/dx2
  Eigen::MatrixXd d2fdx2_numerical = sia::d2fdxx(f, x);
  Eigen::MatrixXd d2fdx2_analytic = A + A.transpose();
  E = d2fdx2_numerical - d2fdx2_analytic;
  EXPECT_NEAR(E.norm(), 0.0, 1.0e-2);
}

TEST(Math, FxuScalarDerivatives) {
  Eigen::MatrixXd A(2, 2), B(2, 2);
  A << 0.1, 0.2, 0.3, 0.4;
  B << 0.8, -0.4, -0.2, 0.1;

  Eigen::VectorXd x(2), u(2);
  x << 1, 2;
  u << -3, 4;

  auto f = [A, B](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    const Eigen::VectorXd Ax = A * x;
    const Eigen::VectorXd Bu = B * u;
    return x.dot(Ax) + u.dot(Bu) + x.dot(u);
  };

  // error
  Eigen::VectorXd e = Eigen::VectorXd::Zero(2);
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(2, 2);

  // df/dx
  Eigen::VectorXd dfdx_numerical = sia::dfdx(f, x, u);
  Eigen::VectorXd dfdx_analytic =
      x.transpose() * (A + A.transpose()) + u.transpose();
  e = dfdx_numerical - dfdx_analytic;
  EXPECT_NEAR(e.dot(e), 0.0, 1.0e-8);

  // df/du
  Eigen::VectorXd dfdu_numerical = sia::dfdu(f, x, u);
  Eigen::VectorXd dfdu_analytic =
      u.transpose() * (B + B.transpose()) + x.transpose();
  e = dfdu_numerical - dfdu_analytic;
  EXPECT_NEAR(e.dot(e), 0.0, 1.0e-8);

  // d2f/dx2
  Eigen::MatrixXd d2fdx2_numerical = sia::d2fdxx(f, x, u);
  Eigen::MatrixXd d2fdx2_analytic = A + A.transpose();
  E = d2fdx2_numerical - d2fdx2_analytic;
  EXPECT_NEAR(E.norm(), 0.0, 1.0e-2);

  // d2f/du2
  Eigen::MatrixXd d2fdu2_numerical = sia::d2fduu(f, x, u);
  Eigen::MatrixXd d2fdu2_analytic = B + B.transpose();
  E = d2fdu2_numerical - d2fdu2_analytic;
  EXPECT_NEAR(E.norm(), 0.0, 1.0e-2);

  // d2f/dudx
  Eigen::MatrixXd d2fdux_numerical = sia::d2fdux(f, x, u);
  Eigen::MatrixXd d2fdux_analytic = Eigen::MatrixXd::Identity(2, 2);
  E = d2fdu2_numerical - d2fdu2_analytic;
  EXPECT_NEAR(E.norm(), 0.0, 1.0e-2);
}

TEST(Math, FxuVectorDerivatives) {
  Eigen::MatrixXd A(2, 2), B(2, 2);
  A << 0.1, 0.2, 0.3, 0.4;
  B << 0.8, -0.4, -0.2, 0.1;

  Eigen::VectorXd x(2), u(2);
  x << 1, 2;
  u << -3, 4;

  auto f = [A, B](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    return A * x + B * u;
  };

  // error
  Eigen::MatrixXd E = Eigen::MatrixXd::Zero(2, 2);

  // df/dx
  Eigen::MatrixXd dfdx_numerical = sia::dfdx(f, x, u);
  Eigen::MatrixXd dfdx_analytic = A;
  E = dfdx_numerical - dfdx_analytic;
  EXPECT_NEAR(E.norm(), 0.0, 1.0e-2);

  // df/du
  Eigen::MatrixXd dfdu_numerical = sia::dfdu(f, x, u);
  Eigen::MatrixXd dfdu_analytic = B;
  E = dfdu_numerical - dfdu_analytic;
  EXPECT_NEAR(E.norm(), 0.0, 1.0e-2);
}
