/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "tests/helpers.h"

#include <gtest/gtest.h>
#include <sia/sia.h>
#include <iostream>

TEST(Runner, Buffer) {
  sia::Buffer buffer(2, 10);
  EXPECT_EQ(buffer.length(), 10);
  EXPECT_EQ(buffer.dimension(), 2);
  EXPECT_TRUE(buffer.data().isApprox(Eigen::MatrixXd::Zero(2, 10)));

  Eigen::Vector2d x(1.0, 2.0);
  EXPECT_TRUE(buffer.record(x));

  Eigen::MatrixXd X = x.asDiagonal() * Eigen::MatrixXd::Ones(2, 10);
  EXPECT_TRUE(buffer.data().isApprox(X));

  Eigen::Vector2d y(3.0, 4.0);
  EXPECT_TRUE(buffer.record(y));
  EXPECT_TRUE(buffer.previous(0).isApprox(y));
  EXPECT_TRUE(buffer.previous(1).isApprox(x));

  EXPECT_TRUE(buffer.future(0).isApprox(x));
  EXPECT_TRUE(buffer.future(1).isApprox(x));
}

TEST(Runner, Runner) {
  sia::LinearGaussian system = createTestSystem();
  sia::Gaussian prior(0, 10);
  sia::KF a(system, prior);
  sia::KF b(system, prior);

  Eigen::Matrix<double, 1, 1> u = Eigen::Matrix<double, 1, 1>::Ones();
  Eigen::VectorXd x0 = Eigen::VectorXd::Ones(1);

  std::size_t buffer = 10;
  sia::EstimatorMap estimators = {{"kf", a}};
  sia::Runner runner(estimators, buffer);
  sia::Recorder& recorder = runner.recorder();
  Eigen::VectorXd xkp1 = runner.stepAndEstimate(system, x0, u);

  const auto& Y = recorder.getObservations();
  const auto& U = recorder.getControls();
  const auto& X = recorder.getStates();
  const auto& EMU = recorder.getEstimateMeans("kf");
  const auto& EMO = recorder.getEstimateModes("kf");
  const auto& EVAR = recorder.getEstimateVariances("kf");
  const auto& EMPTY = recorder.getEstimateMeans("missing_filter");

  ASSERT_EQ(Y.rows(), 1);
  ASSERT_EQ(U.rows(), 1);
  ASSERT_EQ(X.rows(), 1);
  ASSERT_EQ(EMU.rows(), 1);
  ASSERT_EQ(EMO.rows(), 1);
  ASSERT_EQ(EVAR.rows(), 1);
  ASSERT_EQ(EMPTY.rows(), 0);

  ASSERT_EQ(Y.cols(), buffer);
  ASSERT_EQ(U.cols(), buffer);
  ASSERT_EQ(X.cols(), buffer);
  ASSERT_EQ(EMU.cols(), buffer);
  ASSERT_EQ(EMO.cols(), buffer);
  ASSERT_EQ(EVAR.cols(), buffer);
  ASSERT_EQ(EMPTY.cols(), 0);

  EXPECT_DOUBLE_EQ(xkp1(0), X(0, 0));
  EXPECT_DOUBLE_EQ(u(0), U(0, 0));
  auto belief = b.estimate(Y.col(0), U.col(0));
  EXPECT_DOUBLE_EQ(belief.mean()(0), EMU(0, 0));
  EXPECT_DOUBLE_EQ(belief.mode()(0), EMO(0, 0));
  EXPECT_DOUBLE_EQ(belief.covariance()(0, 0), EVAR(0, 0));
}
