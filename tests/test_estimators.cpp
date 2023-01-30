/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "tests/helpers.h"

#include <gtest/gtest.h>
#include <sia/sia.h>

TEST(Estimators, KF) {
  sia::LinearGaussianDynamics dynamics = createTestDynamics();
  sia::LinearGaussianMeasurement measurement = createTestMeasurement();
  sia::Gaussian prior(0, 10);
  sia::KF kf(dynamics, measurement, prior);

  sia::Gaussian belief = kf.belief();
  EXPECT_DOUBLE_EQ(prior.mean()(0), belief.mean()(0));
  EXPECT_DOUBLE_EQ(prior.covariance()(0, 0), belief.covariance()(0, 0));

  Eigen::Matrix<double, 1, 1> y, u;
  y << 0.1;
  u << 1;
  belief = kf.estimate(y, u);
  const auto& metrics = kf.metrics();
  EXPECT_GT(metrics.elapsed_us, 0);

  EXPECT_NE(prior.mean()(0), belief.mean()(0));
  EXPECT_NE(prior.covariance()(0, 0), belief.covariance()(0, 0));
}

TEST(Estimators, EKF) {
  sia::LinearGaussianDynamics dynamics = createTestDynamics();
  sia::LinearGaussianMeasurement measurement = createTestMeasurement();
  sia::Gaussian prior(0, 10);
  sia::EKF ekf(dynamics, measurement, prior);

  sia::Gaussian belief = ekf.belief();
  EXPECT_DOUBLE_EQ(prior.mean()(0), belief.mean()(0));
  EXPECT_DOUBLE_EQ(prior.covariance()(0, 0), belief.covariance()(0, 0));

  Eigen::Matrix<double, 1, 1> y, u;
  y << 0.1;
  u << 1;
  belief = ekf.estimate(y, u);
  const auto& metrics = ekf.metrics();
  EXPECT_GT(metrics.elapsed_us, 0);

  EXPECT_NE(prior.mean()(0), belief.mean()(0));
  EXPECT_NE(prior.covariance()(0, 0), belief.covariance()(0, 0));
}

TEST(Estimators, PF) {
  sia::LinearGaussianDynamics dynamics = createTestDynamics();
  sia::LinearGaussianMeasurement measurement = createTestMeasurement();
  Eigen::Matrix<double, 1, 1> mu, sigma;
  mu << 0;
  sigma << 10;
  sia::Particles prior = sia::Particles::gaussian(mu, sigma, 1000);
  sia::PF::Options options{};
  options.resample_threshold = 1.0;
  options.roughening_factor = 0.0;
  sia::PF pf(dynamics, measurement, prior, options);

  sia::Particles belief = pf.belief();
  EXPECT_DOUBLE_EQ(prior.mean()(0), belief.mean()(0));
  EXPECT_DOUBLE_EQ(prior.covariance()(0, 0), belief.covariance()(0, 0));

  Eigen::Matrix<double, 1, 1> y, u;
  y << 0.1;
  u << 1;
  belief = pf.estimate(y, u);
  const auto& metrics = pf.metrics();
  EXPECT_GT(metrics.elapsed_us, 0);

  EXPECT_NE(prior.mean()(0), belief.mean()(0));
  EXPECT_NE(prior.covariance()(0, 0), belief.covariance()(0, 0));

  // Run a second pass so that the roughening and resampling and performed
  sia::Particles belief2 = pf.estimate(y, u);
  EXPECT_NE(belief.mean()(0), belief2.mean()(0));
  EXPECT_NE(belief.covariance()(0, 0), belief2.covariance()(0, 0));
}
