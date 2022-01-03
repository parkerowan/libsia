/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "tests/helpers.h"

#include <gtest/gtest.h>
#include <sia/sia.h>

TEST(Estimators, KF) {
  sia::LinearGaussianDynamics dynamics = createTestDynamics();
  sia::LinearGaussianMeasurement measurement = createTestMeasurement();
  sia::Gaussian prior(0, 10);
  sia::KF kf(dynamics, measurement, prior);

  Eigen::Matrix<double, 1, 1> y, u;
  y << 0.1;
  u << 1;
  sia::Gaussian belief = kf.estimate(y, u);
  EXPECT_NE(prior.mean()(0), belief.mean()(0));
  EXPECT_NE(prior.covariance()(0, 0), belief.covariance()(0, 0));

  kf.reset(prior);
  belief = kf.getBelief();
  EXPECT_DOUBLE_EQ(prior.mean()(0), belief.mean()(0));
  EXPECT_DOUBLE_EQ(prior.covariance()(0, 0), belief.covariance()(0, 0));
}

TEST(Estimators, EKF) {
  sia::LinearGaussianDynamics dynamics = createTestDynamics();
  sia::LinearGaussianMeasurement measurement = createTestMeasurement();
  sia::Gaussian prior(0, 10);
  sia::EKF ekf(dynamics, measurement, prior);

  Eigen::Matrix<double, 1, 1> y, u;
  y << 0.1;
  u << 1;
  sia::Gaussian belief = ekf.estimate(y, u);
  EXPECT_NE(prior.mean()(0), belief.mean()(0));
  EXPECT_NE(prior.covariance()(0, 0), belief.covariance()(0, 0));

  ekf.reset(prior);
  belief = ekf.getBelief();
  EXPECT_DOUBLE_EQ(prior.mean()(0), belief.mean()(0));
  EXPECT_DOUBLE_EQ(prior.covariance()(0, 0), belief.covariance()(0, 0));
}

TEST(Estimators, PF) {
  sia::LinearGaussianDynamics dynamics = createTestDynamics();
  sia::LinearGaussianMeasurement measurement = createTestMeasurement();
  Eigen::Matrix<double, 1, 1> mu, sigma;
  mu << 0;
  sigma << 10;
  sia::Particles prior = sia::Particles::gaussian(mu, sigma, 1000);
  sia::PF pf(dynamics, measurement, prior, 1.0, 0.01);

  Eigen::Matrix<double, 1, 1> y, u;
  y << 0.1;
  u << 1;
  sia::Particles belief = pf.estimate(y, u);
  EXPECT_NE(prior.mean()(0), belief.mean()(0));
  EXPECT_NE(prior.covariance()(0, 0), belief.covariance()(0, 0));

  // Run a second pass so that the roughening and resampling and performed
  sia::Particles belief2 = pf.estimate(y, u);
  EXPECT_NE(belief.mean()(0), belief2.mean()(0));
  EXPECT_NE(belief.covariance()(0, 0), belief2.covariance()(0, 0));

  pf.reset(prior);
  belief = pf.getBelief();
  EXPECT_DOUBLE_EQ(prior.mean()(0), belief.mean()(0));
  EXPECT_DOUBLE_EQ(prior.covariance()(0, 0), belief.covariance()(0, 0));
}
