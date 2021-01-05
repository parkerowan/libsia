/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>

TEST(Belief, gaussian) {
  sia::Gaussian a(1);
  ASSERT_EQ(a.dimension(), 1);
  EXPECT_DOUBLE_EQ(a.mean()(0), 0);
  EXPECT_DOUBLE_EQ(a.covariance()(0, 0), 1);

  EXPECT_TRUE(a.setMean(5 * Eigen::VectorXd::Ones(1)));
  EXPECT_TRUE(a.setCovariance(2 * Eigen::MatrixXd::Ones(1, 1)));
  EXPECT_DOUBLE_EQ(a.mean()(0), 5);
  EXPECT_DOUBLE_EQ(a.covariance()(0, 0), 2);

  sia::Gaussian b(10, 3);
  ASSERT_EQ(b.dimension(), 1);
  EXPECT_DOUBLE_EQ(b.mean()(0), 10);
  EXPECT_DOUBLE_EQ(b.covariance()(0, 0), 3);

  double x = 8;
  double mahal = sqrt(pow(x - 10, 2) / 3);
  double logprob = log(1 / (sqrt(3 * 2 * M_PI))) - pow(mahal, 2) / 2;
  EXPECT_DOUBLE_EQ(b.mahalanobis(x * Eigen::VectorXd::Ones(1)), mahal);
  EXPECT_DOUBLE_EQ(b.logProb(x * Eigen::VectorXd::Ones(1)), logprob);

  std::size_t ns = 10000;
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(1, ns);
  for (std::size_t i = 0; i < ns; ++i) {
    s.col(i) = b.sample();
  }
  double n = static_cast<double>(ns);
  const Eigen::VectorXd mean = s.rowwise().sum() / n;
  const Eigen::MatrixXd e = (s.array().colwise() - b.mean().array()).matrix();
  const Eigen::MatrixXd cov = e * e.transpose() / (n - 1);
  EXPECT_NEAR(b.mean()(0), mean(0), 5e-2);
  EXPECT_NEAR(b.mode()(0), mean(0), 5e-2);
  EXPECT_NEAR(b.covariance()(0, 0), cov(0, 0), 5e-2);

  Eigen::Vector2d mu;
  mu << 1, 2;
  Eigen::Matrix2d sigma;
  sigma << 5, 1, 1, 3;
  sia::Gaussian c(mu, sigma);
  ASSERT_EQ(c.dimension(), 2);
  EXPECT_TRUE(c.mean().isApprox(mu));
  EXPECT_TRUE(c.mode().isApprox(mu));
  EXPECT_TRUE(c.covariance().isApprox(sigma));

  const auto v = c.vectorize();
  EXPECT_TRUE(c.devectorize(v));
  EXPECT_TRUE(c.mean().isApprox(mu));
  EXPECT_TRUE(c.covariance().isApprox(sigma));
}

TEST(Belief, uniform) {
  sia::Uniform a(1);
  ASSERT_EQ(a.dimension(), 1);
  EXPECT_DOUBLE_EQ(a.lower()(0), 0);
  EXPECT_DOUBLE_EQ(a.upper()(0), 1);

  EXPECT_TRUE(a.setLower(-2 * Eigen::VectorXd::Ones(1)));
  EXPECT_TRUE(a.setUpper(2 * Eigen::VectorXd::Ones(1)));
  EXPECT_DOUBLE_EQ(a.lower()(0), -2);
  EXPECT_DOUBLE_EQ(a.upper()(0), 2);

  sia::Uniform b(1, 3);
  ASSERT_EQ(b.dimension(), 1);
  EXPECT_DOUBLE_EQ(b.lower()(0), 1);
  EXPECT_DOUBLE_EQ(b.upper()(0), 3);

  double x = 2;
  double logprob = log(1.0 / (3.0 - 1.0));
  EXPECT_DOUBLE_EQ(b.logProb(x * Eigen::VectorXd::Ones(1)), logprob);

  // Expect samples outside of the support to return -inf
  EXPECT_DOUBLE_EQ(b.logProb(4.0 * Eigen::VectorXd::Ones(1)), -INFINITY);

  std::size_t ns = 10000;
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(1, ns);
  for (std::size_t i = 0; i < ns; ++i) {
    s.col(i) = b.sample();
  }
  double n = static_cast<double>(ns);
  const Eigen::VectorXd mean = s.rowwise().sum() / n;
  const Eigen::MatrixXd e = (s.array().colwise() - b.mean().array()).matrix();
  const Eigen::MatrixXd cov = e * e.transpose() / (n - 1);
  EXPECT_NEAR(b.mean()(0), mean(0), 5e-2);
  EXPECT_NEAR(b.mode()(0), mean(0), 5e-2);
  EXPECT_NEAR(b.covariance()(0, 0), cov(0, 0), 5e-2);

  Eigen::Vector2d lower;
  lower << 1, 2;
  Eigen::Vector2d upper;
  upper << 4, 5;
  sia::Uniform c(lower, upper);
  ASSERT_EQ(c.dimension(), 2);
  EXPECT_TRUE(c.lower().isApprox(lower));
  EXPECT_TRUE(c.upper().isApprox(upper));

  const auto v = c.vectorize();
  EXPECT_TRUE(c.devectorize(v));
  EXPECT_TRUE(c.lower().isApprox(lower));
  EXPECT_TRUE(c.upper().isApprox(upper));
}

TEST(Belief, particles) {
  sia::Particles a(2, 100);
  EXPECT_EQ(a.dimension(), 2);
  EXPECT_EQ(a.numParticles(), 100);
  EXPECT_TRUE(a.values().isApprox(Eigen::MatrixXd::Zero(2, 100)));
  EXPECT_TRUE(a.weights().isApprox(Eigen::VectorXd::Ones(100) / 100));

  a.setUseWeightedStats(true);
  EXPECT_EQ(a.getUseWeightedStats(), true);

  Eigen::MatrixXd values(2, 3);
  values << 1, 2, 4, 1, 3, 6;
  Eigen::VectorXd weights = Eigen::VectorXd::Ones(3) / 3;

  // Use unweighted stats
  sia::Particles b(values, weights, false);
  EXPECT_EQ(b.dimension(), 2);
  EXPECT_EQ(b.numParticles(), 3);
  EXPECT_TRUE(b.value(0).isApprox(values.col(0)));
  EXPECT_TRUE(b.value(1).isApprox(values.col(1)));
  EXPECT_EQ(b.values().rows(), 2);
  EXPECT_EQ(b.values().cols(), 3);
  EXPECT_DOUBLE_EQ(b.weight(0), 1.0 / 3.0);
  EXPECT_TRUE(b.weights().isApprox(Eigen::VectorXd::Ones(3) / 3));

  const auto v = b.vectorize();
  EXPECT_TRUE(b.devectorize(v));
  EXPECT_TRUE(b.values().isApprox(values));
  EXPECT_TRUE(b.weights().isApprox(weights));

  // Assign deterministic weights and check lookups
  b.setWeights(Eigen::Vector3d(0, 1, 0));
  EXPECT_TRUE(b.sample().isApprox(values.col(1)));
  EXPECT_DOUBLE_EQ(b.logProb(values.col(0)), -INFINITY);
  EXPECT_DOUBLE_EQ(b.logProb(values.col(1)), log(1));
  EXPECT_DOUBLE_EQ(b.logProb(values.col(2)), -INFINITY);

  // Mean and covariance computed using numpy mean and cov functions
  EXPECT_TRUE(b.mean().isApprox(Eigen::Vector2d(7.0, 10.0) / 3.0));
  EXPECT_TRUE(b.mode().isApprox(Eigen::Vector2d(7.0, 10.0) / 3.0));
  Eigen::Matrix2d cov;
  cov << 7.0, 11.5, 11.5, 19.0;
  cov /= 3.0;
  EXPECT_TRUE(b.covariance().isApprox(cov));

  // Use weighted stats - computed using numpy mean and cov functions
  b.setWeights(Eigen::Vector3d(0.1, 0.4, 0.5));
  EXPECT_FALSE(b.getUseWeightedStats());
  b.setUseWeightedStats(true);
  EXPECT_TRUE(b.getUseWeightedStats());
  EXPECT_TRUE(b.mean().isApprox(Eigen::Vector2d(2.9, 4.3)));
  EXPECT_TRUE(b.mode().isApprox(Eigen::Vector2d(4.0, 6.0)));
  cov << 2.224137931034483, 3.5, 3.5, 5.53448275862069;
  EXPECT_TRUE(b.covariance().isApprox(cov));

  std::size_t ns = 10000;
  auto c = sia::Particles::gaussian(Eigen::VectorXd::Zero(1),
                                    Eigen::MatrixXd::Ones(1, 1), ns);
  sia::Gaussian g(0, 1);
  ASSERT_EQ(c.dimension(), 1);
  EXPECT_NEAR(c.mean()(0), g.mean()(0), 5e-2);
  EXPECT_NEAR(c.covariance()(0, 0), g.covariance()(0, 0), 5e-2);

  auto d = sia::Particles::uniform(Eigen::VectorXd::Zero(1),
                                   Eigen::VectorXd::Ones(1), ns);
  sia::Uniform u(0, 1);
  ASSERT_EQ(d.dimension(), 1);
  EXPECT_NEAR(d.mean()(0), u.mean()(0), 5e-2);
  EXPECT_NEAR(d.covariance()(0, 0), u.covariance()(0, 0), 5e-2);
}

TEST(Belief, kernel) {
  Eigen::VectorXd z = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd dz = 0.01 * Eigen::VectorXd::Ones(3);

  sia::UniformKernel a(z.size());
  EXPECT_EQ(a.type(), sia::Kernel::UNIFORM);
  EXPECT_GE(a.evaluate(z), a.evaluate(z + dz));
  EXPECT_GE(a.evaluate(z), a.evaluate(z - dz));
  EXPECT_DOUBLE_EQ(a.evaluate(z + dz), a.evaluate(z - dz));

  sia::GaussianKernel b(z.size());
  EXPECT_EQ(b.type(), sia::Kernel::GAUSSIAN);
  EXPECT_GT(b.evaluate(z), b.evaluate(z + dz));
  EXPECT_GT(b.evaluate(z), b.evaluate(z - dz));
  EXPECT_DOUBLE_EQ(b.evaluate(z + dz), b.evaluate(z - dz));

  sia::EpanechnikovKernel c(z.size());
  EXPECT_EQ(c.type(), sia::Kernel::EPANECHNIKOV);
  EXPECT_GT(c.evaluate(z), c.evaluate(z + dz));
  EXPECT_GT(c.evaluate(z), c.evaluate(z - dz));
  EXPECT_DOUBLE_EQ(c.evaluate(z + dz), c.evaluate(z - dz));
}

TEST(Belief, kernelDensity) {
  auto samples = sia::Particles::uniform(Eigen::Vector2d(-1, -2),
                                         Eigen::Vector2d(3, 4), 100);

  sia::KernelDensity a(samples.values(), samples.weights());
  EXPECT_EQ(a.getKernelType(), sia::Kernel::EPANECHNIKOV);
  EXPECT_GT(a.probability(samples.value(0)), 0);
  ASSERT_EQ(a.dimension(), 2);
  EXPECT_EQ(a.numParticles(), 100);
  EXPECT_GT(a.logProb(samples.value(0)), -INFINITY);
  EXPECT_FALSE(a.mean().isApprox(Eigen::Vector2d::Zero()));
  EXPECT_FALSE(a.mode().isApprox(Eigen::Vector2d::Zero()));
  EXPECT_FALSE(a.covariance().isApprox(Eigen::Matrix2d::Zero()));
  EXPECT_FALSE(a.covariance().isApprox(Eigen::Matrix2d::Identity()));

  const auto h = a.bandwidth();
  EXPECT_DOUBLE_EQ(a.getBandwidthScaling(), 1.0);
  EXPECT_EQ(a.getBandwidthMode(), sia::KernelDensity::SCOTT_RULE);
  a.setValues(a.values());
  EXPECT_TRUE(a.bandwidth().isApprox(h));

  a.setBandwidthScaling(1.5);
  EXPECT_DOUBLE_EQ(a.getBandwidthScaling(), 1.5);
  EXPECT_TRUE(a.bandwidth().isApprox(1.5 * h));

  a.setBandwidth(1.0);
  EXPECT_TRUE(a.bandwidth().isApprox(Eigen::Matrix2d::Identity()));
  EXPECT_EQ(a.getBandwidthMode(), sia::KernelDensity::USER_SPECIFIED);
  EXPECT_DOUBLE_EQ(a.getBandwidthScaling(), 1.0);

  a.setKernelType(sia::Kernel::UNIFORM);
  EXPECT_EQ(a.getKernelType(), sia::Kernel::UNIFORM);

  // Expect if user specified that silverman is used as initialize bandwidth
  sia::KernelDensity c(samples, sia::Kernel::GAUSSIAN,
                       sia::KernelDensity::USER_SPECIFIED);
  EXPECT_EQ(c.getKernelType(), sia::Kernel::GAUSSIAN);
  EXPECT_EQ(c.getBandwidthMode(), sia::KernelDensity::USER_SPECIFIED);
  EXPECT_TRUE(c.bandwidth().isApprox(h));
  EXPECT_DOUBLE_EQ(c.getBandwidthScaling(), 1.0);

  const auto H = c.bandwidth();
  const auto v = c.vectorize();
  EXPECT_TRUE(c.devectorize(v));
  EXPECT_TRUE(c.values().isApprox(samples.values()));
  EXPECT_TRUE(c.weights().isApprox(samples.weights()));
  EXPECT_TRUE(c.bandwidth().isApprox(H));
}
