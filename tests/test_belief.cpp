/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>
#include <limits>

TEST(Belief, Generator) {
  std::uniform_int_distribution<long> distribution(
      0, std::numeric_limits<long>::max());

  sia::Generator::instance().seed(0);
  std::size_t nsamples = 100;
  long p0[nsamples] = {};
  for (std::size_t i = 0; i < nsamples; ++i) {
    p0[i] = distribution(sia::Generator::instance().engine());
  }

  sia::Generator::instance().seed(0);
  long p1[nsamples] = {};
  for (std::size_t i = 0; i < nsamples; ++i) {
    p1[i] = distribution(sia::Generator::instance().engine());
  }

  sia::Generator::instance().seed(1);
  long p2[nsamples] = {};
  for (std::size_t i = 0; i < nsamples; ++i) {
    p2[i] = distribution(sia::Generator::instance().engine());
  }

  for (std::size_t i = 0; i < nsamples; ++i) {
    EXPECT_EQ(p0[i], p1[i]);  // same seed
    EXPECT_NE(p0[i], p2[i]);  // different seed
  }
}

TEST(Belief, Gaussian) {
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
  EXPECT_DOUBLE_EQ(c.logProb(mu), c.maxLogProb());

  const auto v = c.vectorize();
  EXPECT_TRUE(c.devectorize(v));
  EXPECT_TRUE(c.mean().isApprox(mu));
  EXPECT_TRUE(c.covariance().isApprox(sigma));

  // Case where original LDLT decomposition failed
  mu << 1.04865, -0.312572;
  sigma << 0.149517, 0.0102682, 0.0102682, 0.370523;
  sia::Gaussian g(mu, sigma);
  EXPECT_DOUBLE_EQ(g.logProb(mu), g.maxLogProb());
}

TEST(Belief, Uniform) {
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

TEST(Belief, Particles) {
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

TEST(Belief, Kernel) {
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

TEST(Belief, KernelDensity) {
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

TEST(Belief, GMM) {
  sia::GMM a(1, 1);
  EXPECT_EQ(a.numClusters(), 1);
  ASSERT_EQ(a.dimension(), 1);
  EXPECT_DOUBLE_EQ(a.mean()(0), 0);
  EXPECT_DOUBLE_EQ(a.mode()(0), 0);
  EXPECT_DOUBLE_EQ(a.covariance()(0, 0), 1);
  EXPECT_DOUBLE_EQ(a.prior(0), 1);
  EXPECT_DOUBLE_EQ(a.gaussian(0).mean()(0), 0);
  EXPECT_DOUBLE_EQ(a.gaussian(0).covariance()(0, 0), 1);

  sia::GMM b(2, 5);
  EXPECT_EQ(b.numClusters(), 2);
  ASSERT_EQ(b.dimension(), 5);

  sia::Gaussian g0(0.0, 2.0);
  sia::Gaussian g1(1.0, 1.0);
  std::vector<sia::Gaussian> gaussians{g0, g1};
  std::vector<double> priors{0.4, 0.6};
  sia::GMM c(gaussians, priors);

  EXPECT_EQ(c.numClusters(), 2);
  ASSERT_EQ(c.dimension(), 1);

  double mu = 0.4 * 0.0 + 0.6 * 1.0;
  double sigma = 0.4 * (2.0 + (0.0 - mu) * (0.0 - mu)) +
                 0.6 * (1.0 + (1.0 - mu) * (1.0 - mu));

  Eigen::VectorXd x(1);
  x << mu;
  double logProb = log(0.4 * exp(g0.logProb(x)) + 0.6 * exp(g1.logProb(x)));
  EXPECT_DOUBLE_EQ(c.mode()(0), 1.0);
  EXPECT_DOUBLE_EQ(c.mean()(0), mu);
  EXPECT_DOUBLE_EQ(c.covariance()(0, 0), sigma);
  EXPECT_DOUBLE_EQ(c.logProb(x), logProb);

  std::size_t ns = 10000;
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(1, ns);
  for (std::size_t i = 0; i < ns; ++i) {
    s.col(i) = c.sample();
  }
  double n = static_cast<double>(ns);
  const Eigen::VectorXd mean = s.rowwise().sum() / n;
  const Eigen::MatrixXd e = (s.array().colwise() - c.mean().array()).matrix();
  const Eigen::MatrixXd cov = e * e.transpose() / (n - 1);
  EXPECT_NEAR(c.mean()(0), mean(0), 5e-2);
  EXPECT_NEAR(c.mode()(0), 1.0, 5e-2);
  EXPECT_NEAR(c.covariance()(0, 0), cov(0, 0), 5e-2);

  const auto v = c.vectorize();
  EXPECT_TRUE(c.devectorize(v));
  EXPECT_DOUBLE_EQ(c.mean()(0), mu);
  EXPECT_DOUBLE_EQ(c.covariance()(0, 0), sigma);

  // Classification
  sia::GMM d(std::vector<sia::Gaussian>{sia::Gaussian(0.0, 1.0),
                                        sia::Gaussian(1.0, 1.0)},
             std::vector<double>{0.5, 0.5});
  EXPECT_EQ(d.classify(0.0 * Eigen::VectorXd::Ones(1)), 0);
  EXPECT_EQ(d.classify(1.0 * Eigen::VectorXd::Ones(1)), 1);

  // Fit
  Eigen::MatrixXd S(2, 5);
  S << 1, 3, 5, 7, 9, 2, 4, 6, 8, 10;
  gaussians = std::vector<sia::Gaussian>{
      sia::Gaussian(Eigen::Vector2d{3, 3}, Eigen::Matrix2d::Identity()),
      sia::Gaussian(Eigen::Vector2d{6, 6}, Eigen::Matrix2d::Identity()),
      sia::Gaussian(Eigen::Vector2d{9, 9}, Eigen::Matrix2d::Identity())};
  priors = std::vector<double>{0.3, 0.3, 0.4};
  EXPECT_EQ(sia::GMM::fit(S, gaussians, priors, gaussians.size(),
                          sia::GMM::KMEANS, sia::GMM::WARM_START),
            1);
  EXPECT_EQ(gaussians.size(), 3);
  EXPECT_EQ(priors.size(), 3);

  EXPECT_EQ(sia::GMM::fit(S, gaussians, priors, gaussians.size(),
                          sia::GMM::GAUSSIAN_LIKELIHOOD, sia::GMM::WARM_START),
            1);
  EXPECT_EQ(gaussians.size(), 3);
  EXPECT_EQ(priors.size(), 3);

  EXPECT_LE(sia::GMM::fit(S, gaussians, priors, gaussians.size(),
                          sia::GMM::KMEANS, sia::GMM::STANDARD_RANDOM),
            2);
  EXPECT_EQ(gaussians.size(), 3);
  EXPECT_EQ(priors.size(), 3);

  EXPECT_EQ(
      sia::GMM::fit(S, gaussians, priors, gaussians.size(),
                    sia::GMM::GAUSSIAN_LIKELIHOOD, sia::GMM::STANDARD_RANDOM),
      1);
  EXPECT_EQ(gaussians.size(), 3);
  EXPECT_EQ(priors.size(), 3);

  sia::Particles samples = sia::Particles::gaussian(
      Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity(), 10);
  sia::GMM gmm_new(samples.values(), 3);
  EXPECT_EQ(gmm_new.numClusters(), 3);
  EXPECT_EQ(gmm_new.gaussians().size(), 3);
  EXPECT_EQ(gmm_new.priors().size(), 3);
  EXPECT_EQ(gmm_new.dimension(), 2);
}

TEST(Belief, GMR) {
  sia::Gaussian g0(Eigen::Vector2d{0, -7}, 1 * Eigen::Matrix2d::Identity());
  sia::Gaussian g1(Eigen::Vector2d{6, -1}, 2 * Eigen::Matrix2d::Identity());
  sia::GMM gmm(std::vector<sia::Gaussian>{g0, g1},
               std::vector<double>{0.5, 0.5});

  std::vector<std::size_t> ix{0};
  std::vector<std::size_t> ox{1};
  sia::GMR gmr(gmm, ix, ox);

  sia::Gaussian y0 = gmr.predict(0 * Eigen::VectorXd::Ones(1));
  sia::Gaussian y1 = gmr.predict(6 * Eigen::VectorXd::Ones(1));
  ASSERT_EQ(y0.dimension(), 1);
  ASSERT_EQ(y1.dimension(), 1);
  EXPECT_NEAR(y0.mean()(0), -7.0, 1e-3);
  EXPECT_NEAR(y1.mean()(0), -1.0, 1e-3);
  EXPECT_NEAR(y0.covariance()(0, 0), 1.0, 1e-3);
  EXPECT_NEAR(y1.covariance()(0, 0), 2.0, 1e-3);
}