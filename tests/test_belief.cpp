/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>
#include <cmath>
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

TEST(Belief, Deterministic) {
  sia::Deterministic a(1);
  ASSERT_EQ(a.dimension(), 1);
  EXPECT_DOUBLE_EQ(a.mean()(0), 1);
  EXPECT_DOUBLE_EQ(a.mode()(0), 1);

  a.setValue(-2 * Eigen::VectorXd::Ones(1));
  EXPECT_DOUBLE_EQ(a.mean()(0), -2);
  EXPECT_DOUBLE_EQ(a.mode()(0), -2);

  Eigen::Vector2d value{1, 2};
  sia::Deterministic b(value);
  ASSERT_EQ(b.dimension(), 2);
  EXPECT_DOUBLE_EQ(b.mean()(0), 1);
  EXPECT_DOUBLE_EQ(b.mean()(1), 2);
  EXPECT_TRUE(b.sample().isApprox(value));
  EXPECT_DOUBLE_EQ(b.logProb(value), 0);
  EXPECT_EQ(b.logProb(1.01 * value), -INFINITY);
  EXPECT_DOUBLE_EQ(b.covariance()(0, 0), 0);
  EXPECT_DOUBLE_EQ(b.covariance()(0, 1), 0);
  EXPECT_DOUBLE_EQ(b.covariance()(1, 0), 0);
  EXPECT_DOUBLE_EQ(b.covariance()(1, 1), 0);
}

TEST(Belief, Gaussian) {
  sia::Gaussian a(1);
  ASSERT_EQ(a.dimension(), 1);
  EXPECT_DOUBLE_EQ(a.mean()(0), 0);
  EXPECT_DOUBLE_EQ(a.covariance()(0, 0), 1);

  a.setMean(3 * Eigen::VectorXd::Ones(1));
  a.setCovariance(7 * Eigen::MatrixXd::Ones(1, 1));
  EXPECT_DOUBLE_EQ(a.mean()(0), 3);
  EXPECT_DOUBLE_EQ(a.covariance()(0, 0), 7);

  a.setMeanAndCov(5 * Eigen::VectorXd::Ones(1),
                  2 * Eigen::MatrixXd::Ones(1, 1));
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

  a.setLower(-2 * Eigen::VectorXd::Ones(1));
  a.setUpper(2 * Eigen::VectorXd::Ones(1));
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

TEST(Belief, Dirichlet) {
  sia::Dirichlet a(2);
  ASSERT_EQ(a.dimension(), 2);
  EXPECT_DOUBLE_EQ(a.alpha()(0), 1);
  EXPECT_DOUBLE_EQ(a.alpha()(1), 1);

  a.setAlpha(Eigen::Vector2d{2, 3});
  EXPECT_DOUBLE_EQ(a.alpha()(0), 2);
  EXPECT_DOUBLE_EQ(a.alpha()(1), 3);
  EXPECT_EQ(a.classify(), 1);

  // The mean and the mode are the same when the concentrations are the same
  double alpha = 3;
  double beta = 3;
  sia::Dirichlet b(alpha, beta);
  ASSERT_EQ(b.dimension(), 2);
  EXPECT_DOUBLE_EQ(b.alpha()(0), alpha);
  EXPECT_DOUBLE_EQ(b.alpha()(1), beta);

  Eigen::Vector2d x{0.1, 0.9};
  double logprob =
      log(pow(x(0), alpha - 1) * pow(x(1), beta - 1) / std::beta(alpha, beta));
  EXPECT_NEAR(b.logProb(x), logprob, 1e-6);

  // Expect samples outside of the support to return -inf
  EXPECT_DOUBLE_EQ(b.logProb(Eigen::Vector2d{-0.1, 0}), -INFINITY);

  std::size_t ns = 10000;
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(2, ns);
  for (std::size_t i = 0; i < ns; ++i) {
    s.col(i) = b.sample();
  }
  double n = static_cast<double>(ns);
  const Eigen::VectorXd mean = s.rowwise().sum() / n;
  const Eigen::MatrixXd e = (s.array().colwise() - b.mean().array()).matrix();
  const Eigen::MatrixXd cov = e * e.transpose() / (n - 1);
  EXPECT_NEAR(b.mean()(0), mean(0), 5e-2);
  EXPECT_NEAR(b.mean()(1), mean(1), 5e-2);
  EXPECT_NEAR(b.mode()(0), mean(0), 5e-2);
  EXPECT_NEAR(b.mode()(1), mean(1), 5e-2);
  EXPECT_NEAR(b.covariance()(0, 0), cov(0, 0), 5e-2);
  EXPECT_NEAR(b.covariance()(0, 1), cov(0, 1), 5e-2);
  EXPECT_NEAR(b.covariance()(1, 0), cov(1, 0), 5e-2);
  EXPECT_NEAR(b.covariance()(1, 1), cov(1, 1), 5e-2);

  const auto v = a.vectorize();
  EXPECT_TRUE(b.devectorize(v));
  EXPECT_TRUE(b.alpha().isApprox(a.alpha()));

  sia::Dirichlet c(b.alpha());
  EXPECT_TRUE(c.alpha().isApprox(b.alpha()));

  sia::Dirichlet d(Eigen::Vector2d{0.370833, 0.321336});
  EXPECT_NE(d.logProb(Eigen::Vector2d{1, 0}), INFINITY);
  EXPECT_NE(d.logProb(Eigen::Vector2d{1, 0}), -INFINITY);
}

TEST(Belief, Categorical) {
  sia::Categorical a(2);
  ASSERT_EQ(a.dimension(), 2);

  sia::Categorical b(Eigen::Vector2d{0.4, 0.6});
  ASSERT_EQ(b.dimension(), 2);
  EXPECT_DOUBLE_EQ(b.probs()(0), 0.4);
  EXPECT_DOUBLE_EQ(b.probs()(1), 0.6);

  EXPECT_DOUBLE_EQ(b.mean()(0), 0.4);
  EXPECT_DOUBLE_EQ(b.mean()(1), 0.6);

  EXPECT_DOUBLE_EQ(b.mode()(0), 0);
  EXPECT_DOUBLE_EQ(b.mode()(1), 1);

  EXPECT_EQ(b.classify(), 1);
  EXPECT_TRUE(b.oneHot(b.classify()).isApprox(b.mode()));
  EXPECT_EQ(b.category(b.probs()), b.classify());

  Eigen::Vector2d x{0.3, 0.7};
  double logprob = log(0.3 * 0.4 + 0.7 * 0.6);
  EXPECT_NEAR(b.logProb(x), logprob, 1e-6);

  std::size_t ns = 10000;
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(2, ns);
  for (std::size_t i = 0; i < ns; ++i) {
    s.col(i) = b.sample();
  }
  double n = static_cast<double>(ns);
  const Eigen::VectorXd mean = s.rowwise().sum() / n;
  const Eigen::MatrixXd e = (s.array().colwise() - b.mean().array()).matrix();
  const Eigen::MatrixXd cov = e * e.transpose() / (n - 1);
  EXPECT_NEAR(b.mean()(0), mean(0), 5e-2);
  EXPECT_NEAR(b.mean()(1), mean(1), 5e-2);
  EXPECT_NEAR(b.covariance()(0, 0), cov(0, 0), 5e-2);
  EXPECT_NEAR(b.covariance()(0, 1), cov(0, 1), 5e-2);
  EXPECT_NEAR(b.covariance()(1, 0), cov(1, 0), 5e-2);
  EXPECT_NEAR(b.covariance()(1, 1), cov(1, 1), 5e-2);

  const auto v = b.vectorize();
  EXPECT_TRUE(a.devectorize(v));
  EXPECT_TRUE(a.probs().isApprox(b.probs()));

  Eigen::MatrixXd Y = b.oneHot(Eigen::Vector3i{0, 1, 1});
  EXPECT_DOUBLE_EQ(Y(0, 0), 1);
  EXPECT_DOUBLE_EQ(Y(0, 1), 0);
  EXPECT_DOUBLE_EQ(Y(0, 2), 0);
  EXPECT_DOUBLE_EQ(Y(1, 0), 0);
  EXPECT_DOUBLE_EQ(Y(1, 1), 1);
  EXPECT_DOUBLE_EQ(Y(1, 2), 1);
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

TEST(Belief, KDE) {
  auto samples = sia::Particles::uniform(Eigen::Vector2d(-1, -2),
                                         Eigen::Vector2d(3, 4), 100);

  sia::EpanechnikovKernel epan_kernel(2);
  sia::KDE a(samples.values(), samples.weights(), epan_kernel);
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
  EXPECT_EQ(a.getBandwidthMode(), sia::KDE::BandwidthMode::SCOTT_RULE);
  a.setValues(a.values());
  EXPECT_TRUE(a.bandwidth().isApprox(h));

  a.setBandwidthScaling(1.5);
  EXPECT_DOUBLE_EQ(a.getBandwidthScaling(), 1.5);
  EXPECT_TRUE(a.bandwidth().isApprox(1.5 * h));

  a.setBandwidth(1.0);
  EXPECT_TRUE(a.bandwidth().isApprox(Eigen::Matrix2d::Identity()));
  EXPECT_EQ(a.getBandwidthMode(), sia::KDE::BandwidthMode::USER_SPECIFIED);
  EXPECT_DOUBLE_EQ(a.getBandwidthScaling(), 1.0);
  EXPECT_GT(a.probability(samples.value(0)), 0);

  a.setBandwidthMode(sia::KDE::BandwidthMode::SCOTT_RULE);
  EXPECT_EQ(a.getBandwidthMode(), sia::KDE::BandwidthMode::SCOTT_RULE);

  // Expect if user specified that silverman is used as initialize bandwidth
  sia::GaussianKernel gaussian_kernel(2);
  sia::KDE c(samples, gaussian_kernel, sia::KDE::BandwidthMode::USER_SPECIFIED);
  EXPECT_GT(c.probability(samples.value(0)), 0);
  EXPECT_EQ(c.getBandwidthMode(), sia::KDE::BandwidthMode::USER_SPECIFIED);
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
                          sia::GMM::FitMethod::KMEANS,
                          sia::GMM::InitMethod::WARM_START),
            1);
  EXPECT_EQ(gaussians.size(), 3);
  EXPECT_EQ(priors.size(), 3);

  EXPECT_EQ(sia::GMM::fit(S, gaussians, priors, gaussians.size(),
                          sia::GMM::FitMethod::GAUSSIAN_LIKELIHOOD,
                          sia::GMM::InitMethod::WARM_START),
            1);
  EXPECT_EQ(gaussians.size(), 3);
  EXPECT_EQ(priors.size(), 3);

  EXPECT_LE(sia::GMM::fit(S, gaussians, priors, gaussians.size(),
                          sia::GMM::FitMethod::KMEANS,
                          sia::GMM::InitMethod::STANDARD_RANDOM),
            2);
  EXPECT_EQ(gaussians.size(), 3);
  EXPECT_EQ(priors.size(), 3);

  EXPECT_EQ(sia::GMM::fit(S, gaussians, priors, gaussians.size(),
                          sia::GMM::FitMethod::GAUSSIAN_LIKELIHOOD,
                          sia::GMM::InitMethod::STANDARD_RANDOM),
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
  ASSERT_EQ(gmr.inputDim(), 1);
  ASSERT_EQ(gmr.outputDim(), 1);

  sia::Gaussian y0 = gmr.predict(0 * Eigen::VectorXd::Ones(1));
  sia::Gaussian y1 = gmr.predict(6 * Eigen::VectorXd::Ones(1));
  ASSERT_EQ(y0.dimension(), 1);
  ASSERT_EQ(y1.dimension(), 1);
  EXPECT_NEAR(y0.mean()(0), -7.0, 1e-3);
  EXPECT_NEAR(y1.mean()(0), -1.0, 1e-3);
  EXPECT_NEAR(y0.covariance()(0, 0), 1.0, 1e-3);
  EXPECT_NEAR(y1.covariance()(0, 0), 2.0, 1e-3);

  // Well outside the demonstration
  sia::Gaussian y2 = gmr.predict(100 * Eigen::VectorXd::Ones(1));
  ASSERT_EQ(y2.dimension(), 1);
  EXPECT_NEAR(y2.mean()(0), -1.0, 1e-3);
  EXPECT_NEAR(y2.covariance()(0, 0), 2.0, 1e-3);
}

TEST(Belief, GPR) {
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(3, 10);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Random(2, 10);
  sia::NoiseKernel noise_kernel = sia::NoiseKernel();
  EXPECT_EQ(noise_kernel.numHyperparameters(), 1);

  sia::SEKernel se_kernel = sia::SEKernel();
  EXPECT_EQ(se_kernel.numHyperparameters(), 2);

  sia::CompositeKernel kernel = se_kernel + noise_kernel;
  EXPECT_EQ(kernel.numHyperparameters(), 3);

  sia::GPR gpr(X, Y, kernel);

  EXPECT_EQ(gpr.numSamples(), 10);
  EXPECT_EQ(gpr.inputDim(), 3);
  EXPECT_EQ(gpr.outputDim(), 2);

  // Check the hyperparameters are written
  Eigen::VectorXd p = Eigen::Vector3d{0.1, 1.0, 0.1};
  gpr.setHyperparameters(p);
  const auto& pn = gpr.hyperparameters();
  EXPECT_TRUE(pn.isApprox(p));

  // Expect the log loss gradient to equate to the numerical approx
  const double GRAD_TOLERANCE = 1e-4;
  Eigen::VectorXd grad = gpr.negLogMarginalLikGrad();
  auto loss = [&](const Eigen::VectorXd& x) {
    gpr.setHyperparameters(x);
    return gpr.negLogMarginalLik();
  };
  Eigen::VectorXd grad_approx = sia::dfdx(loss, gpr.hyperparameters());
  Eigen::VectorXd e = grad - grad_approx;
  EXPECT_NEAR(sqrt(e.dot(e)), 0, GRAD_TOLERANCE);

  // Is there a theoretical bound on the error given the hyperparameters?
  const double EVAL_TOLERANCE = 1e-1;
  for (std::size_t i = 0; i < 10; ++i) {
    const auto& x = X.col(i);
    const auto& y = Y.col(i);
    sia::Gaussian g = gpr.predict(x);
    ASSERT_EQ(g.dimension(), 2);
    EXPECT_NEAR(g.mean()(0), y(0), EVAL_TOLERANCE);
    EXPECT_NEAR(g.mean()(1), y(1), EVAL_TOLERANCE);
  }

  // Expect after training that the log marginal likelihood has been reduced and
  // that the gradient is small
  grad = gpr.negLogMarginalLikGrad();
  double log_marg_loss = gpr.negLogMarginalLik();
  gpr.train();
  EXPECT_LT(gpr.negLogMarginalLik(), log_marg_loss);
  EXPECT_LT(gpr.negLogMarginalLikGrad().norm(), grad.norm());

  // Reset hyperparameters and only train a subset
  gpr.setHyperparameters(p);
  std::vector<std::size_t> hp_indices{0, 1};
  gpr.train(hp_indices);
  EXPECT_NE(gpr.hyperparameters()(0), p(0));
  EXPECT_NE(gpr.hyperparameters()(1), p(1));
  EXPECT_DOUBLE_EQ(gpr.hyperparameters()(2), p(2));

  // Test that GPR computes for a variety of hyperameters
  Eigen::MatrixXd Xt_offense{};
  Eigen::MatrixXd Yt_offense{};
  Eigen::VectorXd x_offense{};
  Eigen::VectorXd hp_offense{};
  std::size_t n_x = 1;
  std::size_t n_y = 1;
  std::size_t n_hparams = 100;
  auto n_train = std::vector<std::size_t>{0, 1, 5, 100};
  sia::Uniform log_domain(Eigen::Vector3d{-6, -6, -6},
                          Eigen::Vector3d{6, 6, 6});
  Eigen::MatrixXd Xt = Eigen::MatrixXd::Random(n_x, 100);
  Eigen::MatrixXd Yt = Eigen::MatrixXd::Random(n_y, 100);
  Eigen::VectorXd x = Eigen::VectorXd::Random(n_x);
  for (std::size_t i = 0; i < n_hparams; ++i) {
    for (std::size_t j = 0; j < n_train.size(); ++j) {
      Eigen::VectorXd hp = log_domain.sample();
      double length = pow(10.0, hp(0));
      double signal_var = pow(10.0, hp(1));
      double noise_var = pow(10.0, hp(2));

      noise_kernel = sia::NoiseKernel(noise_var);
      se_kernel = sia::SEKernel(length, signal_var);
      auto kernel2 = noise_kernel + se_kernel;
      sia::GPR gpr3(n_x, n_y, kernel2);

      // Test that GPR will compute with j data points
      if (n_train.at(j) > 0) {
        gpr3.setData(Xt.leftCols(n_train.at(j)), Yt.leftCols(n_train.at(j)));
      }
      EXPECT_NO_THROW(gpr3.predict(x));
      for (int j = 0; j < Xt.cols(); ++j) {
        EXPECT_NO_THROW(gpr3.predict(Xt.col(j)));
      }
    }
  }

  // Test other kernels
  auto var_function = [](const Eigen::VectorXd& x) {
    Eigen::VectorXd y(1);
    y << x.dot(x);
    return y;
  };
  sia::VariableNoiseKernel kernel_1(var_function);
  Eigen::Vector2d a, b;
  a << 1, 0;
  b << 1, 0;
  EXPECT_DOUBLE_EQ(kernel_1.eval(a, 0), 1.0);
  EXPECT_DOUBLE_EQ(kernel_1.eval(a, b, 0), 0.0);
  b << 2, 0;
  EXPECT_DOUBLE_EQ(kernel_1.eval(a, b, 0), 0.0);
  EXPECT_EQ(kernel_1.grad(a, b, 0).size(), 0);
  EXPECT_EQ(kernel_1.hyperparameters().size(), 0);
  EXPECT_NO_THROW(kernel_1.setHyperparameters(Eigen::VectorXd(0)));
  EXPECT_THROW(kernel_1.setHyperparameters(Eigen::VectorXd(1)),
               std::runtime_error);
}

TEST(Belief, GPC) {
  double alpha = 0.1;
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(3, 10);
  Eigen::VectorXi Y = Eigen::VectorXi::Zero(10);
  for (std::size_t i = 0; i < 10; i += 2) {
    Y(i) = 1;
  }
  auto kernel = sia::SEKernel();
  sia::GPC gpc(X, Y, kernel, alpha);
  EXPECT_DOUBLE_EQ(gpc.alpha(), alpha);

  gpc.setAlpha(0.001);
  EXPECT_DOUBLE_EQ(gpc.alpha(), 0.001);

  EXPECT_EQ(gpc.numSamples(), 10);
  EXPECT_EQ(gpc.inputDim(), 3);
  EXPECT_EQ(gpc.outputDim(), 2);

  ASSERT_EQ(gpc.numHyperparameters(), 2);
  Eigen::VectorXd p = Eigen::Vector2d{0.1, 1.0};
  gpc.setHyperparameters(p);
  const auto& pn = gpc.hyperparameters();
  EXPECT_TRUE(pn.isApprox(p));

  double loss = gpc.negLogMarginalLik();
  gpc.train();
  EXPECT_LT(gpc.negLogMarginalLik(), loss);

  const double GRAD_TOLERANCE = 2e-2;
  Eigen::VectorXd grad = gpc.negLogMarginalLikGrad();
  EXPECT_NEAR(grad.norm(), 0, GRAD_TOLERANCE);

  sia::Categorical cat(gpc.outputDim());
  Eigen::MatrixXd Yoh = cat.oneHot(Y);
  for (std::size_t i = 0; i < 10; ++i) {
    const auto& x = X.col(i);
    sia::Dirichlet p = gpc.predict(x);
    ASSERT_EQ(p.dimension(), 2);
    EXPECT_EQ(p.classify(), Y(i));
  }
}
