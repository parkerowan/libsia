/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gmm.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

#include <algorithm>

namespace sia {

GMM::GMM(std::size_t K, std::size_t dimension)
    : Distribution(Generator::instance()),
      m_belief(K),
      m_num_clusters(K),
      m_dimension(dimension) {
  for (std::size_t i = 0; i < K; ++i) {
    m_gaussians.emplace_back(Gaussian(dimension));
    m_priors.emplace_back(1.0 / double(K));
  }
}

GMM::GMM(const std::vector<Gaussian>& gaussians,
         const std::vector<double>& priors)
    : Distribution(Generator::instance()),
      m_belief(gaussians.size()),
      m_gaussians(gaussians),
      m_priors(priors) {
  bool r1 = m_gaussians.size() == m_priors.size();
  bool r2 = m_gaussians.size() > 0;
  SIA_THROW_IF_NOT(
      r1, "Number of clusters and priors in GMM constructor do not match");
  SIA_THROW_IF_NOT(r2, "GMM constructor needs one or more clusters");
  m_num_clusters = gaussians.size();
  m_dimension = gaussians[0].dimension();
}

GMM::GMM(const Eigen::MatrixXd& samples, std::size_t K, double regularization)
    : Distribution(Generator::instance()),
      m_belief(K),
      m_num_clusters(K),
      m_dimension(samples.rows()) {
  // Initialize via kmeans
  fit(samples, m_gaussians, m_priors, K, GMM::FitMethod::KMEANS,
      GMM::InitMethod::STANDARD_RANDOM, regularization);

  // Perform EM to learn full covariance
  fit(samples, m_gaussians, m_priors, K, GMM::FitMethod::GAUSSIAN_LIKELIHOOD,
      GMM::InitMethod::WARM_START, regularization);
}

std::size_t GMM::dimension() const {
  assert(m_gaussians.size() > 0);
  return m_dimension;
}

const Eigen::VectorXd GMM::sample() {
  // Use inverse CDF sampling to sample from mixture priors
  std::vector<double> cdf(m_priors.size());
  double prev_weight = 0;
  for (std::size_t i = 0; i < m_priors.size(); ++i) {
    cdf[i] = m_priors[i] + prev_weight;
    prev_weight = cdf[i];
  }

  // set up a uniform sample generator \in [0,1] and sample the class
  double s = m_uniform(m_rng);
  unsigned mixture = 0;
  while (s > cdf[mixture]) {
    mixture++;
  }

  // Sample from associated Gaussian
  return m_gaussians[mixture].sample();
}

double GMM::logProb(const Eigen::VectorXd& x) const {
  // logProb = log(sum w_i N_i)
  double prob = 0;
  for (std::size_t i = 0; i < m_priors.size(); ++i) {
    prob += m_priors[i] * exp(m_gaussians[i].logProb(x));
  }
  return log(prob);
}

const Eigen::VectorXd GMM::mean() const {
  // Weighted sum of gaussian means
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(dimension());
  for (std::size_t i = 0; i < m_priors.size(); ++i) {
    mu += m_priors[i] * m_gaussians[i].mean();
  }
  return mu;
}

const Eigen::VectorXd GMM::mode() const {
  // Compute the weighted max likelihood for each cluster
  Eigen::VectorXd max_lik = Eigen::VectorXd::Zero(m_priors.size());
  for (std::size_t i = 0; i < m_priors.size(); ++i) {
    max_lik(i) = m_priors[i] * exp(m_gaussians[i].maxLogProb());
  }

  // Mean/mode of highest likelihood
  int r, c;
  max_lik.maxCoeff(&r, &c);
  return m_gaussians[r].mode();
}

const Eigen::MatrixXd GMM::covariance() const {
  // \sum_i w_i (Sigma_i + (\mu_i - \mu) (\mu_i - \mu)')
  Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(dimension(), dimension());
  const Eigen::VectorXd mu = mean();
  for (std::size_t i = 0; i < m_priors.size(); ++i) {
    const Eigen::MatrixXd cov = m_gaussians[i].covariance();
    const Eigen::VectorXd err = m_gaussians[i].mean() - mu;
    sigma += m_priors[i] * (cov + err * err.transpose());
  }
  return sigma;
}

const Eigen::VectorXd GMM::vectorize() const {
  auto K = numClusters();
  auto n = dimension();
  Eigen::VectorXd y = Eigen::VectorXd::Zero(K * (n * (n + 1) + 1));
  for (std::size_t i = 0; i < K; ++i) {
    y(i) = m_priors[i];
    auto v = m_gaussians[i].vectorize();
    y.segment(K + i * (n * (n + 1)), n * (n + 1)) = v;
  }
  return y;
}

bool GMM::devectorize(const Eigen::VectorXd& data) {
  auto K = numClusters();
  auto n = dimension();
  std::size_t d = data.size();
  if (d != K * (n * (n + 1) + 1)) {
    SIA_WARN("Devectorization failed, expected vector size "
             << K * (n * (n + 1) + 1) << ", received " << d);
    return false;
  }

  for (std::size_t i = 0; i < K; ++i) {
    m_priors[i] = data(i);
    m_gaussians[i].devectorize(
        data.segment(K + i * (n * (n + 1)), n * (n + 1)));
  }
  return true;
}

const Categorical& GMM::predict(const Eigen::VectorXd& x) {
  // Compute the weighted log likelihood for each cluster
  Eigen::VectorXd lik = Eigen::VectorXd::Zero(m_priors.size());
  for (std::size_t i = 0; i < m_priors.size(); ++i) {
    lik(i) = m_priors[i] * exp(m_gaussians[i].logProb(x));
  }
  lik /= lik.sum();
  m_belief.setProbs(lik);
  return m_belief;
}

std::size_t GMM::inputDim() const {
  return dimension();
}

std::size_t GMM::outputDim() const {
  return numClusters();
}

std::size_t GMM::classify(const Eigen::VectorXd& x) {
  return predict(x).classify();
}

std::size_t GMM::numClusters() const {
  return m_num_clusters;
}

double GMM::prior(std::size_t i) const {
  return m_priors[i];
}

const std::vector<double>& GMM::priors() const {
  return m_priors;
}

const Gaussian& GMM::gaussian(std::size_t i) const {
  return m_gaussians[i];
}

const std::vector<Gaussian>& GMM::gaussians() const {
  return m_gaussians;
}

/// Returns a sub matrix of samples where k is in clusters
const Eigen::MatrixXd extractClusteredSamples(
    const Eigen::MatrixXd& samples,
    const std::vector<std::size_t>& clusters,
    std::size_t k) {
  std::size_t n = samples.cols();
  assert(n == clusters.size());
  std::vector<std::size_t> indices;
  for (std::size_t i = 0; i < n; ++i) {
    if (clusters[i] == k) {
      indices.emplace_back(i);
    }
  }
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(samples.rows(), indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    y.col(i) = samples.col(indices[i]);
  }
  return y;
}

/// Returns if all cluster assignments are equivalent
bool assignmentsEqual(const std::vector<std::size_t>& a,
                      const std::vector<std::size_t>& b) {
  assert(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

void getAssociations(const Eigen::MatrixXd& classes,
                     std::vector<std::size_t>& associations) {
  std::size_t n = classes.cols();
  associations.clear();
  associations.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    classes.col(i).maxCoeff(&associations[i]);
  }
}

std::size_t GMM::fit(const Eigen::MatrixXd& samples,
                     std::vector<Gaussian>& gaussians,
                     std::vector<double>& priors,
                     std::size_t K,
                     GMM::FitMethod fit_method,
                     GMM::InitMethod init_method,
                     double regularization) {
  std::size_t n = samples.cols();
  std::size_t d = samples.rows();
  assert(n >= K);  // # clusters less than samples doesn't make any sense

  // Initialization
  if (init_method == GMM::InitMethod::WARM_START) {
    assert(priors.size() == K);
    assert(gaussians.size() == K);
  } else if (init_method == GMM::InitMethod::STANDARD_RANDOM) {
    priors.clear();
    gaussians.clear();

    std::vector<Eigen::VectorXd> vector_samples(n);
    for (std::size_t i = 0; i < n; ++i) {
      vector_samples[i] = samples.col(i);
    }
    std::vector<Eigen::VectorXd> out;
    std::sample(vector_samples.begin(), vector_samples.end(),
                std::back_inserter(out), K,
                std::mt19937{std::random_device{}()});

    for (std::size_t k = 0; k < K; ++k) {
      gaussians.emplace_back(Gaussian(out[k], Eigen::MatrixXd::Identity(d, d)));
      priors.emplace_back(1.0 / double(K));
    }
  } else {
    SIA_ERROR("Unsupported initialization method "
              << static_cast<int>(init_method));
    return 0;
  }

  // Create regularization covariance to ensure positive definite
  const Eigen::MatrixXd lambda =
      regularization * Eigen::MatrixXd::Identity(d, d);

  // Create K x n matrix of one-hot class associations (cols sum to 1)
  Eigen::MatrixXd weights = Eigen::MatrixXd::Ones(K, n) / K;

  // Run E-M
  bool done = false;
  std::vector<std::size_t> classes, prev_classes;
  classes.resize(n);
  getAssociations(weights, prev_classes);
  std::size_t iter = 0;
  while (not done) {
    // Association
    for (std::size_t k = 0; k < K; ++k) {
      for (std::size_t i = 0; i < n; ++i) {
        if (fit_method == GMM::FitMethod::KMEANS) {
          weights(k, i) =
              1.0 /
              ((gaussians[k].mean() - samples.col(i)).squaredNorm() + 1e-6);
        } else if (fit_method == GMM::FitMethod::GAUSSIAN_LIKELIHOOD) {
          weights(k, i) = priors[k] * exp(gaussians[k].logProb(samples.col(i)));
        } else {
          SIA_ERROR("Unsupported fit method " << static_cast<int>(fit_method));
          return 0;
        }
      }
    }

    const Eigen::VectorXd wsum = weights.colwise().sum();
    for (std::size_t i = 0; i < n; ++i) {
      weights.col(i) /= wsum(i);
    }
    getAssociations(weights, classes);

    // Compute mean, covariance, and prior for each cluster (MLE fit)
    for (std::size_t k = 0; k < K; ++k) {
      const Eigen::MatrixXd samples_k =
          extractClusteredSamples(samples, classes, k);
      if (samples_k.cols() == 0) {
        SIA_WARN("No samples found for cluster " << k);
      } else {
        double nk = double(samples_k.cols());
        const Eigen::VectorXd mu = samples_k.rowwise().sum() / nk;
        const Eigen::MatrixXd e =
            (samples_k.array().colwise() - mu.array()).matrix();
        const Eigen::MatrixXd cov = e * e.transpose() / nk + lambda;
        priors[k] = nk / double(n);
        gaussians[k] = Gaussian(mu, cov);
      }
    }

    // Check if cluster assignments changed
    if (assignmentsEqual(classes, prev_classes)) {
      done = true;
    } else {
      prev_classes = classes;
      iter++;
    }
  }
  return iter;
}

}  // namespace sia
