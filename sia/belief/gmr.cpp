/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gmr.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

#include <glog/logging.h>
#include <limits>

namespace sia {

// Note: the initial cached x is set to 1 here so that it differs from the first
// evaluation and the GMR is reset.
GMR::GMR(const std::vector<Gaussian>& gaussians,
         const std::vector<double>& priors,
         std::vector<std::size_t> input_indices,
         std::vector<std::size_t> output_indices,
         double regularization)
    : m_gmm(gaussians, priors),
      m_belief(input_indices.size()),
      m_input_indices(input_indices),
      m_output_indices(output_indices),
      m_regularization(regularization),
      m_cached_test_x(Eigen::VectorXd::Ones(inputDimension())) {
  cacheRegressionModels();
}

GMR::GMR(const GMM& gmm,
         std::vector<std::size_t> input_indices,
         std::vector<std::size_t> output_indices,
         double regularization)
    : m_gmm(gmm.gaussians(), gmm.priors()),
      m_belief(input_indices.size()),
      m_input_indices(input_indices),
      m_output_indices(output_indices),
      m_regularization(regularization),
      m_cached_test_x(Eigen::VectorXd::Ones(inputDimension())) {
  cacheRegressionModels();
}

const Gaussian& GMR::predict(const Eigen::VectorXd& x) {
  if (x.isApprox(m_cached_test_x)) {
    return m_belief;
  }
  m_cached_test_x = x;

  // Zero out the outputs
  std::size_t d = m_output_indices.size();
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(d);
  Eigen::MatrixXd sig = Eigen::MatrixXd::Zero(d, d);

  // Initialize cumulative priors to smallest double to avoid divide by zero
  double cweight = std::numeric_limits<double>::epsilon();

  // For each Gaussian dist
  const auto& priors = m_gmm.priors();
  for (std::size_t k = 0; k < m_gmm.numClusters(); ++k) {
    const auto& model = m_models[k];

    // Compute the weight based on proximity of input to input mean
    double weight = priors[k] * exp(model.m_gx.logProb(x));
    cweight += weight;

    // Compute the weighted local mean & covariance
    mu += weight *
          (model.m_mu_y + model.m_sigma_yx_sigma_xx_inv * (x - model.m_mu_x));
    sig += pow(weight, 2.0) * model.m_sigma;
  }

  // Normalize
  mu = mu / cweight;
  sig = sig / pow(cweight, 2.0);

  // enforce symmetry of covariance and add regularization for positive definite
  sig = (sig + sig.transpose()) / 2.0;
  sig += m_regularization * Eigen::MatrixXd::Identity(d, d);

  // Create Gaussian and output
  m_belief = Gaussian(mu, sig);
  return m_belief;
}

double GMR::negLogLik(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
  SIA_EXCEPTION(X.cols() == Y.cols(),
                "Test data X, Y expected to have sample number of cols");
  SIA_EXCEPTION(std::size_t(X.rows()) == inputDimension(),
                "Test data X rows expected to be input dimension");
  SIA_EXCEPTION(std::size_t(Y.rows()) == outputDimension(),
                "Test data Y rows expected to be output dimension");
  double neg_log_lik = 0;
  for (int i = 0; i < X.cols(); ++i) {
    const Gaussian g = predict(X.col(i));
    neg_log_lik -= g.logProb(Y.col(i));
  }
  return neg_log_lik;
}

std::size_t GMR::inputDimension() const {
  return m_input_indices.size();
}

std::size_t GMR::outputDimension() const {
  return m_output_indices.size();
}

GMM& GMR::gmm() {
  return m_gmm;
}

void GMR::cacheRegressionModels() {
  std::size_t K = m_gmm.numClusters();
  m_models.clear();
  m_models.reserve(K);

  // For each Gaussian dist
  const auto& gaussians = m_gmm.gaussians();
  for (std::size_t k = 0; k < K; ++k) {
    Eigen::MatrixXd cov = gaussians[k].covariance();
    Eigen::VectorXd mu = gaussians[k].mean();

    // Extract Gaussian conditioning terms
    const Eigen::VectorXd mu_x = slice(mu, m_input_indices);
    const Eigen::VectorXd mu_y = slice(mu, m_output_indices);
    const Eigen::MatrixXd sigma_xx =
        slice(cov, m_input_indices, m_input_indices);
    const Eigen::MatrixXd sigma_xy =
        slice(cov, m_input_indices, m_output_indices);
    const Eigen::MatrixXd sigma_yx =
        slice(cov, m_output_indices, m_input_indices);
    const Eigen::MatrixXd sigma_yy =
        slice(cov, m_output_indices, m_output_indices);

    // Build the local regression model
    m_models.emplace_back(GMR::RegressionModel(mu_x, mu_y, sigma_xx, sigma_xy,
                                               sigma_yx, sigma_yy));
  }

  // Evaluate the initial output for x = 0.  Note the initial x = 1
  predict(Eigen::VectorXd::Zero(inputDimension()));
}

GMR::RegressionModel::RegressionModel(const Eigen::VectorXd& mu_x,
                                      const Eigen::VectorXd& mu_y,
                                      const Eigen::MatrixXd& sigma_xx,
                                      const Eigen::MatrixXd& sigma_xy,
                                      const Eigen::MatrixXd& sigma_yx,
                                      const Eigen::MatrixXd& sigma_yy)
    : m_mu_x(mu_x), m_mu_y(mu_y), m_gx(mu_x, sigma_xx) {
  // Compute inverse
  Eigen::MatrixXd sigma_xx_inv;
  bool r = svdInverse(sigma_xx, sigma_xx_inv);
  SIA_EXCEPTION(r, "Failed to compute SVD of sigma_xx");

  // Compute Gaussian conditioning
  m_sigma_yx_sigma_xx_inv = sigma_yx * sigma_xx_inv;

  // Enforce symmetry of local covariance
  m_sigma = sigma_yy - sigma_yx * sigma_xx_inv * sigma_xy;
  m_sigma = (m_sigma + m_sigma.transpose()) / 2.0;
}

}  // namespace sia
