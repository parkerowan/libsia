/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gmr.h"
#include <limits>
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

namespace sia {

GMR::GMR(const std::vector<Gaussian>& gaussians,
         const std::vector<double>& priors,
         std::vector<std::size_t> input_indices,
         std::vector<std::size_t> output_indices,
         double regularization)
    : m_gmm(gaussians, priors),
      m_belief(input_indices.size()),
      m_input_indices(input_indices),
      m_output_indices(output_indices),
      m_regularization(regularization) {
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
      m_regularization(regularization) {
  cacheRegressionModels();
}

// TODO: Check if x has changed, if not return cached values
const Gaussian& GMR::predict(const Eigen::VectorXd& x) {
  // Zero out the outputs
  std::size_t d = m_output_indices.size();
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(d);
  Eigen::MatrixXd sig = Eigen::MatrixXd::Zero(d, d);

  // Compute the weights
  Eigen::VectorXd log_weights = Eigen::VectorXd::Zero(m_gmm.numClusters());
  const auto& priors = m_gmm.priors();
  for (std::size_t k = 0; k < m_gmm.numClusters(); ++k) {
    const auto& model = m_models[k];
    log_weights(k) = log(priors[k]) + model.m_gx.logProb(x);
  }

  // Normalize the weights, add a small value to avoid divide by zero
  Eigen::VectorXd weights = log_weights.array().exp();
  double eps = std::numeric_limits<double>::epsilon();
  if (weights.sum() < eps) {
    int i;
    log_weights.maxCoeff(&i);
    weights[i] = eps;
  }
  weights = weights / weights.sum();

  // Compute the weighted local mean & covariance
  for (std::size_t k = 0; k < m_gmm.numClusters(); ++k) {
    const auto& model = m_models[k];
    mu += weights(k) *
          (model.m_mu_y + model.m_sigma_yx_sigma_xx_inv * (x - model.m_mu_x));
    sig += pow(weights(k), 2.0) * model.m_sigma;
  }

  // enforce symmetry of covariance and add regularization for positive
  // definite
  sig = (sig + sig.transpose()) / 2.0;
  sig += m_regularization * Eigen::MatrixXd::Identity(d, d);

  // Create Gaussian and output
  m_belief = Gaussian(mu, sig);
  return m_belief;
}

std::size_t GMR::inputDim() const {
  return m_input_indices.size();
}

std::size_t GMR::outputDim() const {
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
  SIA_THROW_IF_NOT(r, "Failed to compute SVD of sigma_xx");

  // Compute Gaussian conditioning
  m_sigma_yx_sigma_xx_inv = sigma_yx * sigma_xx_inv;

  // Enforce symmetry of local covariance
  m_sigma = sigma_yy - sigma_yx * sigma_xx_inv * sigma_xy;
  m_sigma = (m_sigma + m_sigma.transpose()) / 2.0;
}

}  // namespace sia
