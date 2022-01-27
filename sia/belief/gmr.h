/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gmm.h"

#include <Eigen/Dense>

namespace sia {

/// Gaussian mixture regression builds on GMM.  Gaussian mixture regression
/// performs Gaussian conditioning on a subset of variables in the GMM to
/// predict an output.  While GMM predicts $p(s)$, GMR predicts $p(y|x)$, where
/// $s = (x, y)'$ is a composite vector of both inputs and outputs.
///
/// References:
/// [1] http://www.stat.rice.edu/~hgsung/thesis.pdf
class GMR : public Inference {
 public:
  /// Initialize GMR from a GMM.  The input indices define which part of the
  /// GMM state vector is an input $x$, and output indices define which slice of
  /// the state vector is an output $y$ for the regression $y = gmr(x)$.  The
  /// regularization term on the output covariance $cov += (scalar >= 0) I$ is
  /// to improve positive definiteness.
  explicit GMR(const std::vector<Gaussian>& gaussians,
               const std::vector<double>& weights,
               std::vector<std::size_t> input_indices,
               std::vector<std::size_t> output_indices,
               double regularization = GMM::DEFAULT_REGULARIZATION);

  explicit GMR(const GMM& gmm,
               std::vector<std::size_t> input_indices,
               std::vector<std::size_t> output_indices,
               double regularization = GMM::DEFAULT_REGULARIZATION);

  /// Performs the inference $p(y|x)$
  const Gaussian& predict(const Eigen::VectorXd& x) override;
  std::size_t inputDimension() const override;
  std::size_t outputDimension() const override;

  /// Access to the GMM
  GMM& gmm();

 private:
  void cacheRegressionModels();

  struct RegressionModel {
    explicit RegressionModel(const Eigen::VectorXd& mu_x,
                             const Eigen::VectorXd& mu_y,
                             const Eigen::MatrixXd& sigma_xx,
                             const Eigen::MatrixXd& sigma_xy,
                             const Eigen::MatrixXd& sigma_yx,
                             const Eigen::MatrixXd& sigma_yy);

    Eigen::VectorXd m_mu_x;                   // need for local weight
    Eigen::VectorXd m_mu_y;                   // need for local mean
    Eigen::MatrixXd m_sigma_yx_sigma_xx_inv;  // need for local mean
    Eigen::MatrixXd m_sigma;                  // need for local covariance
    Gaussian m_gx;                            // need for local weight
  };

  GMM m_gmm;
  Gaussian m_belief;
  std::vector<std::size_t> m_input_indices;
  std::vector<std::size_t> m_output_indices;
  std::vector<RegressionModel> m_models;
  double m_regularization;
};

}  // namespace sia
