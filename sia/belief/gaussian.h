/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

/// Gaussian (normal) distribution defined by mean and covariance.  Uses the
/// fast Cholesky decomposition (LDLT) of the covariance matrix.  Supports
/// positive semi-definite covariance matrices.
class Gaussian : public Distribution {
 public:
  /// Creates a standard multivariate normal (mu = 0, sigma = 1).
  explicit Gaussian(std::size_t dimension);

  /// Creates a univariate normal from mean and variance.
  explicit Gaussian(double mean, double variance);

  /// Creates a multivariate normal from mean and covariance.  The covariance
  /// matrix must be symmetric and positive semi-definite.
  explicit Gaussian(const Eigen::VectorXd& mean,
                    const Eigen::MatrixXd& covariance);
  virtual ~Gaussian() = default;
  std::size_t dimension() const override;
  const Eigen::VectorXd sample() override;
  double logProb(const Eigen::VectorXd& x) const override;
  const Eigen::VectorXd mean() const override;
  const Eigen::VectorXd mode() const override;
  const Eigen::MatrixXd covariance() const override;
  const Eigen::VectorXd vectorize() const override;
  bool devectorize(const Eigen::VectorXd& data) override;

  bool setMean(const Eigen::VectorXd& mean);

  /// The covariance matrix must be symmetric and positive semi-definite.
  bool setCovariance(const Eigen::MatrixXd& covariance);

  /// Computes the distance $\sqrt{(x-\mu)^\top \Sigma^{-1} (x-\mu)}$.
  double mahalanobis(const Eigen::VectorXd& x) const;

  /// Returns the log probability when $x = \mu$.
  double maxLogProb() const;
  bool checkDimensions(const Eigen::VectorXd& mu,
                       const Eigen::MatrixXd& sigma) const;

 protected:
  /// Computes the Cholesky (LDLT) decomposition of the covariance matrix and
  /// caches the resultant L matrix.
  bool cacheSigmaChol();

 private:
  Gaussian() = default;
  Eigen::VectorXd m_mu;
  Eigen::MatrixXd m_sigma;
  Eigen::MatrixXd m_cached_sigma_L;
  Eigen::MatrixXd m_cached_sigma_L_inv;
  std::normal_distribution<double> m_standard_normal{0.0, 1.0};
};

}  // namespace sia
