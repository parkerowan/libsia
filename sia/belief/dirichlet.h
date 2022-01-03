/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/categorical.h"
#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

/// Dirichlet distribution defined by concentrations alpha > 0.  Note for a
/// sample x outside of the distribution support [0, 1], the logProb returns
/// -INFINITY
class Dirichlet : public Distribution {
 public:
  /// Creates a default multivariate Dirichlet (alpha = 1).
  explicit Dirichlet(std::size_t dimension);

  /// Creates a beta distribution from two concentrations (2D Dirichlet).
  explicit Dirichlet(double alpha, double beta);

  /// Creates a multivariate Dirichlet from vector of concentrations.
  explicit Dirichlet(const Eigen::VectorXd& alpha);
  virtual ~Dirichlet() = default;
  std::size_t dimension() const override;
  const Eigen::VectorXd sample() override;
  double logProb(const Eigen::VectorXd& x) const override;
  const Eigen::VectorXd mean() const override;
  const Eigen::VectorXd mode() const override;
  const Eigen::MatrixXd covariance() const override;
  const Eigen::VectorXd vectorize() const override;
  bool devectorize(const Eigen::VectorXd& data) override;

  Categorical categorical() const;
  std::size_t classify() const;

  const Eigen::VectorXd& alpha() const;
  void setAlpha(const Eigen::VectorXd& alpha);

 private:
  Eigen::VectorXd normalizeInput(const Eigen::VectorXd& x) const;

  Eigen::VectorXd m_alpha;
  std::vector<std::gamma_distribution<double>> m_gamma_dists;
};

}  // namespace sia
