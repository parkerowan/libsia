/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

/// Uniform distribution defined by lower and upper support.  Note for a sample
/// x outside of the distribution support, the logProb returns -INFINITY
class Uniform : public Distribution {
 public:
  /// Creates a standard multivariate uniform (lower = 0, upper = 1).
  explicit Uniform(std::size_t dimension);

  /// Creates a univariate uniform from lower and upper ranges.
  explicit Uniform(double lower, double upper);

  /// Creates a multivariate uniform from lower and upper ranges.
  explicit Uniform(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper);
  virtual ~Uniform() = default;
  std::size_t dimension() const override;
  const Eigen::VectorXd sample() override;
  double logProb(const Eigen::VectorXd& x) const override;
  const Eigen::VectorXd mean() const override;

  /// For uniform distributions, the mode can be anywhere between the lower and
  /// upper ranges.  This method returns the mean as the mode.
  const Eigen::VectorXd mode() const override;
  const Eigen::MatrixXd covariance() const override;
  const Eigen::VectorXd& lower() const;
  const Eigen::VectorXd& upper() const;
  bool setLower(const Eigen::VectorXd& lower);
  bool setUpper(const Eigen::VectorXd& upper);
  bool checkDimensions(const Eigen::VectorXd& lower,
                       const Eigen::VectorXd& upper) const;

 private:
  Eigen::VectorXd m_lower;
  Eigen::VectorXd m_upper;
  std::uniform_real_distribution<double> m_standard_uniform{0.0, 1.0};
};

}  // namespace sia
