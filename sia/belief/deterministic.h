/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

/// Deterministic distribution defined by a vector value.  The log prob
/// computation uses the Dirac measure, i.e. logProb = 1 when x equals the value
/// (within a tolerance) and -Infinity elsewhere.
class Deterministic : public Distribution {
 public:
  explicit Deterministic(double value);
  explicit Deterministic(const Eigen::VectorXd& value);
  virtual ~Deterministic() = default;
  std::size_t dimension() const override;
  const Eigen::VectorXd sample() override;
  double logProb(const Eigen::VectorXd& x) const override;
  const Eigen::VectorXd mean() const override;
  const Eigen::VectorXd mode() const override;
  const Eigen::MatrixXd covariance() const override;
  const Eigen::VectorXd vectorize() const override;
  bool devectorize(const Eigen::VectorXd& data) override;

  void setValue(const Eigen::VectorXd& value);

 private:
  Eigen::VectorXd m_value;
};

}  // namespace sia
