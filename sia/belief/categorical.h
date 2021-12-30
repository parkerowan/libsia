/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

/// Categorical distribution defined by probabilities such that
/// \sum probs = 1
/// 0 < prob_i < 1
/// Note that the Dirichlet distribution is the conjugate prior to the
/// Categorical distribution.  As such, the Categorical distribution can be
/// initialized via probs ~ Dir(alpha).
class Categorical : public Distribution {
 public:
  /// Creates a default multivariate Categorical with equal probs
  explicit Categorical(std::size_t dimension);

  /// Creates a multivariate Categorical with discrete probabilities such that
  /// \sum probs = 1
  explicit Categorical(const Eigen::VectorXd& probs);
  virtual ~Categorical() = default;
  std::size_t dimension() const override;
  const Eigen::VectorXd sample() override;
  double logProb(const Eigen::VectorXd& x) const override;
  const Eigen::VectorXd mean() const override;
  const Eigen::VectorXd mode() const override;
  const Eigen::MatrixXd covariance() const override;
  const Eigen::VectorXd vectorize() const override;
  bool devectorize(const Eigen::VectorXd& data) override;

  std::size_t classify() const;

  const Eigen::VectorXd& probs() const;
  void setProbs(const Eigen::VectorXd& probs);

  /// Converts the category to a one-hot encoding probability
  Eigen::VectorXd oneHot(std::size_t category) const;
  Eigen::MatrixXd oneHot(const Eigen::VectorXi& category) const;

  /// Converts a one-hot encoding probability to a category
  std::size_t category(const Eigen::VectorXd& probs) const;

 private:
  std::size_t sampleInverseCdf() const;

  Eigen::VectorXd m_probs;
};

}  // namespace sia
