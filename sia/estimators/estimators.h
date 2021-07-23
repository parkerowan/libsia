/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

/// Recursive Bayesian estimators perform a 2 step dynamics-based prediction,
/// and measurement-based correction on the prior state belief.  The prediction
/// step propogates the prior belief through the dynamics equation, which in
/// general increases the belief uncertainty.  The correction step adjusts the
/// belief based on the measurement, which in general descreases the belief
/// uncertainty.
/// - Predict: $ p(x_k) = \int p(x_k | k_k-1, u_k) d x_k-1$
/// - Correct: $ p(x_k | z_0:k) \propto p(y | x) \int p(x_k)$
/// where $x$ is state, $y$ is measurement, and $u$ is control.  In the
/// literature, the predict/correct steps are typically shown together, however
/// in practice it is useful to call them separately, e.g. when measurements
/// don't arrive every time step.
class Estimator {
 public:
  Estimator() = default;
  virtual ~Estimator() = default;
  virtual const Distribution& getBelief() const = 0;

  /// Performs the combined prediction and correction.
  virtual const Distribution& estimate(const Eigen::VectorXd& observation,
                                       const Eigen::VectorXd& control) = 0;

  /// Propogates the belief through model dynamics.
  virtual const Distribution& predict(const Eigen::VectorXd& control) = 0;

  /// Corrects the belief with the measurement.
  virtual const Distribution& correct(const Eigen::VectorXd& observation) = 0;
};

}  // namespace sia
