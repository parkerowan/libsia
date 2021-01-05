/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/estimators/estimators.h"
#include "sia/models/linear_gaussian.h"

#include <Eigen/Dense>

namespace sia {

/// The Kalman filter (KF) optimally estimates belief for Linear Gaussian
/// systems.  The system covariance matrices Q, R control the tradeoff between
/// prediction and correction.
class KalmanFilter : public RecursiveBayesEstimator {
 public:
  explicit KalmanFilter(LinearGaussian& system, const Gaussian& state);
  virtual ~KalmanFilter() = default;
  void reset(const Gaussian& state);
  const Gaussian& getBelief() const override;

  /// Performs the combined prediction and correction.
  const Gaussian& estimate(const Eigen::VectorXd& observation,
                           const Eigen::VectorXd& control) override;

  /// Propogates the belief through model dynamics.
  const Gaussian& predict(const Eigen::VectorXd& control) override;

  /// Corrects the belief with the measurement.
  const Gaussian& correct(const Eigen::VectorXd& observation) override;

 private:
  LinearGaussian& m_system;
  Gaussian m_belief;
};

}  // namespace sia
