/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
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
class KF : public Estimator {
 public:
  struct Metrics : public BaseMetrics {
    double kalman_gain_norm{0};
  };

  explicit KF(LinearGaussianDynamics& dynamics,
              LinearGaussianMeasurement& measurement,
              const Gaussian& state);
  virtual ~KF() = default;
  const Gaussian& belief() const override;

  /// Performs the combined prediction and correction.
  const Gaussian& estimate(const Eigen::VectorXd& observation,
                           const Eigen::VectorXd& control) override;

  /// Propogates the belief through model dynamics.
  const Gaussian& predict(const Eigen::VectorXd& control) override;

  /// Corrects the belief with the measurement.
  const Gaussian& correct(const Eigen::VectorXd& observation) override;

  /// Return metrics from the latest step
  const Metrics& metrics() const override;

 private:
  LinearGaussianDynamics& m_dynamics;
  LinearGaussianMeasurement& m_measurement;
  Gaussian m_belief;
  Metrics m_metrics;
};

}  // namespace sia
