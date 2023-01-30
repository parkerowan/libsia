/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/estimators/estimators.h"
#include "sia/models/models.h"

#include <Eigen/Dense>

namespace sia {

/// The extended Kalman filter (EKF) sub-optimally estimates belief for
/// nonlinear Gaussian systems using 1st order linearization.  It exhibits poor
/// performance for highly nonlinear systems, especially when bifurcations in
/// state exhibited by chaotic systems yield multimodal distributions.  The
/// system covariance matrices Q, R control the tradeoff between prediction and
/// correction.
class EKF : public Estimator {
 public:
  struct Metrics : public BaseMetrics {
    double kalman_gain_norm{0};
  };

  explicit EKF(LinearizableDynamics& dynamics,
               LinearizableMeasurement& measurement,
               const Gaussian& state);
  virtual ~EKF() = default;
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
  LinearizableDynamics& m_dynamics;
  LinearizableMeasurement& m_measurement;
  Gaussian m_belief;
  Metrics m_metrics;
};

}  // namespace sia
