/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/estimators/ekf.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

namespace sia {

EKF::EKF(LinearizableDynamics& dynamics,
         LinearizableMeasurement& measurement,
         const Gaussian& state)
    : m_dynamics(dynamics), m_measurement(measurement), m_belief(state) {}

const Gaussian& EKF::belief() const {
  return m_belief;
}

const Gaussian& EKF::estimate(const Eigen::VectorXd& observation,
                              const Eigen::VectorXd& control) {
  m_metrics = EKF::Metrics();
  m_belief = predict(control);
  m_belief = correct(observation);
  m_metrics.clockElapsedUs();
  return m_belief;
}

const Gaussian& EKF::predict(const Eigen::VectorXd& control) {
  // From Burgard et. al. 2006, pp. 59, Table 3.3.
  auto x = m_belief.mean();
  auto P = m_belief.covariance();
  const auto& u = control;
  const auto F = m_dynamics.F(x, u);
  const auto Q = m_dynamics.Q(x, u);

  // Propogate
  x = m_dynamics.f(x, u);
  P = F * P * F.transpose() + Q;

  m_belief.setMean(x);
  m_belief.setCovariance(P);
  return m_belief;
}

const Gaussian& EKF::correct(const Eigen::VectorXd& observation) {
  // From Burgard et. al. 2006, pp. 59, Table 3.3.
  auto x = m_belief.mean();
  auto P = m_belief.covariance();
  const auto& y = observation;
  const auto H = m_measurement.H(x);
  const auto R = m_measurement.R(x);

  // Gain
  Eigen::MatrixXd HPHTRinv;
  bool r = svdInverse(H * P * H.transpose() + R, HPHTRinv);
  SIA_THROW_IF_NOT(r, "Matrix inversion failed in EKF gain computation");
  const Eigen::MatrixXd K = P * H.transpose() * HPHTRinv;
  m_metrics.kalman_gain_norm = K.norm();

  // Update
  x += K * (y - m_measurement.h(x));
  P = (Eigen::MatrixXd::Identity(P.rows(), P.cols()) - K * H) * P;

  m_belief.setMean(x);
  m_belief.setCovariance(P);
  return m_belief;
}

const EKF::Metrics& EKF::metrics() const {
  return m_metrics;
}

}  // namespace sia
