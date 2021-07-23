/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/estimators/ekf.h"
#include "sia/math/math.h"

#include <glog/logging.h>

namespace sia {

EKF::EKF(LinearizableDynamics& dynamics,
         LinearizableMeasurement& measurement,
         const Gaussian& state)
    : m_dynamics(dynamics), m_measurement(measurement), m_belief(state) {}

void EKF::reset(const Gaussian& state) {
  m_belief = state;
}

const Gaussian& EKF::getBelief() const {
  return m_belief;
}

const Gaussian& EKF::estimate(const Eigen::VectorXd& observation,
                              const Eigen::VectorXd& control) {
  m_belief = predict(control);
  m_belief = correct(observation);
  return m_belief;
}

const Gaussian& EKF::predict(const Eigen::VectorXd& control) {
  // From Burgard et. al. 2006, pp. 59, Table 3.3.
  auto x = m_belief.mean();
  auto P = m_belief.covariance();
  const auto& u = control;
  const auto Q = m_dynamics.Q(x, control);

  // Propogate
  x = m_dynamics.f(x, u);
  const auto F = m_dynamics.F(x, u);
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
  const auto R = m_measurement.R(x);

  // Gain
  const auto H = m_measurement.H(x);
  Eigen::MatrixXd HPHTRinv;
  if (not svdInverse(H * P * H.transpose() + R, HPHTRinv)) {
    LOG(ERROR) << "Matrix inversion failed in Kalman gain computation";
  }
  const Eigen::MatrixXd K = P * H.transpose() * HPHTRinv;

  // Update
  x += K * (y - m_measurement.h(x));
  P = (Eigen::MatrixXd::Identity(P.rows(), P.cols()) - K * H) * P;

  m_belief.setMean(x);
  m_belief.setCovariance(P);
  return m_belief;
}

}  // namespace sia
