/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/estimators/extended_kalman_filter.h"
#include "sia/math/math.h"

#include <glog/logging.h>

namespace sia {

ExtendedKalmanFilter::ExtendedKalmanFilter(NonlinearGaussian& system,
                                           const Gaussian& state)
    : m_system(system), m_belief(state) {}

void ExtendedKalmanFilter::reset(const Gaussian& state) {
  m_belief = state;
}

const Gaussian& ExtendedKalmanFilter::getBelief() const {
  return m_belief;
}

const Gaussian& ExtendedKalmanFilter::estimate(
    const Eigen::VectorXd& observation,
    const Eigen::VectorXd& control) {
  m_belief = predict(control);
  m_belief = correct(observation);
  return m_belief;
}

const Gaussian& ExtendedKalmanFilter::predict(const Eigen::VectorXd& control) {
  // From Burgard et. al. 2006, pp. 59, Table 3.3.
  auto x = m_belief.mean();
  auto P = m_belief.covariance();
  const auto& u = control;
  const auto& C = m_system.C();
  const auto& Q = m_system.Q();

  // Propogate
  x = m_system.f(x, u);
  const auto F = m_system.F(x, u);
  P = F * P * F.transpose() + C * Q * C.transpose();

  m_belief.setMean(x);
  m_belief.setCovariance(P);
  return m_belief;
}

const Gaussian& ExtendedKalmanFilter::correct(
    const Eigen::VectorXd& observation) {
  // From Burgard et. al. 2006, pp. 59, Table 3.3.
  auto x = m_belief.mean();
  auto P = m_belief.covariance();
  const auto& y = observation;
  const auto& R = m_system.R();

  // Gain
  const auto H = m_system.H(x);
  Eigen::MatrixXd HPHTRinv;
  if (not svdInverse(H * P * H.transpose() + R, HPHTRinv)) {
    LOG(ERROR) << "Matrix inversion failed in Kalman gain computation";
  }
  const Eigen::MatrixXd K = P * H.transpose() * HPHTRinv;

  // Update
  x += K * (y - m_system.h(x));
  P = (Eigen::MatrixXd::Identity(P.rows(), P.cols()) - K * H) * P;

  m_belief.setMean(x);
  m_belief.setCovariance(P);
  return m_belief;
}

}  // namespace sia
