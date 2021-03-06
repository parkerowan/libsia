/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/estimators/kalman_filter.h"
#include "sia/math/math.h"

#include <glog/logging.h>

namespace sia {

KalmanFilter::KalmanFilter(LinearGaussian& system, const Gaussian& state)
    : m_system(system), m_belief(state) {}

void KalmanFilter::reset(const Gaussian& state) {
  m_belief = state;
}

const Gaussian& KalmanFilter::getBelief() const {
  return m_belief;
}

const Gaussian& KalmanFilter::estimate(const Eigen::VectorXd& observation,
                                       const Eigen::VectorXd& control) {
  m_belief = predict(control);
  m_belief = correct(observation);
  return m_belief;
}

const Gaussian& KalmanFilter::predict(const Eigen::VectorXd& control) {
  // From Burgard et. al. 2006, pp. 42, Table 3.1.
  auto x = m_belief.mean();
  auto P = m_belief.covariance();
  const auto& u = control;
  const auto& F = m_system.F();
  const auto& G = m_system.G();
  const auto& C = m_system.C();
  const auto& Q = m_system.Q();

  // Propogate
  x = F * x + G * u;
  P = F * P * F.transpose() + C * Q * C.transpose();

  m_belief.setMean(x);
  m_belief.setCovariance(P);
  return m_belief;
}

const Gaussian& KalmanFilter::correct(const Eigen::VectorXd& observation) {
  // From Burgard et. al. 2006, pp. 42, Table 3.1.
  auto x = m_belief.mean();
  auto P = m_belief.covariance();
  const auto& y = observation;
  const auto& H = m_system.H();
  const auto& R = m_system.R();

  // Gain
  Eigen::MatrixXd HPHTRinv;
  if (not svdInverse(H * P * H.transpose() + R, HPHTRinv)) {
    LOG(ERROR) << "Matrix inversion failed in Kalman gain computation";
  }
  const Eigen::MatrixXd K = P * H.transpose() * HPHTRinv;

  // Update
  x += K * (y - H * x);
  P = (Eigen::MatrixXd::Identity(P.rows(), P.cols()) - K * H) * P;

  m_belief.setMean(x);
  m_belief.setCovariance(P);
  return m_belief;
}

}  // namespace sia
