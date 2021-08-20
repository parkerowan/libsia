/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/estimators/kf.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

#include <glog/logging.h>

namespace sia {

KF::KF(LinearGaussianDynamics& dynamics,
       LinearGaussianMeasurement& measurement,
       const Gaussian& state)
    : m_dynamics(dynamics), m_measurement(measurement), m_belief(state) {}

void KF::reset(const Gaussian& state) {
  m_belief = state;
}

const Gaussian& KF::getBelief() const {
  return m_belief;
}

const Gaussian& KF::estimate(const Eigen::VectorXd& observation,
                             const Eigen::VectorXd& control) {
  m_belief = predict(control);
  m_belief = correct(observation);
  return m_belief;
}

const Gaussian& KF::predict(const Eigen::VectorXd& control) {
  // From Burgard et. al. 2006, pp. 42, Table 3.1.
  auto x = m_belief.mean();
  auto P = m_belief.covariance();
  const auto& u = control;
  const auto& F = m_dynamics.F();
  const auto& G = m_dynamics.G();
  const auto& Q = m_dynamics.Q();

  // Propogate
  x = F * x + G * u;
  P = F * P * F.transpose() + Q;

  m_belief.setMean(x);
  m_belief.setCovariance(P);
  return m_belief;
}

const Gaussian& KF::correct(const Eigen::VectorXd& observation) {
  // From Burgard et. al. 2006, pp. 42, Table 3.1.
  auto x = m_belief.mean();
  auto P = m_belief.covariance();
  const auto& y = observation;
  const auto& H = m_measurement.H();
  const auto& R = m_measurement.R();

  // Gain
  Eigen::MatrixXd HPHTRinv;
  bool r = svdInverse(H * P * H.transpose() + R, HPHTRinv);
  SIA_EXCEPTION(r, "Matrix inversion failed in KF gain computation");
  const Eigen::MatrixXd K = P * H.transpose() * HPHTRinv;

  // Update
  x += K * (y - H * x);
  P = (Eigen::MatrixXd::Identity(P.rows(), P.cols()) - K * H) * P;

  m_belief.setMean(x);
  m_belief.setCovariance(P);
  return m_belief;
}

}  // namespace sia
