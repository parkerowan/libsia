/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/nonlinear_gaussian_ct.h"
#include "sia/math/math.h"

namespace sia {

NonlinearGaussianCT::NonlinearGaussianCT(DynamicsEquation dynamics,
                                         MeasurementEquation measurement,
                                         const Eigen::MatrixXd& C,
                                         const Eigen::MatrixXd& Q,
                                         const Eigen::MatrixXd& R,
                                         double dt)
    : NonlinearGaussian(dynamics, measurement, C, Q, R), m_dt(dt) {
  cacheStateCovariance();
}

Gaussian& NonlinearGaussianCT::dynamics(const Eigen::VectorXd& state,
                                        const Eigen::VectorXd& control) {
  m_prob_dynamics.setMean(f(state, control));
  // For efficiency, the covariance is set only when C, Q matrices are updated
  return m_prob_dynamics;
}

const Eigen::VectorXd NonlinearGaussianCT::f(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
  return rk4(m_dynamics, state, control, m_dt);
}

const Eigen::MatrixXd NonlinearGaussianCT::F(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
  using namespace std::placeholders;
  DynamicsEquation f = std::bind(&NonlinearGaussianCT::f, this, _1, _2);
  return dfdx(f, state, control);
}

const Eigen::MatrixXd NonlinearGaussianCT::G(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
  using namespace std::placeholders;
  DynamicsEquation f = std::bind(&NonlinearGaussianCT::f, this, _1, _2);
  return dfdu(f, state, control);
}

void NonlinearGaussianCT::setC(const Eigen::MatrixXd& C) {
  m_process_noise_matrix = C;
  cacheStateCovariance();
}

void NonlinearGaussianCT::setQ(const Eigen::MatrixXd& Q) {
  m_process_covariance = Q;
  cacheStateCovariance();
}

void NonlinearGaussianCT::setR(const Eigen::MatrixXd& R) {
  m_measurement_covariance = R;
  cacheMeasurementCovariance();
}

double NonlinearGaussianCT::getTimeStep() const {
  return m_dt;
}

void NonlinearGaussianCT::setTimeStep(double dt) {
  m_dt = dt;
  cacheStateCovariance();
  cacheMeasurementCovariance();
}

void NonlinearGaussianCT::cacheStateCovariance() {
  // From Crassidis and Junkins, 2012, pg. 172.
  m_prob_dynamics.setCovariance(C() * Q() * C().transpose() * m_dt);
}

void NonlinearGaussianCT::cacheMeasurementCovariance() {
  // From Crassidis and Junkins, 2012, pg. 174.
  m_prob_measurement.setCovariance(R() / m_dt);
}

}  // namespace sia
