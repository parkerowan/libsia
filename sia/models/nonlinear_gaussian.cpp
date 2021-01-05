/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/nonlinear_gaussian.h"
#include "sia/math/math.h"

namespace sia {

NonlinearGaussian::NonlinearGaussian(DynamicsEquation dynamics,
                                     MeasurementEquation measurement,
                                     const Eigen::MatrixXd& C,
                                     const Eigen::MatrixXd& Q,
                                     const Eigen::MatrixXd& R)
    : m_dynamics(dynamics),
      m_measurement(measurement),
      m_process_noise_matrix(C),
      m_process_covariance(Q),
      m_measurement_covariance(R),
      m_prob_dynamics(C.rows()),
      m_prob_measurement(R.rows()) {
  cacheStateCovariance();
  cacheMeasurementCovariance();
}

NonlinearGaussian::NonlinearGaussian(const Eigen::MatrixXd& C,
                                     const Eigen::MatrixXd& Q,
                                     const Eigen::MatrixXd& R)
    : m_process_noise_matrix(C),
      m_process_covariance(Q),
      m_measurement_covariance(R),
      m_prob_dynamics(C.rows()),
      m_prob_measurement(R.rows()) {
  cacheStateCovariance();
  cacheMeasurementCovariance();
}

Gaussian& NonlinearGaussian::dynamics(const Eigen::VectorXd& state,
                                      const Eigen::VectorXd& control) {
  m_prob_dynamics.setMean(f(state, control));
  // For efficiency, the covariance is set only when C, Q matrices are updated
  return m_prob_dynamics;
}

Gaussian& NonlinearGaussian::measurement(const Eigen::VectorXd& state) {
  m_prob_measurement.setMean(h(state));
  // For efficiency, the covariance is set only when R matrix is updated
  return m_prob_measurement;
}

const Eigen::VectorXd NonlinearGaussian::f(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
  return m_dynamics(state, control);
}

const Eigen::MatrixXd NonlinearGaussian::F(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) const {
  using namespace std::placeholders;
  DynamicsEquation f = std::bind(&NonlinearGaussian::f, this, _1, _2);
  return numericalJacobian<const Eigen::VectorXd&>(f, state, control);
}

const Eigen::VectorXd NonlinearGaussian::h(const Eigen::VectorXd& state) const {
  return m_measurement(state);
}

const Eigen::MatrixXd NonlinearGaussian::H(const Eigen::VectorXd& state) const {
  using namespace std::placeholders;
  MeasurementEquation h = std::bind(&NonlinearGaussian::h, this, _1);
  return numericalJacobian<>(h, state);
}

const Eigen::MatrixXd& NonlinearGaussian::C() const {
  return m_process_noise_matrix;
}

const Eigen::MatrixXd& NonlinearGaussian::Q() const {
  return m_process_covariance;
}

const Eigen::MatrixXd& NonlinearGaussian::R() const {
  return m_measurement_covariance;
}

void NonlinearGaussian::setC(const Eigen::MatrixXd& C) {
  m_process_noise_matrix = C;
  cacheStateCovariance();
}

void NonlinearGaussian::setQ(const Eigen::MatrixXd& Q) {
  m_process_covariance = Q;
  cacheStateCovariance();
}

void NonlinearGaussian::setR(const Eigen::MatrixXd& R) {
  m_measurement_covariance = R;
  cacheMeasurementCovariance();
}

void NonlinearGaussian::cacheStateCovariance() {
  m_prob_dynamics.setCovariance(C() * Q() * C().transpose());
}

void NonlinearGaussian::cacheMeasurementCovariance() {
  m_prob_measurement.setCovariance(R());
}

}  // namespace sia