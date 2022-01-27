/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/nonlinear_gaussian.h"
#include "sia/math/math.h"

namespace sia {

NonlinearGaussianDynamics::NonlinearGaussianDynamics(DynamicsEquation dynamics,
                                                     const Eigen::MatrixXd& Q)
    : m_dynamics(dynamics), m_process_covariance(Q), m_prob_dynamics(Q.rows()) {
  cacheStateCovariance();
}

Gaussian& NonlinearGaussianDynamics::dynamics(const Eigen::VectorXd& state,
                                              const Eigen::VectorXd& control) {
  m_prob_dynamics.setMean(f(state, control));
  // For efficiency, the covariance is set only when Q is updated
  return m_prob_dynamics;
}

Eigen::VectorXd NonlinearGaussianDynamics::f(const Eigen::VectorXd& state,
                                             const Eigen::VectorXd& control) {
  return m_dynamics(state, control);
}

Eigen::MatrixXd NonlinearGaussianDynamics::Q(const Eigen::VectorXd& state,
                                             const Eigen::VectorXd& control) {
  (void)(state);
  (void)(control);
  return m_prob_dynamics.covariance();
}

const Eigen::MatrixXd& NonlinearGaussianDynamics::Q() const {
  return m_process_covariance;
}

void NonlinearGaussianDynamics::setQ(const Eigen::MatrixXd& Q) {
  m_process_covariance = Q;
  cacheStateCovariance();
}

void NonlinearGaussianDynamics::cacheStateCovariance() {
  m_prob_dynamics.setCovariance(Q());
}

NonlinearGaussianMeasurement::NonlinearGaussianMeasurement(
    MeasurementEquation measurement,
    const Eigen::MatrixXd& R)
    : m_measurement(measurement),
      m_measurement_covariance(R),
      m_prob_measurement(R.rows()) {
  cacheMeasurementCovariance();
}

Gaussian& NonlinearGaussianMeasurement::measurement(
    const Eigen::VectorXd& state) {
  m_prob_measurement.setMean(h(state));
  // For efficiency, the covariance is set only when R matrix is updated
  return m_prob_measurement;
}

Eigen::VectorXd NonlinearGaussianMeasurement::h(const Eigen::VectorXd& state) {
  return m_measurement(state);
}

Eigen::MatrixXd NonlinearGaussianMeasurement::R(const Eigen::VectorXd& state) {
  (void)(state);
  return R();
}

const Eigen::MatrixXd& NonlinearGaussianMeasurement::R() const {
  return m_measurement_covariance;
}

void NonlinearGaussianMeasurement::setR(const Eigen::MatrixXd& R) {
  m_measurement_covariance = R;
  cacheMeasurementCovariance();
}

void NonlinearGaussianMeasurement::cacheMeasurementCovariance() {
  m_prob_measurement.setCovariance(R());
}

NonlinearGaussianDynamicsCT::NonlinearGaussianDynamicsCT(
    DynamicsEquation dynamics,
    const Eigen::MatrixXd& Qpsd,
    double dt)
    : NonlinearGaussianDynamics(dynamics, toQ(Qpsd, dt)), m_dt(dt) {}

Gaussian& NonlinearGaussianDynamicsCT::dynamics(
    const Eigen::VectorXd& state,
    const Eigen::VectorXd& control) {
  m_prob_dynamics.setMean(f(state, control));
  // For efficiency, the covariance is set only when Q is updated
  return m_prob_dynamics;
}

Eigen::VectorXd NonlinearGaussianDynamicsCT::f(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control) {
  return rk4(m_dynamics, state, control, m_dt);
}

Eigen::MatrixXd NonlinearGaussianDynamicsCT::F(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control) {
  using namespace std::placeholders;
  DynamicsEquation f = std::bind(&NonlinearGaussianDynamicsCT::f, this, _1, _2);
  return dfdx(f, state, control);
}

Eigen::MatrixXd NonlinearGaussianDynamicsCT::G(const Eigen::VectorXd& state,
                                               const Eigen::VectorXd& control) {
  using namespace std::placeholders;
  DynamicsEquation f = std::bind(&NonlinearGaussianDynamicsCT::f, this, _1, _2);
  return dfdu(f, state, control);
}

void NonlinearGaussianDynamicsCT::setQpsd(const Eigen::MatrixXd& Qpsd) {
  // This caches the covariance using the discrete time model
  setQ(toQ(Qpsd, m_dt));
}

double NonlinearGaussianDynamicsCT::getTimeStep() const {
  return m_dt;
}

void NonlinearGaussianDynamicsCT::setTimeStep(double dt) {
  double dt_prev = m_dt;
  m_dt = dt;
  // This caches the covariance using the discrete time model.  Only the
  // covariance is stored, so it must first be converted to a PSD and then back.
  setQ(toQ(toQpsd(Q(), dt_prev), m_dt));
}

NonlinearGaussianMeasurementCT::NonlinearGaussianMeasurementCT(
    MeasurementEquation measurement,
    const Eigen::MatrixXd& Rpsd,
    double dt)
    : NonlinearGaussianMeasurement(measurement, toR(Rpsd, dt)), m_dt(dt) {}

void NonlinearGaussianMeasurementCT::setRpsd(const Eigen::MatrixXd& Rpsd) {
  // This caches the covariance using the discrete time model
  setR(toR(Rpsd, m_dt));
}

double NonlinearGaussianMeasurementCT::getTimeStep() const {
  return m_dt;
}

void NonlinearGaussianMeasurementCT::setTimeStep(double dt) {
  double dt_prev = m_dt;
  m_dt = dt;
  // This caches the covariance using the discrete time model.  Only the
  // covariance is stored, so it must first be converted to a PSD and then back.
  setR(toR(toRpsd(R(), dt_prev), m_dt));
}

}  // namespace sia
