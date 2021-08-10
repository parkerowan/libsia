/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/linear_gaussian.h"
#include "sia/math/math.h"

#include <glog/logging.h>

namespace sia {

LinearGaussianDynamics::LinearGaussianDynamics(const Eigen::MatrixXd& F,
                                               const Eigen::MatrixXd& G,
                                               const Eigen::MatrixXd& Q)
    : m_dynamics_matrix(F),
      m_input_matrix(G),
      m_process_covariance(Q),
      m_prob_dynamics(Q.rows()) {
  cacheStateCovariance();
}

LinearGaussianDynamics::LinearGaussianDynamics(const Eigen::MatrixXd& Q)
    : m_process_covariance(Q), m_prob_dynamics(Q.rows()) {}

Gaussian& LinearGaussianDynamics::dynamics(const Eigen::VectorXd& state,
                                           const Eigen::VectorXd& control) {
  m_prob_dynamics.setMean(f(state, control));
  // For efficiency, the covariance is set only when Q is updated
  return m_prob_dynamics;
}

Eigen::VectorXd LinearGaussianDynamics::f(const Eigen::VectorXd& state,
                                          const Eigen::VectorXd& control) {
  return F() * state + G() * control;
}

Eigen::MatrixXd LinearGaussianDynamics::Q(const Eigen::VectorXd& state,
                                          const Eigen::VectorXd& control) {
  (void)(state);
  (void)(control);
  return m_prob_dynamics.covariance();
}

Eigen::MatrixXd LinearGaussianDynamics::F(const Eigen::VectorXd& state,
                                          const Eigen::VectorXd& control) {
  (void)(state);
  (void)(control);
  return F();
}

Eigen::MatrixXd LinearGaussianDynamics::G(const Eigen::VectorXd& state,
                                          const Eigen::VectorXd& control) {
  (void)(state);
  (void)(control);
  return G();
}

const Eigen::MatrixXd& LinearGaussianDynamics::Q() const {
  return m_process_covariance;
}

const Eigen::MatrixXd& LinearGaussianDynamics::F() const {
  return m_dynamics_matrix;
}

const Eigen::MatrixXd& LinearGaussianDynamics::G() const {
  return m_input_matrix;
}

void LinearGaussianDynamics::setQ(const Eigen::MatrixXd& Q) {
  m_process_covariance = Q;
  cacheStateCovariance();
}

void LinearGaussianDynamics::setF(const Eigen::MatrixXd& F) {
  m_dynamics_matrix = F;
}

void LinearGaussianDynamics::setG(const Eigen::MatrixXd& G) {
  m_input_matrix = G;
}

void LinearGaussianDynamics::cacheStateCovariance() {
  m_prob_dynamics.setCovariance(Q());
}

LinearGaussianMeasurement::LinearGaussianMeasurement(const Eigen::MatrixXd& H,
                                                     const Eigen::MatrixXd& R)
    : m_measurement_matrix(H),
      m_measurement_covariance(R),
      m_prob_measurement(R.rows()) {
  cacheMeasurementCovariance();
}

Gaussian& LinearGaussianMeasurement::measurement(const Eigen::VectorXd& state) {
  m_prob_measurement.setMean(h(state));
  // For efficiency, the covariance is set only when R matrix is updated
  return m_prob_measurement;
}

Eigen::VectorXd LinearGaussianMeasurement::h(const Eigen::VectorXd& state) {
  return H() * state;
}

Eigen::MatrixXd LinearGaussianMeasurement::R(const Eigen::VectorXd& state) {
  (void)(state);
  return R();
}

Eigen::MatrixXd LinearGaussianMeasurement::H(const Eigen::VectorXd& state) {
  (void)(state);
  return H();
}

const Eigen::MatrixXd& LinearGaussianMeasurement::R() const {
  return m_measurement_covariance;
}

const Eigen::MatrixXd& LinearGaussianMeasurement::H() const {
  return m_measurement_matrix;
}

void LinearGaussianMeasurement::setR(const Eigen::MatrixXd& R) {
  m_measurement_covariance = R;
  cacheMeasurementCovariance();
}

void LinearGaussianMeasurement::setH(const Eigen::MatrixXd& H) {
  m_measurement_matrix = H;
}

void LinearGaussianMeasurement::cacheMeasurementCovariance() {
  m_prob_measurement.setCovariance(R());
}

LinearGaussianDynamicsCT::LinearGaussianDynamicsCT(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& Qpsd,
    double dt,
    LinearGaussianDynamicsCT::Type type)
    : LinearGaussianDynamics(toQ(Qpsd, dt)),
      m_dynamics_matrix_ct(A),
      m_input_matrix_ct(B),
      m_dt(dt),
      m_type(type) {
  discretizeDynamics();
  cacheStateCovariance();
}

const Eigen::MatrixXd& LinearGaussianDynamicsCT::A() const {
  return m_dynamics_matrix_ct;
}

const Eigen::MatrixXd& LinearGaussianDynamicsCT::B() const {
  return m_input_matrix_ct;
}

void LinearGaussianDynamicsCT::setA(const Eigen::MatrixXd& A) {
  m_dynamics_matrix_ct = A;
  discretizeDynamics();
}

void LinearGaussianDynamicsCT::setB(const Eigen::MatrixXd& B) {
  m_input_matrix_ct = B;
  discretizeDynamics();
}

void LinearGaussianDynamicsCT::setQpsd(const Eigen::MatrixXd& Qpsd) {
  // This caches the covariance using the discrete time model
  setQ(toQ(Qpsd, m_dt));
}

LinearGaussianDynamicsCT::Type LinearGaussianDynamicsCT::getType() const {
  return m_type;
}

void LinearGaussianDynamicsCT::setType(LinearGaussianDynamicsCT::Type type) {
  m_type = type;
  discretizeDynamics();
}

double LinearGaussianDynamicsCT::getTimeStep() const {
  return m_dt;
}

void LinearGaussianDynamicsCT::setTimeStep(double dt) {
  double dt_prev = m_dt;
  m_dt = dt;
  discretizeDynamics();
  // This caches the covariance using the discrete time model.  Only the
  // covariance is stored, so it must first be converted to a PSD and then back.
  setQ(toQ(toQpsd(Q(), dt_prev), m_dt));
}

void LinearGaussianDynamicsCT::discretizeDynamics() {
  const Eigen::MatrixXd& A = m_dynamics_matrix_ct;
  const Eigen::MatrixXd& B = m_input_matrix_ct;
  Eigen::MatrixXd& F = m_dynamics_matrix;
  Eigen::MatrixXd& G = m_input_matrix;
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
  switch (m_type) {
    case BACKWARD_EULER:
      if (svdInverse(I - m_dt * A, F)) {
        G = m_dt * F * B;
        break;
      }
      LOG(WARNING)
          << "LinearGaussianDynamicsCT::discretizeDynamics BACKWARD_EULER "
             "discretization failed, using FORWARD_EULER instead.";
    case FORWARD_EULER:
      F = I + m_dt * A;
      G = m_dt * B;
      break;
    default:
      LOG(ERROR)
          << "LinearGaussianDynamicsCT::discretizeDynamics not implemented "
             "for discretization type "
          << m_type;
  }
}

LinearGaussianMeasurementCT::LinearGaussianMeasurementCT(
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& Rpsd,
    double dt)
    : LinearGaussianMeasurement(H, toR(Rpsd, dt)), m_dt(dt) {}

void LinearGaussianMeasurementCT::setRpsd(const Eigen::MatrixXd& Rpsd) {
  // This caches the covariance using the discrete time model
  setR(toR(Rpsd, m_dt));
}

double LinearGaussianMeasurementCT::getTimeStep() const {
  return m_dt;
}

void LinearGaussianMeasurementCT::setTimeStep(double dt) {
  double dt_prev = m_dt;
  m_dt = dt;
  // This caches the covariance using the discrete time model.  Only the
  // covariance is stored, so it must first be converted to a PSD and then back.
  setR(toR(toRpsd(R(), dt_prev), m_dt));
}

}  // namespace sia
