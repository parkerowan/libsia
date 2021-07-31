/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/linear_gaussian.h"

namespace sia {

LinearGaussian::LinearGaussian(const Eigen::MatrixXd& F,
                               const Eigen::MatrixXd& G,
                               const Eigen::MatrixXd& C,
                               const Eigen::MatrixXd& H,
                               const Eigen::MatrixXd& Q,
                               const Eigen::MatrixXd& R)
    : NonlinearGaussian(C, Q, R),
      m_dynamics_matrix(F),
      m_input_matrix(G),
      m_measurement_matrix(H) {}

LinearGaussian::LinearGaussian(const Eigen::MatrixXd& C,
                               const Eigen::MatrixXd& H,
                               const Eigen::MatrixXd& Q,
                               const Eigen::MatrixXd& R)
    : NonlinearGaussian(C, Q, R), m_measurement_matrix(H) {}

Gaussian& LinearGaussian::dynamics(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& control) {
  m_prob_dynamics.setMean(f(state, control));
  // For efficiency, the covariance is set only when C, Q matrices are updated
  return m_prob_dynamics;
}

Gaussian& LinearGaussian::measurement(const Eigen::VectorXd& state) {
  m_prob_measurement.setMean(h(state));
  // For efficiency, the covariance is set only when R matrix is updated
  return m_prob_measurement;
}

const Eigen::VectorXd LinearGaussian::f(const Eigen::VectorXd& state,
                                        const Eigen::VectorXd& control) const {
  return F() * state + G() * control;
}

const Eigen::MatrixXd LinearGaussian::F(const Eigen::VectorXd& state,
                                        const Eigen::VectorXd& control) const {
  (void)(state);
  (void)(control);
  return F();
}

const Eigen::MatrixXd LinearGaussian::G(const Eigen::VectorXd& state,
                                        const Eigen::VectorXd& control) const {
  (void)(state);
  (void)(control);
  return G();
}

const Eigen::VectorXd LinearGaussian::h(const Eigen::VectorXd& state) const {
  return H() * state;
}

const Eigen::MatrixXd LinearGaussian::H(const Eigen::VectorXd& state) const {
  (void)(state);
  return H();
}

const Eigen::MatrixXd& LinearGaussian::F() const {
  return m_dynamics_matrix;
}

const Eigen::MatrixXd& LinearGaussian::G() const {
  return m_input_matrix;
}

const Eigen::MatrixXd& LinearGaussian::H() const {
  return m_measurement_matrix;
}

void LinearGaussian::setF(const Eigen::MatrixXd& F) {
  m_dynamics_matrix = F;
}

void LinearGaussian::setG(const Eigen::MatrixXd& G) {
  m_input_matrix = G;
}

void LinearGaussian::setH(const Eigen::MatrixXd& H) {
  m_measurement_matrix = H;
}

}  // namespace sia
