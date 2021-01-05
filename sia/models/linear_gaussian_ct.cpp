/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/linear_gaussian_ct.h"
#include "sia/math/math.h"

#include <glog/logging.h>

namespace sia {

LinearGaussianCT::LinearGaussianCT(const Eigen::MatrixXd& A,
                                   const Eigen::MatrixXd& B,
                                   const Eigen::MatrixXd& C,
                                   const Eigen::MatrixXd& H,
                                   const Eigen::MatrixXd& Q,
                                   const Eigen::MatrixXd& R,
                                   double dt,
                                   LinearGaussianCT::Type type)
    : LinearGaussian(C, H, Q, R),
      m_dynamics_matrix_ct(A),
      m_input_matrix_ct(B),
      m_dt(dt),
      m_type(type) {
  discretizeDynamics();
  cacheStateCovariance();
  cacheMeasurementCovariance();
}

const Eigen::MatrixXd& LinearGaussianCT::A() const {
  return m_dynamics_matrix_ct;
}

const Eigen::MatrixXd& LinearGaussianCT::B() const {
  return m_input_matrix_ct;
}

void LinearGaussianCT::setA(const Eigen::MatrixXd& A) {
  m_dynamics_matrix_ct = A;
  discretizeDynamics();
}

void LinearGaussianCT::setB(const Eigen::MatrixXd& B) {
  m_input_matrix_ct = B;
  discretizeDynamics();
}

void LinearGaussianCT::setC(const Eigen::MatrixXd& C) {
  m_process_noise_matrix = C;
  cacheStateCovariance();
}

void LinearGaussianCT::setQ(const Eigen::MatrixXd& Q) {
  m_process_covariance = Q;
  cacheStateCovariance();
}

void LinearGaussianCT::setR(const Eigen::MatrixXd& R) {
  m_measurement_covariance = R;
  cacheMeasurementCovariance();
}

LinearGaussianCT::Type LinearGaussianCT::getType() const {
  return m_type;
}

void LinearGaussianCT::setType(LinearGaussianCT::Type type) {
  m_type = type;
}

double LinearGaussianCT::getTimeStep() const {
  return m_dt;
}

void LinearGaussianCT::setTimeStep(double dt) {
  m_dt = dt;
  discretizeDynamics();
  cacheStateCovariance();
  cacheMeasurementCovariance();
}

void LinearGaussianCT::cacheStateCovariance() {
  // From Crassidis and Junkins, 2012, pg. 172.
  m_prob_dynamics.setCovariance(C() * Q() * C().transpose() * m_dt);
}

void LinearGaussianCT::cacheMeasurementCovariance() {
  // From Crassidis and Junkins, 2012, pg. 174.
  m_prob_measurement.setCovariance(R() / m_dt);
}

void LinearGaussianCT::discretizeDynamics() {
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
      LOG(WARNING) << "LinearGaussianCT::discretizeDynamics BACKWARD_EULER "
                      "discretization failed, using FORWARD_EULER instead.";
    case FORWARD_EULER:
      F = I + m_dt * A;
      G = m_dt * B;
      break;
    default:
      LOG(ERROR) << "LinearGaussianCT::discretizeDynamics not implemented "
                    "for discretization type "
                 << m_type;
  }
}

}  // namespace sia
