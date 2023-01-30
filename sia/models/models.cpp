/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/models.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

namespace sia {

DynamicsModel::DynamicsModel(std::size_t state_dim, std::size_t control_dim)
    : m_state_dim(state_dim), m_control_dim(control_dim) {}

std::size_t DynamicsModel::stateDimension() const {
  return m_state_dim;
}

std::size_t DynamicsModel::controlDimension() const {
  return m_control_dim;
}

void DynamicsModel::checkDimensions(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& control) const {
  SIA_THROW_IF_NOT(state.size() == int(m_state_dim),
                   "Dynamics Model input does not match state dimension");
  SIA_THROW_IF_NOT(control.size() == int(m_control_dim),
                   "Dynamics Model input does not match control dimension");
}

MeasurementModel::MeasurementModel(std::size_t state_dim,
                                   std::size_t measurement_dim)
    : m_state_dim(state_dim), m_measurement_dim(measurement_dim) {}

std::size_t MeasurementModel::stateDimension() const {
  return m_state_dim;
}

std::size_t MeasurementModel::measurementDimension() const {
  return m_measurement_dim;
}

void MeasurementModel::checkDimensions(const Eigen::VectorXd& state) const {
  SIA_THROW_IF_NOT(state.size() == int(m_state_dim),
                   "Measurement Model input does not match state dimension");
}

LinearizableDynamics::LinearizableDynamics(std::size_t state_dim,
                                           std::size_t control_dim)
    : DynamicsModel(state_dim, control_dim) {}

Eigen::MatrixXd LinearizableDynamics::F(const Eigen::VectorXd& state,
                                        const Eigen::VectorXd& control) {
  using namespace std::placeholders;
  DynamicsEquation f = std::bind(&LinearizableDynamics::f, this, _1, _2);
  return dfdx(f, state, control);
}

Eigen::MatrixXd LinearizableDynamics::G(const Eigen::VectorXd& state,
                                        const Eigen::VectorXd& control) {
  using namespace std::placeholders;
  DynamicsEquation f = std::bind(&LinearizableDynamics::f, this, _1, _2);
  return dfdu(f, state, control);
}

LinearizableMeasurement::LinearizableMeasurement(std::size_t state_dim,
                                                 std::size_t measurement_dim)
    : MeasurementModel(state_dim, measurement_dim) {}

Eigen::MatrixXd LinearizableMeasurement::H(const Eigen::VectorXd& state) {
  using namespace std::placeholders;
  MeasurementEquation h = std::bind(&LinearizableMeasurement::h, this, _1);
  return numericalJacobian<>(h, state);
}

Eigen::MatrixXd toQ(const Eigen::MatrixXd& Qpsd, double dt) {
  // From Crassidis and Junkins, 2012, pg. 172.
  return Qpsd * dt;
}

Eigen::MatrixXd toQpsd(const Eigen::MatrixXd& Q, double dt) {
  // From Crassidis and Junkins, 2012, pg. 172.
  return Q / dt;
}

Eigen::MatrixXd toR(const Eigen::MatrixXd& Rpsd, double dt) {
  // From Crassidis and Junkins, 2012, pg. 174.
  return Rpsd / dt;
}

Eigen::MatrixXd toRpsd(const Eigen::MatrixXd& R, double dt) {
  // From Crassidis and Junkins, 2012, pg. 174.
  return R * dt;
}

}  // namespace sia
