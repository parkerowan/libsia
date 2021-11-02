/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/models.h"
#include "sia/math/math.h"

namespace sia {

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
