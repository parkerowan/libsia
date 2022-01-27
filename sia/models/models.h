/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

/// The dynamics equation predicts the deterministic state propogation:
/// - continuous time case $\dot{x} = f(x, u)$
/// - discrete time case $x_k+1 = f(x_k, u_k)$
/// where $x$ is state that evolves over time interval $k$, and $u$ is the
/// control.
using DynamicsEquation =
    std::function<const Eigen::VectorXd(const Eigen::VectorXd&,
                                        const Eigen::VectorXd&)>;

/// The measurement equation predicts the deterministic measurement:
/// $y = h(x)$ where $x$ is state and $y$ is observation.
using MeasurementEquation =
    std::function<const Eigen::VectorXd(const Eigen::VectorXd&)>;

/// The dynamics Jacobian predicts the partial derivative of the dynamics
/// equation wrt to the state:
/// - continuous time case $J = df(x, u) / dx$
/// - discrete time case $J = df(x_k, u_k) / dx_k$
/// where $x$ is state that evolves over time interval $k$, $f$ is the dynamics
/// equation and $u$ is the control.
using DynamicsJacobian =
    std::function<const Eigen::MatrixXd(const Eigen::VectorXd&,
                                        const Eigen::VectorXd&)>;

/// The measurement Jacobian predicts the partial derivative of the measurement
/// equation wrt to the state: $J = dh(x) / dx$ where $x$ is state, $h$ is the
/// measurement equation, and $y$ is observation.
using MeasurementJacobian =
    std::function<const Eigen::MatrixXd(const Eigen::VectorXd&)>;

/// A Markov system that predicts the statistical discrete time state transition
/// $p(x_k+1 | x_k, u_k)$.
/// - $x$ State that evolves over time step $k$.
/// - $u$ Known control action applied to affect the state $x$.
class DynamicsModel {
 public:
  DynamicsModel() = default;
  virtual ~DynamicsModel() = default;

  /// Predicts the statistical state transition $p(x_k+1 | x_k, u_k)$.
  virtual Distribution& dynamics(const Eigen::VectorXd& state,
                                 const Eigen::VectorXd& control) = 0;
};

/// A system that predicts the statistical observation $p(y | x)$.
/// - $x$ Input state.
/// - $y$ Observation generated from the state $x$.
class MeasurementModel {
 public:
  MeasurementModel() = default;
  virtual ~MeasurementModel() = default;

  /// Predicts the statistical observation $p(y | x)$.
  virtual Distribution& measurement(const Eigen::VectorXd& state) = 0;
};

/// Linearizable dynamics model.  Default Jacobians use central difference.
class LinearizableDynamics : public DynamicsModel {
 public:
  LinearizableDynamics() = default;
  virtual ~LinearizableDynamics() = default;

  /// Expected discrete time dynamics $E[x_k+1] = f(x_k, u_k)$.
  virtual Eigen::VectorXd f(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& control) = 0;

  /// Process noise covariance $V[x_k+1] = Q(x_k, u_k)$.
  virtual Eigen::MatrixXd Q(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& control) = 0;

  /// Jacobian $F = df(x, u)/dx$.
  virtual Eigen::MatrixXd F(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& control);

  /// Jacobian $G = df(x, u)/du$.
  virtual Eigen::MatrixXd G(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& control);
};

/// Linearizable measurement model.  Default Jacobians use central difference.
class LinearizableMeasurement : public MeasurementModel {
 public:
  LinearizableMeasurement() = default;
  virtual ~LinearizableMeasurement() = default;

  /// Expected observation $E[y] = h(x)$.
  virtual Eigen::VectorXd h(const Eigen::VectorXd& state) = 0;

  /// Measurement noise covariance $V[y] = R(x)$.
  virtual Eigen::MatrixXd R(const Eigen::VectorXd& state) = 0;

  /// Jacobian $H = dh(x)/dx$.
  virtual Eigen::MatrixXd H(const Eigen::VectorXd& state);
};

/// Convert from a process noise power spectral density (continuous time) to a
/// covariance (discrete time). From Crassidis and Junkins, 2012, pg. 172.
Eigen::MatrixXd toQ(const Eigen::MatrixXd& Qpsd, double dt);

/// Convert to a process noise power spectral density (continuous time) from a
/// covariance (discrete time). From Crassidis and Junkins, 2012, pg. 172.
Eigen::MatrixXd toQpsd(const Eigen::MatrixXd& Q, double dt);

/// Convert from a measurement noise power spectral density (continuous time) to
/// a covariance (discrete time). From Crassidis and Junkins, 2012, pg. 174.
Eigen::MatrixXd toR(const Eigen::MatrixXd& Rpsd, double dt);

/// Convert to a measurement noise power spectral density (continuous time) from
/// a covariance (discrete time). From Crassidis and Junkins, 2012, pg. 174.
Eigen::MatrixXd toRpsd(const Eigen::MatrixXd& R, double dt);

// TODO:

// nonlinear_gmr.h
// class GMRDynamics : public LinearizableDynamics {};
// class GMRMeasurement : public LinearizableMeasurement {};

// nonlinear_gpr.h
// class GPRDynamics : public LinearizableDynamics {};
// class GPRMeasurement : public LinearizableMeasurement {};

}  // namespace sia
