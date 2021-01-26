/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

/// The dynamics equation predicts the deterministic state propogation:
/// - continuous time case $\dot{x} = f(x, u)$
/// - discrete time case $x_k = f(x_k-1, u_k)$
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
/// - discrete time case $J = df(x_k-1, u_k) / dx$
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

/// A Markov process is a statistical system defined by two equations: (i) the
/// dynamical system that describes state evolution, and (ii) the measurement
/// equation that describes the observations.
/// - $x$ Markov state that evolves over time step $k$.
/// - $u$ Known control actions applied to influence the state $x$.
/// - $y$ Measurements generated from the state $x$.
class MarkovProcess {
 public:
  /// Returns the distribution predicted by the discrete time state transition
  /// (dynamics) model $p(x_k | x_k-1, u_k)$.
  virtual Distribution& dynamics(const Eigen::VectorXd& state,
                                 const Eigen::VectorXd& control) = 0;

  /// Returns the distribution predicted by the measurement (observation) model
  /// $p(y_k | x_k)$.
  virtual Distribution& measurement(const Eigen::VectorXd& state) = 0;
};

}  // namespace sia
