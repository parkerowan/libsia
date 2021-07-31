/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/models/nonlinear_gaussian.h"

#include <Eigen/Dense>

namespace sia {

/// A Markov process with a continuous time non-linear Gaussian system of
/// equations where the dynamics equation is
/// - $\dot{x} = f(x, u) + C w, w \sim N(0, Q)$,
/// and the measurement equation is
/// - $y = h(x) + v, v \sim N(0, R)$.
/// Numerical differentiation is used to compute Jacobians.  Note that the
/// dynamics, Q, and R values represent continuous time processes that must be
/// discretized.  The dynamics equation is discretized with rk4 integration. The
/// matrices Q, and R represent power spectrial densities and are converted to
/// discrete time covariance matrices using 1st order approximations from
/// Crassidis and Junkins, 2012, pg. 171-175.
class NonlinearGaussianCT : public NonlinearGaussian {
 public:
  explicit NonlinearGaussianCT(DynamicsEquation dynamics,
                               MeasurementEquation measurement,
                               const Eigen::MatrixXd& C,
                               const Eigen::MatrixXd& Q,
                               const Eigen::MatrixXd& R,
                               double dt);

  /// Returns the distribution predicted by the discrete time state transition
  /// (dynamics) model $p(x_k | x_k-1, u_k)$.
  Gaussian& dynamics(const Eigen::VectorXd& state,
                     const Eigen::VectorXd& control) override;

  /// Returns the deterministic discrete time state transition (dynamics)
  /// predicted by the system, i.e. $x_k = f(x_k-1, u_k)$.
  const Eigen::VectorXd f(const Eigen::VectorXd& state,
                          const Eigen::VectorXd& control) const override;

  /// Returns the Jacobian (linearization) of the dynamics w.r.t. $x$, i.e.
  /// $F = df(x_k-1, u_k) / dx_k-1$.
  const Eigen::MatrixXd F(const Eigen::VectorXd& state,
                          const Eigen::VectorXd& control) const override;

  /// Returns the Jacobian (linearization) of the dynamics w.r.t. $u$, i.e.
  /// $F = df(x_k-1, u_k) / du_k$.
  const Eigen::MatrixXd G(const Eigen::VectorXd& state,
                          const Eigen::VectorXd& control) const override;

  void setC(const Eigen::MatrixXd& C) override;
  void setQ(const Eigen::MatrixXd& Q) override;
  void setR(const Eigen::MatrixXd& R) override;
  double getTimeStep() const;
  void setTimeStep(double dt);

 protected:
  void cacheStateCovariance() override;
  void cacheMeasurementCovariance() override;

  double m_dt;
};

}  // namespace sia