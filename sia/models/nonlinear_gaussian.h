/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/models/models.h"

#include <Eigen/Dense>

namespace sia {

/// A Markov process with a discrete time non-linear Gaussian system of
/// equations where the dynamics equation is
/// - $x_k = f(x_k-1, u_k) + C w_k, w_k \sim N(0, Q_k)$,
/// and the measurement equation is
/// - $y_k = h(x_k) + v_k, v_k \sim N(0, R_k)$.
/// Numerical differentiation is used to compute Jacobians.
class NonlinearGaussian : public MarkovProcess {
 public:
  explicit NonlinearGaussian(DynamicsEquation dynamics,
                             MeasurementEquation measurement,
                             const Eigen::MatrixXd& C,
                             const Eigen::MatrixXd& Q,
                             const Eigen::MatrixXd& R);

  /// Returns the distribution predicted by the discrete time state transition
  /// (dynamics) model $p(x_k | x_k-1, u_k)$.
  Gaussian& dynamics(const Eigen::VectorXd& state,
                     const Eigen::VectorXd& control) override;

  /// Returns the distribution predicted by the measurement (observation) model
  /// $p(y_k | x_k)$.
  Gaussian& measurement(const Eigen::VectorXd& state) override;

  /// Returns the deterministic discrete time state transition (dynamics)
  /// predicted by the system, i.e. $x_k = f(x_k-1, u_k)$.
  virtual const Eigen::VectorXd f(const Eigen::VectorXd& state,
                                  const Eigen::VectorXd& control) const;

  /// Returns the Jacobian (linearization) of the dynamics w.r.t. $x$, i.e.
  /// $F = df(x_k-1, u_k) / dx_k-1$.
  virtual const Eigen::MatrixXd F(const Eigen::VectorXd& state,
                                  const Eigen::VectorXd& control) const;

  /// Returns the Jacobian (linearization) of the dynamics w.r.t. $u$, i.e.
  /// $F = df(x_k-1, u_k) / du_k$.
  virtual const Eigen::MatrixXd G(const Eigen::VectorXd& state,
                                  const Eigen::VectorXd& control) const;

  /// Returns the deterministic measurement (observation) predicted by the
  /// system, i.e. $y = h(x)$.
  virtual const Eigen::VectorXd h(const Eigen::VectorXd& state) const;

  /// Returns the Jacobian (linearization) of the deterministic measurement
  /// (observation) predicted by the system, i.e. $H = dh(x) / dx$.
  virtual const Eigen::MatrixXd H(const Eigen::VectorXd& state) const;

  const Eigen::MatrixXd& C() const;
  const Eigen::MatrixXd& Q() const;
  const Eigen::MatrixXd& R() const;
  virtual void setC(const Eigen::MatrixXd& C);
  virtual void setQ(const Eigen::MatrixXd& Q);
  virtual void setR(const Eigen::MatrixXd& R);

 protected:
  // Used to initialize derived classes
  explicit NonlinearGaussian(const Eigen::MatrixXd& C,
                             const Eigen::MatrixXd& Q,
                             const Eigen::MatrixXd& R);
  virtual void cacheStateCovariance();
  virtual void cacheMeasurementCovariance();

  DynamicsEquation m_dynamics;
  MeasurementEquation m_measurement;
  Eigen::MatrixXd m_process_noise_matrix;
  Eigen::MatrixXd m_process_covariance;
  Eigen::MatrixXd m_measurement_covariance;
  Gaussian m_prob_dynamics;
  Gaussian m_prob_measurement;
};

}  // namespace sia
