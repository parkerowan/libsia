/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/models/nonlinear_gaussian.h"

#include <Eigen/Dense>

namespace sia {

/// A Markov process with a discrete time linear Gaussian system of equations
/// (specialization of the nonlinear case) where the dynamics equation is
/// - $x_k = F x_k-1 + G u_k + C w_k, w_k \sim N(0, Q_k)$,
/// and the measurement equation is
/// - $y_k = H x_k + v_k, v_k \sim N(0, R_k)$.
class LinearGaussian : public NonlinearGaussian {
 public:
  explicit LinearGaussian(const Eigen::MatrixXd& F,
                          const Eigen::MatrixXd& G,
                          const Eigen::MatrixXd& C,
                          const Eigen::MatrixXd& H,
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
  const Eigen::VectorXd f(const Eigen::VectorXd& state,
                          const Eigen::VectorXd& control) const override;

  /// Returns the Jacobian (linearization) of the dynamics, which for the linear
  /// case is equivalent to the F matrix, i.e. $F = df(x_k-1, u_k)/dx_k-1$.
  const Eigen::MatrixXd F(const Eigen::VectorXd& state,
                          const Eigen::VectorXd& control) const override;

  /// Returns the deterministic measurement (observation) predicted by the
  /// system, i.e. $y = h(x)$.
  const Eigen::VectorXd h(const Eigen::VectorXd& state) const override;

  /// Returns the Jacobian (linearization) of the measurement, which for the
  /// linear case is equivalent to the H matrix, i.e. $H = dh(x)/dx$.
  const Eigen::MatrixXd H(const Eigen::VectorXd& state) const override;

  const Eigen::MatrixXd& F() const;
  const Eigen::MatrixXd& G() const;
  const Eigen::MatrixXd& H() const;
  void setF(const Eigen::MatrixXd& F);
  void setG(const Eigen::MatrixXd& G);
  void setH(const Eigen::MatrixXd& H);

 protected:
  explicit LinearGaussian(const Eigen::MatrixXd& C,
                          const Eigen::MatrixXd& H,
                          const Eigen::MatrixXd& Q,
                          const Eigen::MatrixXd& R);
  Eigen::MatrixXd m_dynamics_matrix;
  Eigen::MatrixXd m_input_matrix;
  Eigen::MatrixXd m_measurement_matrix;
};

}  // namespace sia
