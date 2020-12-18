/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/models/linear_gaussian.h"

#include <Eigen/Dense>

namespace sia {

/// A Markov process with a continuous time linear Gaussian system of equations
/// (specialization of the linear discrete time case) where the dynamics
/// equation is
/// - $\dot{x} = A x + B u + C w, w \sim N(0, Q)$,
/// and the measurement equation is
/// - $y = H x + v, v \sim N(0, R)$.
/// Note that the dynamics, Q, and R values represent continuous time processes
/// that must be discretized.  The dynamics equation is discretized using
/// backward Euler (improved stability) or forward Euler (cheaper computation).
/// The matrices Q, and R represent power spectrial densities and are converted
/// to discrete time covariance matrices using 1st order approximations from
/// Crassidis and Junkins, 2012, pg. 171-175.
class LinearGaussianCT : public LinearGaussian {
 public:
  /// The discretization type used for the dynamics transformation.
  enum Type {
    FORWARD_EULER,
    BACKWARD_EULER,
  };

  explicit LinearGaussianCT(const Eigen::MatrixXd& A,
                            const Eigen::MatrixXd& B,
                            const Eigen::MatrixXd& C,
                            const Eigen::MatrixXd& H,
                            const Eigen::MatrixXd& Q,
                            const Eigen::MatrixXd& R,
                            double dt,
                            Type type = BACKWARD_EULER);

  /// Returns the continuous time dynamics matrix
  const Eigen::MatrixXd& A() const;

  /// Returns the continuous time input matrix
  const Eigen::MatrixXd& B() const;

  /// Sets the continuous time dynamics matrix
  void setA(const Eigen::MatrixXd& A);

  /// Sets the continuous time input matrix
  void setB(const Eigen::MatrixXd& B);

  /// Sets the process noise matrix
  void setC(const Eigen::MatrixXd& C) override;

  /// Sets the process noise power spectral density
  void setQ(const Eigen::MatrixXd& Q) override;

  /// Sets the measurement noise power spectral density
  void setR(const Eigen::MatrixXd& R) override;

  Type getType() const;
  void setType(Type type);
  double getTimeStep() const;
  void setTimeStep(double dt);

 protected:
  void cacheStateCovariance() override;
  void cacheMeasurementCovariance() override;
  void discretizeDynamics();

  Eigen::MatrixXd m_dynamics_matrix_ct;
  Eigen::MatrixXd m_input_matrix_ct;
  double m_dt;
  Type m_type;
};

}  // namespace sia
