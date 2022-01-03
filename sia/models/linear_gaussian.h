/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/models/nonlinear_gaussian.h"

#include <Eigen/Dense>

namespace sia {

///
/// DISCRETE TIME
///

/// A linear dynamics model with zero-mean additive Gaussian noise
/// $x_k = F x_k-1 + G u_k + w_k, w_k \sim N(0, Q_k)$.
class LinearGaussianDynamics : public LinearizableDynamics {
 public:
  explicit LinearGaussianDynamics(const Eigen::MatrixXd& F,
                                  const Eigen::MatrixXd& G,
                                  const Eigen::MatrixXd& Q);
  virtual ~LinearGaussianDynamics() = default;

  /// Predicts the statistical state transition $p(x_k | x_k-1, u_k)$.
  Gaussian& dynamics(const Eigen::VectorXd& state,
                     const Eigen::VectorXd& control) override;

  /// Expected discrete time dynamics $E[x_k] = f(x_k-1, u_k)$.
  Eigen::VectorXd f(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  /// Process noise covariance $V[x_k] = S(x_k-1, u_k)$.
  Eigen::MatrixXd Q(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  /// Jacobian $F = df(x, u)/dx$.
  Eigen::MatrixXd F(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  /// Jacobian $G = df(x, u)/du$.
  Eigen::MatrixXd G(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  const Eigen::MatrixXd& Q() const;
  const Eigen::MatrixXd& F() const;
  const Eigen::MatrixXd& G() const;
  void setQ(const Eigen::MatrixXd& Q);
  void setF(const Eigen::MatrixXd& F);
  void setG(const Eigen::MatrixXd& G);

 protected:
  LinearGaussianDynamics(const Eigen::MatrixXd& Q);
  void cacheStateCovariance();

  Eigen::MatrixXd m_dynamics_matrix;
  Eigen::MatrixXd m_input_matrix;
  Eigen::MatrixXd m_process_covariance;
  Gaussian m_prob_dynamics;
};

/// A linear measurement model with zero-mean additive Gaussian noise
/// $y_k = H x_k + v_k, v_k \sim N(0, R_k)$.
class LinearGaussianMeasurement : public LinearizableMeasurement {
 public:
  explicit LinearGaussianMeasurement(const Eigen::MatrixXd& H,
                                     const Eigen::MatrixXd& R);
  virtual ~LinearGaussianMeasurement() = default;

  /// Predicts the statistical observation $p(y | x)$.
  Gaussian& measurement(const Eigen::VectorXd& state) override;

  /// Expected observation $E[y] = h(x)$.
  Eigen::VectorXd h(const Eigen::VectorXd& state) override;

  /// Measurement noise covariance $V[y] = R(x)$.
  Eigen::MatrixXd R(const Eigen::VectorXd& state) override;

  /// Jacobian $H = dh(x)/dx$.
  Eigen::MatrixXd H(const Eigen::VectorXd& state) override;

  const Eigen::MatrixXd& R() const;
  const Eigen::MatrixXd& H() const;
  void setR(const Eigen::MatrixXd& R);
  void setH(const Eigen::MatrixXd& H);

 protected:
  void cacheMeasurementCovariance();

  Eigen::MatrixXd m_measurement_matrix;
  Eigen::MatrixXd m_measurement_covariance;
  Gaussian m_prob_measurement;
};

///
/// CONTINUOUS TIME
///

/// A linear continuous-time dynamics model with zero-mean Gaussian noise
/// $\dot{x} = A x + B u + C w, w \sim N(0, Qpsd)$.
/// Note that the dynamics and Qpsd value represent a continuous time process
/// that must be discretized.  The dynamics are discretized the method specified
/// on construction. Qpsd represents a power spectrial density and is converted
/// to a discrete time covariance matrix Q using 1st order approximations from
/// Crassidis and Junkins, 2012, pg. 171-175.
class LinearGaussianDynamicsCT : public LinearGaussianDynamics {
 public:
  /// The discretization type used for the dynamics transformation.
  enum Type {
    FORWARD_EULER,
    BACKWARD_EULER,
  };

  explicit LinearGaussianDynamicsCT(const Eigen::MatrixXd& A,
                                    const Eigen::MatrixXd& B,
                                    const Eigen::MatrixXd& Qpsd,
                                    double dt,
                                    Type type = BACKWARD_EULER);
  virtual ~LinearGaussianDynamicsCT() = default;

  const Eigen::MatrixXd& A() const;
  const Eigen::MatrixXd& B() const;
  void setA(const Eigen::MatrixXd& A);
  void setB(const Eigen::MatrixXd& B);

  /// Sets the process noise power spectral density
  void setQpsd(const Eigen::MatrixXd& Qpsd);

  Type getType() const;
  void setType(Type type);
  double getTimeStep() const;
  void setTimeStep(double dt);

 protected:
  void discretizeDynamics();

  Eigen::MatrixXd m_dynamics_matrix_ct;
  Eigen::MatrixXd m_input_matrix_ct;
  double m_dt;
  Type m_type;
};

/// A linear measurement model with zero-mean additive Gaussian noise
/// $y = H x + v, v \sim N(0, Rpsd)$.
/// Note that the Rpsd value represents a continuous time process that must be
/// discretized.  Rpsd represents a power spectrial density and is converted to
/// a discrete time covariance matrix R using 1st order approximations from
/// Crassidis and Junkins, 2012, pg. 171-175.
class LinearGaussianMeasurementCT : public LinearGaussianMeasurement {
 public:
  explicit LinearGaussianMeasurementCT(const Eigen::MatrixXd& H,
                                       const Eigen::MatrixXd& Rpsd,
                                       double dt);
  virtual ~LinearGaussianMeasurementCT() = default;

  /// Sets the measurement noise power spectral density
  void setRpsd(const Eigen::MatrixXd& Rpsd);
  double getTimeStep() const;
  void setTimeStep(double dt);

 protected:
  double m_dt;
};

}  // namespace sia
