/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/models/models.h"

#include <Eigen/Dense>

namespace sia {

///
/// DISCRETE TIME
///

/// A nonlinear dynamics model with zero-mean additive Gaussian noise
/// $x_k = f(x_k-1, u_k) + w_k, w_k \sim N(0, Q_k)$.
/// Default Jacobians use central difference.
class NonlinearGaussianDynamics : public LinearizableDynamics {
 public:
  /// The dynamics equation is $x_k = f(x_k-1, u_k)$.
  explicit NonlinearGaussianDynamics(DynamicsEquation dynamics,
                                     const Eigen::MatrixXd& Q);
  virtual ~NonlinearGaussianDynamics() = default;

  /// Predicts the statistical state transition $p(x_k | x_k-1, u_k)$.
  Gaussian& dynamics(const Eigen::VectorXd& state,
                     const Eigen::VectorXd& control) override;

  /// Expected discrete time dynamics $E[x_k] = f(x_k-1, u_k)$.
  Eigen::VectorXd f(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  /// Process noise covariance $V[x_k] = Q(x_k-1, u_k)$.
  Eigen::MatrixXd Q(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  const Eigen::MatrixXd& Q() const;
  void setQ(const Eigen::MatrixXd& Q);

 protected:
  void cacheStateCovariance();

  DynamicsEquation m_dynamics;
  Eigen::MatrixXd m_process_covariance;
  Gaussian m_prob_dynamics;
};

/// A nonlinear measurement model with zero-mean additive Gaussian noise
/// $y_k = h(x_k) + v_k, v_k \sim N(0, R_k)$.
/// Default Jacobians use central difference.
class NonlinearGaussianMeasurement : public LinearizableMeasurement {
 public:
  /// The measurement equation is $y = h(x)$.
  explicit NonlinearGaussianMeasurement(MeasurementEquation measurement,
                                        const Eigen::MatrixXd& R);
  virtual ~NonlinearGaussianMeasurement() = default;

  /// Predicts the statistical observation $p(y | x)$.
  Gaussian& measurement(const Eigen::VectorXd& state) override;

  /// Expected observation $E[y] = h(x)$.
  Eigen::VectorXd h(const Eigen::VectorXd& state) override;

  /// Measurement noise covariance $V[y] = R(x)$.
  Eigen::MatrixXd R(const Eigen::VectorXd& state) override;

  const Eigen::MatrixXd& R() const;
  void setR(const Eigen::MatrixXd& R);

 protected:
  void cacheMeasurementCovariance();

  MeasurementEquation m_measurement;
  Eigen::MatrixXd m_measurement_covariance;
  Gaussian m_prob_measurement;
};

///
/// CONTINUOUS TIME
///

/// A nonlinear continuous-time dynamics model with zero-mean Gaussian noise
/// $\dot{x} = a(x, u) + w, w \sim N(0, Qpsd)$.
/// Default Jacobians use central difference.  Note that the dynamics and Qpsd
/// value represent a continuous time process that must be discretized.  The
/// dynamics equation is discretized with rk4 integration. Qpsd represents a
/// power spectrial density and is converted to the discrete time covariance
/// matrix Q using 1st order approximations from Crassidis and Junkins, 2012,
/// pg. 171-175.
class NonlinearGaussianDynamicsCT : public NonlinearGaussianDynamics {
 public:
  /// The dynamics equation is $\dot{x} = a(x, u)$.
  explicit NonlinearGaussianDynamicsCT(DynamicsEquation dynamics,
                                       const Eigen::MatrixXd& Qpsd,
                                       double dt);
  virtual ~NonlinearGaussianDynamicsCT() = default;

  //// Predicts the statistical state transition $p(x_k | x_k-1, u_k)$.
  Gaussian& dynamics(const Eigen::VectorXd& state,
                     const Eigen::VectorXd& control) override;

  /// Expected discrete time dynamics $E[x_k] = f(x_k-1, u_k)$.
  Eigen::VectorXd f(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  /// Jacobian $F = df(x, u)/dx$.
  Eigen::MatrixXd F(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  /// Jacobian $G = df(x, u)/du$.
  Eigen::MatrixXd G(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  /// Set the process noise power spectral density
  void setQpsd(const Eigen::MatrixXd& Qpsd);
  double getTimeStep() const;
  void setTimeStep(double dt);

 protected:
  double m_dt;
};

/// A nonlinear measurement model with zero-mean additive Gaussian noise
/// $y = h(x) + v, v \sim N(0, Rpsd)$.
/// Default Jacobians use central difference.  Note that the Rpsd value
/// represents a continuous time process that must be discretized.  Rpsd
/// represents a power spectrial density and is converted to the discrete time
/// covariance matrix R using 1st order approximations from Crassidis and
/// Junkins, 2012, pg. 171-175.
class NonlinearGaussianMeasurementCT : public NonlinearGaussianMeasurement {
 public:
  explicit NonlinearGaussianMeasurementCT(MeasurementEquation measurement,
                                          const Eigen::MatrixXd& Rpsd,
                                          double dt);
  virtual ~NonlinearGaussianMeasurementCT() = default;

  /// Sets the measurement noise power spectral density
  void setRpsd(const Eigen::MatrixXd& Rpsd);
  double getTimeStep() const;
  void setTimeStep(double dt);

 protected:
  double m_dt;
};

}  // namespace sia
