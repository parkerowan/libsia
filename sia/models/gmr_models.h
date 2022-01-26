/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/belief/gmr.h"
#include "sia/models/models.h"

#include <Eigen/Dense>

namespace sia {

///
/// DISCRETE TIME
///

/// A nonlinear Gaussian dynamics model based on GMR inference
/// $p(x_k+1) = GMR(x_k, u_k)$.
/// Default Jacobians use central difference.
class GMRDynamics : public LinearizableDynamics {
 public:
  /// The dynamics equation is $x_k+1 = f(x_k, u_k)$.  This constructor defines
  /// the model from initial data.  Cols are samples.
  explicit GMRDynamics(const Eigen::MatrixXd& Xk,
                       const Eigen::MatrixXd& Uk,
                       const Eigen::MatrixXd& Xkp1,
                       std::size_t K);
  virtual ~GMRDynamics() = default;

  /// Predicts the statistical state transition $p(x_k+1 | x_k, u_k)$.
  Gaussian& dynamics(const Eigen::VectorXd& state,
                     const Eigen::VectorXd& control) override;

  /// Expected discrete time dynamics $E[x_k+1] = f(x_k, u_k)$.
  Eigen::VectorXd f(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  /// Process noise covariance $V[x_k+1] = Q(x_k, u_k)$.
  Eigen::MatrixXd Q(const Eigen::VectorXd& state,
                    const Eigen::VectorXd& control) override;

  /// Access the underlying GMR
  GMR& gmr();
  double negLogLik(const Eigen::MatrixXd& Xk,
                   const Eigen::MatrixXd& Uk,
                   const Eigen::MatrixXd& Xkp1);

 protected:
  Gaussian m_prob_dynamics;
  std::vector<std::size_t> m_input_indices;
  std::vector<std::size_t> m_output_indices;
  GMR m_gmr;
};

/// A nonlinear Gaussian measurement model based on GMR inference
/// $p(y) = GMR(x)$.
/// Default Jacobians use central difference.
// class GMRMeasurement : public LinearizableMeasurement {
//  public:
//   /// The measurement equation is $y = h(x)$.
//   explicit GMRMeasurement();
//   virtual ~GMRMeasurement() = default;

//   /// Predicts the statistical observation $p(y | x)$.
//   Gaussian& measurement(const Eigen::VectorXd& state) override;

//   /// Expected observation $E[y] = h(x)$.
//   Eigen::VectorXd h(const Eigen::VectorXd& state) override;

//   /// Measurement noise covariance $V[y] = R(x)$.
//   Eigen::MatrixXd R(const Eigen::VectorXd& state) override;

//  protected:
//   Gaussian m_prob_measurement;
// };

}  // namespace sia
