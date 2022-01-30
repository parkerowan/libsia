/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/gaussian.h"
#include "sia/belief/gmm.h"
#include "sia/belief/gmr.h"
#include "sia/models/models.h"

#include <Eigen/Dense>

namespace sia {

///
/// DISCRETE TIME
///

/// A nonlinear Gaussian dynamics model based on GMR inference
///
/// $p(x_k+1) = GMR(x_k, u_k)$.
///
/// The default constructor builds a GMR model with K clusters based on the data
/// D = {x_k, u_k, x_k+1}.  The GMR input is chosen as X = {x_k, u_k} and the
/// output is chosen as Y = {x_k+1 - x_k}.  This choice of normalized output
/// biases the model to predicting small changes in state outside of the
/// demonstration domain.  Default Jacobians use central difference.
///
/// The negative log likelihood is the following for the
///
/// $L = - \sum_i log p(y_i | x_i)$
///
/// Choice of K influences the resolution of the model.  K = 1 yields a linear
/// model, while K = num samples yields a kernel regression.  The appropriate
/// model can be chosen based on AIC, BIC (information criteria) or
/// cross-validation.
class GMRDynamics : public LinearizableDynamics {
 public:
  /// The dynamics equation is $x_k+1 = f(x_k, u_k)$.  This constructor defines
  /// the model from initial data.  Cols are samples.
  explicit GMRDynamics(const Eigen::MatrixXd& Xk,
                       const Eigen::MatrixXd& Uk,
                       const Eigen::MatrixXd& Xkp1,
                       std::size_t K,
                       double regularization = GMM::DEFAULT_REGULARIZATION);
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

  /// Retrain the GMR with new data
  void train(const Eigen::MatrixXd& Xk,
             const Eigen::MatrixXd& Uk,
             const Eigen::MatrixXd& Xkp1,
             GMM::FitMethod fit_method = GMM::GAUSSIAN_LIKELIHOOD,
             GMM::InitMethod init_method = GMM::WARM_START,
             double regularization = GMM::DEFAULT_REGULARIZATION);

  /// Computes the negative log likelihood loss on test data using the GMR
  /// negLogLik routine.  Colums are samples.
  double negLogLik(const Eigen::MatrixXd& Xk,
                   const Eigen::MatrixXd& Uk,
                   const Eigen::MatrixXd& Xkp1);

  /// Computes the mean squared error loss on test data using the GMR mse
  /// routine.  Colums are samples.
  double mse(const Eigen::MatrixXd& Xk,
             const Eigen::MatrixXd& Uk,
             const Eigen::MatrixXd& Xkp1);

  /// Access the underlying GMR
  GMR& gmr();

 protected:
  GMR createGMR(const Eigen::MatrixXd& Xk,
                const Eigen::MatrixXd& Uk,
                const Eigen::MatrixXd& Xkp1,
                std::size_t K,
                double regularization) const;

  Gaussian m_prob_dynamics;
  GMR m_gmr;
};

/// A nonlinear Gaussian measurement model based on GMR inference
///
/// $p(y) = GMR(x)$.
///
/// The default constructor builds a GMR model with K clusters based on the data
/// D = {x_k, y_k}.  Default Jacobians use central difference.
///
/// The negative log likelihood is the following for the
///
/// $L = - \sum_i log p(y_i | x_i)$
///
/// Choice of K influences the resolution of the model.  K = 1 yields a linear
/// model, while K = num samples yields a kernel regression.  The appropriate
/// model can be chosen based on AIC, BIC (information criteria) or
/// cross-validation.
class GMRMeasurement : public LinearizableMeasurement {
 public:
  /// The measurement equation is $y = h(x)$.  This constructor defines
  /// the model from initial data.  Cols are samples.
  explicit GMRMeasurement(const Eigen::MatrixXd& X,
                          const Eigen::MatrixXd& Y,
                          std::size_t K,
                          double regularization = GMM::DEFAULT_REGULARIZATION);
  virtual ~GMRMeasurement() = default;

  /// Predicts the statistical observation $p(y | x)$.
  Gaussian& measurement(const Eigen::VectorXd& state) override;

  /// Expected observation $E[y] = h(x)$.
  Eigen::VectorXd h(const Eigen::VectorXd& state) override;

  /// Measurement noise covariance $V[y] = R(x)$.
  Eigen::MatrixXd R(const Eigen::VectorXd& state) override;

  /// Retrain the GMR with new data
  void train(const Eigen::MatrixXd& X,
             const Eigen::MatrixXd& Y,
             GMM::FitMethod fit_method = GMM::GAUSSIAN_LIKELIHOOD,
             GMM::InitMethod init_method = GMM::WARM_START,
             double regularization = GMM::DEFAULT_REGULARIZATION);

  /// Computes the negative log likelihood loss on test data using the GMR
  /// negLogLik routine.  Colums are samples.
  double negLogLik(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);

  /// Computes the mean squared error loss on test data using the GMR mse
  /// routine.  Colums are samples.
  double mse(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);

  /// Access the underlying GMR
  GMR& gmr();

 protected:
  GMR createGMR(const Eigen::MatrixXd& X,
                const Eigen::MatrixXd& Y,
                std::size_t K,
                double regularization) const;

  Gaussian m_prob_measurement;
  GMR m_gmr;
};

}  // namespace sia
