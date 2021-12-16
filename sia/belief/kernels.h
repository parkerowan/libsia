/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <Eigen/Dense>

namespace sia {

/// Kernel basis function base class.  Kernels are symmetric and positive
/// definite.  The gradient functions returns the Jacobian w.r.t. to the kernel
/// hyperarameters.
class CovarianceFunction {
 public:
  /// Determines which kernel function is used
  enum Type {
    SQUARED_EXPONENTIAL,
  };

  virtual double eval(const Eigen::VectorXd& a,
                      const Eigen::VectorXd& b) const = 0;
  virtual Eigen::VectorXd grad(const Eigen::VectorXd& a,
                               const Eigen::VectorXd& b) const = 0;
  virtual Eigen::VectorXd hyperparameters() const = 0;
  virtual void setHyperparameters(const Eigen::VectorXd& p) = 0;
  std::size_t numHyperparameters() const;

  /// Evaluates k(a, b) for various permutations of a, b
  Eigen::VectorXd evalVector(const Eigen::MatrixXd& a,
                             const Eigen::VectorXd& b);
  Eigen::MatrixXd evalMatrix(const Eigen::MatrixXd& a,
                             const Eigen::MatrixXd& b);
  std::vector<Eigen::MatrixXd> gradTensor(const Eigen::MatrixXd& a,
                                          const Eigen::MatrixXd& b);

 protected:
  /// Kernel function factory
  static CovarianceFunction* create(Type type);
};

/// The squared exponential function.
/// - length controls the kernel basis blending
/// - noise_var controls the measurement likelihood uncertainty
/// - signal_var controls the marginal variance of the Gaussian prior
class SquaredExponential : public CovarianceFunction {
 public:
  explicit SquaredExponential(double length = 1.0,
                              double signal_var = 1.0,
                              double noise_var = 0.1);
  virtual ~SquaredExponential() = default;
  double eval(const Eigen::VectorXd& a,
              const Eigen::VectorXd& b) const override;
  Eigen::VectorXd grad(const Eigen::VectorXd& a,
                       const Eigen::VectorXd& b) const override;
  Eigen::VectorXd hyperparameters() const override;
  void setHyperparameters(const Eigen::VectorXd& p) override;

 private:
  double m_length;
  double m_signal_var;
  double m_noise_var;
};

}  // namespace sia
