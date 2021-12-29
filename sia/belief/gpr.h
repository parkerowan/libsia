/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"

#include <Eigen/Dense>

namespace sia {

/// Gaussian Process Regression performs Gaussian regression to predict
/// $p(y|x)$ using a GP kernel prior.  This class implements algorithm 2.1
/// from Rasmussen and Williams, and assumes a zero mean prior.
///
/// References:
/// [1]: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
class GPR : public Inference {
 public:
  /// Kernel function used for the GP covariance
  enum KernelType { SQUARED_EXPONENTIAL };

  /// Function used for the GP measurement noise
  enum NoiseType {
    SCALAR,          // Single variance for all channels
    VECTOR,          // Single variance for each channel
    HETEROSKEDASTIC  // Unique variance for each channel and data point
  };

  /// Initialize GPR from training data.  Each column of input and output
  /// training samples is a sample.
  explicit GPR(const Eigen::MatrixXd& input_samples,
               const Eigen::MatrixXd& output_samples,
               KernelType kernel_type = SQUARED_EXPONENTIAL,
               NoiseType noise_type = SCALAR);
  virtual ~GPR();

  /// Performs the inference $p(y|x)$
  const Gaussian& predict(const Eigen::VectorXd& x) override;

  /// Computes the negative log marginal likelihood loss and gradients
  double negLogMarginalLik();
  Eigen::VectorXd negLogMarginalLikGrad();

  /// Train the hyperparameters
  void train();

  /// Dimensions
  std::size_t inputDimension() const override;
  std::size_t outputDimension() const override;
  std::size_t numSamples() const;

  /// Access the hyperparameters
  Eigen::VectorXd hyperparameters() const;
  void setHyperparameters(const Eigen::VectorXd& p);
  std::size_t numHyperparameters() const;

  /// Set the white noise variance hyperparameters
  void setScalarNoise(double var);
  void setVectorNoise(const Eigen::VectorXd& var);
  void setHeteroskedasticNoise(const Eigen::MatrixXd& var);

  // Forward declarations
  struct KernelFunction;
  struct NoiseFunction;
  struct RegressionModel;

 private:
  void cacheRegressionModel();

  Eigen::MatrixXd m_input_samples;
  Eigen::MatrixXd m_output_samples;
  Gaussian m_belief;
  NoiseType m_noise_type;
  KernelFunction* m_kernel{nullptr};
  std::vector<RegressionModel> m_models;
};

// Terms for a 1D regression model
struct GPR::RegressionModel {
  explicit RegressionModel(const Eigen::MatrixXd& L,
                           const Eigen::MatrixXd& Linv,
                           const Eigen::MatrixXd& Kinv,
                           const Eigen::VectorXd& alpha,
                           const std::vector<Eigen::MatrixXd>& grad);
  Eigen::MatrixXd cached_L;
  Eigen::MatrixXd cached_L_inv;
  Eigen::MatrixXd cached_K_inv;
  Eigen::VectorXd cached_alpha;
  std::vector<Eigen::MatrixXd> cached_grad;
};

// Kernel basis function base class.  Kernels are symmetric and positive
// definite.  The gradient functions returns the Jacobian w.r.t. to the kernel
// hyperarameters.
struct GPR::KernelFunction {
  virtual ~KernelFunction() = default;
  virtual double eval(const Eigen::VectorXd& a,
                      const Eigen::VectorXd& b) const = 0;
  virtual Eigen::VectorXd grad(const Eigen::VectorXd& a,
                               const Eigen::VectorXd& b) const = 0;
  virtual Eigen::VectorXd hyperparameters() const = 0;
  virtual void setHyperparameters(const Eigen::VectorXd& p) = 0;
  std::size_t numHyperparameters() const;
};

// The squared exponential function.
// - length controls the kernel basis blending
// - signal_var controls the marginal variance of the Gaussian prior
class SquaredExponential : public GPR::KernelFunction {
 public:
  explicit SquaredExponential(double length = 1.0, double signal_var = 1.0);
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
};

}  // namespace sia
