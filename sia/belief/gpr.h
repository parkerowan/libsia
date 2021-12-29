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
  std::size_t inputDimension() const override;
  std::size_t outputDimension() const override;
  std::size_t numSamples() const;

  /// Train the hyperparameters
  void train();

  /// Access the hyperparameters
  Eigen::VectorXd hyperparameters() const;
  void setHyperparameters(const Eigen::VectorXd& p);
  std::size_t numHyperparameters() const;

  /// Set the white noise variance hyperparameters
  void setScalarNoise(double var);
  void setVectorNoise(const Eigen::VectorXd& var);
  void setHeteroskedasticNoise(const Eigen::MatrixXd& var);

  /// Computes the negative log marginal likelihood loss and gradients
  double negLogMarginalLik();
  Eigen::VectorXd negLogMarginalLikGrad();

  /// Covariance function
  struct CovarianceFunction;

 private:
  void cacheRegressionModels();

  // Terms for a 1D regression model
  struct RegressionModel {
    explicit RegressionModel(CovarianceFunction* kernel,
                             const Eigen::MatrixXd& X,
                             const Eigen::VectorXd& y);
    double logMarginalLik() const;
    Eigen::VectorXd logMarginalLikGrad() const;
    Eigen::MatrixXd m_X;
    Eigen::VectorXd m_y;
    Eigen::MatrixXd m_cached_L;
    Eigen::MatrixXd m_cached_L_inv;
    Eigen::MatrixXd m_cached_K_inv;
    Eigen::VectorXd m_cached_alpha;
    CovarianceFunction* m_kernel;
  };

  Eigen::MatrixXd m_input_samples;
  Eigen::MatrixXd m_output_samples;
  Gaussian m_belief;
  NoiseType m_noise_type;
  CovarianceFunction* m_kernel{nullptr};
  std::vector<RegressionModel> m_models;
};

// Kernel basis function base class.  Kernels are symmetric and positive
// definite.  The gradient functions returns the Jacobian w.r.t. to the kernel
// hyperarameters.
struct GPR::CovarianceFunction {
  virtual ~CovarianceFunction() = default;
  virtual double eval(const Eigen::VectorXd& a,
                      const Eigen::VectorXd& b) const = 0;
  virtual Eigen::VectorXd grad(const Eigen::VectorXd& a,
                               const Eigen::VectorXd& b) const = 0;
  virtual Eigen::VectorXd hyperparameters() const = 0;
  virtual void setHyperparameters(const Eigen::VectorXd& p) = 0;
  std::size_t numHyperparameters() const;

  // Evalutes the kernel to construct the na x 1 kernel vector K(a, b) where a,
  // b are input samples with cols equal to samples.
  Eigen::VectorXd evalVector(const Eigen::MatrixXd& a,
                             const Eigen::VectorXd& b);

  // Evaluates the kernel to construct the na x nb kernel matrix K(a, b) where
  // a, b are input samples with cols equal to samples.
  Eigen::MatrixXd evalMatrix(const Eigen::MatrixXd& a,
                             const Eigen::MatrixXd& b);

  // Evaluates the kernel gradient w.r.t. hyperparameters to construct the
  // tensor na x nb kernel matrix dK(a, b)/dp where the number of elements of
  // the output vector correspond to the number of hyperparameters.
  std::vector<Eigen::MatrixXd> gradTensor(const Eigen::MatrixXd& a,
                                          const Eigen::MatrixXd& b);

  // Factory
  static GPR::CovarianceFunction* create(GPR::KernelType kernel_type);
};

// The squared exponential function.
// - length controls the kernel basis blending
// - signal_var controls the marginal variance of the Gaussian prior
class SquaredExponential : public GPR::CovarianceFunction {
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
