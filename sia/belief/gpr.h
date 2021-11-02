/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"

#include <Eigen/Dense>
#include <memory>

namespace sia {

/// Gaussian Process Regression performs Gaussian regression to predict
/// $p(y|x)$ using a GP kernel prior.  This class implements algorithm 2.1
/// from Rasmussen and Williams, and assumes a zero mean prior.
///
/// Hyperparameters for Kernel Types:
/// - SE_KERNEL: (length, signal_var)
///
/// References:
/// [1]: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
class GPR : public Inference {
 public:
  /// Kernel function used for the GP covariance
  enum KernelType {
    SE_KERNEL  // Squared exponential kernel
  };

  /// Function used for the GP measurement noise
  enum NoiseType {
    SCALAR_NOISE,          // Single variance for all channels
    VECTOR_NOISE,          // Single variance for each channel
    HETEROSKEDASTIC_NOISE  // Unique variance for each channel and sample
  };

  /// Initialize GPR from training data.  Each column of input and output
  /// training samples is a sample.
  explicit GPR(const Eigen::MatrixXd& input_samples,
               const Eigen::MatrixXd& output_samples,
               KernelType kernel_type = SE_KERNEL,
               NoiseType noise_type = SCALAR_NOISE);
  explicit GPR(const Eigen::MatrixXd& input_samples,
               const Eigen::MatrixXd& output_samples,
               const Eigen::VectorXd& hyperparameters,
               double noise_variance,
               KernelType kernel_type = SE_KERNEL);
  virtual ~GPR() = default;

  /// Recompute with new data but same hyperparameters
  void setData(const Eigen::MatrixXd& input_samples,
               const Eigen::MatrixXd& output_samples);

  /// Performs the inference $p(y|x)$
  const Gaussian& predict(const Eigen::VectorXd& x) override;

  /// Computes the negative log marginal likelihood loss and gradients
  double negLogMarginalLik() const;
  Eigen::VectorXd negLogMarginalLikGrad() const;

  /// Train the hyperparameters
  void train();

  /// Dimensions
  std::size_t inputDimension() const override;
  std::size_t outputDimension() const override;
  std::size_t numSamples() const;

  /// Access the hyperparameters
  Eigen::VectorXd hyperparameters() const;
  void setHyperparameters(const Eigen::VectorXd& hyperparameters);
  std::size_t numHyperparameters() const;

  /// Set the white noise variance hyperparameters
  void setScalarNoise(double variance);
  void setVectorNoise(const Eigen::VectorXd& variance);
  void setHeteroskedasticNoise(const Eigen::MatrixXd& variance);

  // Forward declarations
  struct KernelFunction;
  struct NoiseFunction;

 private:
  void cacheRegressionModels();

  // Terms for a 1D regression model
  struct RegressionModel {
    explicit RegressionModel(const Eigen::MatrixXd& L,
                             const Eigen::MatrixXd& Linv,
                             const Eigen::MatrixXd& Kinv,
                             const Eigen::VectorXd& alpha,
                             const std::vector<Eigen::MatrixXd>& grad);
    virtual ~RegressionModel() = default;
    Eigen::MatrixXd cached_L;
    Eigen::MatrixXd cached_L_inv;
    Eigen::MatrixXd cached_K_inv;
    Eigen::VectorXd cached_alpha;
    std::vector<Eigen::MatrixXd> cached_grad;
  };

  Eigen::MatrixXd m_input_samples;
  Eigen::MatrixXd m_output_samples;
  Gaussian m_belief;
  std::shared_ptr<KernelFunction> m_kernel{nullptr};
  std::shared_ptr<NoiseFunction> m_noise{nullptr};
  std::vector<RegressionModel> m_models;
};

}  // namespace sia
