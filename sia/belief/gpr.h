/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"
#include "sia/optimizers/gd.h"

#include <Eigen/Dense>
#include <memory>

namespace sia {

/// Kernel function base class.  Kernels are symmetric and positive definite.
/// The gradient function returns the Jacobian w.r.t. to the kernel
/// hyperparameters.  For details on how to choose kernels, see The Kernel
/// Cookbook https://www.cs.toronto.edu/~duvenaud/cookbook/
struct Kernel {
  Kernel() = default;
  virtual ~Kernel() = default;
  virtual double eval(const Eigen::VectorXd& x,
                      std::size_t output_index) const = 0;
  virtual double eval(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& y,
                      std::size_t output_index) const = 0;
  virtual Eigen::VectorXd grad(const Eigen::VectorXd& x,
                               const Eigen::VectorXd& y,
                               std::size_t output_index) const = 0;

  virtual Eigen::VectorXd hyperparameters() const = 0;
  virtual void setHyperparameters(const Eigen::VectorXd& p) = 0;
  virtual std::size_t numHyperparameters() const = 0;
};

/// Composite kernel function, forms the basis for combining kernels.
/// See: https://www.cs.toronto.edu/~duvenaud/cookbook/
class CompositeKernel : public Kernel {
 public:
  static CompositeKernel multiply(Kernel& a, Kernel& b);
  static CompositeKernel add(Kernel& a, Kernel& b);
  virtual ~CompositeKernel() = default;
  double eval(const Eigen::VectorXd& x,
              std::size_t output_index) const override;
  double eval(const Eigen::VectorXd& x,
              const Eigen::VectorXd& y,
              std::size_t output_index) const override;
  Eigen::VectorXd grad(const Eigen::VectorXd& x,
                       const Eigen::VectorXd& y,
                       std::size_t output_index) const override;
  Eigen::VectorXd hyperparameters() const override;
  void setHyperparameters(const Eigen::VectorXd& p) override;
  std::size_t numHyperparameters() const override;

 private:
  enum Operation {
    MULTIPLY,
    ADD,
  };
  CompositeKernel(Kernel& a, Kernel& b, Operation operation);
  Kernel& m_kernel_a;
  Kernel& m_kernel_b;
  Operation m_operation;
};

/// Multiply two kernels
CompositeKernel operator*(Kernel& a, Kernel& b);

/// Add two kernels
CompositeKernel operator+(Kernel& a, Kernel& b);

/// Squared exponential (SE, RBF) kernel
///
/// Hyperparameters are 2 scalars: (length, signal_var)
/// - length controls the kernel basis blending
/// - signal_var controls the marginal variance of the Gaussian prior
class SEKernel : public Kernel {
 public:
  /// Default hyperparameter values
  static constexpr double DEFAULT_LENGTH = 1.0;
  static constexpr double DEFAULT_SIGNAL_VAR = 1.0;

  explicit SEKernel(double length = DEFAULT_LENGTH,
                    double signal_var = DEFAULT_SIGNAL_VAR);
  explicit SEKernel(const Eigen::Vector2d& hyperparameters);
  virtual ~SEKernel() = default;
  double eval(const Eigen::VectorXd& x,
              std::size_t output_index) const override;
  double eval(const Eigen::VectorXd& x,
              const Eigen::VectorXd& y,
              std::size_t output_index) const override;
  Eigen::VectorXd grad(const Eigen::VectorXd& x,
                       const Eigen::VectorXd& y,
                       std::size_t output_index) const override;
  Eigen::VectorXd hyperparameters() const override;
  void setHyperparameters(const Eigen::VectorXd& p) override;
  std::size_t numHyperparameters() const override;

 private:
  double m_length;
  double m_signal_var;
};

/// Constant white noise kernel
///
/// Hyperparameter noise_var controls the noise variance of the Gaussian
/// posterior
class NoiseKernel : public Kernel {
 public:
  /// Default hyperparameter value
  static constexpr double DEFAULT_NOISE_VAR = 0.1;

  explicit NoiseKernel(double noise_var = DEFAULT_NOISE_VAR);
  virtual ~NoiseKernel() = default;
  double eval(const Eigen::VectorXd& x,
              std::size_t output_index) const override;
  double eval(const Eigen::VectorXd& x,
              const Eigen::VectorXd& y,
              std::size_t output_index) const override;
  Eigen::VectorXd grad(const Eigen::VectorXd& x,
                       const Eigen::VectorXd& y,
                       std::size_t output_index) const override;
  Eigen::VectorXd hyperparameters() const override;
  void setHyperparameters(const Eigen::VectorXd& p) override;
  std::size_t numHyperparameters() const override;

 private:
  double m_noise_var;
};

/// Heteroscedastic white noise kernel, i.e. var = f(x)
///
/// There are no hyperparameters, a known function to describe the variance is
/// provided on construction.
class VariableNoiseKernel : public Kernel {
 public:
  using VarianceFunction =
      std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
  explicit VariableNoiseKernel(VarianceFunction var_function);
  virtual ~VariableNoiseKernel() = default;
  double eval(const Eigen::VectorXd& x,
              std::size_t output_index) const override;
  double eval(const Eigen::VectorXd& x,
              const Eigen::VectorXd& y,
              std::size_t output_index) const override;
  Eigen::VectorXd grad(const Eigen::VectorXd& x,
                       const Eigen::VectorXd& y,
                       std::size_t output_index) const override;
  Eigen::VectorXd hyperparameters() const override;
  void setHyperparameters(const Eigen::VectorXd& p) override;
  std::size_t numHyperparameters() const override;

 private:
  VarianceFunction m_var_function;
};

// ----------------------------------------------------------------------------

/// Gaussian Process Regression performs Gaussian regression to predict
/// $p(y|x)$ using a GP kernel prior.  This class implements algorithm 2.1
/// from Rasmussen and Williams, and assumes a zero mean prior.
///
/// References:
/// [1]: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
class GPR : public Inference {
 public:
  /// Regularization applied to covariance matrices during prediction.
  static constexpr double DEFAULT_REGULARIZATION = 1e-10;

  /// Default hyperparameter max and min values
  static constexpr double DEFAULT_HP_MIN = 1e-3;
  static constexpr double DEFAULT_HP_MAX = 1e3;

  /// Initialize GPR from training data or vector sizes.  Each column of input
  /// and output training samples is a sample.
  explicit GPR(const Eigen::MatrixXd& input_samples,
               const Eigen::MatrixXd& output_samples,
               Kernel& kernel,
               double regularization = DEFAULT_REGULARIZATION);
  explicit GPR(std::size_t input_dim,
               std::size_t output_dim,
               Kernel& kernel,
               double regularization = DEFAULT_REGULARIZATION);
  virtual ~GPR() = default;

  /// Recompute with new data but same hyperparameters
  void setData(const Eigen::MatrixXd& input_samples,
               const Eigen::MatrixXd& output_samples);

  /// Performs the inference $p(y|x)$
  const Gaussian& predict(const Eigen::VectorXd& x) override;

  /// Computes the negative log marginal likelihood loss and gradients
  double negLogMarginalLik() const;
  Eigen::VectorXd negLogMarginalLikGrad() const;

  /// Train the hyperparameters.  A list of trainable hyperparameter indices can
  /// be provided.  If the list is empty (default), all hyperparameters are
  /// optimized.
  void train(const std::vector<std::size_t>& hp_indices = {},
             double hp_min = DEFAULT_HP_MIN,
             double hp_max = DEFAULT_HP_MAX,
             const GD::Options& options = GD::Options());

  /// Dimensions
  std::size_t inputDim() const override;
  std::size_t outputDim() const override;
  std::size_t numSamples() const;

  /// Access the kernel
  const Kernel& kernel() const;

  /// Access the hyperparameters
  Eigen::VectorXd hyperparameters() const;
  void setHyperparameters(const Eigen::VectorXd& hyperparameters);
  std::size_t numHyperparameters() const;

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

  std::size_t m_input_dim;
  std::size_t m_output_dim;
  Eigen::MatrixXd m_input_samples;
  Eigen::MatrixXd m_output_samples;
  Gaussian m_belief;
  Kernel& m_kernel;
  double m_regularization;
  std::vector<RegressionModel> m_models;
};

}  // namespace sia
