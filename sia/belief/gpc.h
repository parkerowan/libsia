/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/dirichlet.h"
#include "sia/belief/distribution.h"
#include "sia/belief/gpr.h"

#include <Eigen/Dense>
#include <memory>

namespace sia {

/// Gaussian Process Classification performs classification using GPR to predict
/// $p(c|x)$, where c is a discrete index representing a category.  This class
/// implements GPC with Dirichlet distributions, where the underlying GPR
/// predicts approximate concentration parameters of the Dirichlet distribution.
/// This yields much faster inference than the more widely used exact Laplacian
/// GPC approach developed in Rasmussen and Williams [2]
///
/// See GPR for Kernel Type hyperparameter options.
///
/// References:
/// [1]: D. Milios et. al., "Dirichlet-based Gaussian Processes for Large-Scale
/// Calibrated Gaussian Process Calibration," NeurIPS, 2018.
/// [2]: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
class GPC : public Inference {
 public:
  /// Default parameter for the concentration offset
  static constexpr double DEFAULT_CONCENTRATION = 1e-2;

  /// Initialize GPC from training data.  Each column of input training data is
  /// a sample.  The output training data is a vector of class indices.  The
  /// dimension of the underlying classifier will be determined by the max value
  /// found in output_samples, with an assumption that class indices are
  /// 0-indexed.  1 > alpha > 0 controls the minimum value of the Dirichlet
  /// concentrations.  The kernel type determines the kernel basis.
  explicit GPC(const Eigen::MatrixXd& input_samples,
               const Eigen::VectorXi& output_samples,
               Kernel& kernel,
               double alpha = DEFAULT_CONCENTRATION,
               double regularization = GPR::DEFAULT_REGULARIZATION);
  explicit GPC(std::size_t input_dim,
               std::size_t output_dim,
               Kernel& kernel,
               double alpha = DEFAULT_CONCENTRATION,
               double regularization = GPR::DEFAULT_REGULARIZATION);
  virtual ~GPC() = default;

  /// Recompute with new data but same hyperparameters
  void setData(const Eigen::MatrixXd& input_samples,
               const Eigen::VectorXi& output_samples);

  /// Performs the inference $p(y|x)$
  const Dirichlet& predict(const Eigen::VectorXd& x) override;

  /// Computes the negative log marginal likelihood loss and gradients
  double negLogMarginalLik() const;
  Eigen::VectorXd negLogMarginalLikGrad() const;

  /// Train the hyperparameters.  A list of trainable hyperparameter indices can
  /// be provided.  If the list is empty (default), all hyperparameters are
  /// optimized.
  void train(const std::vector<std::size_t>& hp_indices = {},
             double hp_min = GPR::DEFAULT_HP_MIN,
             double hp_max = GPR::DEFAULT_HP_MAX,
             const GD::Options& options = GD::Options());

  /// Dimensions
  std::size_t inputDim() const override;
  std::size_t outputDim() const override;
  std::size_t numSamples() const;

  /// Access the kernel
  const Kernel& kernel() const;

  /// Access the hyperparameters
  Eigen::VectorXd hyperparameters() const;
  void setHyperparameters(const Eigen::VectorXd& p);
  std::size_t numHyperparameters() const;

  /// Set the alpha value
  void setAlpha(double alpha);
  double alpha() const;

 private:
  void cacheRegressionModel();
  static std::size_t getNumClasses(const Eigen::VectorXi& x);
  Eigen::VectorXd noiseFunction(const Eigen::VectorXd& x);

  Eigen::MatrixXd m_input_samples;
  Eigen::VectorXi m_output_samples;
  Eigen::MatrixXd m_output_noise;
  Dirichlet m_belief;
  Kernel& m_kernel;
  VariableNoiseKernel m_noise_kernel;
  CompositeKernel m_composite_kernel;
  GPR m_gpr;
  double m_alpha;
};

}  // namespace sia
