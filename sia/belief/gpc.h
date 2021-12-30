/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
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
/// References:
/// [1]: D. Milios et. al., "Dirichlet-based Gaussian Processes for Large-Scale
/// Calibrated Gaussian Process Calibration," NeurIPS, 2018.
/// [2]: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
class GPC : public Inference {
 public:
  /// Initialize GPC from training data.  Each column of input
  /// training data is a sample.  The output training data is a vector of class
  /// indices.  The dimension of the underlying classifier will be determined by
  /// the max value found in output_samples, with an assumption that class
  /// indices are 0-indexed.  1 > alpha > 0 controls the minimum value of the
  /// Dirichlet concentrations.  The kernel type determines the kernel basis.
  explicit GPC(const Eigen::MatrixXd& input_samples,
               const Eigen::VectorXi& output_samples,
               double alpha = 0.01,
               GPR::KernelType kernel_type = GPR::SE_KERNEL);
  explicit GPC(const Eigen::MatrixXd& input_samples,
               const std::vector<int>& output_samples,
               double alpha = 0.01,
               GPR::KernelType kernel_type = GPR::SE_KERNEL);
  virtual ~GPC() = default;

  /// Performs the inference $p(y|x)$
  const Dirichlet& predict(const Eigen::VectorXd& x) override;

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
  void setHyperparameters(const Eigen::VectorXd& p);
  std::size_t numHyperparameters() const;

  /// Set the alpha value
  void setAlpha(double alpha);

 private:
  void cacheRegressionModel();
  static std::size_t getNumClasses(const Eigen::VectorXi& x);

  Dirichlet m_belief;
  std::shared_ptr<GPR> m_gpr;
  Eigen::MatrixXd m_input_samples;
  Eigen::VectorXi m_output_samples;
  double m_alpha;
  GPR::KernelType m_kernel_type;
};

}  // namespace sia
