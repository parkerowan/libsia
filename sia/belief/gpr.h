/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"

#include <Eigen/Dense>

namespace sia {

/// Gaussian Process Regression performs Gaussian regression to predict $p(y|x)$
/// using a GP kernel prior.  This class implements algorithm 2.1 from Rasmussen
/// and Williams, and assumes a zero mean prior.
///
/// References:
/// [1]: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
class GPR : public Regression {
 public:
  /// Supported basis/kernel functions for covariance
  enum CovFunction { SQUARED_EXPONENTIAL };

  /// Initialize GPR from training data.  Each column of input and output
  /// training samples is a sample.  varf controls the uncertainty of the
  /// Gaussian prior (i.e. outside of the training domain); varn controls the
  /// measurement likelihood uncertainty; and length controls the kernel basis
  /// blending.  The covariance type determines the kernel basis.
  explicit GPR(const Eigen::MatrixXd& input_samples,
               const Eigen::MatrixXd& output_samples,
               double varf,
               double varn,
               double length,
               CovFunction type = SQUARED_EXPONENTIAL);
  virtual ~GPR();

  /// Performs the regression $p(y|x)$.
  const Gaussian& predict(const Eigen::VectorXd& x) override;
  std::size_t inputDimension() const override;
  std::size_t outputDimension() const override;

  std::size_t numSamples() const;

 private:
  void cacheRegressionModels();

  // Forward declaration
  class Kernel;

  /// Terms for a 1D regression model
  struct RegressionModel {
    explicit RegressionModel(Kernel* kernel,
                             const Eigen::MatrixXd& X,
                             const Eigen::VectorXd& y,
                             double varf,
                             double varn,
                             double length);
    Eigen::MatrixXd m_cached_L_inv;
    Eigen::VectorXd m_cached_alpha;
  };

  Gaussian m_belief;
  Kernel* m_kernel{nullptr};
  std::vector<RegressionModel> m_models;
  Eigen::MatrixXd m_input_samples;
  Eigen::MatrixXd m_output_samples;
  double m_varf;
  double m_varn;
  double m_length;
};

}  // namespace sia
