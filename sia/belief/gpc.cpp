/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gpc.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

#include <limits>

namespace sia {

GPC::GPC(const Eigen::MatrixXd& input_samples,
         const Eigen::VectorXi& output_samples,
         Kernel& kernel,
         double alpha,
         double regularization)
    : GPC(input_samples.rows(),
          getNumClasses(output_samples),
          kernel,
          alpha,
          regularization) {
  setData(input_samples, output_samples);
}

// Bind the noise function to a callback for the VariableNoiseKernel
using namespace std::placeholders;
GPC::GPC(std::size_t input_dim,
         std::size_t output_dim,
         Kernel& kernel,
         double alpha,
         double regularization)
    : m_belief(output_dim),
      m_kernel(kernel),
      m_noise_kernel(std::bind(&GPC::noiseFunction, this, _1)),
      m_composite_kernel(CompositeKernel::add(m_kernel, m_noise_kernel)),
      m_gpr(input_dim, output_dim, m_composite_kernel, regularization),
      m_alpha(alpha) {}

void GPC::setData(const Eigen::MatrixXd& input_samples,
                  const Eigen::VectorXi& output_samples) {
  SIA_THROW_IF_NOT(input_samples.cols() == output_samples.size(),
                   "Inconsistent number of input cols to output length");
  m_input_samples = input_samples;
  m_output_samples = output_samples;
  cacheRegressionModel();
}

const Dirichlet& GPC::predict(const Eigen::VectorXd& x) {
  // Section 4 in: https://arxiv.org/pdf/1805.10915.pdf
  // Infer the log concentration for each output class, eqn. 6
  Gaussian py = m_gpr.predict(x);
  Eigen::VectorXd alpha = py.mean().array().exp();

  // Generate the expectation via softmax, eqn. 7
  m_belief.setAlpha(alpha);
  return m_belief;
}

double GPC::negLogMarginalLik() const {
  return m_gpr.negLogMarginalLik();
}

Eigen::VectorXd GPC::negLogMarginalLikGrad() const {
  return m_gpr.negLogMarginalLikGrad();
}

void GPC::train(const std::vector<std::size_t>& hp_indices,
                double hp_min,
                double hp_max,
                const GradientDescent::Options& options) {
  return m_gpr.train(hp_indices, hp_min, hp_max, options);
}

std::size_t GPC::inputDimension() const {
  return m_gpr.inputDimension();
}

std::size_t GPC::outputDimension() const {
  return m_gpr.outputDimension();
}

std::size_t GPC::numSamples() const {
  return m_gpr.numSamples();
}

const Kernel& GPC::kernel() const {
  return m_kernel;
}

Eigen::VectorXd GPC::hyperparameters() const {
  return m_gpr.hyperparameters();
}

void GPC::setHyperparameters(const Eigen::VectorXd& p) {
  return m_gpr.setHyperparameters(p);
}

std::size_t GPC::numHyperparameters() const {
  return m_gpr.numHyperparameters();
}

void GPC::setAlpha(double alpha) {
  // TODO: validate alpha (>0?)
  m_alpha = alpha;
  cacheRegressionModel();
}

double GPC::alpha() const {
  return m_alpha;
}

void GPC::cacheRegressionModel() {
  // Section 4 in: https://arxiv.org/pdf/1805.10915.pdf
  Categorical c(outputDimension());
  const Eigen::MatrixXd A = c.oneHot(m_output_samples).array() + m_alpha;

  // Transform concentrations to lognormal distribution, eqn. 5
  // Noise is passed to noiseFunction that is bound to VariableNoiseKernel
  m_output_noise = (1 / A.array() + 1).array().log();
  const Eigen::MatrixXd yi = A.array().log() - m_output_noise.array() / 2;

  // Create multivariate GPR for each log concentration, eqn. 6
  m_gpr.setData(m_input_samples, yi);
}

std::size_t GPC::getNumClasses(const Eigen::VectorXi& x) {
  return x.maxCoeff() + 1;
}

Eigen::VectorXd GPC::noiseFunction(const Eigen::VectorXd& x) {
  // Nearest neighbor lookup - this is horribly inefficient
  Eigen::VectorXd dist = Eigen::VectorXd::Zero(m_input_samples.cols());
  for (int i = 0; i < dist.size(); ++i) {
    dist(i) = (x - m_input_samples.col(i)).norm();
  }
  int istar = 0;
  dist.minCoeff(&istar);
  return m_output_noise.col(istar);
}

}  // namespace sia
