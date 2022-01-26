/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gpc.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

#include <glog/logging.h>
#include <limits>

namespace sia {

GPC::GPC(const Eigen::MatrixXd& input_samples,
         const Eigen::VectorXi& output_samples,
         double alpha,
         GPR::KernelType kernel_type)
    : m_belief(getNumClasses(output_samples)),
      m_input_samples(input_samples),
      m_output_samples(output_samples),
      m_alpha(alpha),
      m_kernel_type(kernel_type) {
  SIA_EXCEPTION(input_samples.cols() == output_samples.size(),
                "Inconsistent number of input cols to output length");

  cacheRegressionModel();
}

GPC::GPC(const Eigen::MatrixXd& input_samples,
         const Eigen::VectorXi& output_samples,
         const Eigen::VectorXd& hyperparameters,
         double alpha,
         GPR::KernelType kernel_type)
    : m_belief(getNumClasses(output_samples)),
      m_input_samples(input_samples),
      m_output_samples(output_samples),
      m_alpha(alpha),
      m_kernel_type(kernel_type) {
  SIA_EXCEPTION(input_samples.cols() == output_samples.size(),
                "Inconsistent number of input cols to output length");

  cacheRegressionModel();
  m_gpr->setHyperparameters(hyperparameters);
}

const Dirichlet& GPC::predict(const Eigen::VectorXd& x) {
  assert(m_gpr != nullptr);

  // Section 4 in: https://arxiv.org/pdf/1805.10915.pdf
  // Infer the log concentration for each output class, eqn. 6
  Gaussian py = m_gpr->predict(x);
  Eigen::VectorXd alpha = py.mean().array().exp();

  // Generate the expectation via softmax, eqn. 7
  m_belief.setAlpha(alpha);
  return m_belief;
}

double GPC::negLogMarginalLik() const {
  assert(m_gpr != nullptr);
  return m_gpr->negLogMarginalLik();
}

Eigen::VectorXd GPC::negLogMarginalLikGrad() const {
  assert(m_gpr != nullptr);
  return m_gpr->negLogMarginalLikGrad();
}

double GPC::negLogLik(const Eigen::MatrixXd& X, const Eigen::VectorXi& Y) {
  SIA_EXCEPTION(X.cols() == Y.size(),
                "Test data X, Y expected to have sample number of cols");
  SIA_EXCEPTION(std::size_t(X.rows()) == inputDimension(),
                "Test data X rows expected to be input dimension");
  double neg_log_lik = 0;
  for (int i = 0; i < X.cols(); ++i) {
    const Categorical c = predict(X.col(i)).categorical();
    neg_log_lik -= c.logProb(c.oneHot(Y(i)));
  }
  return neg_log_lik;
}

void GPC::train() {
  assert(m_gpr != nullptr);
  return m_gpr->train();
}

std::size_t GPC::inputDimension() const {
  return m_input_samples.rows();
}

std::size_t GPC::outputDimension() const {
  return getNumClasses(m_output_samples);
}

std::size_t GPC::numSamples() const {
  return m_input_samples.cols();
}

Eigen::VectorXd GPC::hyperparameters() const {
  assert(m_gpr != nullptr);
  return m_gpr->hyperparameters();
}

void GPC::setHyperparameters(const Eigen::VectorXd& p) {
  assert(m_gpr != nullptr);
  return m_gpr->setHyperparameters(p);
}

std::size_t GPC::numHyperparameters() const {
  assert(m_gpr != nullptr);
  return m_gpr->numHyperparameters();
}

void GPC::setAlpha(double alpha) {
  m_alpha = alpha;
  cacheRegressionModel();
}

void GPC::cacheRegressionModel() {
  // Section 4 in: https://arxiv.org/pdf/1805.10915.pdf
  Categorical c(outputDimension());
  const Eigen::MatrixXd A = c.oneHot(m_output_samples).array() + m_alpha;

  // Transform concentrations to lognormal distribution, eqn. 5
  const Eigen::MatrixXd s2i = (1 / A.array() + 1).array().log();
  const Eigen::MatrixXd yi = A.array().log() - s2i.array() / 2;

  // Create multivariate GPR for each log concentration, eqn. 6
  m_gpr = std::make_shared<GPR>(m_input_samples, yi, m_kernel_type,
                                GPR::HETEROSKEDASTIC_NOISE);
  m_gpr->setHeteroskedasticNoise(s2i);
}

std::size_t GPC::getNumClasses(const Eigen::VectorXi& x) {
  return x.maxCoeff() + 1;
}

}  // namespace sia
