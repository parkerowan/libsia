/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
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
         double varf,
         double length)
    : m_belief(getNumClasses(output_samples)),
      m_input_samples(input_samples),
      m_output_samples(output_samples),
      m_alpha(alpha),
      m_varf(varf),
      m_length(length) {
  SIA_EXCEPTION(input_samples.cols() == output_samples.size(),
                "Inconsistent number of input cols to output length");

  cacheRegressionModel();
}

GPC::GPC(const Eigen::MatrixXd& input_samples,
         const std::vector<int>& output_samples,
         double alpha,
         double varf,
         double length)
    : GPC(input_samples,
          Eigen::Map<const Eigen::VectorXi>(output_samples.data(),
                                            output_samples.size()),
          alpha,
          varf,
          length) {}

GPC::~GPC() {
  delete m_gpr;
}

const Dirichlet& GPC::predict(const Eigen::VectorXd& x) {
  SIA_EXCEPTION(m_gpr != nullptr, "Gaussian Process is uninitialized");

  // Section 4 in: https://arxiv.org/pdf/1805.10915.pdf
  // Infer the log concentration for each output class, eqn. 6
  Gaussian py = m_gpr->predict(x);
  Eigen::VectorXd alpha = py.mean().array().exp();

  // Generate the expectation via softmax, eqn. 7
  m_belief.setAlpha(alpha);
  return m_belief;
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

double GPC::negLogLikLoss() {
  double neg_log_lik = 0;
  std::size_t c = outputDimension();
  const Eigen::MatrixXd Y = getOneHot(m_output_samples, c);
  for (std::size_t i = 0; i < numSamples(); ++i) {
    const auto& x = m_input_samples.col(i);
    const auto& y = Y.col(i);
    neg_log_lik -= predict(x).logProb(y);
  }
  return neg_log_lik;
}

Eigen::VectorXd GPC::getHyperparameters() const {
  return Eigen::Vector3d(m_alpha, m_varf, m_length);
}

void GPC::setHyperparameters(const Eigen::VectorXd& p) {
  SIA_EXCEPTION(p.size() == 3, "Hyperparameter vector dim expected to be 3");

  m_alpha = p(0);
  m_varf = p(1);
  m_length = p(2);

  cacheRegressionModel();
}

// Returns a matrix of one-hot classifications [num classes x sum samples]
// TODO: move this to a discrete Categorical representation
Eigen::MatrixXd GPC::getOneHot(const Eigen::VectorXi& x,
                               std::size_t num_classes) {
  int max_x = x.maxCoeff();
  SIA_EXCEPTION(max_x < (int)num_classes,
                "GPC expects the indices in x to be < number of classes");

  int min_x = x.minCoeff();
  SIA_EXCEPTION(min_x >= 0, "GPC expects the indices in x to be >= 0");

  std::size_t n = x.size();
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(num_classes, n);
  for (std::size_t i = 0; i < n; ++i) {
    assert(x(i) < (int)num_classes);
    assert(x(i) >= 0);
    Y(x(i), i) = 1.0;
  }
  return Y;
}

void GPC::cacheRegressionModel() {
  // Section 4 in: https://arxiv.org/pdf/1805.10915.pdf
  std::size_t c = outputDimension();
  const Eigen::MatrixXd A = getOneHot(m_output_samples, c).array() + m_alpha;

  // Transform concentrations to lognormal distribution, eqn. 5
  const Eigen::MatrixXd s2i = (1 / A.array() + 1).array().log();
  const Eigen::MatrixXd yi = A.array().log() - s2i.array() / 2;

  // Create multivariate GPR for each log concentration, eqn. 6
  m_gpr = new GPR(m_input_samples, yi, s2i, m_varf, m_length);
}

std::size_t GPC::getNumClasses(const Eigen::VectorXi& x) {
  return x.maxCoeff() + 1;
}

}  // namespace sia
