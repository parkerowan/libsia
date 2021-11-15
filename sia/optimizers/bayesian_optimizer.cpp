/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/bayesian_optimizer.h"
#include "sia/common/exception.h"
#include "sia/optimizers/gradient_descent.h"

namespace sia {

BayesianOptimizer::BayesianOptimizer(const Eigen::VectorXd& lower,
                                     const Eigen::VectorXd& upper)
    : m_sampler(lower, upper) {}

Eigen::VectorXd BayesianOptimizer::selectNextSample() {
  // If there is no model yet, just sample uniformly
  if (m_gpr == nullptr) {
    return m_sampler.sample();
  }

  // Optimize the acquisition model
  // Hard-coded upper-confidence bound (UCB) with 0.1\sigma
  GradientDescent optm(m_sampler.lower(), m_sampler.upper(), m_tol, m_beta);
  auto acquisition_fcn = [=](const Eigen::VectorXd& x) {
    double beta = 0.1;
    Gaussian g = m_gpr->predict(x);
    return -(g.mean()(0) + beta * sqrt(g.covariance()(0, 0)));
  };
  return optm.minimize(acquisition_fcn, m_sampler.samples(m_nstarts));
}

void BayesianOptimizer::addDataPoint(const Eigen::VectorXd& x, double y) {
  m_input_data.emplace_back(x);
  m_output_data.emplace_back(y);
}

void BayesianOptimizer::updateModel() {
  // TODO:
  // - abstract the surrogate objective function model (support for regression
  // vs binary classification)
  // - add flag to optimize the hyperparameters
  // - add efficient incremental update to the GPR class
  std::size_t ndim = m_input_data[0].size();
  std::size_t nsamples = m_input_data.size();
  Eigen::MatrixXd X = Eigen::MatrixXd(ndim, nsamples);
  for (std::size_t i = 0; i < nsamples; ++i) {
    X.col(i) = m_input_data[i];
  }
  Eigen::MatrixXd y = Eigen::Map<Eigen::MatrixXd>(m_output_data.data(), 1,
                                                  m_output_data.size());
  m_gpr = new GPR(X, y, 1e-4, 1, 1);
}

Eigen::VectorXd BayesianOptimizer::getSolution() {
  if (m_gpr == nullptr) {
    return m_sampler.sample();
  }

  std::size_t ntest = 100;
  std::vector<Eigen::VectorXd> xtest = m_sampler.samples(ntest);
  std::vector<double> ytest;
  for (std::size_t i = 0; i < ntest; ++i) {
    // Compute the acquisition model for a sample
    Gaussian g = m_gpr->predict(xtest[i]);
    double mu = g.mean()(0);
    ytest.emplace_back(mu);
  }

  // Select the max of the function samples
  Eigen::VectorXd Y = Eigen::Map<Eigen::VectorXd>(ytest.data(), ytest.size());
  int imax;
  Y.maxCoeff(&imax);
  return xtest[imax];
}

GPR& BayesianOptimizer::gpr() {
  return *m_gpr;
}

}  // namespace sia
