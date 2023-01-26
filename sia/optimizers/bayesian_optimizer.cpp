/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/bayesian_optimizer.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"

#include <cmath>

namespace sia {

// The surrogate model provides a statistical approximation of the objective
// function and a corresponding acquisition function.
// - Objective: Function R^n -> p(y) that approximates the true objective
// - Acqusition: Utility function R^n -> R for selecting the next data point
// https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf

BayesianOptimizer::BayesianOptimizer(
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,
    Kernel& kernel,
    std::size_t cond_inputs_dim,
    BayesianOptimizer::AcquisitionType acquisition,
    double beta,
    const GradientDescent::Options& options)
    : m_sampler(lower, upper),
      m_optimizer(lower, upper, options),
      m_cond_inputs_dim(cond_inputs_dim),
      m_acquisition_type(acquisition),
      m_gpr(lower.size(), 1, kernel),
      m_beta(beta) {}

Eigen::VectorXd BayesianOptimizer::selectNextSample(const Eigen::VectorXd& u) {
  // If there is no model yet, just sample uniformly
  if (m_gpr.numSamples() == 0) {
    return m_sampler.sample();
  }

  // Optimize the acquisition model
  double target = objective(getSolution(u), u).mean()(0);
  auto f = [=](const Eigen::VectorXd& x) {
    return -acquisition(x, target, m_acquisition_type, u);
  };
  return m_optimizer.minimize(f);
}

void BayesianOptimizer::addDataPoint(const Eigen::VectorXd& x,
                                     double y,
                                     const Eigen::VectorXd& u) {
  SIA_EXCEPTION(
      x.size() == int(m_sampler.dimension()),
      "BayesianOptimizer expects input data to have same dimension as bounds");
  SIA_EXCEPTION(u.size() == int(m_cond_inputs_dim),
                "BayesianOptimizer expects u to have size cond_inputs_dim");
  m_input_data.emplace_back(x);
  m_output_data.emplace_back(y);
  m_cond_input_data.emplace_back(u);
}

void BayesianOptimizer::updateModel(bool train) {
  SIA_EXCEPTION(m_input_data.size() > 0,
                "BayesianOptimizer expects data points to be added before "
                "calling updateModel()");
  assert(m_input_data.size() == m_output_data.size());
  assert(m_input_data.size() == m_cond_input_data.size());

  // TODO: Add efficient incremental update to the GPR class
  std::size_t n_input_dim = m_sampler.dimension();
  std::size_t n_samples = m_input_data.size();
  Eigen::MatrixXd X =
      Eigen::MatrixXd(n_input_dim + m_cond_inputs_dim, n_samples);
  for (std::size_t i = 0; i < n_samples; ++i) {
    Eigen::VectorXd xu = Eigen::VectorXd(n_input_dim + m_cond_inputs_dim);
    xu.head(n_input_dim) = m_input_data[i];
    xu.tail(m_cond_inputs_dim) = m_cond_input_data[i];
    X.col(i) = xu;
  }
  Eigen::MatrixXd y = Eigen::Map<Eigen::MatrixXd>(m_output_data.data(), 1,
                                                  m_output_data.size());

  // Update the GPR data
  m_gpr.setData(X, y);

  // Optionally train the GPR hyperparameters
  if (train) {
    m_gpr.train();
  }

  // Notify getSolution() that it needs to optimize the new model
  m_dirty_solution = true;
}

Eigen::VectorXd BayesianOptimizer::getSolution(const Eigen::VectorXd& u) {
  if (m_gpr.numSamples() == 0) {
    return m_sampler.sample();
  }

  if (!m_dirty_solution) {
    return m_cached_solution;
  }

  // Optimize the objective function model
  auto f = [=](const Eigen::VectorXd& x) { return -objective(x, u).mean()(0); };
  m_cached_solution = m_optimizer.minimize(f);
  m_dirty_solution = false;
  return m_cached_solution;
}

GradientDescent& BayesianOptimizer::optimizer() {
  return m_optimizer;
}

const Gaussian& BayesianOptimizer::objective(const Eigen::VectorXd& x,
                                             const Eigen::VectorXd& u) {
  std::size_t n_input_dim = m_sampler.dimension();
  SIA_EXCEPTION(
      x.size() == int(n_input_dim),
      "BayesianOptimizer expects input data to have same dimension as bounds");
  SIA_EXCEPTION(u.size() == int(m_cond_inputs_dim),
                "BayesianOptimizer expects u to have size cond_inputs_dim");
  Eigen::VectorXd xu = Eigen::VectorXd(n_input_dim + m_cond_inputs_dim);
  xu.head(n_input_dim) = x;
  xu.tail(m_cond_inputs_dim) = u;
  return m_gpr.predict(xu);
}

double BayesianOptimizer::acquisition(const Eigen::VectorXd& x,
                                      const Eigen::VectorXd& u) {
  double target = objective(getSolution(u), u).mean()(0);
  return acquisition(x, target, m_acquisition_type, u);
}

double BayesianOptimizer::acquisition(const Eigen::VectorXd& x,
                                      double target,
                                      BayesianOptimizer::AcquisitionType type,
                                      const Eigen::VectorXd& u) {
  // Evaluate the acquisition function
  const auto& p = objective(x, u);
  double mu = p.mean()(0);
  double std = sqrt(p.covariance()(0, 0));

  switch (type) {
    case BayesianOptimizer::AcquisitionType::PROBABILITY_IMPROVEMENT:
      return Gaussian::cdf((mu - target) / std);
    case BayesianOptimizer::AcquisitionType::EXPECTED_IMPROVEMENT:
      return (mu - target) * Gaussian::cdf((mu - target) / std) +
             std * Gaussian::pdf((mu - target) / std);
    case BayesianOptimizer::AcquisitionType::UPPER_CONFIDENCE_BOUND:
      return mu + m_beta * std;
  }

  SIA_ERROR("GPRSurrogateModel received unsupported AcquisitionType");
  return 0;
}

}  // namespace sia
