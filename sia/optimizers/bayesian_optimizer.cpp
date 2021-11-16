/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/bayesian_optimizer.h"
#include "sia/common/exception.h"

#include <cmath>

namespace sia {

static double pdf(double x) {
  return 1 / sqrt(2 * M_PI) * exp(-pow(x, 2) / 2);
}

static double cdf(double x) {
  return (1 + erf(x / sqrt(2))) / 2;
}

BayesianOptimizer::BayesianOptimizer(const Eigen::VectorXd& lower,
                                     const Eigen::VectorXd& upper,
                                     ObjectiveType objective,
                                     AcquistionType acquisition,
                                     std::size_t nstarts)
    : m_sampler(lower, upper),
      m_optimizer(lower, upper),
      m_acquisition_type(acquisition),
      m_nstarts(nstarts) {
  switch (objective) {
    case GPR_OBJECTIVE:
      m_surrogate = new GPRSurrogateModel();
      break;
  }
  SIA_EXCEPTION(m_surrogate != nullptr,
                "BayesianOptimizer received unsupported ObjectiveType");
}

Eigen::VectorXd BayesianOptimizer::selectNextSample() {
  // If there is no model yet, just sample uniformly
  if (!m_surrogate->initialized()) {
    return m_sampler.sample();
  }

  // Optimize the acquisition model
  double target = m_surrogate->objective(getSolution()).mean()(0);
  auto f = [=](const Eigen::VectorXd& x) {
    return -m_surrogate->acquisition(x, target, m_acquisition_type);
  };
  return m_optimizer.minimize(f, m_sampler.samples(m_nstarts));
}

void BayesianOptimizer::addDataPoint(const Eigen::VectorXd& x, double y) {
  m_surrogate->addDataPoint(x, y);
}

void BayesianOptimizer::updateModel() {
  m_surrogate->updateModel();
}

Eigen::VectorXd BayesianOptimizer::getSolution() {
  if (!m_surrogate->initialized()) {
    return m_sampler.sample();
  }

  // Optimize the objective function model
  auto f = [=](const Eigen::VectorXd& x) {
    return -m_surrogate->objective(x).mean()(0);
  };
  return m_optimizer.minimize(f, m_sampler.samples(m_nstarts));
}

GradientDescent& BayesianOptimizer::optimizer() {
  return m_optimizer;
}

SurrogateModel& BayesianOptimizer::surrogate() {
  return *m_surrogate;
}

void SurrogateModel::addDataPoint(const Eigen::VectorXd& x, double y) {
  m_input_data.emplace_back(x);
  m_output_data.emplace_back(y);
}

GPRSurrogateModel::GPRSurrogateModel(double varn,
                                     double varf,
                                     double length,
                                     double beta)
    : m_varn(varn), m_varf(varf), m_length(length), m_beta(beta) {}

bool GPRSurrogateModel::initialized() const {
  return m_gpr != nullptr;
}

void GPRSurrogateModel::updateModel() {
  SIA_EXCEPTION(m_input_data.size() > 0,
                "GPRSurrogateModel expects data points to be added before "
                "calling updateModel()");
  assert(m_input_data.size() == m_output_data.size());

  // TODO:
  // - add flag to optimize the GPR hyperparameters
  // - add efficient incremental update to the GPR class
  std::size_t ndim = m_input_data[0].size();
  std::size_t nsamples = m_input_data.size();
  Eigen::MatrixXd X = Eigen::MatrixXd(ndim, nsamples);
  for (std::size_t i = 0; i < nsamples; ++i) {
    X.col(i) = m_input_data[i];
  }
  Eigen::MatrixXd y = Eigen::Map<Eigen::MatrixXd>(m_output_data.data(), 1,
                                                  m_output_data.size());
  m_gpr = new GPR(X, y, m_varn, m_varf, m_length);
}

const Gaussian& GPRSurrogateModel::objective(const Eigen::VectorXd& x) {
  SIA_EXCEPTION(initialized(),
                "GPRSurrogateModel has not be initialized, call updateModel()");
  return m_gpr->predict(x);
}

double GPRSurrogateModel::acquisition(const Eigen::VectorXd& x,
                                      double target,
                                      AcquistionType type) {
  SIA_EXCEPTION(initialized(),
                "GPRSurrogateModel has not be initialized, call updateModel()");

  // Evaluate the acquisition function
  const auto& p = objective(x);
  double mu = p.mean()(0);
  double std = sqrt(p.covariance()(0, 0));

  switch (type) {
    case PROBABILITY_IMPROVEMENT:
      return cdf((mu - target) / std);
    case EXPECTED_IMPROVEMENT:
      return (mu - target) * cdf((mu - target) / std) +
             std * pdf((mu - target) / std);
    case UPPER_CONFIDENCE_BOUND:
      return mu + m_beta * std;
  }

  LOG(ERROR) << "GPRSurrogateModel received unsupported AcquistionType";
  return 0;
}

}  // namespace sia
