/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/bayesian_optimizer.h"
#include "sia/common/exception.h"
#include "sia/optimizers/gradient_descent.h"

#include <cmath>

namespace sia {

static double pdf(double x) {
  return 1 / sqrt(2 * M_PI) * exp(-pow(x, 2) / 2);
}

static double cdf(double x) {
  return (1 + erf(x / sqrt(2))) / 2;
}

BayesianOptimizer::BayesianOptimizer(ObjectiveModel& objective,
                                     AcquisitionModel& acquisition,
                                     const Eigen::VectorXd& lower,
                                     const Eigen::VectorXd& upper,
                                     std::size_t nstarts,
                                     double tol,
                                     double eta,
                                     double delta)
    : m_objective(objective),
      m_acquisition(acquisition),
      m_sampler(lower, upper),
      m_nstarts(nstarts),
      m_tol(tol),
      m_eta(eta),
      m_delta(delta) {}

Eigen::VectorXd BayesianOptimizer::selectNextSample() {
  // If there is no model yet, just sample uniformly
  if (!m_objective.initialized()) {
    return m_sampler.sample();
  }

  // Optimize the acquisition model
  double target = m_objective.evaluate(getSolution()).mean()(0);
  GradientDescent optm(m_sampler.lower(), m_sampler.upper(), m_tol, m_eta,
                       m_delta);
  auto f = [=](const Eigen::VectorXd& x) {
    return -m_acquisition.evaluate(x, target);
  };
  return optm.minimize(f, m_sampler.samples(m_nstarts));
}

void BayesianOptimizer::addDataPoint(const Eigen::VectorXd& x, double y) {
  m_objective.addDataPoint(x, y);
}

void BayesianOptimizer::updateModel() {
  m_objective.updateModel();
}

Eigen::VectorXd BayesianOptimizer::getSolution() {
  if (!m_objective.initialized()) {
    return m_sampler.sample();
  }

  // Optimize the objective function model
  GradientDescent optm(m_sampler.lower(), m_sampler.upper(), m_tol, m_eta,
                       m_delta);
  auto f = [=](const Eigen::VectorXd& x) {
    return -m_objective.evaluate(x).mean()(0);
  };
  return optm.minimize(f, m_sampler.samples(m_nstarts));
}

ObjectiveModel& BayesianOptimizer::objective() {
  return m_objective;
}

AcquisitionModel& BayesianOptimizer::acquisition() {
  return m_acquisition;
}

GPRObjectiveModel::GPRObjectiveModel(double varn, double varf, double length)
    : m_varn(varn), m_varf(varf), m_length(length) {}

const Gaussian& GPRObjectiveModel::evaluate(const Eigen::VectorXd& x) {
  SIA_EXCEPTION(initialized(),
                "GPRObjectiveModel has not be initialized, call updateModel()");
  return m_gpr->predict(x);
}

void GPRObjectiveModel::addDataPoint(const Eigen::VectorXd& x, double y) {
  m_input_data.emplace_back(x);
  m_output_data.emplace_back(y);
}

void GPRObjectiveModel::updateModel() {
  SIA_EXCEPTION(m_input_data.size() > 0,
                "GPRObjectiveModel expects data points to be added before "
                "calling updateModel()");
  assert(m_input_data.size() == m_output_data.size());

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
  m_gpr = new GPR(X, y, m_varn, m_varf, m_length);
}

bool GPRObjectiveModel::initialized() const {
  return m_gpr != nullptr;
}

AcquisitionModel::AcquisitionModel(ObjectiveModel& objective)
    : m_objective(objective) {}

ProbabilityImprovement::ProbabilityImprovement(ObjectiveModel& objective)
    : AcquisitionModel(objective) {}

double ProbabilityImprovement::evaluate(const Eigen::VectorXd& x,
                                        double target) {
  const auto& p = m_objective.evaluate(x);
  double mu = p.mean()(0);
  double std = sqrt(p.covariance()(0, 0));
  return cdf((mu - target) / std);
}

ExpectedImprovement::ExpectedImprovement(ObjectiveModel& objective)
    : AcquisitionModel(objective) {}

double ExpectedImprovement::evaluate(const Eigen::VectorXd& x, double target) {
  const auto& p = m_objective.evaluate(x);
  double mu = p.mean()(0);
  double std = sqrt(p.covariance()(0, 0));
  return (mu - target) * cdf((mu - target) / std) +
         std * pdf((mu - target) / std);
}

UpperConfidenceBound::UpperConfidenceBound(ObjectiveModel& objective,
                                           double beta)
    : AcquisitionModel(objective), m_beta(beta) {}

double UpperConfidenceBound::evaluate(const Eigen::VectorXd& x, double target) {
  (void)(target);
  const auto& p = m_objective.evaluate(x);
  double mu = p.mean()(0);
  double std = sqrt(p.covariance()(0, 0));
  return mu + m_beta * std;
}

}  // namespace sia
