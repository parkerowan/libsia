/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/bayesian_optimizer.h"
#include "sia/common/exception.h"

#include <cmath>

namespace sia {

// ----------------------------------------------------------------------------

// The surrogate model provides a statistical approximation of the objective
// function and a corresponding acquisition function.
// - Objective: Function R^n -> p(y) that approximates the true objective
// - Acqusition: Utility function R^n -> R for selecting the next data point
// https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf
class BayesianOptimizer::SurrogateModel {
 public:
  virtual ~SurrogateModel() = default;
  virtual bool initialized() const = 0;
  virtual void updateModel(bool train) = 0;
  virtual const Distribution& objective(const Eigen::VectorXd& x) = 0;
  virtual double acquisition(const Eigen::VectorXd& x,
                             double target,
                             BayesianOptimizer::AcquisitionType type) = 0;
  void addDataPoint(const Eigen::VectorXd& x, double y);
  const std::vector<Eigen::VectorXd>& inputData() const;
  const std::vector<double>& outputData() const;
  static std::shared_ptr<SurrogateModel> create(
      BayesianOptimizer::ObjectiveType type);

 protected:
  std::vector<Eigen::VectorXd> m_input_data;
  std::vector<double> m_output_data;
};

// Surrogate using a GPR to model the objective.
class GPRSurrogateModel : public BayesianOptimizer::SurrogateModel {
 public:
  explicit GPRSurrogateModel(double beta = 1);
  virtual ~GPRSurrogateModel() = default;
  bool initialized() const override;
  void updateModel(bool train) override;
  const Gaussian& objective(const Eigen::VectorXd& x) override;
  double acquisition(const Eigen::VectorXd& x,
                     double target,
                     BayesianOptimizer::AcquisitionType type) override;

 private:
  double m_beta;
  std::shared_ptr<GPR> m_gpr{nullptr};
};

// TODO: Add support for Binary classification objective

// ----------------------------------------------------------------------------

static double pdf(double x) {
  return 1 / sqrt(2 * M_PI) * exp(-pow(x, 2) / 2);
}

static double cdf(double x) {
  return (1 + erf(x / sqrt(2))) / 2;
}

// ----------------------------------------------------------------------------

BayesianOptimizer::BayesianOptimizer(
    const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper,
    BayesianOptimizer::ObjectiveType objective,
    BayesianOptimizer::AcquisitionType acquisition,
    std::size_t n_starts)
    : m_sampler(lower, upper),
      m_optimizer(lower, upper),
      m_acquisition_type(acquisition) {
  m_surrogate = SurrogateModel::create(objective);
  assert(m_surrogate != nullptr);

  GradientDescent::Options opts = m_optimizer.options();
  opts.n_starts = n_starts;
  m_optimizer.setOptions(opts);
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
  return m_optimizer.minimize(f);
}

void BayesianOptimizer::addDataPoint(const Eigen::VectorXd& x, double y) {
  m_surrogate->addDataPoint(x, y);
}

void BayesianOptimizer::updateModel(bool train) {
  m_surrogate->updateModel(train);
  m_dirty_solution = true;
}

Eigen::VectorXd BayesianOptimizer::getSolution() {
  if (!m_surrogate->initialized()) {
    return m_sampler.sample();
  }

  if (!m_dirty_solution) {
    return m_cached_solution;
  }

  // Optimize the objective function model
  auto f = [=](const Eigen::VectorXd& x) {
    return -m_surrogate->objective(x).mean()(0);
  };
  m_cached_solution = m_optimizer.minimize(f);
  m_dirty_solution = false;
  return m_cached_solution;
}

GradientDescent& BayesianOptimizer::optimizer() {
  return m_optimizer;
}

const Distribution& BayesianOptimizer::objective(const Eigen::VectorXd& x) {
  return m_surrogate->objective(x);
}

double BayesianOptimizer::acquisition(const Eigen::VectorXd& x) {
  double target = m_surrogate->objective(getSolution()).mean()(0);
  return m_surrogate->acquisition(x, target, m_acquisition_type);
}

// ----------------------------------------------------------------------------

void BayesianOptimizer::SurrogateModel::addDataPoint(const Eigen::VectorXd& x,
                                                     double y) {
  m_input_data.emplace_back(x);
  m_output_data.emplace_back(y);
}

const std::vector<Eigen::VectorXd>&
BayesianOptimizer::SurrogateModel::inputData() const {
  return m_input_data;
}

const std::vector<double>& BayesianOptimizer::SurrogateModel::outputData()
    const {
  return m_output_data;
}

std::shared_ptr<BayesianOptimizer::SurrogateModel>
BayesianOptimizer::SurrogateModel::create(
    BayesianOptimizer::ObjectiveType type) {
  switch (type) {
    case BayesianOptimizer::GPR_OBJECTIVE:
      return std::make_shared<GPRSurrogateModel>();
    default:
      SIA_EXCEPTION(false,
                    "BayesianOptimizer::SurrogateModel encountered unsupported "
                    "BayesianOptimizer::ObjectiveType");
  }
}

GPRSurrogateModel::GPRSurrogateModel(double beta) : m_beta(beta) {}

bool GPRSurrogateModel::initialized() const {
  return m_gpr != nullptr;
}

void GPRSurrogateModel::updateModel(bool train) {
  SIA_EXCEPTION(m_input_data.size() > 0,
                "GPRSurrogateModel expects data points to be added before "
                "calling updateModel()");
  assert(m_input_data.size() == m_output_data.size());

  // TODO:
  // - add efficient incremental update to the GPR class
  std::size_t ndim = m_input_data[0].size();
  std::size_t nsamples = m_input_data.size();
  Eigen::MatrixXd X = Eigen::MatrixXd(ndim, nsamples);
  for (std::size_t i = 0; i < nsamples; ++i) {
    X.col(i) = m_input_data[i];
  }
  Eigen::MatrixXd y = Eigen::Map<Eigen::MatrixXd>(m_output_data.data(), 1,
                                                  m_output_data.size());

  // Update the GPR instance
  if (m_gpr == nullptr) {
    m_gpr = std::make_shared<GPR>(X, y);
  } else {
    m_gpr->setData(X, y);
  }

  // Optionally train the GPR hyperparameters
  if (train) {
    m_gpr->train();
  }
}

const Gaussian& GPRSurrogateModel::objective(const Eigen::VectorXd& x) {
  SIA_EXCEPTION(initialized(),
                "GPRSurrogateModel has not be initialized, call updateModel()");
  return m_gpr->predict(x);
}

double GPRSurrogateModel::acquisition(const Eigen::VectorXd& x,
                                      double target,
                                      BayesianOptimizer::AcquisitionType type) {
  SIA_EXCEPTION(initialized(),
                "GPRSurrogateModel has not be initialized, call updateModel()");

  // Evaluate the acquisition function
  const auto& p = objective(x);
  double mu = p.mean()(0);
  double std = sqrt(p.covariance()(0, 0));

  switch (type) {
    case BayesianOptimizer::PROBABILITY_IMPROVEMENT:
      return cdf((mu - target) / std);
    case BayesianOptimizer::EXPECTED_IMPROVEMENT:
      return (mu - target) * cdf((mu - target) / std) +
             std * pdf((mu - target) / std);
    case BayesianOptimizer::UPPER_CONFIDENCE_BOUND:
      return mu + m_beta * std;
  }

  LOG(ERROR) << "GPRSurrogateModel received unsupported AcquisitionType";
  return 0;
}

}  // namespace sia
