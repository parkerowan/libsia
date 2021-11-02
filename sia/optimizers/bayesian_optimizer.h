/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/gpr.h"
#include "sia/belief/uniform.h"
#include "sia/optimizers/gradient_descent.h"

#include <Eigen/Dense>

namespace sia {

/// Bayesian Optimization is an active learning procedure for global joint
/// optimization of expensive-to-evaluate objective functions.  It is useful for
/// applications such as parameter tuning and continuous-action space Bandit
/// learning problems. The method solves the optimization problem
///
/// $x^* = \argmax f(x)$
///
/// by building a surrogate statistical model of f(x) and an acquisition model
/// a(x) for which next test point to sample. The objective f(x) is assumed
/// expensive to evaluate and may be noisy.  The main implementation is
/// Algorithm 1 in [1], which involves iteratively repeating:
/// 1. x = BayesianOptimizer::selectNextSample() which returns a test value x
/// found by optimizing the internal acquisition function.
/// 2. BayesianOptimizer::addDataPoint(x, y) which adds x and a corresponding
/// sample y ~ f(x) to the objective function training data.
/// 3. BayesianOptimizer::updateModel() which updates the internal surrogate
/// objective function with the training data.
///
/// At any point, the current optimum can get found by calling
/// xopt = BayesianOptimizer::getSolution()
///
/// References:
/// [1] B. Shahriari et. al., "Taking the Human Out of the Loop: A Review of
/// Bayesian Optimization," Proceedings of the IEEE, 104(1), 2016.
class BayesianOptimizer {
 public:
  /// Type of objective model
  enum ObjectiveType {
    GPR_OBJECTIVE,
  };

  /// Type of acquisition model
  enum AcquisitionType {
    PROBABILITY_IMPROVEMENT,
    EXPECTED_IMPROVEMENT,
    UPPER_CONFIDENCE_BOUND,
  };

  /// Initialize the optimizer with lower and upper bounds on the parameters
  BayesianOptimizer(const Eigen::VectorXd& lower,
                    const Eigen::VectorXd& upper,
                    ObjectiveType objective = GPR_OBJECTIVE,
                    AcquisitionType acquisition = EXPECTED_IMPROVEMENT,
                    std::size_t n_starts = 10);
  virtual ~BayesianOptimizer() = default;

  Eigen::VectorXd selectNextSample();
  void addDataPoint(const Eigen::VectorXd& x, double y);
  void updateModel(bool train = true);
  Eigen::VectorXd getSolution();
  GradientDescent& optimizer();
  const Distribution& objective(const Eigen::VectorXd& x);
  double acquisition(const Eigen::VectorXd& x);

  // Forward declaration
  class SurrogateModel;

 private:
  Uniform m_sampler;
  GradientDescent m_optimizer;
  std::shared_ptr<SurrogateModel> m_surrogate{nullptr};
  AcquisitionType m_acquisition_type;
  Eigen::VectorXd m_cached_solution{0};
  bool m_dirty_solution{true};
};

}  // namespace sia
