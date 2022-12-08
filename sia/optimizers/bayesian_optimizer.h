/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

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
/// expensive to evaluate and may be noisy. The main implementation is Algorithm
/// 1 in [1], which involves iteratively repeating:
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
/// Note that SIA also provides support for Contextual Bandits which is
/// accomplished by including a conditioning input u.  The class then solves the
/// problem using that same procedure above
///
/// $x^* = \argmax f(x | u)$
///
/// References:
/// [1] B. Shahriari et. al., "Taking the Human Out of the Loop: A Review of
/// Bayesian Optimization," Proceedings of the IEEE, 104(1), 2016.
class BayesianOptimizer {
 public:
  /// Type of acquisition model
  enum class AcquisitionType {
    PROBABILITY_IMPROVEMENT,
    EXPECTED_IMPROVEMENT,
    UPPER_CONFIDENCE_BOUND,
  };

  /// Initialize the optimizer with lower and upper bounds on the parameters
  BayesianOptimizer(
      const Eigen::VectorXd& lower,
      const Eigen::VectorXd& upper,
      Kernel& kernel,
      std::size_t cond_inputs_dim = 0,
      AcquisitionType acquisition = AcquisitionType::EXPECTED_IMPROVEMENT,
      double beta = 1,
      const GradientDescent::Options& options = GradientDescent::Options());
  virtual ~BayesianOptimizer() = default;

  Eigen::VectorXd selectNextSample(
      const Eigen::VectorXd& u = Eigen::VectorXd{});
  void addDataPoint(const Eigen::VectorXd& x,
                    double y,
                    const Eigen::VectorXd& u = Eigen::VectorXd{});
  void updateModel(bool train = false);
  Eigen::VectorXd getSolution(const Eigen::VectorXd& u = Eigen::VectorXd{});
  GradientDescent& optimizer();
  const Gaussian& objective(const Eigen::VectorXd& x,
                            const Eigen::VectorXd& u = Eigen::VectorXd{});
  double acquisition(const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u = Eigen::VectorXd{});
  double acquisition(const Eigen::VectorXd& x,
                     double target,
                     BayesianOptimizer::AcquisitionType type,
                     const Eigen::VectorXd& u = Eigen::VectorXd{});

 private:
  Uniform m_sampler;
  GradientDescent m_optimizer;
  std::size_t m_cond_inputs_dim;
  AcquisitionType m_acquisition_type;
  Eigen::VectorXd m_cached_solution{0};
  GPR m_gpr;
  double m_beta;
  bool m_dirty_solution{true};
  std::vector<Eigen::VectorXd> m_cond_input_data;
  std::vector<Eigen::VectorXd> m_input_data;
  std::vector<double> m_output_data;
};

}  // namespace sia
