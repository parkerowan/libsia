/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/gpr.h"
#include "sia/belief/uniform.h"
#include "sia/optimizers/gradient_descent.h"

#include <Eigen/Dense>

namespace sia {

class ObjectiveModel;

/// Type of objective model
enum ObjectiveType {
  GPR_OBJECTIVE,
};

/// Type of acquisition model
enum AcquistionType {
  PROBABILITY_IMPROVEMENT,
  EXPECTED_IMPROVEMENT,
  UPPER_CONFIDENCE_BOUND,
};

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
  BayesianOptimizer(const Eigen::VectorXd& lower,
                    const Eigen::VectorXd& upper,
                    ObjectiveType objective = GPR_OBJECTIVE,
                    AcquistionType acquisition = EXPECTED_IMPROVEMENT,
                    std::size_t nstarts = 10);
  virtual ~BayesianOptimizer() = default;

  Eigen::VectorXd selectNextSample();
  void addDataPoint(const Eigen::VectorXd& x, double y);
  void updateModel();
  Eigen::VectorXd getSolution();

  // Access to optimizer
  GradientDescent& optimizer();

  // Access to objective model
  ObjectiveModel& objective();

 private:
  Uniform m_sampler;
  GradientDescent m_optimizer;
  ObjectiveModel* m_objective{nullptr};
  AcquistionType m_acquisition_type;
  std::size_t m_nstarts{10};
};

//
// Objective models
//

class ObjectiveModel {
 public:
  virtual void addDataPoint(const Eigen::VectorXd& x, double y);
  virtual bool initialized() const = 0;
  virtual void updateModel() = 0;

  /// Function R^n -> p(y) that statistically approximates the objective
  virtual const Distribution& objective(const Eigen::VectorXd& x) = 0;

  /// Utility function R^n -> R for selecting the next data point
  virtual double acquisition(const Eigen::VectorXd& x,
                             double target,
                             AcquistionType type) = 0;

 protected:
  std::vector<Eigen::VectorXd> m_input_data;
  std::vector<double> m_output_data;
};

/// Models the objective function using a GPR regression model.
class GPRObjectiveModel : public ObjectiveModel {
 public:
  explicit GPRObjectiveModel(double varn = 1e-4,
                             double varf = 1,
                             double length = 1,
                             double beta = 1);
  virtual ~GPRObjectiveModel() = default;
  bool initialized() const override;
  void updateModel() override;
  const Gaussian& objective(const Eigen::VectorXd& x) override;
  double acquisition(const Eigen::VectorXd& x,
                     double target,
                     AcquistionType type) override;

 private:
  double m_varn;
  double m_varf;
  double m_length;
  double m_beta;
  GPR* m_gpr{nullptr};
};

}  // namespace sia
