/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/gpr.h"
#include "sia/belief/uniform.h"

#include <Eigen/Dense>

namespace sia {

/// Function R^n -> p(y) that statistically approximates the objective
// class ObjectiveModel {
//  public:
//   ObjectiveModel() = default;
//   virtual ~ObjectiveModel() = default;
//   virtual const Distribution& evaluate(const Eigen::VectorXd& x) = 0;
//   virtual void addDataPoint(const Eigen::VectorXd& x) = 0;
// };

// class GPRObjectiveModel : public ObjectiveModel {
//  public:
//   GPRObjectiveModel() = default;
//   virtual ~GPRObjectiveModel() = default;
//   const Gaussian& evaluate(const Eigen::VectorXd& x) override;
//   const Gaussian& addDataPoint(const Eigen::VectorXd& x) override;

//  private:
//   GPR* m_gpr{nullptr};
// };

// class GPCObjectiveModel : public ObjectiveModel {};

/// Utility function R^n -> R for selecting the next data point
// class AcquisitionModel {
//  public:
//   explicit AcquisitionModel(ObjectiveModel& objective);
//   virtual ~AcquisitionModel() = default;
//   virtual double evaluate(const Eigen::VectorXd& x) = 0;

//  protected:
//   ObjectiveModel& m_objective;
// };

// class ProbabilityImprovement : public AcquisitionModel {};

// class ExpectedImprovement : public AcquisitionModel {};

// class UpperConfidenceBound : public AcquisitionModel {};

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
  // explicit BayesianOptimizer(ObjectiveModel& objective,
  //                            AcquisitionModel& acquisition);
  BayesianOptimizer(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper);
  virtual ~BayesianOptimizer() = default;

  Eigen::VectorXd selectNextSample();
  void addDataPoint(const Eigen::VectorXd& x, double y);
  void updateModel();
  Eigen::VectorXd getSolution();

  GPR& gpr();

 private:
  std::vector<Eigen::VectorXd> m_input_data;
  std::vector<double> m_output_data;
  Uniform m_sampler;
  GPR* m_gpr{nullptr};

  double m_tol{1e-4};
  double m_beta{0.5};
  std::size_t m_nstarts{10};
};

}  // namespace sia
