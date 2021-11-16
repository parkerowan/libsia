/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/gpr.h"
#include "sia/belief/uniform.h"

#include <Eigen/Dense>

namespace sia {

class ObjectiveModel;
class AcquisitionModel;

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
  BayesianOptimizer(ObjectiveModel& objective,
                    AcquisitionModel& acquisition,
                    const Eigen::VectorXd& lower,
                    const Eigen::VectorXd& upper,
                    std::size_t nstarts = 10,
                    double tol = 1e-4,
                    double eta = 0.5,
                    double delta = 0.5);
  virtual ~BayesianOptimizer() = default;

  Eigen::VectorXd selectNextSample();
  void addDataPoint(const Eigen::VectorXd& x, double y);
  void updateModel();
  Eigen::VectorXd getSolution();
  ObjectiveModel& objective();
  AcquisitionModel& acquisition();

 private:
  ObjectiveModel& m_objective;
  AcquisitionModel& m_acquisition;
  Uniform m_sampler;

  // GD optimizer parameters
  std::size_t m_nstarts{10};
  double m_tol{1e-4};
  double m_eta{0.5};
  double m_delta{0.5};
};

//
// Objective models
//

/// Function R^n -> p(y) that statistically approximates the objective
class ObjectiveModel {
 public:
  ObjectiveModel() = default;
  virtual ~ObjectiveModel() = default;
  virtual const Distribution& evaluate(const Eigen::VectorXd& x) = 0;
  virtual void addDataPoint(const Eigen::VectorXd& x, double y) = 0;
  virtual void updateModel() = 0;
  virtual bool initialized() const = 0;
};

/// Models the objective function using a GPR regression model.
class GPRObjectiveModel : public ObjectiveModel {
 public:
  explicit GPRObjectiveModel(double varn = 1e-4,
                             double varf = 1,
                             double length = 1);
  virtual ~GPRObjectiveModel() = default;
  const Gaussian& evaluate(const Eigen::VectorXd& x) override;
  void addDataPoint(const Eigen::VectorXd& x, double y) override;
  void updateModel() override;
  bool initialized() const override;

 private:
  double m_varn;
  double m_varf;
  double m_length;
  std::vector<Eigen::VectorXd> m_input_data;
  std::vector<double> m_output_data;
  GPR* m_gpr{nullptr};
};

//
// Acqusition models
//

/// Utility function R^n -> R for selecting the next data point
class AcquisitionModel {
 public:
  explicit AcquisitionModel(ObjectiveModel& objective);
  virtual ~AcquisitionModel() = default;
  virtual double evaluate(const Eigen::VectorXd& x, double target) = 0;

 protected:
  ObjectiveModel& m_objective;
};

/// Probability of Improvement (PI)
class ProbabilityImprovement : public AcquisitionModel {
 public:
  explicit ProbabilityImprovement(ObjectiveModel& objective);
  virtual ~ProbabilityImprovement() = default;
  double evaluate(const Eigen::VectorXd& x, double target) override;
};

/// Expected Improvement (EI)
class ExpectedImprovement : public AcquisitionModel {
 public:
  explicit ExpectedImprovement(ObjectiveModel& objective);
  virtual ~ExpectedImprovement() = default;
  double evaluate(const Eigen::VectorXd& x, double target) override;
};

/// Upper Confidence Bound (UCB)
class UpperConfidenceBound : public AcquisitionModel {
 public:
  UpperConfidenceBound(ObjectiveModel& objective, double beta = 0.1);
  virtual ~UpperConfidenceBound() = default;
  double evaluate(const Eigen::VectorXd& x, double target) override;

 private:
  double m_beta;
};

}  // namespace sia
