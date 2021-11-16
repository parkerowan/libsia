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

// Forward declaration
class SurrogateModel;

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
                    AcquisitionType acquisition = EXPECTED_IMPROVEMENT,
                    std::size_t nstarts = 10);
  virtual ~BayesianOptimizer() = default;

  Eigen::VectorXd selectNextSample();
  void addDataPoint(const Eigen::VectorXd& x, double y);
  void updateModel();
  Eigen::VectorXd getSolution();
  GradientDescent& optimizer();
  SurrogateModel& surrogate();

 private:
  Uniform m_sampler;
  GradientDescent m_optimizer;
  SurrogateModel* m_surrogate{nullptr};
  AcquisitionType m_acquisition_type;
  std::size_t m_nstarts{10};
};

/// The surrogate model provides a statistical approximation of the objective
/// function and a corresponding acquisition function.
/// - Objective: Function R^n -> p(y) that approximates the true objective
/// - Acqusition: Utility function R^n -> R for selecting the next data point
/// https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf
class SurrogateModel {
 public:
  void addDataPoint(const Eigen::VectorXd& x, double y);
  const std::vector<Eigen::VectorXd>& inputData() const;
  const std::vector<double>& outputData() const;
  virtual bool initialized() const = 0;
  virtual void updateModel() = 0;
  virtual const Distribution& objective(const Eigen::VectorXd& x) = 0;
  virtual double acquisition(const Eigen::VectorXd& x,
                             double target,
                             AcquisitionType type) = 0;

 protected:
  std::vector<Eigen::VectorXd> m_input_data;
  std::vector<double> m_output_data;
};

/// Surrogate using a GPR to model the objective.
class GPRSurrogateModel : public SurrogateModel {
 public:
  explicit GPRSurrogateModel(double varn = 1e-3,
                             double varf = 1,
                             double length = 0.1,
                             double beta = 1);
  virtual ~GPRSurrogateModel() = default;
  bool initialized() const override;
  void updateModel() override;
  const Gaussian& objective(const Eigen::VectorXd& x) override;
  double acquisition(const Eigen::VectorXd& x,
                     double target,
                     AcquisitionType type) override;

 private:
  double m_varn;
  double m_varf;
  double m_length;
  double m_beta;
  GPR* m_gpr{nullptr};
};

// TODO: Add support for Binary classification objective

}  // namespace sia
