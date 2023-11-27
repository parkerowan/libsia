/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace sia {

/// Optimizers find a solution to the function minimization $min f(x)$.  For
/// optimizers that make use of gradients, if the optional gradient is not
/// provided, it is estimated through numerical differencing.
class Optimizer {
 public:
  /// Returns the cost f(x)
  using Cost = std::function<double(const Eigen::VectorXd&)>;

  /// Returns the Gradient of the cost df(x)/dx.  Note the gradient is always
  /// called immediately after the cost is evaluated, so the input state can be
  /// safely ignored for code optimizations.
  using Gradient = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

  /// Returns a custom criteria b(x) indicating the minimization converged
  using Convergence = std::function<bool(const Eigen::VectorXd&)>;

  /// ftol is a tolerance on the relative change in cost function, max_iter is
  /// the maximum number of iterations allowed before early termination
  explicit Optimizer(std::size_t dimension, double ftol, std::size_t max_iter);
  virtual ~Optimizer() = default;
  std::size_t dimension() const;

  /// Resets internal optimizer states
  virtual void reset() {}

  /// Performs a single iteration of the optimizer to minimize the cost
  /// function.  Note that reset() must be called prior to running step().
  virtual Eigen::VectorXd step(Cost f,
                               const Eigen::VectorXd& x0,
                               Gradient gradient = nullptr) = 0;

  /// Finds a local inflection point of f given an initial guess x0 with an
  /// optional gradient and optional custom convergence criteria.  Returns true
  /// if the optimization converged within the max number of steps.  Internally
  /// calls reset prior to running the optimization.
  Eigen::VectorXd minimize(Cost f,
                           const Eigen::VectorXd& x0,
                           Gradient gradient = nullptr,
                           Convergence convergence = nullptr);

  /// Finds a local inflection point of f using multiple initial guesses for x0
  /// with an optional gradient and optional custom convergence criteria.
  /// Returns true if the optimization converged within the max number of steps.
  Eigen::VectorXd minimize(Cost f,
                           const std::vector<Eigen::VectorXd>& x0,
                           Gradient gradient = nullptr,
                           Convergence convergence = nullptr);

 private:
  std::size_t m_dimension;
  double m_ftol;
  std::size_t m_max_iter;
};

}  // namespace sia
