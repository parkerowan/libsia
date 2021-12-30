/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/uniform.h"

#include <Eigen/Dense>

namespace sia {

/// Projected Gradient Descent with simple box constraints.
///
/// References:
/// [1] W. Hager and H. Zhang, "A New Active Set Algorithm for Box Constrained
/// Optimization," SIAM, 2006.
/// [2] https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
class GradientDescent {
 public:
  /// Solver parameters
  /// n_starts > 0 number of multi-starts using uniform random initial guesses
  /// max_iter > 0 maximum number of iterations
  /// tol > 0 termination based on change in f(x)
  /// 1 > eta > 0 decay rate of backtracking line search
  /// 1 > delta > 0 descent of backtracking line search
  struct Options {
    explicit Options() {}  // Needed due to clang compiler issue
    std::size_t n_starts = 1;
    std::size_t max_iter = 500;
    double tol = 1e-6;
    double eta = 0.5;
    double delta = 0.5;
  };

  GradientDescent(const Eigen::VectorXd& lower,
                  const Eigen::VectorXd& upper,
                  const Options& options = Options());
  virtual ~GradientDescent() = default;
  std::size_t dimension() const;
  const Eigen::VectorXd& lower() const;
  const Eigen::VectorXd& upper() const;
  const Options& options() const;
  void setOptions(const Options& options);

  /// Returns the cost f(x).
  using Cost = std::function<double(const Eigen::VectorXd&)>;

  /// Returns the Jacobian of the cost df(x)/dx.  Note the Jacobian is always
  /// called immediately after the cost is evaluated, so the input state can be
  /// safely ignored for code optimizations.
  using Jacobian = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

  /// Finds a local inflection point of f given an initial guess x0
  Eigen::VectorXd minimize(Cost f,
                           const Eigen::VectorXd& x0,
                           Jacobian jacobian = nullptr) const;

  /// Finds a local inflection point of f using n_starts random initial guesses
  Eigen::VectorXd minimize(Cost f, Jacobian jacobian = nullptr);

  /// Finds a local inflection point of f using multiple starts
  Eigen::VectorXd minimize(Cost f,
                           const std::vector<Eigen::VectorXd>& x0,
                           Jacobian jacobian = nullptr) const;

 private:
  Uniform m_sampler;
  Options m_options;
};

}  // namespace sia
