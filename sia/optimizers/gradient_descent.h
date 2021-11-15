/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <Eigen/Dense>

namespace sia {

/// Projected Gradient Descent with simple box constraints.
///
/// tol > 0 termination based on change in f(x)
/// 1 > eta > 0 decay rate of backtracking line search
/// 1 > delta > 0 descent of backtracking line search
///
/// References:
/// [1] W. Hager and H. Zhang, "A New Active Set Algorithm for Box Constrained
/// Optimization," SIAM, 2006.
/// [2] https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
class GradientDescent {
 public:
  GradientDescent(const Eigen::VectorXd& lower,
                  const Eigen::VectorXd& upper,
                  double tol = 1e-6,
                  double eta = 0.5,
                  double delta = 0.5);
  virtual ~GradientDescent() = default;
  std::size_t dimension() const;

  /// Finds a local inflection point of f given an initial guess x0
  Eigen::VectorXd minimize(std::function<double(const Eigen::VectorXd&)> f,
                           const Eigen::VectorXd& x0) const;

  /// Finds a local inflection point of f using multiple starts
  Eigen::VectorXd minimize(std::function<double(const Eigen::VectorXd&)> f,
                           const std::vector<Eigen::VectorXd>& x0) const;

 private:
  Eigen::VectorXd m_lower;
  Eigen::VectorXd m_upper;
  double m_tol;
  double m_eta;
  double m_delta;
};

}  // namespace sia
