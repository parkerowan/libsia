/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <Eigen/Dense>
#include "sia/belief/uniform.h"
#include "sia/optimizers/optimizers.h"

namespace sia {

/// Projected Gradient Descent (GD) with simple box constraints.
///
/// More information on the parameters is available in [1].
/// - max_iter: (>0) maximum number of iterations
/// - ftol: (>0) termination based on relative change in f(x)
/// - eta: (>0, <1) decay rate of backtracking line search
/// - delta: (>0, <1) descent of backtracking line search
///
/// References:
/// [1] W. Hager and H. Zhang, "A New Active Set Algorithm for Box Constrained
/// Optimization," SIAM, 2006.
/// [2] https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
class GD : public Optimizer {
 public:
  /// Algorithm options
  struct Options {
    explicit Options() {}
    std::size_t max_iter = 500;
    double ftol = 1e-6;
    double eta = 0.5;
    double delta = 0.5;
  };

  explicit GD(const Eigen::VectorXd& lower,
              const Eigen::VectorXd& upper,
              const Options& options = Options());
  virtual ~GD() = default;
  const Eigen::VectorXd& lower() const;
  const Eigen::VectorXd& upper() const;

  /// Performs a single iteration of the optimizer to minimize the cost
  /// function.  Note that reset() must be called prior to running step().
  Eigen::VectorXd step(Cost f,
                       const Eigen::VectorXd& x0,
                       Gradient gradient = nullptr) override;

 private:
  Eigen::VectorXd m_lower;
  Eigen::VectorXd m_upper;
  Options m_options;
};

}  // namespace sia
