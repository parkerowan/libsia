/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <Eigen/Dense>
#include "sia/belief/gaussian.h"

namespace sia {

/// Covariance matrix adaptation evolutionary strategy (CMA-ES).  CMA is a
/// derivative free sampling-based method that convergences using the covariance
/// of the top samples.  It is particularly good with high dimensions and rugged
/// cost landscapes.
///
/// More information on the parameters is available in [2].
/// - n_samples: (>1) number of sample draws per iteration
/// - init_stdev: (>0) initial standard deviation of the search space
/// - max_iter: (>0) maximum number of iterations
/// - tol: (>0) termination based on change in f(x)
/// - max_cov_norm: (>>0) maximum covariance norm
///
/// References:
/// [1]
/// https://tiezhongyu2005.github.io/resources/popularization/CMA-ES_2003.pdf
/// [2] https://en.wikipedia.org/wiki/CMA-ES
class CovarianceAdaptation {
 public:
  /// Algorithm options
  struct Options {
    explicit Options() {}
    std::size_t n_samples = 100;
    double init_stdev = 0.3;
    std::size_t max_iter = 100;
    double tol = 1e-6;
    double max_cov_norm = 1e12;
  };

  explicit CovarianceAdaptation(std::size_t dimension,
                                const Options& options = Options());
  virtual ~CovarianceAdaptation() = default;
  std::size_t dimension() const;

  /// Returns the cost f(x).
  using Cost = std::function<double(const Eigen::VectorXd&)>;

  /// Finds the minima of f given an initial guess x0
  Eigen::VectorXd minimize(Cost f, const Eigen::VectorXd& x0) const;

 private:
  std::size_t m_dimension;
  Options m_options;
};

}  // namespace sia
