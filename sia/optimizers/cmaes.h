/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <Eigen/Dense>
#include "sia/belief/gaussian.h"
#include "sia/optimizers/optimizers.h"

namespace sia {

/// Covariance matrix adaptation evolutionary strategy (CMA-ES).  CMA is a
/// gradient free sampling-based method that convergences using the covariance
/// of the top samples.  It is particularly good with high dimensions and rugged
/// cost landscapes.  There are no bounds on the decision space.
///
/// More information on the parameters is available in [2].
/// - max_iter: (>0) maximum number of iterations
/// - ftol: (>0) termination based on relative change in f(x)
/// - n_samples: (>1) number of sample draws per iteration
/// - init_stdev: (>0) initial standard deviation of the search space
/// - max_cov_norm: (>>0) maximum covariance norm
///
/// References:
/// [1]
/// https://tiezhongyu2005.github.io/resources/popularization/CMA-ES_2003.pdf
/// [2] https://en.wikipedia.org/wiki/CMA-ES
class CMAES : public Optimizer {
 public:
  /// Algorithm options
  struct Options {
    explicit Options() {}
    std::size_t max_iter = 100;
    double ftol = 1e-6;
    std::size_t n_samples = 100;
    double init_stdev = 0.3;
    double max_cov_norm = 1e12;
  };

  explicit CMAES(std::size_t dimension, const Options& options = Options());
  virtual ~CMAES() = default;
  const std::vector<std::pair<double, Eigen::VectorXd>>& getSamples() const;

  /// Resets internal optimizer states
  void reset() override;

  /// Performs a single iteration of the optimizer to minimize the cost
  /// function.  Note that reset() must be called prior to running step().
  Eigen::VectorXd step(Cost f,
                       const Eigen::VectorXd& x0,
                       Gradient gradient = nullptr) override;

 private:
  // Computed once at the beginning of the algorithm
  struct Constants {
    double lambda;
    std::size_t mu;
    Eigen::VectorXd weights;
    double mueff;
    double N;
    double cc;
    double cs;
    double c1;
    double cmu;
    double damps;
  } _c;

  // Recomputed every iteration
  struct States {
    std::size_t iter;
    double sigma;
    double chiN;
    Eigen::MatrixXd C;
    Eigen::VectorXd pc;
    Eigen::VectorXd ps;
  } _s;

  Options m_options;
  std::vector<std::pair<double, Eigen::VectorXd>> m_samples{};
};

}  // namespace sia
