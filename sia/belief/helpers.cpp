/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/helpers.h"

namespace sia {

const Eigen::VectorXd logProb(const Distribution& distribution,
                              const Eigen::MatrixXd& x) {
  std::size_t n = x.cols();
  Eigen::VectorXd p = Eigen::VectorXd::Zero(n);
  for (std::size_t i = 0; i < n; ++i) {
    p(i) = distribution.logProb(x.col(i));
  }
  return p;
}

const Eigen::VectorXd logProb1d(const Distribution& distribution,
                                const Eigen::VectorXd& x) {
  return logProb(distribution, x.transpose());
}

const Eigen::VectorXd logProb2d(const Distribution& distribution,
                                const Eigen::VectorXd& x,
                                const Eigen::VectorXd& y) {
  std::size_t n = x.size();
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(2, n);
  for (std::size_t i = 0; i < n; ++i) {
    X.col(i) << x(i), y(i);
  }
  return logProb(distribution, X);
}

}  // namespace sia