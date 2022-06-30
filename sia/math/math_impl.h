/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/math/math.h"

namespace sia {

template <typename... Args>
const Eigen::MatrixXd numericalJacobian(
    std::function<const Eigen::VectorXd(const Eigen::VectorXd&, Args...)> f,
    const Eigen::VectorXd& x,
    Args... args) {
  std::size_t n = x.size();
  Eigen::MatrixXd J;
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
    dx(i) = NUMERICAL_DERIVATIVE_STEP;
    Eigen::VectorXd fp = f(x + dx, args...);
    Eigen::VectorXd fn = f(x - dx, args...);
    if (i == 0) {
      J = Eigen::MatrixXd::Zero(fp.size(), n);
    }
    J.col(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  return J;
}

}  // namespace sia
