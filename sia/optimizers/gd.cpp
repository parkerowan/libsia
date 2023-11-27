/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/gd.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

namespace sia {

GD::GD(const Eigen::VectorXd& lower,
       const Eigen::VectorXd& upper,
       const GD::Options& options)
    : Optimizer(lower.size(), options.ftol, options.max_iter),
      m_lower(lower),
      m_upper(upper),
      m_options(options) {
  SIA_THROW_IF_NOT(options.eta > 0, "GD eta expected to be > 0");
  SIA_THROW_IF_NOT(options.eta < 1, "GD eta expected to be < 1");
  SIA_THROW_IF_NOT(options.delta > 0, "GD delta expected to be > 0");
  SIA_THROW_IF_NOT(options.delta < 1, "GD delta expected to be < 1");
}

const Eigen::VectorXd& GD::lower() const {
  return m_lower;
}

const Eigen::VectorXd& GD::upper() const {
  return m_upper;
}

Eigen::VectorXd GD::step(Cost f, const Eigen::VectorXd& x0, Gradient gradient) {
  SIA_THROW_IF_NOT(dimension() == (std::size_t)x0.size(),
                   "GD x expected to match dimension");

  // Monotone Gradient Projection Algorithm
  // See: https://www.math.lsu.edu/~hozhang/papers/63522-gg.pdf
  // Compute gradient
  Eigen::VectorXd g = (gradient != nullptr) ? gradient(x0) : dfdx(f, x0);

  // Compute projection P and search direction d
  double alpha = 1;
  Eigen::VectorXd xref = x0 - alpha * g;
  Eigen::VectorXd P = xref.cwiseMin(upper()).cwiseMax(lower());
  Eigen::VectorXd d = P - x0;

  // Backtracing Armijo line search for step size
  // https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
  double fref = f(x0);
  while (f(x0 + alpha * d) > (fref + alpha * m_options.delta * g.dot(d))) {
    alpha *= m_options.eta;
  }

  // Update x
  return x0 + alpha * d;
}

}  // namespace sia
