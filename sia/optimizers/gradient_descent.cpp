/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/gradient_descent.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

namespace sia {

GradientDescent::GradientDescent(const Eigen::VectorXd& lower,
                                 const Eigen::VectorXd& upper,
                                 double tol,
                                 double eta,
                                 double delta)
    : m_lower(lower), m_upper(upper), m_tol(tol), m_eta(eta), m_delta(delta) {
  SIA_EXCEPTION(lower.size() == upper.size(),
                "Lower and upper bounds expected to be equal size");
  SIA_EXCEPTION(tol > 0, "Gradient descent tol expected to be > 0");
  SIA_EXCEPTION(eta > 0, "Gradient descent eta expected to be > 0");
  SIA_EXCEPTION(eta < 1, "Gradient descent eta expected to be < 1");
  SIA_EXCEPTION(delta > 0, "Gradient descent delta expected to be > 0");
  SIA_EXCEPTION(delta < 1, "Gradient descent delta expected to be < 1");
}

std::size_t GradientDescent::dimension() const {
  return m_lower.size();
}

Eigen::VectorXd GradientDescent::minimize(
    std::function<double(const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x0) const {
  SIA_EXCEPTION(dimension() == (std::size_t)x0.size(),
                "x0 expected to be same dimension as bounds");

  // Monotone Gradient Projection Algorithm
  // See: https://www.math.lsu.edu/~hozhang/papers/63522-gg.pdf
  Eigen::VectorXd x = x0;
  double fref = f(x);
  do {
    // Compute gradient
    Eigen::VectorXd g = dfdx(f, x);
    fref = f(x);

    // Compute projection P and search direction d
    double alpha = 1;
    Eigen::VectorXd xref = x - alpha * g;
    Eigen::VectorXd P = xref.cwiseMin(m_upper).cwiseMax(m_lower);
    Eigen::VectorXd d = P - x;

    // Backtracing Armijo line search for step size
    // https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
    while (f(x + alpha * d) > (fref + alpha * m_delta * g.dot(d))) {
      alpha *= m_eta;
    }

    // Update x
    x += alpha * d;
  } while (abs(fref - f(x)) > m_tol);

  return x;
}

Eigen::VectorXd GradientDescent::minimize(
    std::function<double(const Eigen::VectorXd&)> f,
    const std::vector<Eigen::VectorXd>& x0) const {
  std::vector<Eigen::VectorXd> x_sol;
  std::vector<double> f_sol;
  for (const auto& x : x0) {
    Eigen::VectorXd x_opt = minimize(f, x);
    x_sol.emplace_back(x_opt);
    f_sol.emplace_back(f(x_opt));
  }
  Eigen::VectorXd f_x = Eigen::Map<Eigen::VectorXd>(f_sol.data(), f_sol.size());
  int imin;
  f_x.minCoeff(&imin);
  return x_sol[imin];
}

}  // namespace sia
