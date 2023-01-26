/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/gradient_descent.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

namespace sia {

GradientDescent::GradientDescent(const Eigen::VectorXd& lower,
                                 const Eigen::VectorXd& upper,
                                 const GradientDescent::Options& options)
    : m_sampler(lower, upper) {
  setOptions(options);
}

std::size_t GradientDescent::dimension() const {
  return m_sampler.dimension();
}

const Eigen::VectorXd& GradientDescent::lower() const {
  return m_sampler.lower();
}

const Eigen::VectorXd& GradientDescent::upper() const {
  return m_sampler.upper();
}

const GradientDescent::Options& GradientDescent::options() const {
  return m_options;
}

void GradientDescent::setOptions(const GradientDescent::Options& options) {
  SIA_EXCEPTION(options.n_starts > 0,
                "GradientDescent n_starts expected to be > 0");
  SIA_EXCEPTION(options.max_iter > 0,
                "GradientDescent max_iter expected to be > 0");
  SIA_EXCEPTION(options.tol > 0, "GradientDescent tol expected to be > 0");
  SIA_EXCEPTION(options.eta > 0, "GradientDescent eta expected to be > 0");
  SIA_EXCEPTION(options.eta < 1, "GradientDescent eta expected to be < 1");
  SIA_EXCEPTION(options.delta > 0, "GradientDescent delta expected to be > 0");
  SIA_EXCEPTION(options.delta < 1, "GradientDescent delta expected to be < 1");
  m_options = options;
}

Eigen::VectorXd GradientDescent::minimize(
    GradientDescent::Cost f,
    const Eigen::VectorXd& x0,
    GradientDescent::Jacobian jacobian) const {
  SIA_EXCEPTION(dimension() == (std::size_t)x0.size(),
                "x0 expected to be same dimension as bounds");

  // Monotone Gradient Projection Algorithm
  // See: https://www.math.lsu.edu/~hozhang/papers/63522-gg.pdf
  Eigen::VectorXd x = x0;
  std::size_t i = 0;
  double fref = f(x);
  double fref_prev = fref;
  do {
    // Compute gradient
    Eigen::VectorXd g = (jacobian != nullptr) ? jacobian(x) : dfdx(f, x);

    // Compute projection P and search direction d
    double alpha = 1;
    Eigen::VectorXd xref = x - alpha * g;
    Eigen::VectorXd P = xref.cwiseMin(upper()).cwiseMax(lower());
    Eigen::VectorXd d = P - x;

    // Backtracing Armijo line search for step size
    // https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
    while (f(x + alpha * d) > (fref + alpha * m_options.delta * g.dot(d))) {
      alpha *= m_options.eta;
    }

    // Update x
    fref_prev = fref;
    x += alpha * d;
    fref = f(x);
    i++;
  } while ((abs(fref_prev - fref) > m_options.tol) && (i < m_options.max_iter));

  if (i >= m_options.max_iter) {
    SIA_WARN("GradientDescent reached max_iter=" << m_options.max_iter);
  }

  return x;
}

Eigen::VectorXd GradientDescent::minimize(GradientDescent::Cost f,
                                          GradientDescent::Jacobian jacobian) {
  std::vector<Eigen::VectorXd> x0 = m_sampler.samples(m_options.n_starts);
  return minimize(f, x0, jacobian);
}

Eigen::VectorXd GradientDescent::minimize(
    GradientDescent::Cost f,
    const std::vector<Eigen::VectorXd>& x0,
    GradientDescent::Jacobian jacobian) const {
  std::vector<Eigen::VectorXd> x_sol;
  std::vector<double> f_sol;
  for (const auto& x : x0) {
    Eigen::VectorXd x_opt = minimize(f, x, jacobian);
    x_sol.emplace_back(x_opt);
    f_sol.emplace_back(f(x_opt));
  }
  Eigen::VectorXd f_x = Eigen::Map<Eigen::VectorXd>(f_sol.data(), f_sol.size());
  int imin;
  f_x.minCoeff(&imin);
  return x_sol[imin];
}

}  // namespace sia
