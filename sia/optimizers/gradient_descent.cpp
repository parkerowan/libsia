/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/gradient_descent.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

namespace sia {

GradientDescent::GradientDescent(const Eigen::VectorXd& lower,
                                 const Eigen::VectorXd& upper,
                                 std::size_t n_starts,
                                 std::size_t max_iter,
                                 double tol,
                                 double eta,
                                 double delta)
    : m_sampler(lower, upper) {
  setNstarts(n_starts);
  setMaxIter(max_iter);
  setTol(tol);
  setEta(eta);
  setDelta(delta);
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

void GradientDescent::setNstarts(std::size_t n_starts) {
  SIA_EXCEPTION(n_starts > 0, "GradientDescent n_starts expected to be > 0");
  m_n_starts = n_starts;
}

std::size_t GradientDescent::nstarts() const {
  return m_n_starts;
}

void GradientDescent::setMaxIter(std::size_t max_iter) {
  SIA_EXCEPTION(max_iter > 0, "GradientDescent max_iter expected to be > 0");
  m_max_iter = max_iter;
}

std::size_t GradientDescent::maxIter() const {
  return m_max_iter;
}

void GradientDescent::setTol(double tol) {
  SIA_EXCEPTION(tol > 0, "GradientDescent tol expected to be > 0");
  m_tol = tol;
}

double GradientDescent::tol() const {
  return m_tol;
}

void GradientDescent::setEta(double eta) {
  SIA_EXCEPTION(eta > 0, "GradientDescent eta expected to be > 0");
  SIA_EXCEPTION(eta < 1, "GradientDescent eta expected to be < 1");
  m_eta = eta;
}

double GradientDescent::eta() const {
  return m_eta;
}

void GradientDescent::setDelta(double delta) {
  SIA_EXCEPTION(delta > 0, "GradientDescent delta expected to be > 0");
  SIA_EXCEPTION(delta < 1, "GradientDescent delta expected to be < 1");
  m_delta = delta;
}

double GradientDescent::delta() const {
  return m_delta;
}

Eigen::VectorXd GradientDescent::minimize(
    std::function<double(const Eigen::VectorXd&)> f) {
  std::vector<Eigen::VectorXd> x0 = m_sampler.samples(m_n_starts);
  return minimize(f, x0);
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
  std::size_t i = 0;
  do {
    // Compute gradient
    Eigen::VectorXd g = dfdx(f, x);
    fref = f(x);

    // Compute projection P and search direction d
    double alpha = 1;
    Eigen::VectorXd xref = x - alpha * g;
    Eigen::VectorXd P = xref.cwiseMin(upper()).cwiseMax(lower());
    Eigen::VectorXd d = P - x;

    // Backtracing Armijo line search for step size
    // https://www.cs.cmu.edu/~ggordon/10725-F12/slides/05-gd-revisited.pdf
    while (f(x + alpha * d) > (fref + alpha * m_delta * g.dot(d))) {
      alpha *= m_eta;
    }

    // Update x
    x += alpha * d;
    i++;
  } while ((abs(fref - f(x)) > m_tol) && (i < m_max_iter));

  if (i >= m_max_iter) {
    LOG(WARNING) << "GradientDescent reached max_iter=" << m_max_iter;
  }

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
