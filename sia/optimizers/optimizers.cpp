/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/optimizers.h"
#include <cmath>
#include "sia/common/exception.h"

namespace sia {

Optimizer::Optimizer(std::size_t dimension, double ftol, std::size_t max_iter)
    : m_dimension(dimension), m_ftol(ftol), m_max_iter(max_iter) {
  SIA_THROW_IF_NOT(m_dimension > 0, "Optimizer dimension expected to be > 0");
  SIA_THROW_IF_NOT(m_ftol > 0, "Optimizer ftol expected to be > 0");
  SIA_THROW_IF_NOT(m_max_iter > 0, "Optimizer max_iter expected to be > 0");
}

std::size_t Optimizer::dimension() const {
  return m_dimension;
}

Eigen::VectorXd Optimizer::minimize(Optimizer::Cost f,
                                    const Eigen::VectorXd& x0,
                                    Optimizer::Gradient gradient,
                                    Optimizer::Convergence convergence) {
  SIA_THROW_IF_NOT(x0.size() == int(dimension()),
                   "Optimizer x0 size expected to match dimension");

  reset();

  // TODO: Figure out the custom convergence.  Specifically, we want to avoid
  // even checking on the condition if it is unused.  Perhaps it's just some
  // clever logic in the while statement below.
  // bool custom_converged = false;

  // Perform the minimization
  Eigen::VectorXd x = x0;
  std::size_t i = 0;
  double fref = f(x);
  double fref_prev = fref;
  do {
    // Run a single step of the optimization
    x = step(f, x, gradient);

    // Update iterations to check for convergence
    fref_prev = fref;
    fref = f(x);
    i++;
    // custom_converged = (convergence != nullptr) ? convergence(x) : true;
  } while ((abs(fref_prev - fref) > m_ftol) && (i < m_max_iter));

  SIA_THROW_IF_NOT(i < m_max_iter, "Optimizer failed to converge in max_steps");

  return x;
}

Eigen::VectorXd Optimizer::minimize(Optimizer::Cost f,
                                    const std::vector<Eigen::VectorXd>& x0,
                                    Optimizer::Gradient gradient,
                                    Optimizer::Convergence convergence) {
  std::vector<Eigen::VectorXd> x_sol;
  std::vector<double> f_sol;
  for (const auto& x : x0) {
    try {
      Eigen::VectorXd x_opt = minimize(f, x, gradient, convergence);
      x_sol.push_back(x_opt);
      f_sol.push_back(f(x_opt));
    } catch (...) {
    }
  }

  SIA_THROW_IF(x_sol.empty(),
               "Optimizer failed to find a solution from multiple seeds");

  Eigen::VectorXd f_x = Eigen::Map<Eigen::VectorXd>(f_sol.data(), f_sol.size());
  int imin;
  f_x.minCoeff(&imin);
  return x_sol[imin];
}

}  // namespace sia
