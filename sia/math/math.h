/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <Eigen/Dense>
#include <limits>

namespace sia {

/// Default tolerance used for singular values
const static double DEFAULT_SINGULAR_TOLERANCE = 1e-8;

/// Default step size for numerical derivative
const static double NUMERICAL_DERIVATIVE_STEP = 1e-6;

/// Slices the input vector by the given indices
const Eigen::VectorXd slice(const Eigen::VectorXd& x,
                            const std::vector<std::size_t>& indices);

/// Slices the input matrix by the given indices
const Eigen::MatrixXd slice(const Eigen::MatrixXd& X,
                            const std::vector<std::size_t>& rows,
                            const std::vector<std::size_t>& cols);

/// Computes the SVD of matrix A.  If the singular values are less than the
/// tolerance, returns false.  Otherwise returns true.
bool svd(const Eigen::MatrixXd& A,
         Eigen::MatrixXd& U,
         Eigen::VectorXd& S,
         Eigen::MatrixXd& V,
         double tolerance = DEFAULT_SINGULAR_TOLERANCE);

/// Computes the SVD-based inverse of matrix A.  If the singular values are
/// less than the tolerance, returns false.  Otherwise returns true.
bool svdInverse(const Eigen::MatrixXd& A,
                Eigen::MatrixXd& Ainv,
                double tolerance = DEFAULT_SINGULAR_TOLERANCE);

/// 4th order Runge-Kutta integrator where the prototype dynamical system has
/// the form \dot{x} = f(x, u)
const Eigen::VectorXd rk4(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> dynamical_system,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u,
    double dt);

/// Computes a numerical Jacobian df(x) / dx using 1st order central difference
template <typename... Args>
const Eigen::MatrixXd numericalJacobian(
    std::function<const Eigen::VectorXd(const Eigen::VectorXd&, Args...)> f,
    const Eigen::VectorXd& x,
    Args... args);

}  // namespace sia

#include "sia/math/math_impl.h"
