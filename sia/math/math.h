/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
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

/// Stacks two matrices on top of each other, must have same num cols
Eigen::MatrixXd stack(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

/// Construct a vector of indices from [m to n)
std::vector<std::size_t> indices(std::size_t m, std::size_t n);

/// Concatenate vectors of data
template <typename T>
std::vector<T> concat(const std::vector<T>& a, const std::vector<T>& b);

/// Computes the LLT decomposition of positive definite matrix A, returns true
/// on success.
bool llt(const Eigen::MatrixXd& A, Eigen::MatrixXd& L);

/// Computes the factorization M M' = A for positive semi-definite matrix A
/// using LDLT decomposition.  Returns true on success.
bool ldltSqrt(const Eigen::MatrixXd& A, Eigen::MatrixXd& M);

/// Computes the SVD of matrix A.  If the singular values are less than the
/// tolerance, returns false.  Otherwise returns true.
bool svd(const Eigen::MatrixXd& A,
         Eigen::MatrixXd& U,
         Eigen::VectorXd& S,
         Eigen::MatrixXd& V,
         double tolerance = DEFAULT_SINGULAR_TOLERANCE);

/// Computes the matrix inverse from SVD matrices
const Eigen::MatrixXd svdInverse(const Eigen::MatrixXd& U,
                                 const Eigen::VectorXd& S,
                                 const Eigen::MatrixXd& V);

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

/// Compute gradient df(x)/dx via central difference for scalar f(x)
const Eigen::VectorXd dfdx(std::function<double(const Eigen::VectorXd&)> f,
                           const Eigen::VectorXd& x);

/// Compute gradient df(x, u)/dx via central difference for scalar f(x, u)
const Eigen::VectorXd dfdx(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u);

/// Compute gradient df(x, u)/dx via central difference for vector f(x, u)
const Eigen::MatrixXd dfdx(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u);

/// Compute gradient df(x, u/du via central difference for scalar f(x, u)
const Eigen::VectorXd dfdu(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u);

/// Compute gradient df(x, u)/du via central difference for vector f(x, u)
const Eigen::MatrixXd dfdu(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u);

/// Compute Hessian d2f(x)/dx2 via central difference for scalar f(x)
const Eigen::MatrixXd d2fdxx(std::function<double(const Eigen::VectorXd&)> f,
                             const Eigen::VectorXd& x);

/// Compute Hessian d2f(x, u)/dx2 via central difference for scalar f(x, u)
const Eigen::MatrixXd d2fdxx(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u);

/// Compute Hessian d2f(x, u)/du2 via central difference for scalar f(x, u)
const Eigen::MatrixXd d2fduu(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u);

/// Compute Hessian d2f(x, u)/dudx via central difference for scalar f(x, u)
const Eigen::MatrixXd d2fdux(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u);

/// Computes a numerical Jacobian df(x) / dx using 1st order central difference
template <typename... Args>
const Eigen::MatrixXd numericalJacobian(
    std::function<const Eigen::VectorXd(const Eigen::VectorXd&, Args...)> f,
    const Eigen::VectorXd& x,
    Args... args);

/// Computes the normalized frobenius norm of a square matrix, such that the
/// norm squared = 1 for the identity matrix.
double frobNormSquared(const Eigen::MatrixXd& A);

/// Estimate the sample covariance using Ledoit and Wolf shrinkage estimator.
/// Each column is a sample.  Assumes that samples are centered (i.e mean is 0).
const Eigen::MatrixXd estimateCovariance(const Eigen::MatrixXd& samples);

}  // namespace sia

#include "sia/math/math_impl.h"
