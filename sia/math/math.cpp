/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/math/math.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"

#include <Eigen/SVD>

namespace sia {

const Eigen::VectorXd slice(const Eigen::VectorXd& x,
                            const std::vector<std::size_t>& indices) {
  Eigen::VectorXd y = Eigen::VectorXd::Zero(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    y(i) = x(indices.at(i));
  }
  return y;
}

const Eigen::MatrixXd slice(const Eigen::MatrixXd& X,
                            const std::vector<std::size_t>& rows,
                            const std::vector<std::size_t>& cols) {
  Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(rows.size(), cols.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < cols.size(); ++j) {
      Y(i, j) = X(rows.at(i), cols.at(j));
    }
  }
  return Y;
}

Eigen::VectorXd replace(const Eigen::VectorXd& x,
                        const Eigen::VectorXd& u,
                        const std::vector<std::size_t>& indices) {
  SIA_THROW_IF_NOT(indices.empty() || (u.size() == int(indices.size())),
                   "Input vector u and indices not consistent");
  Eigen::VectorXd y = x;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    SIA_THROW_IF_NOT(int(indices.at(i)) < x.size(),
                     "Index value exceeds size of input vector x");
    y(indices.at(i)) = u(i);
  }
  return y;
}

bool llt(const Eigen::MatrixXd& A, Eigen::MatrixXd& L) {
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  L = llt.matrixL();
  if (llt.info() != Eigen::ComputationInfo::Success) {
    SIA_WARN("LLT decomposition of matrix A = " << A << " failed with "
                                                << llt.info());
    return false;
  }
  return true;
}

bool ldltSqrt(const Eigen::MatrixXd& A, Eigen::MatrixXd& M) {
  Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.cols());
  const Eigen::MatrixXd P = ldlt.transpositionsP() * I;
  const Eigen::MatrixXd L = ldlt.matrixL();
  const Eigen::VectorXd D = ldlt.vectorD();
  M = P.transpose() * L * D.array().sqrt().matrix().asDiagonal();
  if (ldlt.info() != Eigen::ComputationInfo::Success) {
    SIA_WARN("LDLT decomposition of matrix A = " << A << " failed with "
                                                 << ldlt.info());
    return false;
  }
  return true;
}

bool svd(const Eigen::MatrixXd& A,
         Eigen::MatrixXd& U,
         Eigen::VectorXd& S,
         Eigen::MatrixXd& V,
         double tolerance) {
  bool result = true;

  // Compute the SVD of A
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd& singular_values = svd.singularValues();

  // Check for singular condition
  for (int i = 0; i < singular_values.size(); ++i) {
    if (singular_values(i) <= tolerance) {
      SIA_WARN("Singular value is less than tolerance");
      result = false;
    }
  }

  // Return SVD matrices
  U = svd.matrixU();
  S = singular_values.array();
  V = svd.matrixV();
  return result;
}

const Eigen::MatrixXd svdInverse(const Eigen::MatrixXd& U,
                                 const Eigen::VectorXd& S,
                                 const Eigen::MatrixXd& V) {
  return V * S.array().inverse().matrix().asDiagonal() * U.transpose();
}

bool symmetric(const Eigen::MatrixXd& A) {
  SIA_THROW_IF_NOT(A.rows() == A.cols(),
                   "A matrix must be square to test symmetric");
  return A.isApprox(A.transpose());
}

bool positiveDefinite(const Eigen::MatrixXd& A) {
  if (!symmetric(A)) {
    return false;
  }

  // If A is positive definite then the diagonal elements of D are all positive
  const auto ldlt = A.selfadjointView<Eigen::Upper>().ldlt();
  Eigen::VectorXd d = ldlt.vectorD();
  return d.minCoeff() > 0;
}

bool svdInverse(const Eigen::MatrixXd& A,
                Eigen::MatrixXd& Ainv,
                double tolerance) {
  Eigen::MatrixXd U, V;
  Eigen::VectorXd S;
  bool result = svd(A, U, S, V, tolerance);

  // Compute the generalized inverse using SVD
  Ainv = svdInverse(U, S, V);
  return result;
}

const Eigen::MatrixXd lltSqrt(const Eigen::MatrixXd& A) {
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  return llt.matrixL();
}

const Eigen::VectorXd rk4(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> dynamical_system,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u,
    double dt) {
  Eigen::VectorXd f1 = dynamical_system(x, u);
  Eigen::VectorXd f2 = dynamical_system(x + dt * f1 / 2, u);
  Eigen::VectorXd f3 = dynamical_system(x + dt * f2 / 2, u);
  Eigen::VectorXd f4 = dynamical_system(x + dt * f3, u);
  return x + dt / 6 * (f1 + 2 * f2 + 2 * f3 + f4);
}

const Eigen::VectorXd dfdx(std::function<double(const Eigen::VectorXd&)> f,
                           const Eigen::VectorXd& x) {
  std::size_t n = x.size();
  Eigen::VectorXd df = Eigen::VectorXd::Zero(n);
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
    dx(i) = NUMERICAL_DERIVATIVE_STEP;
    double fp = f(x + dx);
    double fn = f(x - dx);
    df(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  return df;
}

const Eigen::VectorXd dfdx(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u) {
  std::size_t n = x.size();
  Eigen::VectorXd df = Eigen::VectorXd::Zero(n);
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
    dx(i) = NUMERICAL_DERIVATIVE_STEP;
    double fp = f(x + dx, u);
    double fn = f(x - dx, u);
    df(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  return df;
}

const Eigen::MatrixXd dfdx(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u) {
  std::size_t n = x.size();
  Eigen::MatrixXd Df;
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
    dx(i) = NUMERICAL_DERIVATIVE_STEP;
    Eigen::VectorXd fp = f(x + dx, u);
    Eigen::VectorXd fn = f(x - dx, u);
    if (i == 0) {
      Df = Eigen::MatrixXd::Zero(fp.size(), n);
    }
    Df.col(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  return Df;
}

const Eigen::VectorXd dfdu(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u) {
  std::size_t n = u.size();
  Eigen::VectorXd df = Eigen::VectorXd::Zero(n);
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd du = Eigen::VectorXd::Zero(n);
    du(i) = NUMERICAL_DERIVATIVE_STEP;
    double fp = f(x, u + du);
    double fn = f(x, u - du);
    df(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  return df;
}

const Eigen::MatrixXd dfdu(
    std::function<Eigen::VectorXd(const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u) {
  std::size_t n = u.size();
  Eigen::MatrixXd Df;
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd du = Eigen::VectorXd::Zero(n);
    du(i) = NUMERICAL_DERIVATIVE_STEP;
    Eigen::VectorXd fp = f(x, u + du);
    Eigen::VectorXd fn = f(x, u - du);
    if (i == 0) {
      Df = Eigen::MatrixXd::Zero(fp.size(), n);
    }
    Df.col(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  return Df;
}

const Eigen::MatrixXd d2fdxx(std::function<double(const Eigen::VectorXd&)> f,
                             const Eigen::VectorXd& x) {
  std::size_t n = x.size();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
    dx(i) = NUMERICAL_DERIVATIVE_STEP;
    Eigen::VectorXd fp = dfdx(f, x + dx);
    Eigen::VectorXd fn = dfdx(f, x - dx);
    H.col(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  // Ensure that the Hessian is necessarily symmetric
  return (H + H.transpose()) / 2.0;
}

const Eigen::MatrixXd d2fdxx(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u) {
  std::size_t n = x.size();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);
    dx(i) = NUMERICAL_DERIVATIVE_STEP;
    Eigen::VectorXd fp = dfdx(f, x + dx, u);
    Eigen::VectorXd fn = dfdx(f, x - dx, u);
    H.col(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  // Ensure that the Hessian is necessarily symmetric
  return (H + H.transpose()) / 2.0;
}

const Eigen::MatrixXd d2fduu(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u) {
  std::size_t n = u.size();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd du = Eigen::VectorXd::Zero(n);
    du(i) = NUMERICAL_DERIVATIVE_STEP;
    Eigen::VectorXd fp = dfdu(f, x, u + du);
    Eigen::VectorXd fn = dfdu(f, x, u - du);
    H.col(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  // Ensure that the Hessian is necessarily symmetric
  return (H + H.transpose()) / 2.0;
}

const Eigen::MatrixXd d2fdux(
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> f,
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& u) {
  std::size_t m = x.size();
  std::size_t n = u.size();
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, m);
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd du = Eigen::VectorXd::Zero(n);
    du(i) = NUMERICAL_DERIVATIVE_STEP;
    Eigen::VectorXd fp = dfdx(f, x, u + du);
    Eigen::VectorXd fn = dfdx(f, x, u - du);
    H.row(i) = (fp - fn) / 2 / NUMERICAL_DERIVATIVE_STEP;
  }
  // Ensure that the Hessian is necessarily symmetric
  // return (H + H.transpose()) / 2.0;
  return H;
}

double frobNormSquared(const Eigen::MatrixXd& A) {
  assert(A.rows() == A.cols());
  double p = double(A.rows());
  return (A * A.transpose()).trace() / p;
}

const Eigen::MatrixXd estimateCovariance(const Eigen::MatrixXd& samples) {
  // See: "A Well-Conditioned Estimator for Large-dimensional Covariance
  // Matrices", 2004.
  std::size_t n = samples.cols();
  std::size_t d = samples.rows();
  double p = double(d);
  const Eigen::MatrixXd cov = samples * samples.transpose() / double(n);
  double mu = cov.trace() / p;
  const Eigen::MatrixXd muI = mu * Eigen::MatrixXd::Identity(d, d);
  double d2 = frobNormSquared(cov - muI);
  double b2 = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const Eigen::VectorXd& x = samples.col(i);
    b2 += frobNormSquared(x * x.transpose() - cov);
  }
  b2 /= pow(double(n), 2);
  b2 = std::min(d2, b2);
  double shrinkage = b2 / d2;
  return (1 - shrinkage) * cov + shrinkage * muI;
}

}  // namespace sia
