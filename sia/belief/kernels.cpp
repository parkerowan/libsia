/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/kernels.h"
#include "sia/common/exception.h"

#include <glog/logging.h>

namespace sia {

std::size_t CovarianceFunction::numHyperparameters() const {
  return hyperparameters().size();
}

SquaredExponential::SquaredExponential(double length,
                                       double signal_var,
                                       double noise_var) {
  setHyperparameters(Eigen::Vector3d{length, signal_var, noise_var});
}

Eigen::VectorXd CovarianceFunction::evalVector(const Eigen::MatrixXd& a,
                                               const Eigen::VectorXd& b) {
  std::size_t na = a.cols();
  Eigen::VectorXd k(na);
  for (std::size_t i = 0; i < na; ++i) {
    k(i) = eval(a.col(i), b);
  }
  return k;
}

Eigen::MatrixXd CovarianceFunction::evalMatrix(const Eigen::MatrixXd& a,
                                               const Eigen::MatrixXd& b) {
  std::size_t na = a.cols();
  std::size_t nb = b.cols();
  Eigen::MatrixXd K(na, nb);
  for (std::size_t i = 0; i < nb; ++i) {
    K.col(i) = evalVector(a, b.col(i));
  }
  return K;
}

std::vector<Eigen::MatrixXd> CovarianceFunction::gradTensor(
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b) {
  std::size_t na = a.cols();
  std::size_t nb = b.cols();
  std::size_t np = numHyperparameters();
  std::vector<Eigen::MatrixXd> dK(np, Eigen::MatrixXd(na, nb));
  for (std::size_t i = 0; i < nb; ++i) {
    for (std::size_t j = 0; j < na; ++j) {
      Eigen::VectorXd g = grad(a.col(j), b.col(i));
      for (std::size_t k = 0; k < np; ++k) {
        dK[k](j, i) = g(k);
      }
    }
  }
  return dK;
}

CovarianceFunction* CovarianceFunction::create(Type type) {
  switch (type) {
    case SQUARED_EXPONENTIAL:
      return new SquaredExponential();
    default:
      SIA_EXCEPTION(false,
                    "CovarianceFunction::Type encountered unsupported type");
  }
}

double SquaredExponential::eval(const Eigen::VectorXd& a,
                                const Eigen::VectorXd& b) const {
  const Eigen::VectorXd e = a - b;
  double delta = a.isApprox(b) ? m_noise_var : 0.0;
  return m_signal_var * exp(-e.dot(e) / pow(m_length, 2) / 2) + delta;
}

Eigen::VectorXd SquaredExponential::grad(const Eigen::VectorXd& a,
                                         const Eigen::VectorXd& b) const {
  const Eigen::VectorXd e = (a - b) / m_length;
  double mahal2 = e.dot(e);
  double knorm = exp(-mahal2 / 2);
  double dkdl = m_signal_var * knorm * mahal2 / m_length;
  double dkds = knorm;
  double dkdn = a.isApprox(b) ? 1.0 : 0.0;
  Eigen::VectorXd g(3);
  g << dkdl, dkds, dkdn;
  return g;
}

Eigen::VectorXd SquaredExponential::hyperparameters() const {
  Eigen::VectorXd p(3);
  p << m_length, m_signal_var, m_noise_var;
  return p;
}

void SquaredExponential::setHyperparameters(const Eigen::VectorXd& p) {
  SIA_EXCEPTION(numHyperparameters() == 3,
                "SquaredExponential hyperparameter dim expexted to be 3");
  m_length = p(0);
  m_signal_var = p(1);
  m_noise_var = p(2);
  SIA_EXCEPTION(m_length > 0,
                "CovarianceFunction length scale is expexted > 0");
  SIA_EXCEPTION(m_signal_var > 0,
                "CovarianceFunction signal variance is expexted > 0");
  SIA_EXCEPTION(m_noise_var > 0,
                "CovarianceFunction noise variance is expexted > 0");
}

}  // namespace sia
