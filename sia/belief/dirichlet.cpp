/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/dirichlet.h"
#include "sia/common/exception.h"

#include <glog/logging.h>
#include <cmath>

namespace sia {

#define SMALL_NUMBER 1e-6
#define VERY_SMALL_NUMBER 1e-16

Dirichlet::Dirichlet(std::size_t dimension)
    : Distribution(Generator::instance()) {
  SIA_EXCEPTION(dimension >= 2, "Dirichlet distribution requires dim >= 2");
  setAlpha(Eigen::VectorXd::Ones(dimension));
}

Dirichlet::Dirichlet(double alpha, double beta) : Dirichlet(2) {
  setAlpha(Eigen::Vector2d{alpha, beta});
}

Dirichlet::Dirichlet(const Eigen::VectorXd& alpha) : Dirichlet(alpha.size()) {
  setAlpha(alpha);
}

std::size_t Dirichlet::dimension() const {
  return m_alpha.size();
}

const Eigen::VectorXd Dirichlet::sample() {
  assert(m_gamma_dists.size() == dimension());

  // Sample from K independent gamma distributions
  Eigen::VectorXd x = Eigen::VectorXd::Zero(dimension());
  for (std::size_t i = 0; i < dimension(); ++i) {
    x(i) = m_gamma_dists[i](m_rng);
  }
  return x / x.sum();
}

double Dirichlet::logProb(const Eigen::VectorXd& x) const {
  // Check if x is in the domain of the distribution
  for (std::size_t i = 0; i < dimension(); i++) {
    if ((x(i) > (1.0 + SMALL_NUMBER)) || (x(i) < -SMALL_NUMBER)) {
      LOG(WARNING) << "x(i) is " << x(i) << ", returning -INF";
      return -INFINITY;
    }
  }

  // Check that x sums to 1 and squashes the extrema of x to avoid a degenerate
  // log likelihood
  Eigen::VectorXd xnorm = normalizeInput(x);

  // See: Sec 3 http://jonathan-huang.org/research/dirichlet/dirichlet.pdf
  double a = log(tgamma(m_alpha.sum()));
  double b = 0;
  for (std::size_t i = 0; i < dimension(); ++i) {
    b -= log(tgamma(m_alpha(i)));
  }
  double c = ((m_alpha.array() - 1.0) * xnorm.array().log()).sum();
  return a + b + c;
}

const Eigen::VectorXd Dirichlet::mean() const {
  return m_alpha / m_alpha.sum();
}

const Eigen::VectorXd Dirichlet::mode() const {
  double min_alpha = m_alpha.minCoeff();
  if (min_alpha <= 1.0) {
    LOG(WARNING) << "Mode is invalid for alpha concentrations <= 1";
  }
  return (m_alpha.array() - 1.0) / (m_alpha.sum() - double(m_alpha.size()));
}

const Eigen::MatrixXd Dirichlet::covariance() const {
  double a0 = m_alpha.sum();
  double den = pow(a0, 2) * (a0 + 1);
  Eigen::MatrixXd cov = -m_alpha * m_alpha.transpose() / den;
  cov.diagonal() = m_alpha.array() * (a0 - m_alpha.array()) / den;
  return cov;
}

const Eigen::VectorXd Dirichlet::vectorize() const {
  Eigen::VectorXd data = m_alpha;
  return data;
}

bool Dirichlet::devectorize(const Eigen::VectorXd& data) {
  std::size_t n = dimension();
  std::size_t d = data.size();
  if (d != n) {
    LOG(WARNING) << "Devectorization failed, expected vector size " << n
                 << ", received " << d;
    return false;
  }
  setAlpha(data);
  return true;
}

Categorical Dirichlet::categorical() const {
  return Categorical(mean());
}

std::size_t Dirichlet::classify() const {
  return categorical().classify();
}

const Eigen::VectorXd& Dirichlet::alpha() const {
  return m_alpha;
}

void Dirichlet::setAlpha(const Eigen::VectorXd& alpha) {
  m_alpha = alpha;
  m_gamma_dists.clear();
  for (std::size_t i = 0; i < dimension(); ++i) {
    m_gamma_dists.emplace_back(std::gamma_distribution<double>(alpha(i), 1));
  }
}

Eigen::VectorXd Dirichlet::normalizeInput(const Eigen::VectorXd& x) const {
  // Apply a small correction to avoid the degenerate case at x = 0 where the
  // log likelihood goes to infinity
  Eigen::VectorXd xnorm =
      (1 - 2 * VERY_SMALL_NUMBER) * x.array() + VERY_SMALL_NUMBER;
  if ((abs(x.sum() - 1.0)) > SMALL_NUMBER) {
    LOG(WARNING) << "Sum of x is expected to be 1, applying normalization";
    xnorm /= xnorm.sum();
  }
  return xnorm;
}

}  // namespace sia
