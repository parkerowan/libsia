/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/uniform.h"
#include "sia/common/exception.h"

#include <glog/logging.h>

namespace sia {

Uniform::Uniform(std::size_t dimension)
    : Distribution(Generator::instance()),
      m_lower(Eigen::VectorXd::Zero(dimension)),
      m_upper(Eigen::VectorXd::Ones(dimension)) {}

Uniform::Uniform(double lower, double upper) : Uniform(1) {
  m_lower << lower;
  m_upper << upper;
}

Uniform::Uniform(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper)
    : Uniform(1) {
  checkDimensions(lower, upper);
  m_lower = lower;
  m_upper = upper;
}

std::size_t Uniform::dimension() const {
  return m_lower.size();
}

const Eigen::VectorXd Uniform::sample() {
  // Sample from standard uniform
  Eigen::VectorXd x = Eigen::VectorXd::Zero(dimension());
  for (std::size_t i = 0; i < dimension(); ++i) {
    x(i) = m_standard_uniform(m_rng);
  }

  // Project x onto the defined distribution using affine transform
  return (upper() - lower()).cwiseProduct(x) + lower();
}

double Uniform::logProb(const Eigen::VectorXd& x) const {
  // Check if x is in the domain of the distribution
  for (std::size_t i = 0; i < dimension(); i++) {
    if (x(i) > upper()(i) || (x(i) < lower()(i))) {
      return -INFINITY;
    }
  }

  // Compute the log prob (probability = 1 / sum(b-a))
  Eigen::VectorXd e = upper() - lower();
  return -e.array().log().sum();
}

const Eigen::VectorXd Uniform::mean() const {
  return (upper() + lower()) / 2.0;
}

const Eigen::VectorXd Uniform::mode() const {
  return (upper() + lower()) / 2.0;
}

const Eigen::MatrixXd Uniform::covariance() const {
  return ((upper() - lower()).array().pow(2) / 12.0).matrix().asDiagonal();
}

const Eigen::VectorXd Uniform::vectorize() const {
  std::size_t n = dimension();
  Eigen::VectorXd data = Eigen::VectorXd::Zero(2 * n);
  data.head(n) = lower();
  data.tail(n) = upper();
  return data;
}

bool Uniform::devectorize(const Eigen::VectorXd& data) {
  std::size_t n = dimension();
  std::size_t d = data.size();
  if (d != 2 * n) {
    LOG(WARNING) << "Devectorization failed, expected vector size " << 2 * n
                 << ", received " << d;
    return false;
  }
  setLower(data.head(n));
  setUpper(data.tail(n));
  return true;
}

const Eigen::VectorXd& Uniform::lower() const {
  return m_lower;
}

const Eigen::VectorXd& Uniform::upper() const {
  return m_upper;
}

void Uniform::setLower(const Eigen::VectorXd& lower) {
  checkDimensions(lower, m_upper);
  m_lower = lower;
}

void Uniform::setUpper(const Eigen::VectorXd& upper) {
  checkDimensions(m_lower, upper);
  m_upper = upper;
}

void Uniform::checkDimensions(const Eigen::VectorXd& lower,
                              const Eigen::VectorXd& upper) const {
  std::size_t n = lower.size();
  std::size_t m = upper.size();
  bool r = n == m;
  SIA_EXCEPTION(r, "Inconsistent dimensions between lower and upper");
}

}  // namespace sia
