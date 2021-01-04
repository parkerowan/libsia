/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/uniform.h"

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
  if (checkDimensions(lower, upper)) {
    m_lower = lower;
    m_upper = upper;
  }
}

std::size_t Uniform::dimension() const {
  return m_lower.size();
}

const Eigen::VectorXd Uniform::sample() {
  // Sample from standard uniform
  Eigen::VectorXd x = Eigen::VectorXd::Zero(dimension());
  for (std::size_t i = 0; i < dimension(); ++i) {
    x(i) = m_standard_uniform(m_generator);
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

const Eigen::VectorXd& Uniform::lower() const {
  return m_lower;
}

const Eigen::VectorXd& Uniform::upper() const {
  return m_upper;
}

bool Uniform::setLower(const Eigen::VectorXd& lower) {
  if (not checkDimensions(lower, m_upper)) {
    return false;
  }
  m_lower = lower;
  return true;
}

bool Uniform::setUpper(const Eigen::VectorXd& upper) {
  if (not checkDimensions(m_lower, upper)) {
    return false;
  }
  m_upper = upper;
  return true;
}

bool Uniform::checkDimensions(const Eigen::VectorXd& lower,
                              const Eigen::VectorXd& upper) const {
  std::size_t n = lower.size();
  std::size_t m = upper.size();
  bool result = n == m;
  if (not result) {
    LOG(WARNING) << "Uniform dimensions not compatible, lower(n=" << n
                 << "), upper(m=" << m << ")";
  }
  return result;
}

}  // namespace sia
