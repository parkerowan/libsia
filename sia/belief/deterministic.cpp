/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/deterministic.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

#include <cmath>

#define SMALL_NUMBER 1e-6

namespace sia {

Deterministic::Deterministic(double value)
    : Distribution(Generator::instance()), m_value(Eigen::VectorXd::Zero(1)) {
  m_value << value;
}

Deterministic::Deterministic(const Eigen::VectorXd& value)
    : Distribution(Generator::instance()), m_value(value) {}

std::size_t Deterministic::dimension() const {
  return m_value.size();
}

const Eigen::VectorXd Deterministic::sample() {
  return m_value;
}

double Deterministic::logProb(const Eigen::VectorXd& x) const {
  if ((x - m_value).lpNorm<Eigen::Infinity>() < SMALL_NUMBER) {
    return 0;
  }
  return -INFINITY;
}

const Eigen::VectorXd Deterministic::mean() const {
  return m_value;
}

const Eigen::VectorXd Deterministic::mode() const {
  return m_value;
}

const Eigen::MatrixXd Deterministic::covariance() const {
  return Eigen::MatrixXd::Zero(dimension(), dimension());
}

const Eigen::VectorXd Deterministic::vectorize() const {
  return m_value;
}

bool Deterministic::devectorize(const Eigen::VectorXd& data) {
  setValue(data);
  return true;
}

void Deterministic::setValue(const Eigen::VectorXd& value) {
  m_value = value;
}

}  // namespace sia
