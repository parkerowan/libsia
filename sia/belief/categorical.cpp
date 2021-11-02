/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/categorical.h"
#include "sia/belief/uniform.h"
#include "sia/common/exception.h"

#include <glog/logging.h>
#include <cmath>

namespace sia {

#define SMALL_NUMBER 1e-6

Categorical::Categorical(std::size_t dimension)
    : Distribution(Generator::instance()) {
  SIA_EXCEPTION(dimension >= 1, "Categorical distribution requires dim >= 1");
  setProbs(Eigen::VectorXd::Ones(dimension) / double(dimension));
}

Categorical::Categorical(const Eigen::VectorXd& probs)
    : Categorical(probs.size()) {
  setProbs(probs);
}

std::size_t Categorical::dimension() const {
  return m_probs.size();
}

const Eigen::VectorXd Categorical::sample() {
  std::size_t category = sampleInverseCdf();
  return oneHot(category);
}

double Categorical::logProb(const Eigen::VectorXd& x) const {
  SIA_EXCEPTION(std::size_t(x.size()) == dimension(),
                "Categorical distribution one hot vectors = dim");
  SIA_EXCEPTION((abs(x.sum() - 1.0)) <= SMALL_NUMBER,
                "Categorical distribution expects sum of x = 1");
  return log(m_probs.cwiseProduct(x).sum());
}

const Eigen::VectorXd Categorical::mean() const {
  return m_probs;
}

const Eigen::VectorXd Categorical::mode() const {
  int category;
  m_probs.maxCoeff(&category);
  return oneHot(category);
}

const Eigen::MatrixXd Categorical::covariance() const {
  std::size_t n = dimension();
  Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(n, n);
  cov = -m_probs * m_probs.transpose();
  cov.diagonal() = m_probs.array() * (1.0 - m_probs.array());
  return cov;
}

const Eigen::VectorXd Categorical::vectorize() const {
  Eigen::VectorXd data = m_probs;
  return data;
}

bool Categorical::devectorize(const Eigen::VectorXd& data) {
  std::size_t n = dimension();
  std::size_t d = data.size();
  if (d != n) {
    LOG(WARNING) << "Devectorization failed, expected vector size " << n
                 << ", received " << d;
    return false;
  }
  setProbs(data);
  return true;
}

std::size_t Categorical::classify() const {
  int i;
  mean().maxCoeff(&i);
  return i;
}

const Eigen::VectorXd& Categorical::probs() const {
  return m_probs;
}

void Categorical::setProbs(const Eigen::VectorXd& probs) {
  SIA_EXCEPTION((abs(probs.sum() - 1.0)) <= SMALL_NUMBER,
                "Categorical distribution expects sum of probs = 1");
  m_probs = probs;
}

Eigen::VectorXd Categorical::oneHot(std::size_t category) const {
  SIA_EXCEPTION(category < dimension(),
                "Categorical distribution expects category index to be < dim");
  Eigen::VectorXd probs = Eigen::VectorXd::Zero(dimension());
  probs(category) = 1.0;
  return probs;
}

Eigen::MatrixXd Categorical::oneHot(const Eigen::VectorXi& category) const {
  SIA_EXCEPTION(std::size_t(category.maxCoeff()) < dimension(),
                "Categorical distribution expects category index to be < dim");
  Eigen::MatrixXd probs = Eigen::MatrixXd::Zero(dimension(), category.size());
  for (int i = 0; i < category.size(); ++i) {
    probs.col(i) = oneHot(category(i));
  }
  return probs;
}

std::size_t Categorical::category(const Eigen::VectorXd& probs) const {
  int category;
  probs.maxCoeff(&category);
  return category;
}

std::size_t Categorical::sampleInverseCdf() const {
  Uniform g(0, 1);
  double u = g.sample()(0);
  double c = m_probs(0);
  std::size_t i = 0;
  while (u > c) {
    i++;
    c += m_probs(i);
  }
  return i;
}

}  // namespace sia
