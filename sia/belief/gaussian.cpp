/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gaussian.h"
#include "sia/math/math.h"

#include <glog/logging.h>

namespace sia {

Gaussian::Gaussian(std::size_t dimension)
    : Distribution(Generator::instance()),
      m_mu(Eigen::VectorXd::Zero(dimension)),
      m_sigma(Eigen::MatrixXd::Identity(dimension, dimension)) {
  cacheSigmaChol();
}

Gaussian::Gaussian(double mean, double covariance)
    : Distribution(Generator::instance()),
      m_mu(Eigen::VectorXd::Zero(1)),
      m_sigma(Eigen::MatrixXd::Identity(1, 1)) {
  m_mu << mean;
  m_sigma << covariance;
  cacheSigmaChol();
}

Gaussian::Gaussian(const Eigen::VectorXd& mean,
                   const Eigen::MatrixXd& covariance)
    : Distribution(Generator::instance()) {
  // TODO: Raise exception
  if (checkDimensions(mean, covariance)) {
    m_mu = mean;
    m_sigma = covariance;
    cacheSigmaChol();
  }
}

std::size_t Gaussian::dimension() const {
  return m_mu.size();
}

const Eigen::VectorXd Gaussian::sample() {
  // Sample from standard normal
  Eigen::VectorXd x = Eigen::VectorXd::Zero(dimension());
  for (std::size_t i = 0; i < dimension(); ++i) {
    x(i) = m_standard_normal(m_rng);
  }

  // Project x onto the defined distribution using the Cholesky of sigma
  return m_cached_sigma_L * x + mean();
}

double Gaussian::logProb(const Eigen::VectorXd& x) const {
  // -0.5 * (rank * log_2_pi + log_det + pow(mahalanobix(x), 2));
  return maxLogProb() - 0.5 * pow(mahalanobis(x), 2);
}

const Eigen::VectorXd Gaussian::mean() const {
  return m_mu;
}

const Eigen::VectorXd Gaussian::mode() const {
  return m_mu;
}

const Eigen::MatrixXd Gaussian::covariance() const {
  return m_sigma;
}

const Eigen::VectorXd Gaussian::vectorize() const {
  std::size_t n = dimension();
  Eigen::VectorXd data = Eigen::VectorXd::Zero(n * (n + 1));
  data.head(n) = m_mu;
  data.tail(n * n) = Eigen::VectorXd::Map(m_sigma.data(), n * n);
  return data;
}

bool Gaussian::devectorize(const Eigen::VectorXd& data) {
  std::size_t n = dimension();
  std::size_t d = data.size();
  if (d != n * (n + 1)) {
    LOG(WARNING) << "Devectorization failed, expected vector size "
                 << n * (n + 1) << ", received " << d;
    return false;
  }
  setMean(data.head(n));
  return setCovariance(Eigen::MatrixXd::Map(data.tail(n * n).data(), n, n));
}

bool Gaussian::setMean(const Eigen::VectorXd& mean) {
  if (not checkDimensions(mean, m_sigma)) {
    return false;
  }
  m_mu = mean;
  return true;
}

bool Gaussian::setCovariance(const Eigen::MatrixXd& covariance) {
  if (not checkDimensions(m_mu, covariance)) {
    return false;
  }
  m_sigma = covariance;
  return cacheSigmaChol();
}

double Gaussian::mahalanobis(const Eigen::VectorXd& x) const {
  const Eigen::VectorXd y = m_cached_sigma_L_inv * (x - m_mu);
  return sqrt(y.dot(y));
}

double Gaussian::maxLogProb() const {
  double rank = static_cast<double>(dimension());
  double log_2_pi = log(2 * M_PI);
  double log_det = 2 * m_cached_sigma_L.diagonal().array().log().sum();
  return -0.5 * (rank * log_2_pi + log_det);
}

bool Gaussian::checkDimensions(const Eigen::VectorXd& mu,
                               const Eigen::MatrixXd& sigma) const {
  std::size_t n = mu.size();
  std::size_t m = sigma.rows();
  std::size_t p = sigma.cols();
  bool result = (n == m) && (n == p);
  if (not result) {
    LOG(WARNING) << "Gaussian dimensions not compatible, mu(n=" << n
                 << "), sigma(m=" << m << ",p=" << p << ")";
  }
  return result;
}

bool Gaussian::cacheSigmaChol() {
  // TODO: Raise exception if LDLT fails, or inverse fails
  bool result = true;
  if (not llt(m_sigma, m_cached_sigma_L)) {
    LOG(WARNING) << "Failed to compute LDLT of covariance matrix";
    result = false;
  }
  m_cached_sigma_L_inv = m_cached_sigma_L.triangularView<Eigen::Lower>().solve(
      Eigen::MatrixXd::Identity(dimension(), dimension()));
  return result;
}

}  // namespace sia
