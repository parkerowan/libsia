/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gaussian.h"

#include <glog/logging.h>

namespace sia {

Gaussian::Gaussian(std::size_t dimension)
    : Distribution(Generator::instance()),
      m_mu(Eigen::VectorXd::Zero(dimension)),
      m_sigma(Eigen::MatrixXd::Identity(dimension, dimension)) {
  cacheSigmaChol();
}

Gaussian::Gaussian(double mean, double covariance) : Gaussian(1) {
  m_mu << mean;
  m_sigma << covariance;
  cacheSigmaChol();
}

Gaussian::Gaussian(const Eigen::VectorXd& mean,
                   const Eigen::MatrixXd& covariance)
    : Gaussian(1) {
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
    x(i) = m_standard_normal(m_generator);
  }

  // Project x onto the defined distribution using the Cholesky of sigma
  return m_cached_sigma_L * x + mean();
}

double Gaussian::logProb(const Eigen::VectorXd& x) const {
  double rank = static_cast<double>(dimension());
  double log_2_pi = log(2 * M_PI);
  double log_det = 2 * m_cached_sigma_L.diagonal().array().log().sum();
  return -0.5 * (rank * log_2_pi + pow(mahalanobis(x), 2) + log_det);
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
  Eigen::LLT<Eigen::MatrixXd> llt_of_sigma(m_sigma);
  m_cached_sigma_L = llt_of_sigma.matrixL();
  m_cached_sigma_L_inv = m_cached_sigma_L.triangularView<Eigen::Lower>().solve(
      Eigen::MatrixXd::Identity(dimension(), dimension()));
  if (llt_of_sigma.info() != Eigen::ComputationInfo::Success) {
    LOG(WARNING) << "LLT transform of covariance matrix failed";
    return false;
  }
  return true;
}

}  // namespace sia
