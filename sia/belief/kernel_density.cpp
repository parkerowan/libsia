/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/kernel_density.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/helpers.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

#include <cmath>

namespace sia {

// ----------------------------------------------------------------------------

UniformKernel::UniformKernel(std::size_t dimension) {
  double n = static_cast<double>(dimension);
  m_constant = 1 / pow(2, n);
}

double UniformKernel::evaluate(const Eigen::VectorXd& x) const {
  return (x.array().abs().maxCoeff() <= 1) ? m_constant : 0.0;
}

// ----------------------------------------------------------------------------

GaussianKernel::GaussianKernel(std::size_t dimension) {
  double n = static_cast<double>(dimension);
  m_constant = 1 / sqrt(pow(2 * M_PI, n));
}

double GaussianKernel::evaluate(const Eigen::VectorXd& x) const {
  return m_constant * exp(-x.dot(x) / 2);
}

// ----------------------------------------------------------------------------

EpanechnikovKernel::EpanechnikovKernel(std::size_t dimension) {
  double n = static_cast<double>(dimension);
  double c = pow(M_PI, n / 2) / tgamma(n / 2 + 1);  // Volume of unit n-ball
  m_constant = (n + 2) / (2 * c);
}

double EpanechnikovKernel::evaluate(const Eigen::VectorXd& x) const {
  double xdotx = x.dot(x);
  return (sqrt(xdotx) <= 1) ? m_constant * (1 - xdotx) : 0.0;
}

// ----------------------------------------------------------------------------

KernelDensity::KernelDensity(const Eigen::MatrixXd& values,
                             const Eigen::VectorXd& weights,
                             SmoothingKernel& kernel,
                             KernelDensity::BandwidthMode mode,
                             double bandwidth_scaling)
    : Particles(values, weights),
      m_kernel(kernel),
      m_mode(mode),
      m_bandwidth_scaling(bandwidth_scaling) {
  // If user specified, set the initial bandwidth using Scott's rule
  if (mode == BandwidthMode::USER_SPECIFIED) {
    m_mode = BandwidthMode::SCOTT_RULE;
  }

  autoUpdateBandwidth();
  m_mode = mode;
}

KernelDensity::KernelDensity(const Particles& particles,
                             SmoothingKernel& kernel,
                             BandwidthMode mode,
                             double bandwidth_scaling)
    : KernelDensity(particles.values(),
                    particles.weights(),
                    kernel,
                    mode,
                    bandwidth_scaling) {}

double KernelDensity::probability(const Eigen::VectorXd& x) const {
  double p = 0;
  std::size_t n = m_weights.size();
  for (std::size_t i = 0; i < n; ++i) {
    const auto& c = m_values.col(i);
    double kh = m_kernel.evaluate(m_bandwidth_inv * (x - c));
    p += m_weights(i) * kh;
  }
  return p / m_bandwidth_det;
}

std::size_t KernelDensity::dimension() const {
  return m_values.rows();
}

const Eigen::VectorXd KernelDensity::sample() {
  std::size_t i = sampleInverseCdf();

  // TODO: Should sample from the kernel and project via the bandwidth
  Gaussian g(value(i), m_bandwidth.asDiagonal());
  return g.sample();
}

double KernelDensity::logProb(const Eigen::VectorXd& x) const {
  return log(probability(x));
}

const Eigen::VectorXd KernelDensity::mean() const {
  Eigen::VectorXd p = Eigen::VectorXd::Zero(numParticles());
  for (std::size_t i = 0; i < numParticles(); ++i) {
    p(i) = probability(value(i));
  }
  return values() * p / p.sum();
}

const Eigen::VectorXd KernelDensity::mode() const {
  Eigen::VectorXd p = Eigen::VectorXd::Zero(numParticles());
  for (std::size_t i = 0; i < numParticles(); ++i) {
    p(i) = probability(value(i));
  }
  std::size_t r, c;
  p.maxCoeff(&r, &c);
  return m_values.col(r);
}

const Eigen::MatrixXd KernelDensity::covariance() const {
  const Eigen::MatrixXd e =
      (m_values.array().colwise() - mean().array()).matrix();
  Eigen::VectorXd p = Eigen::VectorXd::Zero(numParticles());
  for (std::size_t i = 0; i < numParticles(); ++i) {
    p(i) = probability(value(i));
  }
  const Eigen::MatrixXd ewe = e * p.asDiagonal() * e.transpose();
  return ewe / (1 - p.array().square().sum());
}

const Eigen::VectorXd KernelDensity::vectorize() const {
  std::size_t n = dimension();
  std::size_t p = numParticles();
  Eigen::VectorXd data = Eigen::VectorXd::Zero(p * (n + 1) + n * n);
  data.head(n * p) = Eigen::VectorXd::Map(m_values.data(), n * p);
  data.segment(n * p, p) = m_weights;
  data.tail(n * n) = Eigen::VectorXd::Map(m_bandwidth.data(), n * n);
  return data;
}

bool KernelDensity::devectorize(const Eigen::VectorXd& data) {
  std::size_t n = dimension();
  std::size_t p = numParticles();
  std::size_t d = data.size();
  if (d != p * (n + 1) + n * n) {
    SIA_WARN("Devectorization failed, expected vector size "
             << p * (n + 1) + n * n << ", received " << d);
    return false;
  }
  setValues(Eigen::MatrixXd::Map(data.head(n * p).data(), n, p));
  setWeights(data.segment(n * p, p));
  setBandwidthMatrix(Eigen::MatrixXd::Map(data.tail(n * n).data(), n, n));
  return true;
}

void KernelDensity::setValues(const Eigen::MatrixXd& values) {
  m_values = values;
  autoUpdateBandwidth();
}

void KernelDensity::setBandwidth(double h) {
  setBandwidth(h * Eigen::VectorXd::Ones(dimension()));
}

void KernelDensity::setBandwidth(const Eigen::VectorXd& h) {
  m_bandwidth = h.asDiagonal();
  m_bandwidth_inv = h.array().inverse().matrix().asDiagonal();
  m_bandwidth_det = h.prod();
  m_bandwidth_scaling = 1.0;
  m_mode = BandwidthMode::USER_SPECIFIED;
}

const Eigen::MatrixXd& KernelDensity::bandwidth() const {
  return m_bandwidth;
}

void KernelDensity::setBandwidthScaling(double scaling) {
  m_bandwidth_scaling = scaling;
  autoUpdateBandwidth();
}

double KernelDensity::getBandwidthScaling() const {
  return m_bandwidth_scaling;
}

void KernelDensity::setBandwidthMode(KernelDensity::BandwidthMode mode) {
  m_mode = mode;
  autoUpdateBandwidth();
}

KernelDensity::BandwidthMode KernelDensity::getBandwidthMode() const {
  return m_mode;
}

SmoothingKernel& KernelDensity::kernel() {
  return m_kernel;
}

void KernelDensity::setBandwidthMatrix(const Eigen::MatrixXd& H) {
  m_bandwidth = H;
  bool r = svdInverse(m_bandwidth, m_bandwidth_inv);
  SIA_EXCEPTION(r, "Failed to compute inverse of bandwidth matrix");
  m_bandwidth_det = m_bandwidth.determinant();
}

void KernelDensity::autoUpdateBandwidth() {
  switch (m_mode) {
    case BandwidthMode::USER_SPECIFIED:
      break;
    case BandwidthMode::SCOTT_RULE:
      const Eigen::MatrixXd Sigma = Particles::covariance();
      bandwidthScottRule(Sigma);
      break;
  }
}

// From eqn. 3.71 Hardle et. al., 2004.
void KernelDensity::bandwidthScottRule(const Eigen::MatrixXd& Sigma) {
  double d = static_cast<double>(Sigma.rows());
  double n = static_cast<double>(numParticles());
  double c = m_bandwidth_scaling * pow(n, -1.0 / (d + 4));
  Eigen::MatrixXd H;
  bool r = llt(Sigma, H);
  SIA_EXCEPTION(r, "Failed to compute cholesky decomposition of covariance");
  H *= c;
  setBandwidthMatrix(H);
}

}  // namespace sia
