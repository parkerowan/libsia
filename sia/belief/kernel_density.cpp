/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/kernel_density.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/helpers.h"

#include <glog/logging.h>
#include <cmath>
#include <iostream>

namespace sia {

Kernel* Kernel::create(Type type, std::size_t dimension) {
  switch (type) {
    case UNIFORM:
      return new UniformKernel(dimension);
    case GAUSSIAN:
      return new GaussianKernel(dimension);
    case EPANECHNIKOV:
      return new EpanechnikovKernel(dimension);
    default:
      LOG(WARNING) << "Kernel::Type " << type
                   << ", unsupported.  Creating EpanechnikovKernel instead";
      return new EpanechnikovKernel(dimension);
  }
}

UniformKernel::UniformKernel(std::size_t dimension) {
  double n = static_cast<double>(dimension);
  m_constant = 1 / pow(2, n);
}

double UniformKernel::evaluate(const Eigen::VectorXd& x) const {
  return (x.array().abs().maxCoeff() <= 1) ? m_constant : 0.0;
}

Kernel::Type UniformKernel::type() const {
  return Kernel::UNIFORM;
}

GaussianKernel::GaussianKernel(std::size_t dimension) {
  double n = static_cast<double>(dimension);
  m_constant = 1 / sqrt(pow(2 * M_PI, n));
}

double GaussianKernel::evaluate(const Eigen::VectorXd& x) const {
  return m_constant * exp(-x.dot(x) / 2);
}

Kernel::Type GaussianKernel::type() const {
  return Kernel::GAUSSIAN;
}

EpanechnikovKernel::EpanechnikovKernel(std::size_t dimension) {
  double n = static_cast<double>(dimension);
  double c = pow(M_PI, n / 2) / tgamma(n / 2 + 1);  // Volume of unit n-ball
  m_constant = (n + 2) / (2 * c);
}

double EpanechnikovKernel::evaluate(const Eigen::VectorXd& x) const {
  double xdotx = x.dot(x);
  return (sqrt(xdotx) <= 1) ? m_constant * (1 - xdotx) : 0.0;
}

Kernel::Type EpanechnikovKernel::type() const {
  return Kernel::EPANECHNIKOV;
}

// From eqn. 3.69 Hardle et. al., 2004.
Eigen::VectorXd bandwidthSilverman(const Eigen::VectorXd& sigma,
                                   std::size_t num_samples) {
  std::size_t dim = sigma.size();
  double d = static_cast<double>(dim);
  double n = static_cast<double>(num_samples);
  Eigen::VectorXd h = Eigen::VectorXd::Zero(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    double power = 1 / (d + 4);
    h(i) = pow(pow(4 / (d + 2), power) * pow(n, -power) * sigma(i), 2);
  }
  return h;
}

// From eqn. 3.70 Hardle et. al., 2004.
Eigen::VectorXd bandwidthScott(const Eigen::VectorXd& sigma,
                               std::size_t num_samples) {
  std::size_t dim = sigma.size();
  double d = static_cast<double>(dim);
  double n = static_cast<double>(num_samples);
  Eigen::VectorXd h = Eigen::VectorXd::Zero(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    h(i) = pow(pow(n, -1 / (d + 4)) * sigma(i), 2);
  }
  return h;
}

KernelDensity::KernelDensity(const Eigen::MatrixXd& values,
                             const Eigen::VectorXd& weights,
                             Kernel::Type type,
                             BandwidthMode mode,
                             double bandwidth_scaling)
    : Particles(values, weights),
      m_mode(mode),
      m_bandwidth_scaling(bandwidth_scaling) {
  m_kernel = Kernel::create(type, values.rows());

  // If user specified, set the initial bandwidth using silverman
  if (mode == USER_SPECIFIED) {
    m_mode = SILVERMAN;
  }
  setValues(values);
  m_mode = mode;
}

KernelDensity::KernelDensity(const Particles& particles,
                             Kernel::Type type,
                             BandwidthMode mode,
                             double bandwidth_scaling)
    : KernelDensity(particles.values(),
                    particles.weights(),
                    type,
                    mode,
                    bandwidth_scaling) {}

KernelDensity::~KernelDensity() {
  delete m_kernel;
}

double KernelDensity::probability(const Eigen::VectorXd& x) const {
  double p = 0;
  std::size_t n = m_weights.size();
  const Eigen::VectorXd h = m_bandwidth.array().sqrt();
  for (std::size_t i = 0; i < n; ++i) {
    const auto& c = m_values.col(i);
    double kh = m_kernel->evaluate((x - c).cwiseQuotient(h));
    p += m_weights(i) * kh;
  }
  p /= h.prod();
  return p;
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

void KernelDensity::setValues(const Eigen::MatrixXd& values) {
  m_values = values;
  if (m_mode == USER_SPECIFIED) {
    return;
  }

  // If not user-specified, update the bandwidth based on the new sample
  // covariance.  Apply scaling.
  const Eigen::VectorXd sigma =
      Particles::covariance().diagonal().array().sqrt();
  if (m_mode == SILVERMAN) {
    m_bandwidth = bandwidthSilverman(sigma, numParticles());
  } else if (m_mode == SCOTT) {
    m_bandwidth = bandwidthScott(sigma, numParticles());
  }
  m_bandwidth *= m_bandwidth_scaling;
}

void KernelDensity::setBandwidth(double h) {
  setBandwidth(h * Eigen::VectorXd::Ones(dimension()));
}

void KernelDensity::setBandwidth(const Eigen::VectorXd& h) {
  m_bandwidth = h;
  m_bandwidth_scaling = 1.0;
  m_mode = USER_SPECIFIED;
  setValues(values());
}

const Eigen::VectorXd KernelDensity::getBandwidth() const {
  return m_bandwidth;
}

void KernelDensity::setBandwidthScaling(double scaling) {
  m_bandwidth_scaling = scaling;
  setValues(values());
}

double KernelDensity::getBandwidthScaling() const {
  return m_bandwidth_scaling;
}

void KernelDensity::setBandwidthMode(KernelDensity::BandwidthMode mode) {
  m_mode = mode;
  setValues(values());
}

KernelDensity::BandwidthMode KernelDensity::getBandwidthMode() const {
  return m_mode;
}

void KernelDensity::setKernelType(Kernel::Type type) {
  m_kernel = Kernel::create(type, m_values.rows());
}

Kernel::Type KernelDensity::getKernelType() const {
  return m_kernel->type();
}

}  // namespace sia
