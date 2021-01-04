/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/particles.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/uniform.h"

#include <glog/logging.h>
#include <iostream>

namespace sia {

Particles::Particles(std::size_t dimension,
                     std::size_t num_particles,
                     bool weighted_stats)
    : Distribution(Generator::instance()), m_weighted_stats(weighted_stats) {
  setValues(Eigen::MatrixXd::Zero(dimension, num_particles));
  setWeights(Eigen::VectorXd::Ones(num_particles) /
             static_cast<double>(num_particles));
}

Particles::Particles(const Eigen::MatrixXd& values,
                     const Eigen::VectorXd& weights,
                     bool weighted_stats)
    : Distribution(Generator::instance()), m_weighted_stats(weighted_stats) {
  setValues(values);
  setWeights(weights);
}

Particles Particles::init(Distribution& distribution,
                          std::size_t num_particles,
                          bool weighted_stats) {
  Particles particles(distribution.dimension(), num_particles, weighted_stats);
  for (std::size_t i = 0; i < num_particles; ++i) {
    particles.m_values.col(i) = distribution.sample();
  }
  return particles;
}

Particles Particles::gaussian(const Eigen::VectorXd& mean,
                              const Eigen::MatrixXd& covariance,
                              std::size_t num_particles,
                              bool weighted_stats) {
  Gaussian gaussian(mean, covariance);
  return init(gaussian, num_particles, weighted_stats);
}

Particles Particles::uniform(const Eigen::VectorXd& lower,
                             const Eigen::VectorXd& upper,
                             std::size_t num_particles,
                             bool weighted_stats) {
  Uniform uniform(lower, upper);
  return init(uniform, num_particles, weighted_stats);
}

std::size_t Particles::dimension() const {
  return m_values.rows();
}

const Eigen::VectorXd Particles::sample() {
  return m_values.col(sampleInverseCdf());
}

double Particles::logProb(const Eigen::VectorXd& x) const {
  return log(m_weights(findNearestNeighbor(x)));
}

const Eigen::VectorXd Particles::mean() const {
  double n = static_cast<double>(m_values.cols());
  if (m_weighted_stats) {
    return m_values * m_weights;
  }
  return m_values.rowwise().sum() / n;
}

const Eigen::VectorXd Particles::mode() const {
  if (m_weighted_stats) {
    std::size_t r, c;
    m_weights.maxCoeff(&r, &c);
    return m_values.col(r);
  }
  return Particles::mean();
}

const Eigen::MatrixXd Particles::covariance() const {
  const Eigen::VectorXd x = Particles::mean();
  const Eigen::MatrixXd e = (m_values.array().colwise() - x.array()).matrix();
  if (m_weighted_stats) {
    const Eigen::MatrixXd ewe = e * m_weights.asDiagonal() * e.transpose();
    return ewe / (1 - m_weights.array().square().sum());
  }
  double n = static_cast<double>(m_values.cols());
  return e * e.transpose() / (n - 1);
}

bool Particles::getUseWeightedStats() const {
  return m_weighted_stats;
}

void Particles::setUseWeightedStats(bool weighted_stats) {
  m_weighted_stats = weighted_stats;
}

std::size_t Particles::numParticles() const {
  return m_values.cols();
}

void Particles::setValues(const Eigen::MatrixXd& values) {
  m_values = values;
}

const Eigen::MatrixXd& Particles::values() const {
  return m_values;
}

const Eigen::VectorXd Particles::value(std::size_t i) const {
  return m_values.col(i);
}

void Particles::setWeights(const Eigen::VectorXd& weights) {
  m_weights = weights;
  double wsum = m_weights.array().sum();

  if (abs(wsum - 1.0) >= RENORMALIZE_WEIGHTS_TOLERANCE) {
    LOG(WARNING) << "Provided weights sum to " << wsum
                 << ", renormalizing to sum to 1";
    m_weights /= wsum;
  }
}

const Eigen::VectorXd& Particles::weights() const {
  return m_weights;
}

double Particles::weight(std::size_t i) const {
  return m_weights(i);
}

std::size_t Particles::findNearestNeighbor(const Eigen::VectorXd& x) const {
  std::size_t r, c;
  const Eigen::MatrixXd e = (m_values.array().colwise() - x.array()).matrix();
  const Eigen::VectorXd d = e.array().square().matrix().colwise().sum();
  d.minCoeff(&r, &c);
  return r;
}

std::size_t Particles::sampleInverseCdf() const {
  Uniform g(0, 1);
  double u = g.sample()(0);
  double c = m_weights(0);
  std::size_t i = 0;
  while (u > c) {
    i++;
    c += m_weights(i);
  }
  return i;
}

}  // namespace sia
