/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/distribution.h"

namespace sia {

std::mt19937 Generator::m_rng;

Generator::Generator() {
  seed(DEFAULT_SEED);
}

Generator& Generator::instance() {
  static Generator generator;
  return generator;
}

void Generator::seed(unsigned seed) {
  m_rng.seed(seed);
}

std::mt19937& Generator::engine() {
  return m_rng;
}

Distribution::Distribution(Generator& generator) : m_rng(generator.engine()) {}

Distribution& Distribution::operator=(const Distribution& other) {
  m_rng = other.m_rng;
  return *this;
}

std::vector<Eigen::VectorXd> Distribution::samples(std::size_t num_samples) {
  std::vector<Eigen::VectorXd> samples;
  samples.reserve(num_samples);
  for (std::size_t i = 0; i < num_samples; ++i) {
    samples.emplace_back(sample());
  }
  return samples;
}

}  // namespace sia
