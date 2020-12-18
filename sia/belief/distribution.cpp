/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/distribution.h"

namespace sia {

Generator::Generator() {
  seed(DEFAULT_SEED);
}

Generator& Generator::instance() {
  static Generator generator;
  return generator;
}

void Generator::seed(unsigned seed) {
  m_generator.seed(seed);
}

std::default_random_engine& Generator::engine() {
  return m_generator;
}

Distribution::Distribution(Generator& generator)
    : m_generator(generator.engine()) {}

Distribution& Distribution::operator=(const Distribution& other) {
  m_generator = other.m_generator;
  return *this;
}

}  // namespace sia
