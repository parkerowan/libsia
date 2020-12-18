/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <Eigen/Dense>
#include <random>

namespace sia {

/// Default seed used for the random number generator
const static unsigned DEFAULT_SEED = 0;

// Forward declaration
class Distribution;

/// A singleton to access the default random engine for seeded randomization
class Generator {
  friend class Distribution;

 public:
  static Generator& instance();
  void seed(unsigned seed = DEFAULT_SEED);

 protected:
  std::default_random_engine& engine();

 private:
  Generator();
  std::default_random_engine m_generator;
};

/// Base probability distribution describing a random variable
class Distribution {
 public:
  explicit Distribution(Generator& generator);
  Distribution& operator=(const Distribution& other);

  virtual std::size_t dimension() const = 0;
  virtual const Eigen::VectorXd sample() = 0;
  virtual double logProb(const Eigen::VectorXd& x) const = 0;
  virtual const Eigen::VectorXd mean() const = 0;
  virtual const Eigen::VectorXd mode() const = 0;
  virtual const Eigen::MatrixXd covariance() const = 0;

 protected:
  std::default_random_engine& m_generator;
};

}  // namespace sia
