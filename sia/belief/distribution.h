/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
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
  std::mt19937& engine();

 private:
  Generator();

  static std::mt19937 m_rng;
};

/// Base probability distribution describing a random variable $x ~ p(x)$
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
  virtual const Eigen::VectorXd vectorize() const = 0;
  virtual bool devectorize(const Eigen::VectorXd& data) = 0;

  std::vector<Eigen::VectorXd> samples(std::size_t num_samples);

 protected:
  std::mt19937& m_rng;
};

/// Base class for prediction $p(y) = p(y|x)$ (regression or classification)
class Inference {
 public:
  Inference() = default;
  virtual ~Inference() = default;

  /// Performs the inference $p(y|x)$
  virtual const Distribution& predict(const Eigen::VectorXd& x) = 0;
  virtual std::size_t inputDimension() const = 0;
  virtual std::size_t outputDimension() const = 0;
};

}  // namespace sia
