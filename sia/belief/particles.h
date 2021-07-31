/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

// Forward declaration
class PF;

/// Constant for tolerance around 1.0 to renormalize weights
static const double RENORMALIZE_WEIGHTS_TOLERANCE = 1e-6;

/// Defines a distribution based on a discrete set of samples drawn from the
/// random variable.  Weights are initialized as equal unless explicitly
/// specified, and normalized to one when set.  Weights are used to compute
/// mean, mode, and covariance if weighted_stats is set.
class Particles : public Distribution {
  friend class PF;

 public:
  /// Creates zero-vector particles with uniform weights.
  explicit Particles(std::size_t dimension,
                     std::size_t num_particles,
                     bool weighted_stats = false);

  /// Creates particles with uniform weights from the vector of states.
  explicit Particles(const Eigen::MatrixXd& values,
                     const Eigen::VectorXd& weights,
                     bool weighted_stats = false);

  /// Creates particles by sampling from the provided distribution.
  static Particles init(Distribution& distribution,
                        std::size_t num_particles,
                        bool weighted_stats = false);

  /// Creates particles by sampling from a Gaussian distribution.
  static Particles gaussian(const Eigen::VectorXd& mean,
                            const Eigen::MatrixXd& covariance,
                            std::size_t num_particles,
                            bool weighted_stats = false);

  /// Creates particles by sampling from a Uniform distribution.
  static Particles uniform(const Eigen::VectorXd& lower,
                           const Eigen::VectorXd& upper,
                           std::size_t num_particles,
                           bool weighted_stats = false);

  std::size_t dimension() const override;

  /// Draws a sample using inverse CDF of the sample weights.
  const Eigen::VectorXd sample() override;

  /// Returns the log of the nearest neighbor sample weight.
  double logProb(const Eigen::VectorXd& x) const override;

  /// Returns the sample mean (or weighted mean if WeightedStats is set).
  const Eigen::VectorXd mean() const override;

  /// Returns the sample mean (or sample associated with max weight if
  /// WeightedStats is set).
  const Eigen::VectorXd mode() const override;

  /// Returns the unbiased sample covariance (or weighted covariance if
  /// WeightedStats is set).
  const Eigen::MatrixXd covariance() const override;

  const Eigen::VectorXd vectorize() const override;
  bool devectorize(const Eigen::VectorXd& data) override;

  bool getUseWeightedStats() const;
  void setUseWeightedStats(bool weighted_stats);
  std::size_t numParticles() const;

  /// Set the sample values as an M x N matrix, where M is state dimension, and
  /// N is number of particles.
  virtual void setValues(const Eigen::MatrixXd& values);
  const Eigen::MatrixXd& values() const;
  const Eigen::VectorXd value(std::size_t i) const;

  /// Set the sample weights, normalize to sum 1 if they aren't already.
  void setWeights(const Eigen::VectorXd& weights);
  const Eigen::VectorXd& weights() const;
  double weight(std::size_t i) const;

 protected:
  std::size_t findNearestNeighbor(const Eigen::VectorXd& x) const;
  std::size_t sampleInverseCdf() const;

  // M x N where: M is state dimension, N is number of particles
  Eigen::MatrixXd m_values;
  Eigen::VectorXd m_weights;
  bool m_weighted_stats;
};

}  // namespace sia
