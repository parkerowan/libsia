/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/categorical.h"
#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"

#include <Eigen/Dense>

namespace sia {

/// Gaussian mixture model (GMM) defined by a weighted combinations of
/// Gaussians.  GMMs are locally linear and used to perform classification or
/// regression (see sia::GMR).
///
/// References:
/// [1] http://www.stat.rice.edu/~hgsung/thesis.pdf
class GMM : public Distribution, public Inference {
 public:
  /// Regularization applied to covariance matrices during GMM fit.
  static constexpr double DEFAULT_REGULARIZATION = 1e-6;

  /// Initialize K equally weighted standard normal Gaussians for n-dim space.
  explicit GMM(std::size_t K, std::size_t dimension);

  /// Creates a GMM from Gaussians and weights, prior weights must sum to 1.
  explicit GMM(const std::vector<Gaussian>& gaussians,
               const std::vector<double>& priors);

  /// Fits a GMM with K clusters to samples using EM initalized with kmeans.
  explicit GMM(const Eigen::MatrixXd& samples,
               std::size_t K,
               double regularization = DEFAULT_REGULARIZATION);

  virtual ~GMM() = default;
  std::size_t dimension() const override;
  const Eigen::VectorXd sample() override;
  double logProb(const Eigen::VectorXd& x) const override;
  const Eigen::VectorXd mean() const override;
  const Eigen::VectorXd mode() const override;
  const Eigen::MatrixXd covariance() const override;
  const Eigen::VectorXd vectorize() const override;
  bool devectorize(const Eigen::VectorXd& data) override;

  /// Performs the inference $p(y|x)$
  const Categorical& predict(const Eigen::VectorXd& x) override;
  std::size_t inputDimension() const override;
  std::size_t outputDimension() const override;

  std::size_t classify(const Eigen::VectorXd& x);

  std::size_t numClusters() const;
  double prior(std::size_t i) const;
  const std::vector<double>& priors() const;
  const Gaussian& gaussian(std::size_t i) const;
  const std::vector<Gaussian>& gaussians() const;

  /// Method used to fit the data
  enum class FitMethod {
    KMEANS,               // Associates samples to clusters using L2 norm
    GAUSSIAN_LIKELIHOOD,  // Associates samples to clusters using Gaussian MLE
  };

  /// Method used to initialize the data
  enum class InitMethod {
    STANDARD_RANDOM,  // "Forgy" method randomly chooses means from samples
    WARM_START,       // Uses values of gaussians and priors passed to function
  };

  /// Fit a GMM of K clusters to the sample data (cols are samples)
  static std::size_t fit(const Eigen::MatrixXd& samples,
                         std::vector<Gaussian>& gaussians,
                         std::vector<double>& priors,
                         std::size_t K,
                         FitMethod fit_method = FitMethod::GAUSSIAN_LIKELIHOOD,
                         InitMethod init_method = InitMethod::STANDARD_RANDOM,
                         double regularization = DEFAULT_REGULARIZATION);

 private:
  Categorical m_belief;
  std::size_t m_num_clusters;
  std::size_t m_dimension;
  std::vector<Gaussian> m_gaussians;
  std::vector<double> m_priors;
  std::uniform_real_distribution<double> m_uniform{0.0, 1.0};
};

}  // namespace sia
