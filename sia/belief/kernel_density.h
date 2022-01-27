/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/particles.h"

#include <Eigen/Dense>
#include <memory>

namespace sia {

/// SmoothingKernel density estimator to smooth a weighted particle density.  If
/// bandwidth mode is USER_SPECIFIED, kernel bandwidth is initialized using
/// Scott's rule.
///
/// References:
/// [1] https://www.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
/// [2] Hardle et. al., "Nonparametric and Semiparametric models," 2004.
class KernelDensity : public Particles {
 public:
  /// Determines how the bandwidth is computed
  enum BandwidthMode { SCOTT_RULE, USER_SPECIFIED };

  /// Smoothing kernel function
  enum KernelType {
    UNIFORM,
    GAUSSIAN,
    EPANECHNIKOV,
  };

  /// Each column of values is a sample.
  explicit KernelDensity(const Eigen::MatrixXd& values,
                         const Eigen::VectorXd& weights,
                         KernelType type = EPANECHNIKOV,
                         BandwidthMode mode = SCOTT_RULE,
                         double bandwidth_scaling = 1.0);

  explicit KernelDensity(const Particles& particles,
                         KernelType type = EPANECHNIKOV,
                         BandwidthMode mode = SCOTT_RULE,
                         double bandwidth_scaling = 1.0);

  virtual ~KernelDensity() = default;

  /// Evaluate the kernel density function (pdf)
  double probability(const Eigen::VectorXd& x) const;

  std::size_t dimension() const override;
  const Eigen::VectorXd sample() override;
  double logProb(const Eigen::VectorXd& x) const override;
  const Eigen::VectorXd mean() const override;
  const Eigen::VectorXd mode() const override;
  const Eigen::MatrixXd covariance() const override;
  const Eigen::VectorXd vectorize() const override;
  bool devectorize(const Eigen::VectorXd& data) override;

  /// Sets new samples and updates the bandwidth if mode is not USER_SPECIFIED.
  void setValues(const Eigen::MatrixXd& values) override;

  /// Sets the bandwidth for all dimensions, sets mode to USER_SPECIFIED, and
  /// bandwidth scaling to 1.
  void setBandwidth(double h);

  /// Sets the multivariate bandwidth, sets mode to USER_SPECIFIED, and
  /// bandwidth scaling to 1.
  void setBandwidth(const Eigen::VectorXd& h);
  const Eigen::MatrixXd& bandwidth() const;

  /// Sets a factor to inflate/deflate the scaling if not USER_SPECIFIED
  void setBandwidthScaling(double scaling);
  double getBandwidthScaling() const;

  void setBandwidthMode(BandwidthMode mode);
  BandwidthMode getBandwidthMode() const;

  void setKernelType(KernelType type);
  KernelType getKernelType() const;

  // Forward declarations
  struct SmoothingKernel;

 protected:
  void setBandwidthMatrix(const Eigen::MatrixXd& H);
  void autoUpdateBandwidth();
  void bandwidthScottRule(const Eigen::MatrixXd& Sigma);

  std::shared_ptr<SmoothingKernel> m_kernel;
  Eigen::MatrixXd m_bandwidth;
  Eigen::MatrixXd m_bandwidth_inv;
  double m_bandwidth_det;
  BandwidthMode m_mode;
  double m_bandwidth_scaling;
};

}  // namespace sia
