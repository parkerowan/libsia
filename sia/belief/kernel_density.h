/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/particles.h"

#include <Eigen/Dense>
#include <memory>

namespace sia {

/// SmoothingKernel base class.  See:
/// https://vision.in.tum.de/_media/teaching/ss2013/ml_ss13/ml4cv_vi.pdf
/// A kernel function maps a vector to a (semi)-positive scalar, is symmetric
/// about x=0, and whose integral is 1.  See: Hansen, Lecture notes on
/// nonparametrics, 2009.
///
/// References:
/// [1] https://www.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
/// [2] W. Hardle et. al., "Nonparametric and Semiparametric models," 2004.
struct SmoothingKernel {
  SmoothingKernel() = default;
  virtual ~SmoothingKernel() = default;
  virtual double evaluate(const Eigen::VectorXd& x) const = 0;
};

/// Multivariate uniform kernel, domain |x| <= 1.
struct UniformKernel : public SmoothingKernel {
  explicit UniformKernel(std::size_t dimension);
  double evaluate(const Eigen::VectorXd& x) const override;

 private:
  double m_constant;
};

/// Multivariate standard normal kernel, domain is infinite.
struct GaussianKernel : public SmoothingKernel {
  explicit GaussianKernel(std::size_t dimension);
  double evaluate(const Eigen::VectorXd& x) const override;

 private:
  double m_constant;
};

/// Multivariate spherical Epanechnikov kernel, domain L2(x)^2 <= 1.
struct EpanechnikovKernel : public SmoothingKernel {
  explicit EpanechnikovKernel(std::size_t dimension);
  double evaluate(const Eigen::VectorXd& x) const override;

 private:
  double m_constant;
};

// ----------------------------------------------------------------------------

/// Kernel density estimator to smooth a weighted particle density.  If
/// bandwidth mode is USER_SPECIFIED, kernel bandwidth is initialized using
/// Scott's rule.
///
/// References:
/// [1] https://www.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
/// [2] Hardle et. al., "Nonparametric and Semiparametric models," 2004.
class KernelDensity : public Particles {
 public:
  /// Determines how the bandwidth is computed
  enum class BandwidthMode { SCOTT_RULE, USER_SPECIFIED };

  /// Each column of values is a sample.
  explicit KernelDensity(const Eigen::MatrixXd& values,
                         const Eigen::VectorXd& weights,
                         SmoothingKernel& kernel,
                         BandwidthMode mode = BandwidthMode::SCOTT_RULE,
                         double bandwidth_scaling = 1.0);

  explicit KernelDensity(const Particles& particles,
                         SmoothingKernel& kernel,
                         BandwidthMode mode = BandwidthMode::SCOTT_RULE,
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

  /// Access the kernel
  SmoothingKernel& kernel();

 protected:
  void setBandwidthMatrix(const Eigen::MatrixXd& H);
  void autoUpdateBandwidth();
  void bandwidthScottRule(const Eigen::MatrixXd& Sigma);

  SmoothingKernel& m_kernel;
  Eigen::MatrixXd m_bandwidth;
  Eigen::MatrixXd m_bandwidth_inv;
  double m_bandwidth_det;
  BandwidthMode m_mode;
  double m_bandwidth_scaling;
};

}  // namespace sia
