/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/particles.h"

#include <Eigen/Dense>

namespace sia {

// Forward declaration
class KernelDensity;

/// A kernel function maps a vector to a (semi)-positive scalar, is symmetric
/// about x=0, and whose integral is 1.  See: Hansen, Lecture notes on
/// nonparametrics, 2009.
/// https://www.ssc.wisc.edu/~bhansen/718/NonParametrics1.pdf
class Kernel {
  friend class KernelDensity;

 public:
  Kernel() = default;
  virtual ~Kernel() = default;

  /// Evaluates the kernel function
  virtual double evaluate(const Eigen::VectorXd& x) const = 0;

  /// Determines which kernel function is used
  enum Type {
    UNIFORM,
    GAUSSIAN,
    EPANECHNIKOV,
  };

  /// Kernel function type
  virtual Type type() const = 0;

 protected:
  /// Kernel function factory
  static Kernel* create(Type type, std::size_t dimension);
};

/// Multivariate uniform kernel, domain |x| <= 1.
struct UniformKernel : public Kernel {
  explicit UniformKernel(std::size_t dimension);
  double evaluate(const Eigen::VectorXd& x) const override;
  Type type() const override;

 private:
  double m_constant;
};

/// Multivariate  standard normal kernel, domain is infinite.
struct GaussianKernel : public Kernel {
  explicit GaussianKernel(std::size_t dimension);
  double evaluate(const Eigen::VectorXd& x) const override;
  Type type() const override;

 private:
  double m_constant;
};

/// Multivariate Epanechnikov kernel, domain L2(x) <= 1.
struct EpanechnikovKernel : public Kernel {
  explicit EpanechnikovKernel(std::size_t dimension);
  double evaluate(const Eigen::VectorXd& x) const override;
  Type type() const override;

 private:
  double m_constant;
};

/// Returns the multivariate bandwidth using Silverman's rule of thumb, where
/// sigma is a vector of the diagonalized sample standard deviations and
/// num_samples is the number of samples that comprise the kernel density.
Eigen::VectorXd bandwidthSilverman(const Eigen::VectorXd& sigma,
                                   std::size_t num_samples);

/// Returns the multivariate bandwidth using Scott's rule of thumb, where sigma
/// is a vector of the diagonalized sample standard deviations and num_samples
/// is the number of samples that comprise the kernel density.
Eigen::VectorXd bandwidthScott(const Eigen::VectorXd& sigma,
                               std::size_t num_samples);

/// Kernel density estimator to smooth a weighted particle density.  If
/// bandwidth mode is USER_SPECIFIED, kernel bandwidth is initialized using
/// Silverman's rule.
class KernelDensity : public Particles {
 public:
  /// Determines how the bandwidth is computed
  enum BandwidthMode { SILVERMAN, SCOTT, USER_SPECIFIED };

  explicit KernelDensity(const Eigen::MatrixXd& values,
                         const Eigen::VectorXd& weights,
                         Kernel::Type type = Kernel::EPANECHNIKOV,
                         BandwidthMode mode = SILVERMAN,
                         double bandwidth_scaling = 1.0);

  explicit KernelDensity(const Particles& particles,
                         Kernel::Type type = Kernel::EPANECHNIKOV,
                         BandwidthMode mode = SILVERMAN,
                         double bandwidth_scaling = 1.0);

  virtual ~KernelDensity();

  /// Evaluate the kernel density function (pdf)
  double probability(const Eigen::VectorXd& x) const;

  std::size_t dimension() const override;
  const Eigen::VectorXd sample() override;
  double logProb(const Eigen::VectorXd& x) const override;
  const Eigen::VectorXd mean() const override;
  const Eigen::VectorXd mode() const override;
  const Eigen::MatrixXd covariance() const override;

  /// Sets new samples and updates the bandwidth if mode is not USER_SPECIFIED.
  void setValues(const Eigen::MatrixXd& values) override;

  /// Sets the bandwidth for all dimensions, sets mode to USER_SPECIFIED, and
  /// bandwidth scaling to 1.
  void setBandwidth(double h);

  /// Sets the multivariate bandwidth, sets mode to USER_SPECIFIED, and
  /// bandwidth scaling to 1.
  void setBandwidth(const Eigen::VectorXd& h);
  const Eigen::VectorXd getBandwidth() const;

  /// Sets a factor to inflate/deflate the scaling if not USER_SPECIFIED
  void setBandwidthScaling(double scaling);
  double getBandwidthScaling() const;

  void setBandwidthMode(BandwidthMode mode);
  BandwidthMode getBandwidthMode() const;

  void setKernelType(Kernel::Type type);
  Kernel::Type getKernelType() const;

 protected:
  Kernel* m_kernel;
  Eigen::VectorXd m_bandwidth;
  BandwidthMode m_mode;
  double m_bandwidth_scaling;
};

}  // namespace sia
