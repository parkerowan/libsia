/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/optimizers/covariance_adaptation.h"
#include <cmath>
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

namespace sia {

CovarianceAdaptation::CovarianceAdaptation(
    std::size_t dimension,
    const CovarianceAdaptation::Options& options)
    : m_dimension(dimension), m_options(options) {
  SIA_THROW_IF_NOT(m_dimension > 0,
                   "CovarianceAdaptation dimension expected to be > 0");
  SIA_THROW_IF_NOT(options.n_samples > 1,
                   "CovarianceAdaptation n_samples expected to be > 1");
  SIA_THROW_IF_NOT(options.init_stdev > 0,
                   "CovarianceAdaptation init_stdev expected to be > 0");
  reset();
}

std::size_t CovarianceAdaptation::dimension() const {
  return m_dimension;
}

Eigen::VectorXd CovarianceAdaptation::minimize(CovarianceAdaptation::Cost f,
                                               const Eigen::VectorXd& x0) {
  SIA_THROW_IF_NOT(x0.size() == int(m_dimension),
                   "CovarianceAdaptation x0 size expected to match dimension");

  // Perform the minimization
  Eigen::VectorXd x = x0;
  double fref = f(x);
  double fref_prev = fref;
  do {
    // Run a single step of the optimization
    x = minimizeSingleStep(f, x);

    // Update iterations to check for convergence
    fref_prev = fref;
    fref = f(x);
  } while ((abs(fref_prev - fref) > m_options.tol) &&
           (_s.iter < m_options.max_iter) &&
           (_s.C.norm() < m_options.max_cov_norm));

  return x;
}

Eigen::VectorXd CovarianceAdaptation::minimizeSingleStep(
    Cost f,
    const Eigen::VectorXd& x0) {
  SIA_THROW_IF_NOT(x0.size() == int(dimension()),
                   "CovarianceAdaptation x0 size expected to match dimension");
  Eigen::VectorXd x = x0;

  // Generate samples of (cost, state) pairs
  Gaussian sampler(x, pow(_s.sigma, 2) * _s.C);
  m_samples.clear();
  m_samples.reserve(m_options.n_samples);
  for (std::size_t i = 0; i < m_options.n_samples; ++i) {
    Eigen::VectorXd xx = sampler.sample();
    m_samples.emplace_back(std::make_pair(f(xx), xx));
  }

  // Sort in ascending order by the cost function and compute weighted
  std::sort(m_samples.begin(), m_samples.end(),
            [](const std::pair<double, Eigen::VectorXd>& a,
               const std::pair<double, Eigen::VectorXd>& b) -> bool {
              return a.first < b.first;
            });
  Eigen::VectorXd x_old = x;
  x = Eigen::VectorXd::Zero(dimension());
  for (std::size_t i = 0; i < _c.mu; ++i) {
    x += _c.weights(i) * m_samples.at(i).second;
  }

  // Cumulation: Update evolution paths
  _s.ps = (1. - _c.cs) * _s.ps + sqrt(_c.cs * (2 - _c.cs) * _c.mueff) *
                                     sampler.Linv() * (x - x_old) / _s.sigma;
  double h_sig =
      _s.ps.norm() /
          sqrt(1. - pow(1. - _c.cs, 2. * double(_s.iter) / _c.lambda)) /
          _s.chiN <
      1.4 + 2 / (_c.N + 1.);
  _s.pc = (1. - _c.cc) * _s.pc + h_sig * sqrt(_c.cc * (2. - _c.cc) * _c.mueff) *
                                     (x - x_old) / _s.sigma;

  // Adapt covariance matrix C
  Eigen::MatrixXd e = Eigen::MatrixXd::Zero(dimension(), _c.mu);
  for (std::size_t i = 0; i < _c.mu; ++i) {
    e.col(i) = (m_samples.at(i).second - x_old) / _s.sigma;
  }
  _s.C = (1. - _c.c1 - _c.cmu) * _s.C +
         _c.c1 * (_s.pc * _s.pc.transpose() +
                  (1. - h_sig) * _c.cc * (2. - _c.cc) * _s.C) +
         _c.cmu * e * _c.weights.asDiagonal() * e.transpose();
  if (_s.C.norm() >= m_options.max_cov_norm) {
    SIA_WARN("CovarianceAdaptation norm exceeded max_cov_norm="
             << m_options.max_cov_norm);
  }

  // Adapt step size sigma
  _s.sigma = _s.sigma * exp((_c.cs / _c.damps) * (_s.ps.norm() / _s.chiN - 1.));

  // Update the iteration
  _s.iter++;
  if (_s.iter >= m_options.max_iter) {
    SIA_WARN("CovarianceAdaptation reached max_iter=" << m_options.max_iter);
  }

  return x;
}

const std::vector<std::pair<double, Eigen::VectorXd>>&
CovarianceAdaptation::getSamples() const {
  return m_samples;
}

void CovarianceAdaptation::reset() {
  // Strategy parameter setting: Selection
  _c.lambda = m_options.n_samples;
  _c.mu = std::floor(_c.lambda / 2.0);
  _c.weights = Eigen::VectorXd::Zero(_c.mu);
  for (std::size_t i = 0; i < _c.mu; ++i) {
    _c.weights(i) = log(double(_c.mu) + 0.5) - log(double(i + 1));
  }
  _c.weights /= _c.weights.sum();
  _c.mueff = 1 / _c.weights.array().square().sum();

  // Strategy parameter setting: Adaptation
  _c.N = dimension();
  _c.cc = (4. + _c.mueff) / (_c.N + 4. + 2. * _c.mueff / _c.N);
  _c.cs = (_c.mueff + 2.) / (_c.N + _c.mueff + 5.);
  _c.c1 = 2. / (pow(_c.N + 1.3, 2) + _c.mueff);
  _c.cmu = std::min(1. - _c.c1, 2. * (_c.mueff - 2. + 1. / _c.mueff) /
                                    (pow(_c.N + 2., 2) + _c.mueff));
  _c.damps =
      1. + 2. * std::max(0., sqrt((_c.mueff - 1.) / (_c.N + 1.)) - 1.) + _c.cs;

  // Initialize dynamic (internal) strategy parameters
  _s.sigma = m_options.init_stdev;
  _s.chiN =
      pow(_c.N, 0.5) * (1. - 1. / (4. * _c.N) + 1. / (21. * pow(_c.N, 2)));
  _s.C = Eigen::MatrixXd::Identity(dimension(), dimension());
  _s.pc = Eigen::VectorXd::Zero(dimension());
  _s.ps = Eigen::VectorXd::Zero(dimension());

  _s.iter = 0;
}

}  // namespace sia
