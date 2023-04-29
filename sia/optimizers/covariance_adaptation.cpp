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
    : m_dimension(dimension) {
  SIA_THROW_IF_NOT(m_dimension > 0,
                   "CovarianceAdaptation dimension expected to be > 0");
  SIA_THROW_IF_NOT(options.n_samples > 1,
                   "CovarianceAdaptation n_samples expected to be > 1");
  SIA_THROW_IF_NOT(options.init_stdev > 0,
                   "CovarianceAdaptation init_stdev expected to be > 0");
}

std::size_t CovarianceAdaptation::dimension() const {
  return m_dimension;
}

Eigen::VectorXd CovarianceAdaptation::minimize(
    CovarianceAdaptation::Cost f,
    const Eigen::VectorXd& x0) const {
  SIA_THROW_IF_NOT(x0.size() == int(m_dimension),
                   "CovarianceAdaptation x0 size expected to match dimension");
  // Algorithm from: https://en.wikipedia.org/wiki/CMA-ES

  // Strategy parameter setting: Selection
  double lambda = m_options.n_samples;
  std::size_t mu = std::floor(lambda / 2.0);
  Eigen::VectorXd weights = Eigen::VectorXd::Zero(mu);
  for (std::size_t i = 0; i < mu; ++i) {
    weights(i) = log(double(mu) + 0.5) - log(double(i + 1));
  }
  weights /= weights.sum();
  double mueff = 1 / weights.array().square().sum();

  // Strategy parameter setting: Adaptation
  double N = dimension();
  double cc = (4. + mueff) / (N + 4. + 2. * mueff / N);
  double cs = (mueff + 2.) / (N + mueff + 5.);
  double c1 = 2. / (pow(N + 1.3, 2) + mueff);
  double cmu = std::min(
      1. - c1, 2. * (mueff - 2. + 1. / mueff) / (pow(N + 2., 2) + mueff));
  double damps =
      1. + 2. * std::max(0., sqrt((mueff - 1.) / (N + 1.)) - 1.) + cs;

  // Initialize dynamic (internal) strategy parameters and constants
  double sigma = m_options.init_stdev;
  Eigen::MatrixXd C = Eigen::MatrixXd::Identity(dimension(), dimension());
  Eigen::VectorXd pc = Eigen::VectorXd::Zero(dimension());
  Eigen::VectorXd ps = Eigen::VectorXd::Zero(dimension());
  double chiN = pow(N, 0.5) * (1. - 1. / (4. * N) + 1. / (21. * pow(N, 2)));

  // Perform the minimization
  Eigen::VectorXd x = x0;
  std::size_t iter = 0;
  double fref = f(x);
  double fref_prev = fref;
  do {
    // Generate samples of (cost, state) pairs
    Gaussian sampler(x, pow(sigma, 2) * C);
    std::vector<std::pair<double, Eigen::VectorXd>> samples{};
    samples.reserve(m_options.n_samples);
    for (std::size_t i = 0; i < m_options.n_samples; ++i) {
      Eigen::VectorXd x = sampler.sample();
      samples.emplace_back(std::make_pair(f(x), x));
    }

    // Sort in ascending order by the cost function and compute weighted
    std::sort(samples.begin(), samples.end(),
              [](const std::pair<double, Eigen::VectorXd>& a,
                 const std::pair<double, Eigen::VectorXd>& b) -> bool {
                return a.first < b.first;
              });
    Eigen::VectorXd x_old = x;
    x = Eigen::VectorXd::Zero(dimension());
    for (std::size_t i = 0; i < mu; ++i) {
      x += weights(i) * samples.at(i).second;
    }

    // Cumulation: Update evolution paths
    ps = (1. - cs) * ps +
         sqrt(cs * (2 - cs) * mueff) * sampler.Linv() * (x - x_old) / sigma;
    double h_sig =
        ps.norm() / sqrt(1. - pow(1. - cs, 2. * double(iter) / lambda)) / chiN <
        1.4 + 2 / (N + 1.);
    pc = (1. - cc) * pc +
         h_sig * sqrt(cc * (2. - cc) * mueff) * (x - x_old) / sigma;

    // Adapt covariance matrix C
    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(dimension(), mu);
    for (std::size_t i = 0; i < mu; ++i) {
      e.col(i) = (samples.at(i).second - x_old) / sigma;
    }
    C = (1. - c1 - cmu) * C +
        c1 * (pc * pc.transpose() + (1. - h_sig) * cc * (2. - cc) * C) +
        cmu * e * weights.asDiagonal() * e.transpose();

    // Adapt step size sigma
    sigma = sigma * exp((cs / damps) * (ps.norm() / chiN - 1.));

    // Update iterations to check for convergence
    fref_prev = fref;
    fref = f(x);
    iter++;
  } while ((abs(fref_prev - fref) > m_options.tol) &&
           (iter < m_options.max_iter) && (C.norm() < m_options.max_cov_norm));

  if (iter >= m_options.max_iter) {
    SIA_WARN("CovarianceAdaptation reached max_iter=" << m_options.max_iter);
  }

  if (C.norm() >= m_options.max_cov_norm) {
    SIA_WARN("CovarianceAdaptation norm exceeded max_cov_norm="
             << m_options.max_cov_norm);
  }

  return x;
}

}  // namespace sia
