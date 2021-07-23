/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/controllers/mppi.h"

#include <glog/logging.h>

namespace sia {

using Trajectory = std::vector<Eigen::VectorXd>;

MPPI::MPPI(DynamicsModel& dynamics,
           CostFunction& cost,
           const std::vector<Eigen::VectorXd>& u0,
           std::size_t num_samples,
           const Eigen::MatrixXd& sigma,
           double lam)
    : m_dynamics(dynamics),
      m_cost(cost),
      m_horizon(u0.size()),
      m_num_samples(num_samples),
      m_sigma(Gaussian(Eigen::VectorXd::Zero(sigma.rows()), sigma)),
      m_lambda(lam),
      m_controls(u0) {
  cacheSigmaInv();
}

const Eigen::VectorXd& MPPI::policy(const Distribution& state) {
  auto& N = m_horizon;
  auto& K = m_num_samples;

  // Shift the control through the buffer so that we use the previously computed
  // cost to initialize the trajectory.
  m_controls.erase(m_controls.begin());        // remove the first element
  m_controls.emplace_back(m_controls.back());  // copy the last element

  // Rollout a perturbation around the nominal control for each sample
  std::vector<Trajectory> eps;
  eps.reserve(K);
  Eigen::VectorXd S = Eigen::VectorXd::Zero(K);
  for (std::size_t k = 0; k < K; ++k) {
    auto x = state.mean();
    // TODO: can parallelize
    Trajectory samples = m_sigma.samples(N);
    eps.emplace_back(samples);
    for (std::size_t i = 0; i < N - 1; ++i) {
      const auto& u = m_controls.at(i);
      const auto& e = samples.at(i);
      const auto uhat = u + e;
      x = m_dynamics.dynamics(x, uhat).mean();
      S(k) += m_cost.c(x, uhat, i) + m_lambda * u.transpose() * m_sigma_inv * e;
    }
    S(k) += m_cost.cf(x);
  }

  // Compute weights
  double beta = S.minCoeff();
  Eigen::VectorXd w = exp(-(S.array() - beta) / m_lambda);
  w /= w.sum();

  // Update controls
  for (std::size_t i = 0; i < N; ++i) {
    Eigen::VectorXd e = Eigen::VectorXd::Zero(eps.at(0).at(0).size());
    for (std::size_t k = 0; k < K; ++k) {
      e += w(k) * eps.at(k).at(i);
    }
    m_controls.at(i) += e;
  }

  return m_controls.at(0);
}

void MPPI::cacheSigmaInv() {
  const Eigen::MatrixXd& sigma = m_sigma.covariance();
  std::size_t n = sigma.rows();
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  m_sigma_inv = sigma.llt().solve(I);
}

}  // namespace sia
