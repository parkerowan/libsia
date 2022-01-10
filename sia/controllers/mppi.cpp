/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/controllers/mppi.h"

#include <glog/logging.h>

namespace sia {

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
  auto T = m_horizon;
  auto K = m_num_samples;
  auto x = state.mean();

  // Shift the control through the buffer so that we use the previously computed
  // cost to initialize the trajectory.
  m_controls.erase(m_controls.begin());        // remove the first element
  m_controls.emplace_back(m_controls.back());  // copy the last element

  // Clear the states
  m_states.clear();
  m_states.reserve(T);

  // Rollout a perturbation around the nominal control for each sample
  std::vector<Trajectory<Eigen::VectorXd>> eps;
  eps.reserve(K);
  Eigen::VectorXd S = Eigen::VectorXd::Zero(K);
  for (std::size_t k = 0; k < K; ++k) {
    x = state.mean();  // TODO: Add option to sample

    // TODO: can parallelize
    Trajectory<Eigen::VectorXd> samples = m_sigma.samples(T);
    eps.emplace_back(samples);
    for (std::size_t i = 0; i < T - 1; ++i) {
      const auto& u = m_controls.at(i);
      const auto& e = samples.at(i);
      const auto uhat = u + e;
      x = m_dynamics.dynamics(x, uhat).mean();  // TODO: Add option to sample
      S(k) += m_cost.c(x, uhat, i) + m_lambda * u.transpose() * m_sigma_inv * e;
    }
    S(k) += m_cost.cf(x);
  }

  // Compute weights
  double beta = S.minCoeff();
  Eigen::VectorXd w = exp(-(S.array() - beta) / m_lambda);
  w /= w.sum();

  // Update controls and states
  x = state.mean();
  m_states.emplace_back(x);
  for (std::size_t i = 0; i < T - 1; ++i) {
    Eigen::VectorXd e = Eigen::VectorXd::Zero(eps.at(0).at(0).size());
    for (std::size_t k = 0; k < K; ++k) {
      e += w(k) * eps.at(k).at(i);
    }
    m_controls.at(i) += e;
    x = m_dynamics.dynamics(x, m_controls.at(i)).mean();
    m_states.emplace_back(x);
  }
  return m_controls.at(0);
}

const Trajectory<Eigen::VectorXd>& MPPI::controls() const {
  return m_controls;
}

const Trajectory<Eigen::VectorXd>& MPPI::states() const {
  return m_states;
}

void MPPI::cacheSigmaInv() {
  const Eigen::MatrixXd& sigma = m_sigma.covariance();
  std::size_t n = sigma.rows();
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  m_sigma_inv = sigma.llt().solve(I);
}

}  // namespace sia
