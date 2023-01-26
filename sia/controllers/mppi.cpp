/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/controllers/mppi.h"
#include "sia/common/logger.h"

namespace sia {

MPPI::MPPI(DynamicsModel& dynamics,
           CostFunction& cost,
           const std::vector<Eigen::VectorXd>& u0,
           std::size_t num_samples,
           const Eigen::MatrixXd& sample_covariance,
           double temperature)
    : m_dynamics(dynamics),
      m_cost(cost),
      m_horizon(u0.size()),
      m_num_samples(num_samples),
      m_sigma(Gaussian(Eigen::VectorXd::Zero(sample_covariance.rows()),
                       sample_covariance)),
      m_lambda(temperature),
      m_controls(u0),
      m_rollout_weights(Eigen::VectorXd::Zero(num_samples)) {
  cacheSigmaInv();
}

const Eigen::VectorXd& MPPI::policy(const Distribution& state) {
  auto T = m_horizon;
  auto N = m_num_samples;
  auto x = state.mean();

  // Initialize from the previous trajectory
  if (!m_first_pass) {
    m_controls.erase(m_controls.begin());        // remove the first element
    m_controls.emplace_back(m_controls.back());  // copy the last element
  }
  m_first_pass = false;

  // Reset the rollout history
  m_rollout_states.clear();
  m_rollout_states.reserve(N);

  // Rollout a perturbation around the nominal control for each sample
  std::vector<Trajectory<Eigen::VectorXd>> eps;
  eps.reserve(N);
  Eigen::VectorXd S = Eigen::VectorXd::Zero(N);
  for (std::size_t j = 0; j < N; ++j) {
    Trajectory<Eigen::VectorXd> state_rollout;
    state_rollout.reserve(T);

    x = state.mean();  // TODO: Add option to sample
    state_rollout.emplace_back(x);

    // TODO: can parallelize
    Trajectory<Eigen::VectorXd> samples = m_sigma.samples(T);
    eps.emplace_back(samples);
    for (std::size_t i = 0; i < T - 1; ++i) {
      const auto& u = m_controls.at(i);
      const auto& e = samples.at(i);
      const auto uhat = u + e;
      S(j) += m_cost.c(x, uhat, i) + m_lambda * u.transpose() * m_sigma_inv * e;
      x = m_dynamics.dynamics(x, uhat).mean();  // TODO: Add option to sample
      state_rollout.emplace_back(x);
    }
    S(j) += m_cost.cf(x);
    m_rollout_states.emplace_back(state_rollout);
  }

  // Compute weights
  double beta = S.minCoeff();
  m_rollout_weights = exp(-(S.array() - beta) / m_lambda);
  m_rollout_weights /= m_rollout_weights.sum();

  // Clear the states
  m_states.clear();
  m_states.reserve(T);

  // Update controls and states
  x = state.mean();
  m_states.emplace_back(x);
  for (std::size_t i = 0; i < T - 1; ++i) {
    Eigen::VectorXd e = Eigen::VectorXd::Zero(eps.at(0).at(0).size());
    for (std::size_t j = 0; j < N; ++j) {
      e += m_rollout_weights(j) * eps.at(j).at(i);
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

const std::vector<Trajectory<Eigen::VectorXd>>& MPPI::rolloutStates() const {
  return m_rollout_states;
}

const Eigen::VectorXd& MPPI::rolloutWeights() const {
  return m_rollout_weights;
}

void MPPI::cacheSigmaInv() {
  const Eigen::MatrixXd& sigma = m_sigma.covariance();
  std::size_t n = sigma.rows();
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
  m_sigma_inv = sigma.llt().solve(I);
}

}  // namespace sia
