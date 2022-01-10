/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/controllers/lqr.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

#include <glog/logging.h>

namespace sia {

LQR::LQR(LinearGaussianDynamics& dynamics,
         QuadraticCost& cost,
         std::size_t horizon)
    : m_dynamics(dynamics), m_cost(cost), m_horizon(horizon) {}

const Eigen::VectorXd& LQR::policy(const Distribution& state) {
  auto T = m_horizon;
  auto x = state.mean();
  const auto& Qf = m_cost.Qf();
  const auto& Q = m_cost.Q();
  const auto& R = m_cost.R();
  const auto& F = m_dynamics.F();
  const auto& G = m_dynamics.G();

  m_controls.clear();
  m_controls.reserve(T);

  m_states.clear();
  m_states.reserve(T);

  // 1. Backward recursion dynamic Ricatti equation to compute the cost to go
  std::vector<Eigen::MatrixXd> feedback;
  std::vector<Eigen::VectorXd> feedforward;
  feedback.reserve(T);
  feedforward.reserve(T);

  // Initialize terminal value function Gradient and Hessian
  Eigen::MatrixXd P = Qf;
  Eigen::VectorXd v = Qf * m_cost.xd(T - 1);
  Eigen::MatrixXd QuuInv;

  for (int i = T - 1; i >= 0; --i) {
    bool r = svdInverse(R + G.transpose() * P * G, QuuInv);
    SIA_EXCEPTION(r, "Matrix inversion failed in LQR cost to go computation");
    const Eigen::MatrixXd K = QuuInv * G.transpose() * P * F;
    const Eigen::VectorXd k = QuuInv * G.transpose() * v;
    feedback.emplace_back(K);
    feedforward.emplace_back(k);

    const Eigen::MatrixXd FGK = F - G * K;
    P = F.transpose() * P * FGK + Q;
    v = FGK.transpose() * v + Qf * m_cost.xd(i);
  }

  // Flip the gain vector/matrices order to advance forward in time
  std::reverse(feedback.begin(), feedback.end());
  std::reverse(feedforward.begin(), feedforward.end());

  // 2. Forward pass to compute the optimal control and states
  for (std::size_t i = 0; i < T; ++i) {
    const Eigen::VectorXd u = -feedback.at(i) * x + feedforward.at(i);
    m_controls.emplace_back(u);
    m_states.emplace_back(x);
    x = m_dynamics.f(x, u);
  }

  return m_controls.at(0);
}

const Trajectory<Eigen::VectorXd>& LQR::controls() const {
  return m_controls;
}

const Trajectory<Eigen::VectorXd>& LQR::states() const {
  return m_states;
}

}  // namespace sia
