/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/controllers/lqr.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

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

  m_feedforward.clear();
  m_feedforward.reserve(T);
  m_feedback.clear();
  m_feedback.reserve(T);

  // 1. Backward recursion dynamic Ricatti equation to compute the cost to go
  // Initialize terminal value function Gradient and Hessian
  Eigen::MatrixXd P = Qf;
  Eigen::VectorXd v = -Qf * m_cost.xd(T);
  Eigen::MatrixXd QuuInv;

  for (int k = T - 1; k >= 0; --k) {
    const Eigen::MatrixXd Quu = R + G.transpose() * P * G;
    bool r = svdInverse(Quu, QuuInv);
    SIA_THROW_IF_NOT(r,
                     "Matrix inversion failed in LQR cost to go computation");
    const Eigen::MatrixXd K = -QuuInv * G.transpose() * P * F;
    const Eigen::VectorXd d = -QuuInv * G.transpose() * v;
    m_feedback.emplace_back(K);
    m_feedforward.emplace_back(d);

    const Eigen::MatrixXd FGK = F + G * K;
    P = F.transpose() * P * FGK + Q;
    v = FGK.transpose() * v - Q * m_cost.xd(k);
  }

  // Flip the gain vector/matrices order to advance forward in time
  std::reverse(m_feedback.begin(), m_feedback.end());
  std::reverse(m_feedforward.begin(), m_feedforward.end());

  // 2. Forward pass to compute the optimal control and states
  for (std::size_t k = 0; k < T; ++k) {
    const Eigen::VectorXd u = m_feedback.at(k) * x + m_feedforward.at(k);
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

const Trajectory<Eigen::VectorXd>& LQR::feedforward() const {
  return m_feedforward;
}

const Trajectory<Eigen::MatrixXd>& LQR::feedback() const {
  return m_feedback;
}

}  // namespace sia
