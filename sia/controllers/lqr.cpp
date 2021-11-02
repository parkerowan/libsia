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
  auto& N = m_horizon;
  auto x = state.mean();
  const auto& Qf = m_cost.Qf();
  const auto& Q = m_cost.Q();
  const auto& R = m_cost.R();
  const auto& F = m_dynamics.F();
  const auto& G = m_dynamics.G();

  // Backward recursion dynamic Ricatti equation to compute the cost to go
  Eigen::MatrixXd P = Qf;
  Eigen::VectorXd v = Qf * m_cost.xd(N - 1);
  Eigen::MatrixXd QuuInv, K;
  Eigen::VectorXd k;
  for (int i = N - 2; i >= 0; --i) {
    bool r = svdInverse(R + G.transpose() * P * G, QuuInv);
    SIA_EXCEPTION(r, "Matrix inversion failed in LQR cost to go computation");
    K = QuuInv * G.transpose() * P * F;
    k = QuuInv * G.transpose() * v;

    const Eigen::MatrixXd FGK = F - G * K;
    P = F.transpose() * P * FGK + Q;
    v = FGK.transpose() * v + Qf * m_cost.xd(i);
  }

  // Compute the optimal feedback gain
  m_control = -K * x + k;
  return m_control;
}

}  // namespace sia
