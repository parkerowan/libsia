/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/controllers/ilqr.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

#include <glog/logging.h>
#include <chrono>

#define SMALL_NUMBER 1e-12

namespace sia {

// TODO: Make a common function to record metrics for estimators, controllers.
using steady_clock = std::chrono::steady_clock;
static unsigned get_elapsed_us(steady_clock::time_point tic,
                               steady_clock::time_point toc) {
  return std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
      .count();
};

iLQR::iLQR(LinearizableDynamics& dynamics,
           DifferentiableCost& cost,
           const std::vector<Eigen::VectorXd>& u0,
           std::size_t max_iter,
           std::size_t max_backsteps,
           double epsilon,
           double tau,
           double min_z,
           double mu)
    : m_dynamics(dynamics),
      m_cost(cost),
      m_horizon(u0.size()),
      m_max_iter(max_iter),
      m_max_backsteps(max_backsteps),
      m_epsilon(epsilon),
      m_tau(tau),
      m_min_z(min_z),
      m_mu(mu),
      m_controls(u0) {}

const Eigen::VectorXd& iLQR::policy(const Distribution& state) {
  auto tic = steady_clock::now();
  auto T = m_horizon;

  // Shift the control through the buffer so that we use the previously computed
  // cost to initialize the trajectory.
  m_controls.erase(m_controls.begin());        // remove the first element
  m_controls.emplace_back(m_controls.back());  // copy the last element

  // Rollout the dynamics for the initial control
  m_states.clear();
  m_states.reserve(T);
  m_states.emplace_back(state.mean());
  for (std::size_t i = 0; i < T - 1; ++i) {
    m_states.emplace_back(m_dynamics.f(m_states.at(i), m_controls.at(i)));
  }

  // Loop for fixed iteration or until convergence
  std::size_t j = 0;
  double dJ;
  double J0 = m_cost.eval(m_states, m_controls);
  double J1 = 0;
  double z = 0;
  do {
    // 1. Backward pass
    std::vector<Eigen::VectorXd> feedforward;
    std::vector<Eigen::MatrixXd> feedback;
    feedforward.reserve(T);
    feedback.reserve(T);

    // Initialize terminal value function Gradient and Hessian
    double dJa = 0;
    double dJb = 0;
    Eigen::VectorXd Vpx = m_cost.cfx(m_states.at(T - 1));
    Eigen::MatrixXd Vpxx = m_cost.cfxx(m_states.at(T - 1));
    std::size_t n = m_states.at(T - 1).size();
    for (int i = T - 1; i >= 0; --i) {
      const auto& x = m_states.at(i);
      const auto& u = m_controls.at(i);

      // Compute eqns (5) to find the Q value coefficients based on the value at
      // the next time step (initialize above)
      // NOTE: DDP tensor terms are currently ommited to simplify complexity.
      const Eigen::VectorXd lx = m_cost.cx(x, u, i);
      const Eigen::VectorXd lu = m_cost.cu(x, u, i);
      const Eigen::MatrixXd lxx = m_cost.cxx(x, u, i);
      const Eigen::MatrixXd lux = m_cost.cux(x, u, i);
      const Eigen::MatrixXd luu = m_cost.cuu(x, u, i);
      const Eigen::MatrixXd fx = m_dynamics.F(x, u);  // df/dx f(x,u)
      const Eigen::MatrixXd fu = m_dynamics.G(x, u);  // df/du f(x,u)

      const Eigen::MatrixXd muI = m_mu * Eigen::MatrixXd::Identity(n, n);
      const Eigen::VectorXd Qx = lx + fx.transpose() * Vpx;
      const Eigen::VectorXd Qu = lu + fu.transpose() * Vpx;
      const Eigen::MatrixXd Qxx = lxx + fx.transpose() * Vpxx * fx;
      const Eigen::MatrixXd Qux =
          lux.transpose() + fu.transpose() * (Vpxx + muI) * fx;
      const Eigen::MatrixXd Quu = luu + fu.transpose() * (Vpxx + muI) * fu;

      // Check if Quu positive definite
      Eigen::VectorXd v = Eigen::VectorXd::Ones(Quu.rows());
      if (v.transpose() * Quu * v <= 0) {
        LOG(WARNING) << "Quu is not positive definite, increase mu value";
      }

      // Compute the inverse of Quu
      Eigen::MatrixXd QuuInv;
      bool r = svdInverse(Quu, QuuInv);
      SIA_EXCEPTION(r, "Failed to invert Quu in iLQR backward pass");

      // Compute eqns (6) to find the gains k, K for the next forward pass
      const Eigen::MatrixXd K = -QuuInv * Qux;
      const Eigen::VectorXd k = -QuuInv * Qu;
      feedback.emplace_back(K);
      feedforward.emplace_back(k);

      // Compute eqns (11) to update the value function (cost to go
      // approximation) for the backward recursion
      // dV = -Qu * QuuInv * Qu / 2
      // Vpx = Qx - Qu * QuuInv * Qux
      // Vpxx = Qxx - Qux.transpose() * QuuInv * Qux
      Vpx = Qx + K.transpose() * (Quu * k + Qu) + Qux.transpose() * k;
      Vpxx = Qxx + K.transpose() * (Quu * K + Qux) + Qux.transpose() * K;
      Vpxx = 0.5 * (Vpxx + Vpxx.transpose());
      double dVa = k.transpose() * Qu;
      double dVb = 0.5 * k.transpose() * Quu * k;

      // $V(x) = min_u J(x, u)$
      // dJ = dJa + dJb
      dJa += dVa;
      dJb += dVb;
    }

    // Flip the gain vector/matrices order to advance forward in time
    std::reverse(feedforward.begin(), feedforward.end());
    std::reverse(feedback.begin(), feedback.end());

    // 2. Forward pass
    double alpha = 1;
    std::size_t k = 0;
    std::vector<Eigen::VectorXd> new_states, new_controls;
    do {
      new_states.clear();
      new_controls.clear();
      new_states.reserve(T);
      new_controls.reserve(T);
      Eigen::VectorXd xhat = m_states.at(0);
      for (std::size_t i = 0; i < T; ++i) {
        // Compute eqns (8a-b) to find the control law
        const Eigen::VectorXd& u = m_controls.at(i);
        const Eigen::VectorXd& x = m_states.at(i);
        const Eigen::VectorXd& k = feedforward.at(i);
        const Eigen::MatrixXd& K = feedback.at(i);
        Eigen::VectorXd uhat = u + alpha * k + K * (xhat - x);

        // Compute eqn (8c) to integrate the control through system dynamics
        new_controls.emplace_back(uhat);
        new_states.emplace_back(xhat);
        xhat = m_dynamics.f(xhat, uhat);
      }

      // From Y. Tassa et al 2012, "Section II-D Improved Line Search"
      dJ = alpha * (dJa + alpha * dJb);

      // Compute convergence
      J1 = m_cost.eval(new_states, new_controls);
      z = 1;
      if (abs(dJ) > SMALL_NUMBER) {
        z = (J1 - J0) / dJ;
      }

      // Backtracking iteration - shrink alpha (feedfoward contribution)
      alpha *= m_tau;
      k++;
    } while ((k < m_max_backsteps) && (z < m_min_z));

    // Debug iteration
    m_metrics.z = z;
    m_metrics.backstep_iter = k;
    m_metrics.alpha = alpha / m_tau;

    // Accept and update iteration
    m_states = new_states;
    m_controls = new_controls;
    J0 = J1;
    j++;
  } while ((j < m_max_iter) && (abs(dJ) > m_epsilon));

  // Populate metrics
  auto toc = steady_clock::now();
  m_metrics.elapsed_us = get_elapsed_us(tic, toc);
  m_metrics.iter = j;
  m_metrics.dJ = dJ;
  m_metrics.J = J1;
  return m_controls.at(0);
}

const Trajectory<Eigen::VectorXd>& iLQR::controls() const {
  return m_controls;
}

const Trajectory<Eigen::VectorXd>& iLQR::states() const {
  return m_states;
}

const iLQR::Metrics& iLQR::metrics() const {
  return m_metrics;
}

}  // namespace sia
