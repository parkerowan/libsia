/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/controllers/ilqr.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

#include <chrono>

#define SMALL_NUMBER 1e-12

namespace sia {

iLQR::iLQR(LinearizableDynamics& dynamics,
           DifferentiableCost& cost,
           const std::vector<Eigen::VectorXd>& u0,
           const iLQR::Options& options)
    : m_dynamics(dynamics),
      m_cost(cost),
      m_horizon(u0.size()),
      m_options(options),
      m_controls(u0) {}

const Eigen::VectorXd& iLQR::policy(const Distribution& state) {
  m_metrics = iLQR::Metrics();
  auto T = m_horizon;

  // Initialize from the previous trajectory
  if (!m_first_pass) {
    m_controls.erase(m_controls.begin());        // remove the first element
    m_controls.emplace_back(m_controls.back());  // copy the last element
  }
  m_first_pass = false;

  // Rollout the dynamics for the initial control
  m_states.clear();
  m_states.reserve(T);
  m_states.emplace_back(state.mean());
  for (std::size_t k = 0; k < T - 1; ++k) {
    m_states.emplace_back(m_dynamics.f(m_states.at(k), m_controls.at(k)));
  }
  m_final_state = m_dynamics.f(m_states.at(T - 1), m_controls.at(T - 1));

  // Inner LQR loop for fixed iteration or until convergence
  std::size_t lqr_iter = 0;
  double dJ = 0;
  double J = m_cost.eval(m_states, m_controls);
  do {
    // 1. Backward pass
    double dJa = 0;
    double dJb = 0;
    backwardPass(m_dynamics, m_cost, m_states, m_controls, m_final_state,
                 m_feedforward, m_feedback, dJa, dJb, m_options, m_metrics);

    // 2. Forward pass
    forwardPass(m_dynamics, m_cost, m_states, m_controls, m_final_state,
                m_feedforward, m_feedback, dJa, dJb, dJ, J, m_options,
                m_metrics);

    lqr_iter++;
  } while ((lqr_iter < m_options.max_lqr_iter) &&
           (abs(dJ) > m_options.cost_tol));

  // Populate metrics
  m_metrics.lqr_iter = lqr_iter;
  m_metrics.clockElapsedUs();
  return m_controls.at(0);
}

const Trajectory<Eigen::VectorXd>& iLQR::controls() const {
  return m_controls;
}

const Trajectory<Eigen::VectorXd>& iLQR::states() const {
  return m_states;
}

const Trajectory<Eigen::VectorXd>& iLQR::feedforward() const {
  return m_feedforward;
}

const Trajectory<Eigen::MatrixXd>& iLQR::feedback() const {
  return m_feedback;
}

const iLQR::Metrics& iLQR::metrics() const {
  return m_metrics;
}

void backwardPass(LinearizableDynamics& dynamics,
                  DifferentiableCost& cost,
                  const std::vector<Eigen::VectorXd>& states,
                  const std::vector<Eigen::VectorXd>& controls,
                  const Eigen::VectorXd& final_state,
                  std::vector<Eigen::VectorXd>& feedforward,
                  std::vector<Eigen::MatrixXd>& feedback,
                  double& dJa,
                  double& dJb,
                  const iLQR::Options& options,
                  iLQR::Metrics& metrics) {
  std::size_t T = states.size();
  std::size_t m = controls.at(T - 1).size();

  // Regularization parameters
  double rho = options.regularization_init;
  Eigen::MatrixXd Iu = Eigen::MatrixXd::Identity(m, m);

  // Loop until we find a regularization that yeilds pos def Quu
  std::size_t num_attempts = 0;
  bool recompute_pass = false;
  do {
    recompute_pass = false;

    // Equations (38-39) from [3]
    Eigen::VectorXd Vpx = cost.cfx(final_state);
    Eigen::MatrixXd Vpxx = cost.cfxx(final_state);

    // Integrate backwards from terminal goal
    feedforward.clear();
    feedforward.reserve(T);
    feedback.clear();
    feedback.reserve(T);

    dJa = 0;
    dJb = 0;
    for (int k = T - 1; k >= 0; --k) {
      const auto& x = states.at(k);
      const auto& u = controls.at(k);

      const Eigen::VectorXd lx = cost.cx(x, u, k);
      const Eigen::VectorXd lu = cost.cu(x, u, k);
      const Eigen::MatrixXd lxx = cost.cxx(x, u, k);
      const Eigen::MatrixXd lux = cost.cux(x, u, k);
      const Eigen::MatrixXd luu = cost.cuu(x, u, k);
      const Eigen::MatrixXd fx = dynamics.F(x, u);  // d/dx f(x,u)
      const Eigen::MatrixXd fu = dynamics.G(x, u);  // d/du f(x,u)

      // Equations (41-45) from [3]
      Eigen::MatrixXd Qxx = lxx + fx.transpose() * Vpxx * fx;
      Eigen::MatrixXd Quu = luu + fu.transpose() * Vpxx * fu;
      const Eigen::MatrixXd Qux = lux + fu.transpose() * Vpxx * fx;
      const Eigen::VectorXd Qx = lx + fx.transpose() * Vpx;
      const Eigen::VectorXd Qu = lu + fu.transpose() * Vpx;

      // Section III-B-2) from [3]
      // Find a regularization to make Quu converge, re-run the pass
      Eigen::MatrixXd QuuReg = Quu + rho * Iu;
      while (!positiveDefinite(QuuReg) &&
             (num_attempts < options.max_regularization_iter)) {
        recompute_pass = true;
        rho = std::max(rho * options.regularization_rate,
                       options.regularization_min);
        SIA_INFO("Increasing Quu regularization " << rho);
        QuuReg = Quu + rho * Iu;
        k = 0;
        num_attempts++;
      }

      // Invert Quu
      Eigen::MatrixXd QuuInv;
      bool r = svdInverse(QuuReg, QuuInv);
      SIA_THROW_IF_NOT(r, "Failed to invert Quu in iLQR backward pass");

      // Equation (46) from [3]
      const Eigen::MatrixXd K = -QuuInv * Qux;
      const Eigen::VectorXd d = -QuuInv * Qu;
      feedback.emplace_back(K);
      feedforward.emplace_back(d);

      // Equations (47-48) from [3]
      Vpx = Qx + K.transpose() * (Quu * d + Qu) + Qux.transpose() * d;
      Vpxx = Qxx + K.transpose() * (Quu * K + Qux) + Qux.transpose() * K;
      Vpxx = Vpxx.selfadjointView<Eigen::Upper>();

      // Equation (49) from [3]
      double dVa = d.transpose() * Qu;
      double dVb = 0.5 * d.transpose() * Quu * d;
      dJa += dVa;
      dJb += dVb;
    }
  } while (recompute_pass && (num_attempts < options.max_regularization_iter));

  // If we ran out of recomputes, Quu is divergent and the solution not useful
  SIA_THROW_IF_NOT(num_attempts < options.max_regularization_iter,
                   "Failed to find a regularization within max iterations");

  // Record metrics
  metrics.rho.emplace_back(rho);

  // Flip the gain vector/matrices order to advance forward in time
  std::reverse(feedforward.begin(), feedforward.end());
  std::reverse(feedback.begin(), feedback.end());
}

void forwardPass(LinearizableDynamics& dynamics,
                 DifferentiableCost& cost,
                 std::vector<Eigen::VectorXd>& states,
                 std::vector<Eigen::VectorXd>& controls,
                 Eigen::VectorXd& final_state,
                 const std::vector<Eigen::VectorXd>& feedforward,
                 const std::vector<Eigen::MatrixXd>& feedback,
                 double dJa,
                 double dJb,
                 double& dJ,
                 double& J0,
                 const iLQR::Options& options,
                 iLQR::Metrics& metrics) {
  std::size_t T = states.size();
  double J1 = 0;
  double z = 0;

  double alpha = 1;
  std::size_t backtrack_iter = 0;
  std::vector<Eigen::VectorXd> new_states, new_controls;
  do {
    new_states.clear();
    new_controls.clear();
    new_states.reserve(T);
    new_controls.reserve(T);
    Eigen::VectorXd xhat = states.at(0);
    for (std::size_t k = 0; k < T; ++k) {
      // Equations (50-52) from [3]
      const Eigen::VectorXd& u = controls.at(k);
      const Eigen::VectorXd& x = states.at(k);
      const Eigen::VectorXd& d = feedforward.at(k);
      const Eigen::MatrixXd& K = feedback.at(k);
      Eigen::VectorXd uhat = u + alpha * d + K * (xhat - x);

      // Equation (53) from [3] to integrate the control through system dynamics
      new_controls.emplace_back(uhat);
      new_states.emplace_back(xhat);
      xhat = dynamics.f(xhat, uhat);
    }
    final_state = xhat;

    // Equation (55) from [3] Improved Line Search
    dJ = alpha * (dJa + alpha * dJb);

    // Equation (54) from [3] Compute convergence
    J1 = cost.eval(new_states, new_controls);
    z = 1;
    if (abs(dJ) > SMALL_NUMBER) {
      z = (J1 - J0) / dJ;
    }

    // Record metrics
    metrics.dJ.emplace_back(dJ);
    metrics.z.emplace_back(z);
    metrics.alpha.emplace_back(alpha);
    metrics.cost.emplace_back(J1);

    // Backtracking iteration - shrink alpha (feedfoward contribution)
    alpha *= options.linesearch_rate;
    backtrack_iter++;
  } while (
      (backtrack_iter < options.max_linesearch_iter) &&
      ((z < options.linesearch_tol_lb) || (z > options.linesearch_tol_ub)));

  // Accept and update iteration
  states = new_states;
  controls = new_controls;
  J0 = J1;
}

}  // namespace sia
