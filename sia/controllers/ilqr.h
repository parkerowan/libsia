/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/controllers/controllers.h"
#include "sia/controllers/cost.h"
#include "sia/models/nonlinear_gaussian.h"

#include <Eigen/Dense>

namespace sia {

/// This class implements an iterative (i)-LQR controller.  iLQR is an
/// optimization-based controller (MPC) for a nonlinear system and arbitrary
/// cost, minimizing $min J = l_f(x_T) + \sum_{i=0}^{T-1} l(x, u)$ where:
/// - $l_f$ is the final cost
/// - $l$ is the running cost
/// - $T$ is the horizon
///
/// iLQR requires the nonlinear dynamics and the cost function to be
/// differentiable at each time step.  The algorithm is based on [1] Y. Tassa
/// et. al, 2012.  In each policy call, the algorithm performs the following
/// steps
/// 1. Preprocess: shift the previous control history forward by one time step
/// to reuse the previous solution.
/// Repeat until convergence:
/// 2. Backward pass: Compute the Q-function (state-action value) by integrating
/// backwards from the final cost and computing Gradients and Hessians of
/// dynamics and cost.
/// 3. Forward pass: Compute the control (action) by integrating forward using
/// the Q function computed in the backward pass.  Perform backtracking line
/// search to find the amount of feedforward contribution.
///
/// More information on parameters is available in Table I from [3].
/// - max_lqr_iter: (>0) Limit on number of iterations of steps 2 & 3.
/// - cost_tol: (>0) Termination criteria on change in cost after 2 & 3.
/// - max_regularization_iter: (>0) Limit on regularization increases in 2.
/// - regularization_init: (>=0) Initial regularization value in 2.
/// - regularization_min: (>0) Baseline value when adding regularization in 2.
/// - regularization_rate: (>1) Growth rate when increasing regularization in 2.
/// - max_linesearch_iter: (>0) Limit on line search iterations in 3.
/// - linesearch_rate: (>0, <1) Attenuation rate when backstepping in 3.
/// - linesearch_tol_lb: (>0) Limit on cost change vs linear prediction in 3.
/// - linesearch_tol_ub: (>lb) Limit on cost change vs linear prediction in 3.
///
/// References:
/// [1] https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
/// [2] https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
/// [3] https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf
class iLQR : public Controller {
 public:
  struct Metrics {
    unsigned elapsed_us{0};
    std::size_t lqr_iter{0};
    std::vector<double> rho{};     // Quu regularization
    std::vector<double> dJ{0};     // Linearized change in cost
    std::vector<double> z{0};      // (J1 - J0) / dJ
    std::vector<double> alpha{0};  // Feedforward scale
    std::vector<double> J{0};      // Cost based on optimization
  };

  explicit iLQR(LinearizableDynamics& dynamics,
                DifferentiableCost& cost,
                const std::vector<Eigen::VectorXd>& u0,
                std::size_t max_lqr_iter = 50,
                double cost_tol = 1e-4,
                std::size_t max_regularization_iter = 10,
                double regularization_init = 0,
                double regularization_min = 1e-4,
                double regularization_rate = 1.6,
                std::size_t max_linesearch_iter = 10,
                double linesearch_rate = 0.5,
                double linesearch_tol_lb = 1e-8,
                double linesearch_tol_ub = 10);

  virtual ~iLQR() = default;

  /// Performs a single step of the MPC $u = \pi(p(x))$.
  const Eigen::VectorXd& policy(const Distribution& state) override;

  /// Returns the solution control trajectory $U$ over the horizon
  const Trajectory<Eigen::VectorXd>& controls() const override;

  /// Returns the expected solution state trajectory $X$ over the horizon
  const Trajectory<Eigen::VectorXd>& states() const override;

  /// Access policy terms
  const Trajectory<Eigen::VectorXd>& feedforward() const;
  const Trajectory<Eigen::MatrixXd>& feedback() const;

  /// Return the metrics
  const Metrics& metrics() const;

 private:
  LinearizableDynamics& m_dynamics;
  DifferentiableCost& m_cost;
  std::size_t m_horizon;
  std::size_t m_max_lqr_iter;
  double m_cost_tol;
  std::size_t m_max_regularization_iter;
  double m_regularization_init;
  double m_regularization_min;
  double m_regularization_rate;
  std::size_t m_max_linesearch_iter;
  double m_linesearch_rate;
  double m_linesearch_tol_lb;
  double m_linesearch_tol_ub;
  Metrics m_metrics;
  std::vector<Eigen::VectorXd> m_controls;
  std::vector<Eigen::VectorXd> m_states;
  Eigen::VectorXd m_final_state;
  std::vector<Eigen::VectorXd> m_feedforward;
  std::vector<Eigen::MatrixXd> m_feedback;
  bool m_first_pass{true};
};

// Algorithm 1 from [3]
void backwardPass(LinearizableDynamics& dynamics,
                  DifferentiableCost& cost,
                  const std::vector<Eigen::VectorXd>& states,
                  const std::vector<Eigen::VectorXd>& controls,
                  const Eigen::VectorXd& final_state,
                  std::vector<Eigen::VectorXd>& feedforward,
                  std::vector<Eigen::MatrixXd>& feedback,
                  double& dJa,
                  double& dJb,
                  std::size_t max_regularization_iter,
                  double regularization_init,
                  double regularization_min,
                  double regularization_rate,
                  iLQR::Metrics& metrics);

// Algorithm 2 from [3]
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
                 std::size_t max_linesearch_iter,
                 double linesearch_rate,
                 double linesearch_tol_lb,
                 double linesearch_tol_ub,
                 iLQR::Metrics& metrics);

}  // namespace sia
