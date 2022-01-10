/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
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
/// More information on parameters is available in [2].
/// - max_iter: The maximum number of iterations for steps 2 & 3.
/// - max_backsteps: The maximum number of iterations during line search in 3.
/// - epsilon: The threshold of change in cost (dJ) for convergence of 2 & 3.
/// - tau: The backstepping parameter (0 < tau <= 1) in 3.
/// - min_z: The threshold (0 < min_z) of z (actual change in cost over linear
///   predicted change in cost (dJ)) in backstepping in 3.
/// - mu: The regularization (0 <= mu) placing a quadratic state cost around the
///   previous sequence.  Note that [2] describes a regularization schedule,
///   while this implementation uses a constant user-defined regularization
///   parameter.
///
/// References:
/// [1] https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf
/// [2] https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
class iLQR : public Controller {
 public:
  struct Metrics {
    std::size_t iter{0};
    double dJ{0};  // Linearized change in cost
    double J{0};   // Cost based on optimization
    double z{0};   // (J1 - J0) / dJ
    unsigned elapsed_us{0};
    std::size_t backstep_iter{0};
    std::size_t alpha{0};
  };

  explicit iLQR(LinearizableDynamics& dynamics,
                DifferentiableCost& cost,
                const std::vector<Eigen::VectorXd>& u0,
                std::size_t max_iter = 1,
                std::size_t max_backsteps = 1,
                double epsilon = 1e-1,
                double tau = 0.5,
                double min_z = 1e-2,
                double mu = 0);
  virtual ~iLQR() = default;

  /// Performs a single step of the MPC $u = \pi(p(x))$.
  const Eigen::VectorXd& policy(const Distribution& state) override;

  /// Returns the solution control trajectory $U$ over the horizon
  const Trajectory<Eigen::VectorXd>& controls() const override;

  /// Returns the expected solution state trajectory $X$ over the horizon
  const Trajectory<Eigen::VectorXd>& states() const override;

  /// Return the metrics
  const Metrics& getMetrics() const;

 private:
  LinearizableDynamics& m_dynamics;
  DifferentiableCost& m_cost;
  std::size_t m_horizon;
  std::size_t m_max_iter;
  std::size_t m_max_backsteps;
  double m_epsilon;
  double m_tau;
  double m_min_z;
  double m_mu;
  Metrics m_metrics;
  std::vector<Eigen::VectorXd> m_controls;
  std::vector<Eigen::VectorXd> m_states;
};

}  // namespace sia
