/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/belief/gaussian.h"
#include "sia/controllers/controllers.h"
#include "sia/controllers/cost.h"
#include "sia/models/models.h"

#include <Eigen/Dense>

namespace sia {

/// This class implements a finite horizon discrete time model-predictive path
/// integral controller (MPPI).  MPPI is an optimal control (MPC) for a general
/// nonlinear system and cost, minimizing $min J = \phi(x) \sum_i=0^T-1 q(x)$
///
/// MPPI uses a sampling based approach.  The algorithm simulates multiple
/// trajectories forward in time for different perturbations in control.  The
/// trajectories are then weighted based on cost and control, and the returned
/// control is the weighted average of the controls.
///
/// Note that the derivaton of MPPI in [1] does not assume a control cost, while
/// this class accepts cost functions that support cost on controls.  To match
/// the original algorithm results, weight the control cost to 0 in quadratic
/// algorithms or do not implement control cost in custom cost functions.
///
/// More information on parameters is available in [1].
/// - sample_covariance: (>0) Covariance matrix to sample control perturbations.
///   Must be a square matrix with the same dimension as the control.
/// - num_samples: (>0) Number of sample trajectories to rollout.
/// - temperature: (>0) Penalizes the perturbation magnitude.
///
/// References:
/// [1]
/// https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf
class MPPI : public Controller {
 public:
  struct Metrics : public BaseMetrics {
    double cost{0};
  };

  /// Algorithm options
  struct Options {
    explicit Options() {}
    std::size_t num_samples = 10;
    double temperature = 1.0;
  };

  explicit MPPI(DynamicsModel& dynamics,
                CostFunction& cost,
                const std::vector<Eigen::VectorXd>& u0,
                const Eigen::MatrixXd& sample_covariance,
                const Options& options = Options());
  virtual ~MPPI() = default;

  /// Performs a single step of the MPC $u = \pi(p(x))$.
  const Eigen::VectorXd& policy(const Distribution& state) override;

  /// Returns the solution control trajectory $U$ over the horizon
  const Trajectory<Eigen::VectorXd>& controls() const override;

  /// Returns the expected solution state trajectory $X$ over the horizon
  const Trajectory<Eigen::VectorXd>& states() const override;

  /// Return metrics from the latest step
  const Metrics& metrics() const override;

  /// Returns the sampled state trajectories $X$ over the horizon
  const std::vector<Trajectory<Eigen::VectorXd>>& rolloutStates() const;

  /// Returns the weight vector for each sampled rollout
  const Eigen::VectorXd& rolloutWeights() const;

 private:
  void cacheSigmaInv();

  DynamicsModel& m_dynamics;
  CostFunction& m_cost;
  std::size_t m_horizon;
  Options m_options;
  Metrics m_metrics;
  Gaussian m_sigma;
  Eigen::MatrixXd m_sigma_inv;
  Trajectory<Eigen::VectorXd> m_controls;
  Trajectory<Eigen::VectorXd> m_states;
  std::vector<Trajectory<Eigen::VectorXd>> m_rollout_states;
  Eigen::VectorXd m_rollout_weights;
  bool m_first_pass{true};
};

}  // namespace sia
