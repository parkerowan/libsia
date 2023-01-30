/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/controllers/controllers.h"
#include "sia/controllers/cost.h"
#include "sia/models/linear_gaussian.h"

#include <Eigen/Dense>

namespace sia {

/// This class implements a finite horizon discrete time linear quadratic
/// regular (LQR).  LQR is an optimal control (MPC) for a linear system and a
/// quadratic cost, minimizing
/// $min J = x_T' Q_f x_T + \sum_{i=0}^{T-1} (x' Q x + u' R u)$
/// where:
/// - $Q_f$ is the final state cost
/// - $Q$ is the running state cost
/// - $R$ is the running input cost
/// - $T$ is the horizon
///
/// LQR requires linear dynamics and quadratic cost.  The algorithm is based on
/// Table 2 in [1] which extends the traditional LQR to tracking control.
///
/// References:
/// [1] https://web.mst.edu/~bohner/papers/tlqtots.pdf
class LQR : public Controller {
 public:
  struct Metrics : public BaseMetrics {
    double cost{0};
  };

  explicit LQR(LinearGaussianDynamics& dynamics,
               QuadraticCost& cost,
               std::size_t horizon);
  virtual ~LQR() = default;

  /// Performs a single step of the MPC $u = \pi(p(x))$.
  const Eigen::VectorXd& policy(const Distribution& state) override;

  /// Returns the solution control trajectory $U$ over the horizon
  const Trajectory<Eigen::VectorXd>& controls() const override;

  /// Returns the expected solution state trajectory $X$ over the horizon
  const Trajectory<Eigen::VectorXd>& states() const override;

  /// Return metrics from the latest step
  const Metrics& metrics() const override;

  /// Access policy terms
  const Trajectory<Eigen::VectorXd>& feedforward() const;
  const Trajectory<Eigen::MatrixXd>& feedback() const;

 private:
  LinearGaussianDynamics& m_dynamics;
  QuadraticCost& m_cost;
  std::size_t m_horizon;
  Metrics m_metrics;
  Trajectory<Eigen::VectorXd> m_controls;
  Trajectory<Eigen::VectorXd> m_states;
  std::vector<Eigen::VectorXd> m_feedforward;
  std::vector<Eigen::MatrixXd> m_feedback;
};

}  // namespace sia
