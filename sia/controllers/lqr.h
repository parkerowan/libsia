/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
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
  explicit LQR(LinearGaussian& system,
               QuadraticCost& cost,
               std::size_t horizon);
  virtual ~LQR() = default;

  /// Performs a single step of the MPC $u = \pi(p(x))$.
  const Eigen::VectorXd& policy(const Distribution& state) override;

 private:
  LinearGaussian& m_system;
  QuadraticCost& m_cost;
  Eigen::VectorXd m_control;
  std::size_t m_horizon;
};

}  // namespace sia
