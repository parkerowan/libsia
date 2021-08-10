/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/models/models.h"

#include <Eigen/Dense>
#include <map>

namespace sia {

/// Model predictive controllers (MPC) compute the optimal control by simulating
/// the system dynamics for a control trajectory forward in time over a finite
/// horizon.  The first control is applied at the current time step.  MPC is
/// known to be useful for correcting for inaccuracies in the dynamics model,
/// since a new forward simulation is computed with each ne updated state
/// estimate.
class Controller {
 public:
  Controller() = default;
  virtual ~Controller() = default;

  /// Performs a single step of the MPC $u = \pi(p(x))$.
  virtual const Eigen::VectorXd& policy(const Distribution& state) = 0;
};

}  // namespace sia