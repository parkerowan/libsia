/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"
#include "sia/models/models.h"
#include "sia/models/trajectory.h"

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

  /// Returns the solution control trajectory $U$ over the horizon
  virtual const Trajectory<Eigen::VectorXd>& controls() const = 0;

  /// Returns the expected solution state trajectory $X$ over the horizon
  virtual const Trajectory<Eigen::VectorXd>& states() const = 0;
};

}  // namespace sia
