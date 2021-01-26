/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/models/models.h"

#include <Eigen/Dense>
#include <vector>

namespace sia {

/// Collects a trace of states, controls, and measurements from simulating a
/// system forward in time.  Each column is a snapshot in time.
struct Trajectory {
  Eigen::MatrixXd states;
  Eigen::MatrixXd controls;
  Eigen::MatrixXd measurements;
};

/// Collects a set of trajectories
class Trajectories {
 public:
  explicit Trajectories(const std::vector<Trajectory>& data);
  const std::vector<Trajectory>& data() const;
  std::size_t size() const;

  /// Retrieves the states from a set of trajectories at a snapshot $k$.
  const Eigen::MatrixXd states(std::size_t k);

  /// Retrieves the controls from a set of trajectories at a snapshot $k$.
  const Eigen::MatrixXd controls(std::size_t k);

  /// Retrieves the measurements from a set of trajectories at a snapshot $k$.
  const Eigen::MatrixXd measurements(std::size_t k);

 private:
  std::vector<Trajectory> m_data;
};

/// Simulates a Markov process forward in time, starting from an initial state
/// vector $x$ and sequence of control $\{u\}_k$.  Returns a single trajectory.
Trajectory simulate(MarkovProcess& system,
                    const Eigen::VectorXd& state,
                    const Eigen::MatrixXd& controls,
                    bool sample = true);

/// Simulates a Markov process forward in time, starting from a set of initial
/// state vectors $\{x\}_i$ and sequence of control $\{u\}_k$.  Returns a
/// trajectory for each initial state.
Trajectories simulate(MarkovProcess& system,
                      const std::vector<Eigen::VectorXd>& states,
                      const Eigen::MatrixXd& controls,
                      bool sample = true);

/// Simulates a Markov process forward in time, starting from a set of initial
/// state vectors $\{x\}_i$ and set of control sequences $\{\{u\}_k\}_i$.
/// Returns a trajectory for each initial state/control pair.
Trajectories simulate(MarkovProcess& system,
                      const std::vector<Eigen::VectorXd>& states,
                      const std::vector<Eigen::MatrixXd>& controls,
                      bool sample = true);

}  // namespace sia
