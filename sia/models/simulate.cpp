/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/simulate.h"

namespace sia {

Trajectories::Trajectories(const std::vector<Trajectory>& data)
    : m_data(data) {}

const std::vector<Trajectory>& Trajectories::data() const {
  return m_data;
}

std::size_t Trajectories::size() const {
  return m_data.size();
}

const Eigen::MatrixXd Trajectories::states(std::size_t k) {
  assert(m_data.size() > 0);
  std::size_t n = m_data.at(0).states.rows();
  std::size_t m = m_data.size();
  Eigen::MatrixXd states = Eigen::MatrixXd::Zero(n, m);
  for (std::size_t i = 0; i < m; ++i) {
    states.col(i) = m_data.at(i).states.col(k);
  }
  return states;
}

const Eigen::MatrixXd Trajectories::controls(std::size_t k) {
  assert(m_data.size() > 0);
  std::size_t n = m_data.at(0).controls.rows();
  std::size_t m = m_data.size();
  Eigen::MatrixXd controls = Eigen::MatrixXd::Zero(n, m);
  for (std::size_t i = 0; i < m; ++i) {
    controls.col(i) = m_data.at(i).controls.col(k);
  }
  return controls;
}

const Eigen::MatrixXd Trajectories::measurements(std::size_t k) {
  assert(m_data.size() > 0);
  std::size_t n = m_data.at(0).measurements.rows();
  std::size_t m = m_data.size();
  Eigen::MatrixXd measurements = Eigen::MatrixXd::Zero(n, m);
  for (std::size_t i = 0; i < m; ++i) {
    measurements.col(i) = m_data.at(i).measurements.col(k);
  }
  return measurements;
}

Trajectory simulate(DynamicsModel& dynamics,
                    MeasurementModel& measurement,
                    const Eigen::VectorXd& state,
                    const Eigen::MatrixXd& controls,
                    bool sample) {
  Trajectory traj;
  std::size_t n = controls.cols();
  Eigen::VectorXd x = state;
  for (std::size_t i = 0; i < n; ++i) {
    const auto u = controls.col(i);

    // Integrate the dynamics forward one step
    auto& px = dynamics.dynamics(x, u);

    // Optionally sample the new state
    if (sample) {
      x = px.sample();
    } else {
      x = px.mean();
    }

    // Return the measurement distribution
    auto& py = measurement.measurement(x);

    // Optionally sample the measurement
    Eigen::VectorXd y(py.dimension());
    if (sample) {
      y = py.sample();
    } else {
      y = py.mean();
    }

    // Set the sizes
    if (i == 0) {
      traj.states.setZero(x.size(), n);
      traj.controls.setZero(u.size(), n);
      traj.measurements.setZero(y.size(), n);
    }

    // Append
    traj.states.col(i) = x;
    traj.controls.col(i) = u;
    traj.measurements.col(i) = y;
  }

  return traj;
}

Trajectories simulate(DynamicsModel& dynamics,
                      MeasurementModel& measurement,
                      const std::vector<Eigen::VectorXd>& states,
                      const Eigen::MatrixXd& controls,
                      bool sample) {
  std::size_t n = states.size();
  std::vector<Trajectory> traj;
  traj.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    traj.emplace_back(
        simulate(dynamics, measurement, states.at(i), controls, sample));
  }
  return Trajectories(traj);
}

Trajectories simulate(DynamicsModel& dynamics,
                      MeasurementModel& measurement,
                      const std::vector<Eigen::VectorXd>& states,
                      const std::vector<Eigen::MatrixXd>& controls,
                      bool sample) {
  std::size_t n = states.size();
  assert(n == controls.size());
  std::vector<Trajectory> traj;
  traj.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    traj.emplace_back(
        simulate(dynamics, measurement, states.at(i), controls.at(i), sample));
  }
  return Trajectories(traj);
}

}  // namespace sia
