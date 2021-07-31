/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/runner/runner.h"

#include <glog/logging.h>

namespace sia {

Runner::Runner(const EstimatorMap& estimators, std::size_t buffer_size)
    : m_estimators(estimators), m_recorder(buffer_size) {}

void Runner::reset() {
  m_recorder.reset();
}

void Runner::estimate(const Eigen::VectorXd& observation,
                      const Eigen::VectorXd& control) {
  // Record the observation and control
  m_recorder.record(m_recorder.observation, observation);
  m_recorder.record(m_recorder.control, control);

  // Step each estimator and add the state
  for (const auto& it : m_estimators) {
    const std::string& name = it.first;
    Estimator& estimator = it.second;
    const Distribution& b = estimator.estimate(observation, control);

    // Record the belief
    m_recorder.record(m_recorder.estimate_mean[name], b.mean());
    m_recorder.record(m_recorder.estimate_mode[name], b.mode());
    m_recorder.record(m_recorder.estimate_var[name], b.covariance().diagonal());
  }

  // Clear the reset flag
  if (m_recorder.m_first_pass) {
    m_recorder.m_first_pass = false;
  }
}

const Eigen::VectorXd Runner::stepAndEstimate(MarkovProcess& system,
                                              const Eigen::VectorXd& state,
                                              const Eigen::VectorXd& control) {
  // Propogate the state and take a measurement
  const Eigen::VectorXd x = system.dynamics(state, control).sample();
  const Eigen::VectorXd observation = system.measurement(x).sample();

  // Record the state
  m_recorder.record(m_recorder.state, x);

  // Run the estimators
  estimate(observation, control);
  return x;
}

Recorder& Runner::recorder() {
  return m_recorder;
}

}  // namespace sia
