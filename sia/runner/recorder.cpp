/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/runner/recorder.h"

#include <glog/logging.h>

namespace sia {

Recorder::Recorder(std::size_t buffer_size) : m_buffer_size(buffer_size) {}

void Recorder::record(Eigen::MatrixXd& X, const Eigen::VectorXd& x) const {
  // If first time running after reset, set the buffer to current value
  if (m_first_pass) {
    X = x.replicate(1, m_buffer_size);
  }

  // Shift the buffer to the left one index and append the new vector
  X.rightCols(m_buffer_size - 1).swap(X.leftCols(m_buffer_size - 1));
  X.col(m_buffer_size - 1) = x;
}

void Recorder::reset() {
  m_first_pass = true;
}

const Eigen::MatrixXd& Recorder::getObservations() const {
  return observation;
}

const Eigen::MatrixXd& Recorder::getControls() const {
  return control;
}

const Eigen::MatrixXd& Recorder::getStates() const {
  return state;
}

const Eigen::MatrixXd& Recorder::getEstimateMeans(const std::string& name) {
  return estimate_mean[name];
}

const Eigen::MatrixXd& Recorder::getEstimateModes(const std::string& name) {
  return estimate_mode[name];
}

const Eigen::MatrixXd& Recorder::getEstimateVariances(const std::string& name) {
  return estimate_var[name];
}

}  // namespace sia
