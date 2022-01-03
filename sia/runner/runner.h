/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/estimators/estimators.h"
#include "sia/models/models.h"
#include "sia/runner/recorder.h"

#include <Eigen/Dense>

namespace sia {

/// Steps the estimators and records the measurements, control and estimates
/// in the recorder.  Optionally steps the system and records the ground truth
/// state for development purposes.
class Runner {
 public:
  explicit Runner(const EstimatorMap& estimators, std::size_t buffer_size);
  virtual ~Runner() = default;
  void reset();

  /// Steps the estimators and appends the results to the buffer.
  void estimate(const Eigen::VectorXd& observation,
                const Eigen::VectorXd& control);

  /// Steps the dynamics to generate a ground truth state and observation.
  /// Calls estimate() internally.
  const Eigen::VectorXd stepAndEstimate(DynamicsModel& dynamics,
                                        MeasurementModel& measurement,
                                        const Eigen::VectorXd& state,
                                        const Eigen::VectorXd& control);

  /// Returns the internal reference to the trace recorder.
  Recorder& recorder();

 private:
  EstimatorMap m_estimators;
  Recorder m_recorder;
};

}  // namespace sia
