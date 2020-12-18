/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/estimators/estimators.h"

#include <Eigen/Dense>
#include <map>

namespace sia {

/// Map of estimators
using EstimatorMap = std::map<std::string, RecursiveBayesEstimator&>;

// Forward declaration
class Runner;

/// Buffer for recording estimator traces.  Records added to the buffer are
/// accesible via the get<Name>() routines as M x N matrices where M is the
/// vector dimension and N is the buffer length.  Matrices are padded with
/// initial conditions on the first pass.  For estimates, the estimator name is
/// needed.  If it is not found, an empty matrix is returned.
class Recorder {
  friend class Runner;

 public:
  void reset();
  const Eigen::MatrixXd& getObservations() const;
  const Eigen::MatrixXd& getControls() const;
  const Eigen::MatrixXd& getStates() const;
  const Eigen::MatrixXd& getEstimateMeans(const std::string& name);
  const Eigen::MatrixXd& getEstimateModes(const std::string& name);
  const Eigen::MatrixXd& getEstimateVariances(const std::string& name);

 protected:
  explicit Recorder(std::size_t buffer_size);
  void record(Eigen::MatrixXd& X, const Eigen::VectorXd& x) const;

  /// Each matrix here is M x N, where M is signal dim and N is buffer dim
  Eigen::MatrixXd observation;
  Eigen::MatrixXd control;
  Eigen::MatrixXd state;
  std::map<std::string, Eigen::MatrixXd> estimate_mean;
  std::map<std::string, Eigen::MatrixXd> estimate_mode;
  std::map<std::string, Eigen::MatrixXd> estimate_var;
  std::size_t m_buffer_size;
  bool m_first_pass{true};
};

}  // namespace sia
