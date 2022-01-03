/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <Eigen/Dense>

namespace sia {

/// Trajectory buffer used to record rollouts and replay.  Each snapshot in time
/// is a vector that is pushed into a data matrix.  Old values are popped.  The
/// buffer matrix is initialized to zero.
class Buffer {
 public:
  /// Initializes a buffer with the snapshot dimension and buffer length
  explicit Buffer(std::size_t dimension, std::size_t buffer_length);
  virtual ~Buffer() = default;

  void reset();
  bool record(const Eigen::VectorXd& snapshot);

  /// Returns a snapshot for a given index, where 0 is the most recent and
  /// increments to oldest snapshot.
  const Eigen::VectorXd previous(std::size_t index) const;

  /// Returns a snapshot for a given index, where 0 is the oldest and increments
  /// to the most recent snapshot.
  const Eigen::VectorXd future(std::size_t index) const;

  const Eigen::MatrixXd& data() const;
  std::size_t length() const;
  std::size_t dimension() const;

 private:
  // M x N, where M is snapshot dimension and N is buffer length
  Eigen::MatrixXd m_buffer_data;
  bool m_first_pass{true};
  std::size_t m_buffer_length;
};

}  // namespace sia
