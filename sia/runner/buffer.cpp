/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/runner/buffer.h"

#include <glog/logging.h>

namespace sia {

Buffer::Buffer(std::size_t dimension, std::size_t buffer_length)
    : m_buffer_data(Eigen::MatrixXd::Zero(dimension, buffer_length)),
      m_buffer_length(buffer_length) {}

void Buffer::reset() {
  m_first_pass = true;
}

bool Buffer::record(const Eigen::VectorXd& snapshot) {
  if (snapshot.size() != m_buffer_data.rows()) {
    LOG(WARNING) << "Cannot record snapshot, size " << snapshot.size()
                 << " does not match buffer dimension " << m_buffer_data.rows();
    return false;
  }

  if (m_first_pass) {
    // If first time running after reset, set the buffer to current value
    m_buffer_data = snapshot.replicate(1, m_buffer_length);
    m_first_pass = false;
  } else {
    // Shift the buffer to the left one index and append the new vector
    m_buffer_data.rightCols(m_buffer_length - 1)
        .swap(m_buffer_data.leftCols(m_buffer_length - 1));
    m_buffer_data.col(m_buffer_length - 1) = snapshot;
  }

  return true;
}

const Eigen::VectorXd Buffer::previous(const std::size_t index) const {
  return m_buffer_data.col(length() - (index + 1));
}

const Eigen::VectorXd Buffer::future(const std::size_t index) const {
  return m_buffer_data.col(index);
}

const Eigen::MatrixXd& Buffer::data() const {
  return m_buffer_data;
}

std::size_t Buffer::length() const {
  return m_buffer_length;
}

std::size_t Buffer::dimension() const {
  return m_buffer_data.rows();
}

}  // namespace sia
