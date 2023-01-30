/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <chrono>

namespace sia {

/// A base class for algorithm metrics.
class BaseMetrics {
 public:
  using steady_clock = std::chrono::steady_clock;

  /// Creating the metrics object starts the clock
  BaseMetrics();
  virtual ~BaseMetrics() = default;

  /// Computes the elapsed_us from when the object was constructed
  unsigned clockElapsedUs();
  unsigned elapsed_us{0};

 private:
  steady_clock::time_point m_tic;
};

}  // namespace sia
