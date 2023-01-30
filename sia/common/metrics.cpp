/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/common/metrics.h"

namespace sia {

unsigned getElapsedUs(BaseMetrics::steady_clock::time_point tic,
                      BaseMetrics::steady_clock::time_point toc) {
  return std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
      .count();
};

BaseMetrics::BaseMetrics() : m_tic(BaseMetrics::steady_clock::now()) {}

unsigned BaseMetrics::clockElapsedUs() {
  auto toc = BaseMetrics::steady_clock::now();
  elapsed_us = getElapsedUs(m_tic, toc);
  return elapsed_us;
}

}  // namespace sia
