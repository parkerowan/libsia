/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <stdexcept>

#include <glog/logging.h>

namespace sia {

/// Throws an exception with the provided message if evaluation is false
#define SIA_EXCEPTION(evaluation, msg)                     \
  if (!(evaluation)) {                                     \
    std::string what = std::string("SIA Runtime: ") + msg; \
    LOG(ERROR) << what;                                    \
    throw std::runtime_error(what);                        \
  }

}  // namespace sia
