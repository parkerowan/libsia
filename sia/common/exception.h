/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <stdexcept>

namespace sia {

/// Throws an exception with the provided message
#define SIA_THROW(msg) throw std::runtime_error(msg);

/// Throws an exception with the provided message if evaluation is true
#define SIA_THROW_IF(evaluation, msg) \
  if (evaluation) {                   \
    SIA_THROW(msg);                   \
  }

/// Throws an exception with the provided message if evaluation is false
#define SIA_THROW_IF_NOT(evaluation, msg) SIA_THROW_IF(!(evaluation), msg)

}  // namespace sia
