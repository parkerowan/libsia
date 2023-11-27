// Copyright (c) 2023 Parker Owan.  All rights reserved.

#pragma once

#include <sstream>

namespace sia {

/// Formats an expression inplace using stringstream
#define SIA_FMT(expression) \
  static_cast<const std::stringstream&>(std::stringstream() << expression).str()

}  // namespace sia
