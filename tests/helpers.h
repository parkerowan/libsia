/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <sia/sia.h>

// Create a 1st order low pass filter system for test
sia::LinearGaussian createTestSystem();
