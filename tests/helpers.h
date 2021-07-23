/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <sia/sia.h>

// Create a 1st order low pass filter system for test
sia::LinearGaussianDynamics createTestDynamics();

// Create a 1st order integrator system for test
sia::LinearGaussianDynamics createIntegratorDynamics();

// Create a 1st order direct measurement system for test
sia::LinearGaussianMeasurement createTestMeasurement();

// Create a quadratic cost for test
sia::QuadraticCost createTestCost();
