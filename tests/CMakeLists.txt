# Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
# Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

# ------------------------------------------------------------------------------
# Test target
# ------------------------------------------------------------------------------
set(test_target "test-sia")
set(test_deps "sia")
set(test_libs "")
set(test_source
  helpers.h
  helpers.cpp
  test_belief.cpp
  test_estimators.cpp
  test_math.cpp
  test_models.cpp
  test_runner.cpp
)
FUNCTIONS_CREATE_CPP_TEST(
  TARGET "${test_target}"
  DEPENDENCIES ${test_deps}
  LIBRARIES ${test_libs}
  SOURCES ${test_source}
)
