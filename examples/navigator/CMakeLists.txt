# Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
# Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

# ------------------------------------------------------------------------------
# Example target
# ------------------------------------------------------------------------------
set(example_target "example-navigator")
set(example_deps "sia")
set(example_libs "")
set(example_source
  navigator.cpp
)
FUNCTIONS_CREATE_CPP_EXE(
  TARGET "${example_target}"
  DEPENDENCIES ${example_deps}
  LIBRARIES ${example_libs}
  SOURCES ${example_source}
)
