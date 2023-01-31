# Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
# Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

# ------------------------------------------------------------------------------
# Library target
# ------------------------------------------------------------------------------
set(lib_target "sia")

set(lib_libs "")

file (GLOB lib_headers
  "*.h"
  "belief/*.h"
  "common/*.h"
  "controllers/*.h"
  "estimators/*.h"
  "math/*.h"
  "models/*.h"
  "optimizers/*.h"
)

file (GLOB lib_sources
  "belief/*.cpp"
  "common/*.cpp"
  "controllers/*.cpp"
  "estimators/*.cpp"
  "math/*.cpp"
  "models/*.cpp"
  "optimizers/*.cpp"
)

FUNCTIONS_CREATE_CPP_SHARED_LIB(
  TARGET ${lib_target}
  LIBRARIES ${lib_libs}
  HEADERS ${lib_headers}
  SOURCES ${lib_sources}
  HEADER_DEST ${PROJ_NAME}
)
