# Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
# Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

# ------------------------------------------------------------------------------
# Library target
# ------------------------------------------------------------------------------
set(lib_target "sia")
set(lib_libs
  glog
)
set(lib_header
  sia.h
  belief/distribution.h
  belief/gaussian.h
  belief/helpers.h
  belief/kernel_density.h
  belief/particles.h
  belief/uniform.h
  estimators/estimators.h
  estimators/extended_kalman_filter.h
  estimators/kalman_filter.h
  estimators/particle_filter.h
  math/math.h
  math/math_impl.h
  models/linear_gaussian.h
  models/linear_gaussian_ct.h
  models/models.h
  models/nonlinear_gaussian.h
  models/nonlinear_gaussian_ct.h
  runner/recorder.h
  runner/runner.h
)
set(lib_source
  belief/distribution.cpp
  belief/gaussian.cpp
  belief/helpers.cpp
  belief/kernel_density.cpp
  belief/particles.cpp
  belief/uniform.cpp
  estimators/extended_kalman_filter.cpp
  estimators/kalman_filter.cpp
  estimators/particle_filter.cpp
  math/math.cpp
  models/linear_gaussian.cpp
  models/linear_gaussian_ct.cpp
  models/nonlinear_gaussian.cpp
  models/nonlinear_gaussian_ct.cpp
  runner/recorder.cpp
  runner/runner.cpp
)
FUNCTIONS_CREATE_CPP_SHARED_LIB(
  TARGET ${lib_target}
  LIBRARIES ${lib_libs}
  HEADERS ${lib_header}
  SOURCES ${lib_source}
  HEADER_DEST ${PROJ_NAME}
)
