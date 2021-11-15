# Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
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
  belief/dirichlet.h
  belief/distribution.h
  belief/gaussian.h
  belief/gmm.h
  belief/gmr.h
  belief/gpc.h
  belief/gpr.h
  common/exception.h
  belief/helpers.h
  belief/kernel_density.h
  belief/particles.h
  belief/uniform.h
  controllers/controllers.h
  controllers/cost.h
  controllers/ilqr.h
  controllers/lqr.h
  controllers/mppi.h
  estimators/estimators.h
  estimators/ekf.h
  estimators/kf.h
  estimators/pf.h
  math/math.h
  math/math_impl.h
  models/linear_gaussian.h
  models/models.h
  models/nonlinear_gaussian.h
  models/simulate.h
  optimizers/bayesian_optimizer.h
  optimizers/gradient_descent.h
  runner/buffer.h
  runner/recorder.h
  runner/runner.h
)

set(lib_source
  belief/dirichlet.cpp
  belief/distribution.cpp
  belief/gaussian.cpp
  belief/gmm.cpp
  belief/gmr.cpp
  belief/gpc.cpp
  belief/gpr.cpp
  belief/helpers.cpp
  belief/kernel_density.cpp
  belief/particles.cpp
  belief/uniform.cpp
  controllers/cost.cpp
  controllers/ilqr.cpp
  controllers/lqr.cpp
  controllers/mppi.cpp
  estimators/ekf.cpp
  estimators/kf.cpp
  estimators/pf.cpp
  math/math.cpp
  models/linear_gaussian.cpp
  models/models.cpp
  models/nonlinear_gaussian.cpp
  models/simulate.cpp
  optimizers/bayesian_optimizer.cpp
  optimizers/gradient_descent.cpp
  runner/buffer.cpp
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
