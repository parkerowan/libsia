# Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
# Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

# ------------------------------------------------------------------------------
# Python target
# ------------------------------------------------------------------------------
set(py_module "pysia")
set(py_libs
  "sia"
)
set(py_sources
  py_belief.cpp
  py_controllers.cpp
  py_estimators.cpp
  py_math.cpp
  py_models.cpp
  py_optimizers.cpp
  py_runner.cpp
  pysia.cpp
)
FUNCTIONS_CREATE_PYBIND11_MODULE(
  MODULE ${py_module}
  LIBRARIES ${py_libs}
  SOURCES ${py_sources}
)
