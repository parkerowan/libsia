# Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
# Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10.2)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(SET CMP0054 NEW)

# Command line options
option(TEST_COVERAGE "Generate a make coverage command" OFF)
option(BUILD_EXAMPLES "Builds the examples" ON)
option(PYTHON_BINDINGS "Make the python bindings" OFF)
option(SPHINX_DOCS "Make the sphinx documentation" ON)

# CTest
enable_testing()

# Get the project version info
file(READ "version.txt" ver)
string(REGEX MATCH "VERSION_MAJOR ([0-9]*)" _ ${ver})
set(VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_MINOR ([0-9]*)" _ ${ver})
set(VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "VERSION_PATCH ([0-9]*)" _ ${ver})
set(VERSION_PATCH ${CMAKE_MATCH_1})

# Set the project info
set(PROJ_NAME sia)
set(PROJ_VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")

# Create the project
project(${PROJ_NAME} 
  LANGUAGES C CXX
  VERSION ${PROJ_VERSION}
)

# Add compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set build type
if(TEST_COVERAGE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type set to ${CMAKE_BUILD_TYPE}")

# ------------------------------------------------------------------------------
# CMake modules
# ------------------------------------------------------------------------------
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(Functions)

if(TEST_COVERAGE)
  include(CodeCoverage)
  APPEND_COVERAGE_COMPILER_FLAGS()
endif()

# ------------------------------------------------------------------------------
# Find depdendencies
# ------------------------------------------------------------------------------
find_package(Eigen3 3.3.4 REQUIRED)
find_package(Glog 0.4.0 REQUIRED)
find_package(GTest REQUIRED)
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(pybind11 2.6.1 REQUIRED)

message(STATUS "EIGEN3_INCLUDE_DIR = ${EIGEN3_INCLUDE_DIR}")
message(STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
message(STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")

# Dependency include directories
include_directories(
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIR}
)

# ------------------------------------------------------------------------------
# Add the library targets
# ------------------------------------------------------------------------------
add_subdirectory(sia)

# ------------------------------------------------------------------------------
# Add the python bindings
# ------------------------------------------------------------------------------
if(PYTHON_BINDINGS)
  add_subdirectory(python)
endif()

# ------------------------------------------------------------------------------
# Add the test targets
# ------------------------------------------------------------------------------
add_subdirectory(tests)

# ------------------------------------------------------------------------------
# Add the example targets
# ------------------------------------------------------------------------------
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ------------------------------------------------------------------------------
# Setup the coverage reporting
# ------------------------------------------------------------------------------
if(TEST_COVERAGE)
  set(COVERAGE_GCOVR_EXCLUDES
    ${PROJECT_SOURCE_DIR}/examples
    ${PROJECT_SOURCE_DIR}/python
  )

  SETUP_TARGET_FOR_COVERAGE_GCOVR_HTML(
    NAME coverage
    EXECUTABLE ctest
    DEPENDENCIES sia
  )
endif()

# ------------------------------------------------------------------------------
# Setup the Sphinx documentation custom target
# ------------------------------------------------------------------------------
if(SPHINX_DOCS)
  add_custom_target(docs
    COMMAND sphinx-build -b html . public
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
endif()
