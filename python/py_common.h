/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include <pybind11/pybind11.h>

#include "sia/common/logger.h

namespace py = pybind11;

/// LoggerInterface trampoline class
class PyLoggerInterface : public sia::LoggerInterface {
 public:
  // Inherit the constructors
  using sia::LoggerInterface::LoggerInterface;

  // Trampoline (need one for each virtual function)
  void debug(const std::string& msg) const override {
    PYBIND11_OVERRIDE_PURE(void, sia::LoggerInterface, debug, msg);
  }

  // Trampoline (need one for each virtual function)
  void info(const std::string& msg) const override {
    PYBIND11_OVERRIDE_PURE(void, sia::LoggerInterface, info, msg);
  }

  // Trampoline (need one for each virtual function)
  void warn(const std::string& msg) const override {
    PYBIND11_OVERRIDE_PURE(void, sia::LoggerInterface, warn, msg);
  }

  // Trampoline (need one for each virtual function)
  void error(const std::string& msg) const override {
    PYBIND11_OVERRIDE_PURE(void, sia::LoggerInterface, error, msg);
  }

  // Trampoline (need one for each virtual function)
  void critical(const std::string& msg) const override {
    PYBIND11_OVERRIDE_PURE(void, sia::LoggerInterface, critical, msg);
  }
};

// Define module
void export_py_common(py::module& m_sup);
