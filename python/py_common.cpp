/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_common.h"

// Define module
void export_py_common(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::LoggerInterface, PyLoggerInterface>(m, "LoggerInterface")
      .def(py::init<>())
      .def("debug", &sia::LoggerInterface::debug, py::arg("msg"))
      .def("info", &sia::LoggerInterface::info, py::arg("msg"))
      .def("warn", &sia::LoggerInterface::warn, py::arg("msg"))
      .def("error", &sia::LoggerInterface::error, py::arg("msg"))
      .def("critical", &sia::LoggerInterface::critical, py::arg("msg"));

  py::class_<sia::Logger>(m, "Logger")
      .def_static("setCustomLogger", &sia::Logger::setCustomLogger,
                  py::arg("interface"))
      .def_static("debug", &sia::Logger::debug, py::arg("msg"))
      .def_static("info", &sia::Logger::info, py::arg("msg"))
      .def_static("warn", &sia::Logger::warn, py::arg("msg"))
      .def_static("error", &sia::Logger::error, py::arg("msg"))
      .def_static("critical", &sia::Logger::critical, py::arg("msg"))
}
