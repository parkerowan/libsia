/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_common.h"

// Define module
void export_py_common(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::Logger>(m, "Logger")
      .def_static("debug", &sia::Logger::debug, py::arg("msg"))
      .def_static("info", &sia::Logger::info, py::arg("msg"))
      .def_static("warn", &sia::Logger::warn, py::arg("msg"))
      .def_static("error", &sia::Logger::error, py::arg("msg"))
      .def_static("critical", &sia::Logger::critical, py::arg("msg"));

  py::class_<sia::BaseMetrics>(m, "BaseMetrics")
      .def(py::init<>())
      .def("clockElapsedUs", &sia::BaseMetrics::clockElapsedUs)
      .def_readwrite("elapsed_us", &sia::BaseMetrics::elapsed_us);
}
