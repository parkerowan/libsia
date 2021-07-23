/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_runner.h"

// Define module
void export_py_runner(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::Recorder>(m, "Recorder")
      .def("reset", &sia::Recorder::reset)
      .def("getObservations", &sia::Recorder::getObservations)
      .def("getControls", &sia::Recorder::getControls)
      .def("getStates", &sia::Recorder::getStates)
      .def("getEstimateMeans", &sia::Recorder::getEstimateMeans)
      .def("getEstimateModes", &sia::Recorder::getEstimateModes)
      .def("getEstimateVariances", &sia::Recorder::getEstimateVariances);

  py::class_<sia::Runner>(m, "Runner")
      .def(py::init<sia::EstimatorMap&, std::size_t>(), py::arg("estimators"),
           py::arg("buffer_size"))
      .def("reset", &sia::Runner::reset)
      .def("estimate", &sia::Runner::estimate, py::arg("observation"),
           py::arg("control"))
      .def("stepAndEstimate", &sia::Runner::stepAndEstimate,
           py::arg("dynamics"), py::arg("measurement"), py::arg("state"),
           py::arg("control"))
      .def("recorder", &sia::Runner::recorder,
           py::return_value_policy::reference_internal);

  py::class_<sia::Buffer>(m, "Buffer")
      .def(py::init<std::size_t, std::size_t>(), py::arg("dimension"),
           py::arg("buffer_length"))
      .def("reset", &sia::Buffer::reset)
      .def("record", &sia::Buffer::record, py::arg("snapshot"))
      .def("previous", &sia::Buffer::previous, py::arg("index"))
      .def("future", &sia::Buffer::future, py::arg("index"))
      .def("data", &sia::Buffer::data)
      .def("length", &sia::Buffer::length)
      .def("dimension", &sia::Buffer::dimension);
}
