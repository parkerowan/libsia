/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
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
      .def("stepAndEstimate", &sia::Runner::stepAndEstimate, py::arg("system"),
           py::arg("state"), py::arg("control"))
      .def("recorder", &sia::Runner::recorder,
           py::return_value_policy::reference_internal);
}
