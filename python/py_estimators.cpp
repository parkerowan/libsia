/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_estimators.h"

// Define module
void export_py_estimators(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::Estimator, PyRecursiveBayesEstimator>(m, "Estimator")
      .def("getBelief", &sia::Estimator::getBelief)
      .def("estimate", &sia::Estimator::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::Estimator::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::Estimator::correct, py::arg("observation"),
           py::return_value_policy::reference_internal);

  py::class_<sia::KF, sia::Estimator>(m, "KF")
      .def(py::init<sia::LinearGaussian&, const sia::Gaussian&>(),
           py::arg("system"), py::arg("state"))
      .def("reset", &sia::KF::reset, py::arg("state"))
      .def("getBelief", &sia::KF::getBelief)
      .def("estimate", &sia::KF::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::KF::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::KF::correct, py::arg("observation"),
           py::return_value_policy::reference_internal);

  py::class_<sia::EKF, sia::Estimator>(m, "EKF")
      .def(py::init<sia::NonlinearGaussian&, const sia::Gaussian&>(),
           py::arg("system"), py::arg("state"))
      .def("reset", &sia::EKF::reset, py::arg("state"))
      .def("getBelief", &sia::EKF::getBelief)
      .def("estimate", &sia::EKF::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::EKF::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::EKF::correct, py::arg("observation"),
           py::return_value_policy::reference_internal);

  py::class_<sia::PF, sia::Estimator>(m, "PF")
      .def(py::init<sia::MarkovProcess&, const sia::Particles&, double,
                    double>(),
           py::arg("system"), py::arg("particles"),
           py::arg("resample_threshold") = 1, py::arg("roughening_factor") = 0)
      .def("reset", &sia::PF::reset, py::arg("particles"))
      .def("getBelief", &sia::PF::getBelief)
      .def("estimate", &sia::PF::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::PF::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::PF::correct, py::arg("observation"),
           py::return_value_policy::reference_internal);
}
